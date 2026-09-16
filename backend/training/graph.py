"""節點圖執行引擎：解析 graph_spec（節點 + 各節點 inputs 引用其他節點 id），
拓樸排序後依序執行 Feature Node / Label Node / Model Node。

data_loader 用依賴注入的方式傳進來（一個 `(timeframe) -> DataFrame` 的函式），
這個模組完全不知道資料是從 DB 撈出來的還是本機測試用的假資料，方便本機輕量測試。

時間戳對齊（取代舊版「陣列長度一樣就假設對齊」的做法）：
每個節點的輸出除了數值陣列本身，這裡另外維護一份平行的 `timestamps` 字典——
Feature/Label Node 的時間戳就是它所屬 `df` 的（排序、去重後的）DatetimeIndex；
Model Node 融合多個上游輸入時，改成用「時間戳交集」決定要用哪些列，而不是
「從尾端裁到最短長度」——後者只在剛好沒有缺棒、所有上游算到同一根最後K棒時才是對的，
只要中間有一根被丟掉（缺棒/去重），位置對齊就會悄悄對錯而不會報錯。時間戳交集不管
中間有沒有缺棒都能對到正確的那一列。

Model Node 接 Model Node（多模型融合）：
一個 Model Node 的 `inputs` 可以是 Feature Node，也可以是另一個 Model Node（把上游模型的
預測結果當成下游模型的輸入特徵）——每個 Model Node 訓練完，除了回傳 metrics/predict 之外，
也會對自己訓練用的「全範圍」窗格跑一次預測，存進 outputs[nid] 當一個 (n, 1) 的特徵陣列，
下游節點就能像接 Feature Node 一樣接它，不用特別判斷上游是什麼型態；這個輸出也有自己對應
的 timestamps（每一列窗格對應的是「窗格內最後一根K棒」的時間）。

Label Node 拆成 Outcome（算原始數值，見 registry/outcomes.py）+ Labeling Rule（把原始數值
轉成學習目標，見 registry/labeling_rules.py）兩個獨立、可自由組合的積木，不是像舊版
`nbar_return` 那樣把「算報酬率」跟「切成幾類」焊在同一個 function 裡——這樣同一個 outcome
（例如 simple_return）以後可以配不同的 labeling_rule（分類門檻、或直接回歸），也能加
「未來價格」這種新 outcome，不用重寫整個 function。Labeling Rule 會決定 `task_type`
（"classification" 或 "regression"），Model Node 依這個決定要用哪種 loss/輸出頭。

尚未處理：跨時間框架融合（例如 15m 模型接 1h 模型）還沒做重採樣對齊，時間戳交集在
「兩邊時間戳完全不同尺度」的情況下會直接交集出空集合而報錯，不會算出錯的結果，但也還
不能真的融合——這個要做時間框架之間的重採樣規則，這裡先不擅自決定怎麼做。

已知缺口（尚未實作，見 docs/record/archive/2026-09-10-training-v2-redesign-plan.md）：
  - 正規化尚未接（教授要求「val/test 只能用 train 的統計量做 transform，不能重新 fit」這條還沒做進來）
  - train/val 目前仍是隨機切分，教授的三階段協定（Phase 1 random / Phase 2 chronological /
    Phase 3 holdout 推論）還沒做進來，這是下一輪的工作
  - 波動度門檻／分位數切分／成本導向／triple barrier 這些進階 labeling_rule 還沒做，目前只有
    fixed_threshold（分類）跟 identity（回歸）兩種
  - 上游模型接下游模型時，上游模型的「全範圍預測」是用同一個模型（在自己的 train 上 fit 過）
    直接對自己的 train 段也預測一次，這對下游來說等於看到了「上游模型對 train 段的過擬合結果」，
    嚴謹做法應該是 OOF（out-of-fold）預測，這裡先不做，是個已知的資料洩漏風險，留給 Hermes 判斷
這些先不擅自決定怎麼補，留在 checklist 標記給 Hermes。
"""

import time
from typing import Callable

import numpy as np
import pandas as pd

from training.registry.architectures import train_model
from training.registry.features import compute_feature
from training.registry.labeling_rules import apply_labeling_rule, validate_combination
from training.registry.outcomes import compute_outcome


class LazyWindowed:
    """訓練/驗證集不整包攤開存在記憶體——只留正規化過、還沒展開窗格的小陣列
    （`base`，形狀是 n×特徵數，不管資料筆數多大都只有幾十~幾百 MB），每次被索引
    （單一個或一整批）才現切現算出對應的窗格，用完即丟。

    這是這次 NAS 上一連串 OOM 崩潰（`dmesg` 實測 anon-rss ~15GB）的根本解法：切完窗格後
    train_idx（80%）+val_idx（20%）兩個子集加起來剛好等於全部資料，不管怎麼分、怎麼省
    中間的複製，只要「一次性攤開整包存在記憶體」這個設計不變，尖峰用量就跟資料總量
    死死綁在一起（這批 7 特徵/16 欄/window=60 的資料算出來就是 ~10.8GB）。改成這樣之後，
    尖峰記憶體只跟 batch_size 有關，不會再隨資料筆數等比例成長。

    只用在 LSTM（PyTorch 训练迴圈本來就是一個 batch 一個 batch 跑，天生適合這樣做）；
    XGBoost 的 sklearn API 本來就要求真正的 2D 陣列，沒辦法做成這樣，維持原本的作法，
    這條路徑目前還是有同樣的潛在風險，先不動，留著之後有需要再處理。
    """

    def __init__(self, base: "np.ndarray", sample_idx: "np.ndarray", window: int):
        self.base = base
        self.sample_idx = sample_idx
        self.window = window

    @property
    def shape(self):
        return (len(self.sample_idx), self.window, self.base.shape[-1])

    def __len__(self):
        return len(self.sample_idx)

    def __getitem__(self, local_idx):
        starts = self.sample_idx[local_idx]
        if np.ndim(starts) == 0:
            s = int(starts)
            return self.base[s:s + self.window]
        return np.stack([self.base[s:s + self.window] for s in starts])


def _topo_order(nodes: dict) -> list[str]:
    visited: set[str] = set()
    order: list[str] = []

    def visit(nid: str, stack: frozenset[str]):
        if nid in visited:
            return
        if nid in stack:
            raise ValueError(f"graph 有循環依賴：{nid}")
        node = nodes.get(nid)
        if node is None:
            raise ValueError(f"找不到節點：{nid}")
        for dep in node.get("inputs", []) or []:
            visit(dep, stack | {nid})
        if node.get("label"):
            visit(node["label"], stack | {nid})
        visited.add(nid)
        order.append(nid)

    for nid in nodes:
        visit(nid, frozenset())
    return order


def _ensure_time_indexed(df: "pd.DataFrame") -> "pd.DataFrame":
    """確保 df 有排序好、去重的 DatetimeIndex，當作這個時間框架的『真實時間戳』基準。
    同一根K棒可能因為資料管線重跑被寫入多次，這裡保留每個時間點最後一筆
    （跟 services/backtest_demo.py 的 prepare_ohlcv 同一套處理，訓練這邊之前沒有做）。

    本機測試用的假資料如果沒有 datetime 欄位/索引，就退化成用原本的 RangeIndex——
    這種情況沒有真實時間戳可用，只能靠位置對齊，等於維持舊行為，僅供輕量測試。
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df[~df.index.duplicated(keep="last")].sort_index()
    if "datetime" in df.columns:
        d = df.copy()
        d["datetime"] = pd.to_datetime(d["datetime"])
        d = d.set_index("datetime")
        d = d[~d.index.duplicated(keep="last")]
        return d.sort_index()
    return df


PREVIEW_THROTTLE_SEC = 1.5  # 即時視窗預覽最多每這麼多秒送一筆，不是每個 batch/每個 epoch 都送


def run_graph(
    graph_spec: dict,
    data_loader: Callable[[str], "object"],
    on_epoch: Callable[[str, int, dict], None] | None = None,
    on_preview: Callable[[str, dict], None] | None = None,
) -> dict:
    nodes = {n["id"]: n for n in graph_spec["nodes"]}
    order = _topo_order(nodes)

    outputs: dict = {}
    timestamps: dict = {}
    label_task_types: dict = {}
    results: dict = {}

    for nid in order:
        node = nodes[nid]
        ntype = node["type"]

        if ntype == "feature":
            df = _ensure_time_indexed(data_loader(node["timeframe"]))
            outputs[nid] = compute_feature(node["key"], df, node.get("params"))
            timestamps[nid] = np.asarray(df.index.values)

        elif ntype == "label":
            df = _ensure_time_indexed(data_loader(node["timeframe"]))
            params = node.get("params") or {}
            outcome_key = node["outcome"]
            rule_key = node.get("labeling_rule", "fixed_threshold")
            validate_combination(outcome_key, rule_key)
            raw, valid_length = compute_outcome(outcome_key, df, params)
            y, task_type, _meta = apply_labeling_rule(rule_key, raw, valid_length, params)
            outputs[nid] = (y, valid_length)
            timestamps[nid] = np.asarray(df.index.values)
            label_task_types[nid] = task_type

        elif ntype == "model":
            input_ids = node["inputs"]
            input_arrays = [outputs[i] for i in input_ids]
            input_ts = [timestamps[i] for i in input_ids]

            y_raw, valid_length = outputs[node["label"]]
            label_ts = timestamps[node["label"]]
            task_type = label_task_types[node["label"]]

            # 用時間戳交集決定哪些列所有輸入 + Label 都有值可用（不是位置對齊）。
            # label_ts[:valid_length] 先排除 Label 尾端因為看不到未來而是 NaN 的那段。
            common_ts = label_ts[:valid_length]
            for ts in input_ts:
                common_ts = np.intersect1d(common_ts, ts)
            if len(common_ts) == 0:
                raise ValueError(f"節點 {nid}：所有輸入與 Label 沒有共同的時間戳，無法對齊（inputs={input_ids}）")

            def _select(arr: np.ndarray, ts: np.ndarray) -> np.ndarray:
                idx = np.searchsorted(ts, common_ts)
                return arr[idx]

            aligned = [_select(a, ts) for a, ts in zip(input_arrays, input_ts)]
            X_raw = aligned[0] if len(aligned) == 1 else np.concatenate(aligned, axis=1)
            y_common = _select(y_raw, label_ts)

            # TA-Lib 指標有暖機期（例如 RSI(14) 前 14 根算不出值，回傳 NaN），時間戳交集
            # 只對齊「有沒有這個時間點」，不會檢查數值本身是不是 NaN——這些暖機期的列如果被
            # 隨機切分抽進 train_idx，會讓底下算 min/max 正規化統計量時整組變成 NaN，
            # 存進 DB 時因為 JSON 不接受 NaN 直接整個訓練失敗（實測復現：NAS 上兩次真的這樣壞掉）。
            # 在切窗之前就把任何一個輸入特徵是 NaN 的列整排丟掉，從根本避免 NaN 流進任何一個窗格，
            # 不是切完窗再補救——數量對整個資料集而言是資料開頭那一小段，可忽略。
            valid_mask = ~np.isnan(X_raw).any(axis=1)
            if not valid_mask.all():
                common_ts = common_ts[valid_mask]
                X_raw = X_raw[valid_mask]
                y_common = y_common[valid_mask]

            # 切窗格不再用 Python 迴圈把 291 萬根 K 棒（1m 級資料常見量級）逐一疊進 list 再
            # np.asarray() ——這條路徑在 NAS 上實測直接把 DRAM 榨爆到被 kernel OOM killed，
            # `restart: always` 又把容器救回來，job 卻永遠卡在 running（process 是被系統強制砍的，
            # 不是 Python 例外，來不及寫回 failed 狀態）。改用 sliding_window_view 產生的是「view」
            # 不是新記憶體，切窗格本身幾乎不佔額外空間；真正需要整包搬進記憶體的地方延後到
            # 下面 X[train_idx]/X[val_idx] 那一刻才發生，而且是先在還沒展開窗格的 X_raw（形狀只有
            # n×特徵數，比窗格化後的 n×window×特徵數小 window 倍）上做完正規化，才展開成窗格，
            # 不是展開完窗格的大陣列才正規化——這是真正省記憶體的關鍵，不只是換個寫法。
            window = int(node.get("window", 60))
            n = len(common_ts)
            if n <= window:
                raise ValueError(f"節點 {nid}：資料筆數不足以組出任何一個 window={window} 的樣本")
            n_samples = n - window
            sample_ts = common_ts[window - 1:n - 1]
            y = y_common[window - 1:n - 1]
            y = np.asarray(y, dtype=(int if task_type == "classification" else float))

            # 每個樣本的「標籤成熟時間」：它的決策時刻在 common_ts 的位置往後數 horizon 格
            # （horizon 是「後 h 根實際K棒」，不是固定鐘錶時間，所以用位置往後數，
            # 不是用日期加 horizon 天——這是 outcome 節點在算 outcome 時，一開始就用同一種
            # 位置語意定義 horizon，這裡對齊回去才會一致）。超出範圍代表本來就在 valid_length
            # 之外，已經被前面的 NaN／common_ts 交集濾掉，clip 只是防呆不影響結果。
            label_node = nodes[node["label"]]
            horizon = int((label_node.get("params") or {}).get("horizon", 1))
            label_end_pos = np.clip(np.arange(window, n) - 1 + horizon, 0, len(common_ts) - 1)
            label_end_ts = common_ts[label_end_pos]

            split_strategy = node.get("split_strategy", "random")
            val_ratio = float(node.get("val_ratio", 0.2))
            val_size = max(1, int(n_samples * val_ratio))
            n_excluded_boundary = 0

            if split_strategy == "chronological":
                # Phase 2：不隨機打亂，val 固定是時間上最後一段；訓練樣本裡只要「標籤成熟時間」
                # 跨進了 val 開始的時刻，就代表這筆訓練樣本用到了 val 期間才會知道的資訊，排除掉
                # ——這是時間切分的通用規則，跟 horizon/threshold/模型設定本身無關。
                val_idx = np.arange(n_samples - val_size, n_samples)
                candidate_train_idx = np.arange(0, n_samples - val_size)
                val_start_ts = sample_ts[val_idx[0]]
                boundary_ok = label_end_ts[candidate_train_idx] < val_start_ts
                train_idx = candidate_train_idx[boundary_ok]
                n_excluded_boundary = int((~boundary_ok).sum())
            else:
                # Phase 1：隨機打亂切分（現有行為，保留不變，這是既有實驗結果的重現基準）。
                rng = np.random.default_rng(42)
                idx = rng.permutation(n_samples)
                val_idx, train_idx = idx[:val_size], idx[val_size:]

            if len(train_idx) == 0:
                raise ValueError(f"節點 {nid}：{split_strategy} 切分後訓練樣本數為 0（可能是 val_ratio 太大或 horizon 太長導致邊界排除掉太多樣本）")

            # 每一列預測值屬於 train／val／unused（因邊界排除而沒被用在訓練或驗證任何一邊）
            # 哪個來源，逐列標記，不是整個節點貼一個籠統的標籤——這是三方對齊時 Codex 特別要求的
            # 精確度（E 項）：上游全範圍預測混了 train 跟 val，下游／之後查看的人要能分清楚
            # 每一列實際上是不是真的樣本外，不能只憑一個 "in_sample" 字串打發。
            prediction_source = np.full(n_samples, "unused", dtype=object)
            prediction_source[train_idx] = "train"
            prediction_source[val_idx] = "val"

            # 資料標準化（MinMax，train-only fit）：教授要求「val/test 只能用 train 的統計量
            # 做 transform，不能重新 fit」——這裡只用「有被至少一個訓練窗格用到」的原始列算
            # min/max，套用到全部（含 val），不會讓 val 的分布資訊偷偷洩漏進正規化參數裡。
            # fit 出來的 min/scale 存進 preprocessing_state，連同模型產物一起保存，推論新資料時
            # 要套用同一組數字，不是重新算一次（training/inference.py 那邊會讀這個）。
            #
            # 這裡改成在 X_raw（還沒展開成窗格的小陣列）上算，不是在窗格化後的大陣列上算：
            # 樣本 j（訓練窗格）對應的原始列範圍是 [j, j+window)，用差分陣列＋累加一次算出
            # 「至少被一個訓練窗格用到」的原始列遮罩，是全向量化操作，跟窗格化後再攤平比
            # 數學上等價（重疊的列本來就會被同一組 min/max 涵蓋到），但完全不用碰那個大陣列。
            coverage = np.zeros(n + 1, dtype=np.int64)
            np.add.at(coverage, train_idx, 1)
            np.add.at(coverage, train_idx + window, -1)
            train_row_mask = np.cumsum(coverage[:-1]) > 0
            feat_min = X_raw[train_row_mask].min(axis=0)
            feat_max = X_raw[train_row_mask].max(axis=0)
            feat_scale = np.where(feat_max > feat_min, feat_max - feat_min, 1.0)

            # 正規化跟轉 float32（省一半記憶體）都在展開窗格之前做，窗格化只是用
            # sliding_window_view 開一個 view，不搬資料；真正的記憶體只在下面
            # X[train_idx]/X[val_idx] 那一刻，為了各自需要的那個子集才分配。
            X_raw_norm = ((X_raw - feat_min) / feat_scale).astype(np.float32, copy=False)
            X = np.moveaxis(np.lib.stride_tricks.sliding_window_view(X_raw_norm, window_shape=window, axis=0), -1, 1)
            X = X[:-1]  # 最後一個窗格會涵蓋到 common_ts 的最後一列，跟原本迴圈版本一樣刻意排除

            # 即時視窗預覽（依 Codex 最新指示重新設計）：不是固定一個樣本看 100 輪，而是
            # 「正在處理的 batch 抽一筆出來展示」，讓使用者看到訓練實際在跑哪些窗格——隨機
            # 訓練下日期本來就會跳動，這是正常現象，不是 bug。頻率限制在最多每
            # PREVIEW_THROTTLE_SEC 秒送一筆（不是每個 batch/每個 epoch 都送），架構層
            # （architectures.py）只負責把「這一批用了哪個本地索引、模型現在的預測是什麼」
            # 回報回來——這兩個數字都是訓練本來就會算出來的（batch 的輸入索引、forward pass
            # 的輸出），不需要為了預覽多做重運算；节流判斷跟組成完整展示資料（撈真實 K 線）
            # 都在這裡（graph.py），跟架構無關的部分不該讓 architectures.py 重複實作。
            # 這裡準備的 bars/實際答案只透過 on_preview 往外送給 UI/DB，完全不會被寫回
            # X_train_arg/y[train_idx] 這兩個真正餵給訓練的物件，不會流進模型輸入。
            price_tf = None
            for iid in input_ids:
                in_node = nodes.get(iid)
                if in_node is not None and in_node.get("type") == "feature":
                    price_tf = in_node["timeframe"]
                    break
            price_df = _ensure_time_indexed(data_loader(price_tf)) if price_tf is not None else None
            n_classes_for_preview = int((label_node.get("params") or {}).get("n_classes", 2)) if task_type == "classification" else None
            preview_labeling_rule = label_node.get("labeling_rule")
            _last_preview_ts = [0.0]

            def preview_hook(epoch: int, local_idx: int, predicted, _nid=nid, source: str = "train"):
                """architectures.py 每個 batch/每次迭代都可以呼叫這個，節流判斷在這裡做，
                大部分呼叫會在時間還沒到就直接跳過（成本只有一次 time.time() + 比較），
                真正要組資料、查 K 線、寫資料庫的重活只有節流通過時才會執行。任何一步
                失敗都不能讓訓練跟著死，這裡整個包一層 try/except。
                """
                if on_preview is None or price_df is None:
                    return
                now = time.time()
                if now - _last_preview_ts[0] < PREVIEW_THROTTLE_SEC:
                    return
                _last_preview_ts[0] = now
                try:
                    idx_arr = train_idx if source == "train" else val_idx
                    if local_idx >= len(idx_arr):
                        return
                    pos = int(idx_arr[local_idx])
                    window_ts = common_ts[pos:pos + window]
                    if len(window_ts) != window or not {"open", "high", "low", "close"}.issubset(price_df.columns):
                        return
                    bars_df = price_df.reindex(pd.DatetimeIndex(window_ts))
                    if bars_df[["open", "high", "low", "close"]].isna().any().any():
                        return
                    on_preview(_nid, {
                        "epoch": epoch,
                        "source": source,
                        "decision_ts": int(pd.Timestamp(common_ts[pos + window - 1]).timestamp()),
                        "target_ts": int(pd.Timestamp(label_end_ts[pos]).timestamp()),
                        "horizon": horizon,
                        "task_type": task_type,
                        "n_classes": n_classes_for_preview,
                        "labeling_rule": preview_labeling_rule,
                        "bars": [
                            {"time": int(pd.Timestamp(t).timestamp()), "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
                            for t, o, h, l, c in zip(window_ts, bars_df["open"], bars_df["high"], bars_df["low"], bars_df["close"])
                        ],
                        "actual": (int(y[pos]) if task_type == "classification" else float(y[pos])),
                        "predicted": predicted,
                    })
                except Exception:
                    pass  # 預覽失敗（例如剛好抽到邊界資料缺漏）安靜跳過，不影響訓練本身

            def _on_epoch(epoch, metrics, _nid=nid):
                if on_epoch:
                    on_epoch(_nid, epoch, metrics)

            # LSTM 用 LazyWindowed，不整包攤開訓練/驗證集（見類別上的說明，這是這次
            # OOM 崩潰的真正修法）；XGBoost 的 sklearn API 需要真正的 2D 陣列，維持原本
            # 整包攤開的作法，這條路徑目前還是有同樣的潛在風險，先不動。
            if node["key"] == "lstm":
                X_train_arg = LazyWindowed(X_raw_norm, train_idx, window)
                X_val_arg = LazyWindowed(X_raw_norm, val_idx, window)
            else:
                X_train_arg, X_val_arg = X[train_idx], X[val_idx]

            model_result = train_model(
                node["key"],
                X_train_arg, y[train_idx], X_val_arg, y[val_idx],
                node.get("params", {}), _on_epoch,
                task_type=task_type,
                preview_hook=preview_hook,
            )
            # training_meta 記錄實際切分方式跟排除原因，供之後 UI／job 紀錄查詢用，不影響訓練本身。
            model_result["training_meta"] = {
                "split_strategy": split_strategy,
                "n_train": len(train_idx), "n_val": len(val_idx),
                "n_excluded_boundary": n_excluded_boundary,
                "preprocessing": "minmax",
            }
            model_result["task_type"] = task_type
            model_result["prediction_source"] = prediction_source.tolist()
            model_result["preprocessing_state"] = {
                "method": "minmax", "min": feat_min.tolist(), "scale": feat_scale.tolist(),
            }
            results[nid] = model_result
            # 下游節點如果要接這個模型的輸出，用「對自己全範圍窗格跑一次預測」當特徵陣列，
            # 每一列對應 sample_ts 裡同一個位置的時間戳（該窗格最後一根K棒的時間）。
            # output_mode="probability" 只在分類時有意義（機率分佈）；回歸沒有 predict_proba，
            # 一律走預設的 predict() 分支，輸出就是預測的連續數值。
            if task_type == "classification" and node.get("output_mode") == "probability" and "predict_proba" in model_result:
                full_predictions = model_result["predict_proba"](X)
                outputs[nid] = np.asarray(full_predictions, dtype=float)
            else:
                full_predictions = model_result["predict"](X)
                outputs[nid] = np.asarray(full_predictions, dtype=float).reshape(-1, 1)
            timestamps[nid] = np.asarray(sample_ts)

        else:
            raise ValueError(f"未知節點類型：{ntype}")

    return results
