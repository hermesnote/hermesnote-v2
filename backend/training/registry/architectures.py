"""Model Node 登記表：輸入訓練/驗證陣列 + 超參數，訓練一個模型，過程中透過 callback 回報進度。

跟 features.py / outcomes.py / labeling_rules.py 同一套模式：函式登記進字典，用 key 查表呼叫。
每個函式簽名一致：`(X_train, y_train, X_val, y_val, params, on_epoch, task_type) -> dict`
  X_*        — (n_samples, window, n_features) 或 (n_samples, n_features)，依架構而定
  y_*        — (n_samples,)，task_type="classification" 時是類別索引，"regression" 時是連續數值
  params     — 該架構的超參數
  on_epoch   — 呼叫端（graph.py/worker.py）傳進來的 callback：on_epoch(epoch:int, metrics:dict)，
               每個 epoch/checkpoint 呼叫一次；這個函式本身完全不知道 DB/WebSocket 這些東西，
               只負責把訓練過程中的數字回報出去，職責分離。
  task_type  — "classification" 或 "regression"，決定用哪種 loss/輸出頭/評估指標；
               回歸沒有 predict_proba（機率的概念只對分類有意義）。
回傳 dict 至少含 {"final_metrics": {...}, "predict": callable}；另外還會帶
`weights`（bytes，供 model_artifacts 落地保存）跟 `model_config`（dict，重建模型結構要的資訊），
這兩個是 D 項「訓練完自動保存產物」要用的，訓練當下的即時進度/摘要不受影響。

LOAD_REGISTRY 是反過來的：給 `(weights: bytes, model_config: dict)` 重建出一個能直接
`predict`（分類另外有 `predict_proba`，多分類 XGBoost 目前只能給類別、還不能給機率，
見 `load_xgboost` 裡的說明）的物件，不重新訓練——這是「載入舊模型做推論」用的。
"""

import numpy as np

ARCHITECTURE_REGISTRY: dict = {}
LOAD_REGISTRY: dict = {}


def register(key: str):
    def deco(fn):
        ARCHITECTURE_REGISTRY[key] = fn
        return fn
    return deco


def register_loader(key: str):
    def deco(fn):
        LOAD_REGISTRY[key] = fn
        return fn
    return deco


class LSTMNet:
    """延遲定義：實際的 nn.Module 要在有 torch 時才能建立類別，
    用一個 factory function 包起來，訓練跟載入推論共用同一份結構定義，
    不會出現「訓練用一個 class、載入時重新定義另一個」導致 state_dict 對不起來的風險。

    結構比照教授教材（H_時序列模型理論與實作.pdf）的 Keras 範例：
    LSTM → Dropout → Dense(relu) → Dropout → 輸出層——不是只有 LSTM 接一個輸出，
    中間多一層全連接層（`dense` 寬度），前後各自獨立的 dropout（LSTM 內部的跟
    Dense 後面的是兩個不同的數字，不是同一個 dropout 參數套兩次）。
    """

    @staticmethod
    def build(n_features: int, units: int, layers: int, dropout: float, dense: int,
              head_dropout: float, out_dim: int, bidirectional: bool = False):
        import torch.nn.functional as F
        from torch import nn

        class _LSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    n_features, units, layers, batch_first=True,
                    dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional,
                )
                lstm_out = units * (2 if bidirectional else 1)
                self.dense = nn.Linear(lstm_out, dense)
                self.head_dropout = nn.Dropout(head_dropout)
                self.head = nn.Linear(dense, out_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                h = F.relu(self.dense(out[:, -1, :]))
                h = self.head_dropout(h)
                return self.head(h)

        return _LSTMNet()


def _batched_eval(model, X_np, batch_size, device, forward):
    """驗證/推論分批次搬上 GPU，不整包一次丟——資料量一大（幾百萬筆＋高維特徵）
    整包塞 GPU 會直接把 VRAM 塞爆，這裡跟訓練那段一樣分批次算完就丟掉。"""
    import torch

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.tensor(X_np[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(forward(xb))
    return torch.cat(outs, dim=0)


@register("lstm")
def train_lstm(X_train, y_train, X_val, y_val, params: dict, on_epoch, task_type: str = "classification", preview_hook=None) -> dict:
    """LSTM（PyTorch），分類或回歸，結構比照教授教材（LSTM→Dropout→Dense→Dropout→輸出）。
    params: units, layers, dropout, dense, head_dropout, epochs, batch_size, learning_rate,
    l2_lambda（L2 正則化強度，對應 Adam 的 weight_decay）, patience（early stopping，0=不啟用）,
    bidirectional, class_weight（"none" 或 "balanced"，只對分類有意義）, n_classes（分類時用）
    """
    import io

    import torch
    from torch import nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_cls = task_type == "classification"
    n_classes = int(params.get("n_classes", 2))
    units = int(params.get("units", 64))
    layers = int(params.get("layers", 2))
    dropout = float(params.get("dropout", 0.2))
    dense = int(params.get("dense", 32))
    head_dropout = float(params.get("head_dropout", 0.2))
    epochs = int(params.get("epochs", 10))
    batch_size = int(params.get("batch_size", 64))
    lr = float(params.get("learning_rate", 1e-3))
    l2_lambda = float(params.get("l2_lambda", 0.0))
    patience = int(params.get("patience", 0))
    bidirectional = bool(params.get("bidirectional", False))
    class_weight = params.get("class_weight", "none")

    n_features = X_train.shape[-1]
    out_dim = n_classes if is_cls else 1

    model = LSTMNet.build(n_features, units, layers, dropout, dense, head_dropout, out_dim, bidirectional).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

    # 外部工具（nvidia-smi）是定期輪詢，兩次輪詢之間的瞬間空檔會被漏拍，容易誤判成
    # 「GPU 沒在動」；這裡直接從 PyTorch 內部問「這個 process 自己實際在這張卡上佔用
    # 多少記憶體」，是 process 自己的即時數字，不受外部輪詢時間點影響，才是真的證據。
    if device == "cuda":
        print(f"[lstm] 使用 GPU：{torch.cuda.get_device_name(0)}，模型建立後已佔用 "
              f"{torch.cuda.memory_allocated(0) / 1e6:.1f} MB VRAM", flush=True)

    if is_cls and class_weight == "balanced":
        # 類別權重跟教材沒有直接示範，但 V1 有這個選項——用 train 段的類別頻率反比算權重，
        # 頻率越低的類別權重越高，緩解三分類「平」那一類樣本數過多的問題。
        counts = np.bincount(y_train.astype(int), minlength=n_classes).astype(float)
        counts[counts == 0] = 1.0
        weight = torch.tensor(len(y_train) / (n_classes * counts), dtype=torch.float32, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
    else:
        loss_fn = nn.CrossEntropyLoss() if is_cls else nn.MSELoss()

    # 不再把 X_train 整包轉成一個大 tensor——X_train 可能是 graph.py 傳進來的
    # LazyWindowed（資料量一大時，整包攤開會直接把 DRAM 榨爆，這是這次 NAS 上 OOM
    # 崩潰的根因，見 graph.py 的 LazyWindowed 類別說明），也可能是一般的 ndarray
    # （小資料/測試用），兩種都支援同樣的 `X_train[idx]` 索引語法，這裡統一每個 batch
    # 才現切現轉 tensor，不管哪種輸入，尖峰記憶體都只跟 batch_size 有關。
    yt = torch.tensor(y_train, dtype=torch.long if is_cls else torch.float32)

    n = len(X_train)
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0
    train_loss = val_loss = 0.0
    train_metric = val_metric = None
    metric_key = "accuracy" if is_cls else "mae"

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss, correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = torch.tensor(X_train[idx.numpy()], dtype=torch.float32, device=device)
            yb = yt[idx].to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb) if is_cls else loss_fn(out.squeeze(-1), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
            if is_cls:
                correct += (out.argmax(1) == yb).sum().item()
            # 即時視窗預覽：用「這一批正在處理的第一筆」跟它已經算出來的 forward 輸出——
            # 不是額外多做一次運算，out[0] 本來就是這個 batch 已經跑過的結果。哪個樣本被
            # 抽到會隨每個 batch 換（idx 是 perm 打亂後的結果），這就是「持續更換」的來源；
            # 節流判斷在 preview_hook 內部做，這裡每個 batch 都呼叫沒關係，大部分會被直接
            # 跳過，成本只有一次函式呼叫。
            if preview_hook is not None:
                try:
                    preview_hook(epoch, int(idx[0].item()), int(out[0].argmax().item()) if is_cls else float(out[0].item()))
                except Exception:
                    pass
        train_loss = total_loss / n
        train_metric = correct / n if is_cls else None

        val_out = _batched_eval(model, X_val, batch_size, device, model)
        yv = torch.tensor(y_val, dtype=torch.long if is_cls else torch.float32, device=device)
        if is_cls:
            val_loss = loss_fn(val_out, yv).item()
            val_metric = (val_out.argmax(1) == yv).float().mean().item()
        else:
            val_loss = loss_fn(val_out.squeeze(-1), yv).item()
            val_metric = torch.abs(val_out.squeeze(-1) - yv).mean().item()  # MAE

        if on_epoch:
            metrics = {"loss": train_loss, "val_loss": val_loss, f"val_{metric_key}": val_metric}
            if train_metric is not None:
                metrics[metric_key] = train_metric
            if device == "cuda":
                # 每個 epoch 都附上這個 process 實際佔用的 VRAM（MB），跟上面模型剛建立時
                # 的那個數字比對，數字有波動就代表真的有 GPU 運算在發生，不是掛假名。
                metrics["gpu_mem_mb"] = torch.cuda.memory_allocated(0) / 1e6
            on_epoch(epoch, metrics)

        # Early stopping：patience=0 代表不啟用（現有行為，訓練滿 epochs）；
        # 啟用時記錄 val_loss 最好的那次權重，之後沒有改善的 epoch 數超過 patience 就提早停止，
        # 並還原成最好那次的權重（不是用最後一個 epoch 的，那可能已經過擬合了）。
        if patience > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
        val_loss = best_val_loss

    def predict(X):
        out = _batched_eval(model, X, batch_size, device, model)
        return (out.argmax(1) if is_cls else out.squeeze(-1)).cpu().numpy()

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)

    result = {
        "final_metrics": {"loss": train_loss, "val_loss": val_loss, f"val_{metric_key}": val_metric},
        "predict": predict,
        "device": device,
        "weights": buf.getvalue(),
        "model_config": {
            "n_features": n_features, "units": units, "layers": layers, "dropout": dropout,
            "dense": dense, "head_dropout": head_dropout, "bidirectional": bidirectional,
            "out_dim": out_dim, "is_cls": is_cls,
        },
    }
    if is_cls:
        def predict_proba(X):
            out = _batched_eval(model, X, batch_size, device, lambda x: torch.softmax(model(x), dim=1))
            return out.cpu().numpy()
        result["predict_proba"] = predict_proba
    return result


@register_loader("lstm")
def load_lstm(weights: bytes, model_config: dict) -> dict:
    """載入已保存的 LSTM 權重，重建出跟訓練時結構一致的模型，只做推論，不重新訓練。"""
    import io

    import torch

    model = LSTMNet.build(
        model_config["n_features"], model_config["units"], model_config["layers"],
        model_config["dropout"], model_config["dense"], model_config["head_dropout"],
        model_config["out_dim"], model_config.get("bidirectional", False),
    )
    model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
    model.eval()
    is_cls = model_config["is_cls"]

    def predict(X):
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            out = model(x)
            return (out.argmax(1) if is_cls else out.squeeze(-1)).numpy()

    result = {"predict": predict}
    if is_cls:
        def predict_proba(X):
            with torch.no_grad():
                x = torch.tensor(X, dtype=torch.float32)
                return torch.softmax(model(x), dim=1).numpy()
        result["predict_proba"] = predict_proba
    return result


@register("xgboost")
def train_xgboost(X_train, y_train, X_val, y_val, params: dict, on_epoch, task_type: str = "classification", preview_hook=None) -> dict:
    """XGBoost，分類或回歸。params: n_estimators, max_depth, learning_rate, subsample,
    colsample_bytree, patience（early_stopping_rounds，0=不啟用）, n_classes（分類時用）"""
    import xgboost as xgb

    is_cls = task_type == "classification"
    n_classes = int(params.get("n_classes", 2))
    n_estimators = int(params.get("n_estimators", 100))
    subsample = float(params.get("subsample", 1.0))
    colsample_bytree = float(params.get("colsample_bytree", 1.0))
    patience = int(params.get("patience", 0))
    early_stopping_rounds = patience if patience > 0 else None

    # XGBoost 吃 2D 特徵，(n, window, n_features) 攤平成 (n, window*n_features)
    X_train_2d = X_train.reshape(len(X_train), -1)
    X_val_2d = X_val.reshape(len(X_val), -1)
    _preview_rng = np.random.default_rng()

    def _emit_preview(model, epoch):
        """XGBoost 沒有 mini-batch 概念（每輪迭代是對整個 train 集算梯度、長一棵樹），
        沒有「這一批正在處理哪些樣本」可以借用——改成每次呼叫都抽一個新的隨機本地索引
        （不是固定同一個），讓展示的 window 一樣會「持續更換」。節流判斷在 preview_hook
        內部做，這裡多算幾次沒關係，大部分會被跳過。
        """
        if preview_hook is None:
            return
        try:
            local_idx = int(_preview_rng.integers(0, len(X_train)))
            xb_prev = xgb.DMatrix(X_train[local_idx:local_idx + 1].reshape(1, -1))
            pred = model.predict(xb_prev, iteration_range=(0, epoch + 1))
            if is_cls:
                # multi:softmax 的 Booster.predict() 直接回傳類別標籤（不是機率向量）；
                # binary:logistic 回傳的才是機率，門檻 0.5 轉成類別，兩種輸出型態不同，
                # 不能用同一種方式解讀。
                pred_val = int(round(float(pred[0]))) if n_classes > 2 else (1 if float(pred[0]) >= 0.5 else 0)
            else:
                pred_val = float(pred[0])
            preview_hook(epoch, local_idx, pred_val)
        except Exception:
            pass

    if is_cls:
        eval_metric = "merror" if n_classes > 2 else "error"

        class _ProgressCallback(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                if on_epoch:
                    train_err = evals_log.get("train", {}).get(eval_metric, [None])[-1]
                    val_err = evals_log.get("val", {}).get(eval_metric, [None])[-1]
                    metrics = {
                        "loss": train_err, "accuracy": (1 - train_err) if train_err is not None else None,
                        "val_loss": val_err, "val_accuracy": (1 - val_err) if val_err is not None else None,
                    }
                    on_epoch(epoch, metrics)
                _emit_preview(model, epoch)
                return False

        clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            early_stopping_rounds=early_stopping_rounds,
            objective="multi:softmax" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            eval_metric=eval_metric,
            callbacks=[_ProgressCallback()],
        )
        clf.fit(X_train_2d, y_train, eval_set=[(X_train_2d, y_train), (X_val_2d, y_val)], verbose=False)

        val_pred = clf.predict(X_val_2d)
        val_metric = float(np.mean(val_pred == y_val))
        final_metrics = {"val_accuracy": val_metric}

        def predict(X):
            return clf.predict(X.reshape(len(X), -1))

        def predict_proba(X):
            return clf.predict_proba(X.reshape(len(X), -1))

        return {
            "final_metrics": final_metrics, "predict": predict, "predict_proba": predict_proba, "device": "cpu",
            "weights": bytes(clf.get_booster().save_raw()),
            # best_iteration 只有在 early_stopping_rounds 有生效時才有意義；sklearn 的
            # predict() 自動只用到 best_iteration 那棵樹為止，但原生 Booster.predict() 不會
            # 自動限制，要靠 model_config 把這個資訊帶過去給 load_xgboost，不然載入推論會
            # 用到 early stopping 之後那些「已經開始過擬合」的多餘樹，跟訓練當下的預測對不起來。
            "model_config": {
                "is_cls": True, "n_classes": n_classes,
                "best_iteration": int(clf.best_iteration) if early_stopping_rounds else None,
            },
        }

    class _ProgressCallback(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            if on_epoch:
                train_rmse = evals_log.get("train", {}).get("rmse", [None])[-1]
                val_rmse = evals_log.get("val", {}).get("rmse", [None])[-1]
                on_epoch(epoch, {"loss": train_rmse, "val_loss": val_rmse})
            _emit_preview(model, epoch)
            return False

    reg = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
        objective="reg:squarederror",
        eval_metric="rmse",
        callbacks=[_ProgressCallback()],
    )
    reg.fit(X_train_2d, y_train, eval_set=[(X_train_2d, y_train), (X_val_2d, y_val)], verbose=False)

    val_pred = reg.predict(X_val_2d)
    val_mae = float(np.mean(np.abs(val_pred - y_val)))

    def predict(X):
        return reg.predict(X.reshape(len(X), -1))

    return {
        "final_metrics": {"val_mae": val_mae}, "predict": predict, "device": "cpu",
        "weights": bytes(reg.get_booster().save_raw()),
        "model_config": {
            "is_cls": False, "n_classes": 0,
            "best_iteration": int(reg.best_iteration) if early_stopping_rounds else None,
        },
    }


@register_loader("xgboost")
def load_xgboost(weights: bytes, model_config: dict) -> dict:
    """載入已保存的 XGBoost 權重（原生 `save_raw()` 格式，不是純文字 dump），只做推論。

    已知限制：這裡用的是原生 `Booster`，不是訓練時的 sklearn wrapper（`XGBClassifier`/
    `XGBRegressor`），`Booster.predict()` 對 `multi:softmax` 訓練出來的模型只會吐類別標籤、
    吐不出機率分佈（softmax 機率是 sklearn wrapper 內部另外處理的，原生 booster 沒有這層）。
    二元分類（`binary:logistic`）沒有這個問題，`predict()` 本來就回機率，這裡能正確算出
    `predict_proba`；多分類的機率輸出目前推論階段還做不到，這裡明確擋掉並說明原因，
    不是回傳錯誤數字冒充。

    如果訓練時有開 early stopping（`patience` > 0），`best_iteration` 會被記錄下來，
    predict 時用 `iteration_range` 限制只用到那個位置為止——不限制的話，原生 Booster
    預設會用「全部」訓練出來的樹（含 early stopping 之後、已經開始過擬合的那些），
    跟訓練當下 sklearn wrapper 的 `predict()`（自動只用到 best_iteration）對不起來。
    """
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(weights))
    is_cls = model_config["is_cls"]
    n_classes = model_config.get("n_classes", 2)
    best_iteration = model_config.get("best_iteration")
    # (0, 0) 是 xgboost 自己的「不限制，用全部樹」預設值，不能傳 None 進去——
    # 傳 None 會被拿去當 tuple 下標，直接炸掉。
    iteration_range = (0, best_iteration + 1) if best_iteration is not None else (0, 0)

    def predict(X):
        d = xgb.DMatrix(X.reshape(len(X), -1))
        raw = booster.predict(d, iteration_range=iteration_range)
        if not is_cls:
            return raw
        return raw.astype(int) if n_classes > 2 else (raw > 0.5).astype(int)

    result = {"predict": predict}
    if is_cls and n_classes <= 2:
        def predict_proba(X):
            d = xgb.DMatrix(X.reshape(len(X), -1))
            p = booster.predict(d, iteration_range=iteration_range)
            return np.stack([1 - p, p], axis=1)
        result["predict_proba"] = predict_proba
    return result


def list_available() -> list[dict]:
    """給 GET /api/model/registry/architectures 用，Swagger 自動文件化，Agent/MCP 查詢用。"""
    return [{"key": k, "description": (fn.__doc__ or "").strip()} for k, fn in ARCHITECTURE_REGISTRY.items()]


def load_model(key: str, weights: bytes, model_config: dict) -> dict:
    if key not in LOAD_REGISTRY:
        raise ValueError(f"未登記推論載入函式的 architecture key: {key!r}，目前有：{sorted(LOAD_REGISTRY)}")
    return LOAD_REGISTRY[key](weights, model_config)


def train_model(
    key: str, X_train, y_train, X_val, y_val, params: dict, on_epoch=None, task_type: str = "classification",
    preview_hook=None,
) -> dict:
    if key not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"未登記的 architecture key: {key!r}，目前有：{sorted(ARCHITECTURE_REGISTRY)}")
    return ARCHITECTURE_REGISTRY[key](X_train, y_train, X_val, y_val, params, on_epoch, task_type, preview_hook=preview_hook)
