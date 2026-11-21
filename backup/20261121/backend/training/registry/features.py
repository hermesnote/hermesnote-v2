"""Feature Node 登記表：輸入一段 OHLCV 資料 + 參數，輸出特徵陣列。

每個登記項目是一個函式 `(df, params) -> np.ndarray`，用 key 查表呼叫，不寫死 if/else。
新增一種特徵＝在這裡多登記一個函式，不用改任何呼叫端的介面（graph.py 完全不用動）。

df 是 pandas DataFrame，欄位至少含 open/high/low/close/volume，依時間升冪排序。
輸出固定是 2D 陣列 (n_samples, n_feature_dims)——就算只有一維特徵，也 reshape 成 (n, 1)，
這樣 graph.py 串接多個 Feature Node 輸出時可以直接沿著 axis=1 橫向拼接，不用另外判斷維度。
"""

import numpy as np

FEATURE_REGISTRY: dict = {}


def register(key: str):
    def deco(fn):
        FEATURE_REGISTRY[key] = fn
        return fn
    return deco


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


@register("ohlcv")
def ohlcv(df, params: dict | None = None) -> np.ndarray:
    """Custom > OHLCV：開高低收＋成交量一起加入，價跟量綁在同一個特徵、不拆開。
    跟 raw_price／raw_volume 是同一個 Custom 分類底下三個平等的選項——要嘛價量一起加
    （這個），要嘛只要價（raw_price）、只要量（raw_volume），做消融實驗時各自獨立移除。
    """
    return df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)


@register("raw_price")
def raw_price(df, params: dict | None = None) -> np.ndarray:
    """Custom > Raw Price：原始開高低收，不做任何轉換，不需要參數。"""
    return df[["open", "high", "low", "close"]].to_numpy(dtype=float)


@register("raw_volume")
def raw_volume(df, params: dict | None = None) -> np.ndarray:
    """Custom > Raw Volume：原始成交量，不做任何轉換，不需要參數。"""
    return _as_2d(df["volume"].to_numpy(dtype=float))


@register("talib_indicator")
def talib_indicator(df, params: dict) -> np.ndarray:
    """呼叫任一個 TA-Lib 指標。params 至少要有 "name"（例如 "RSI"），其餘 key 當該指標的參數
    （例如 {"name": "RSI", "timeperiod": 14}）。用 TA-Lib 的 abstract API，不用為每個指標各寫一個函式。
    """
    import talib.abstract as ta

    name = params.get("name") or ""
    if not name:
        # 前端理論上會擋掉這種「沒選指標就送」的情況，但 Agent 可能直接打 API 送
        # graph_spec 略過前端表單——空字串丟進 ta.Function 會得到 TA-Lib 自己那句
        # "not supported by TA-LIB."，完全看不出真正原因，這裡先擋下來講清楚。
        raise ValueError('talib_indicator 節點缺少 params.name（要指定指標名稱，例如 "RSI"）')
    kwargs = {k: v for k, v in params.items() if k != "name"}
    fn = ta.Function(name)
    inputs = {
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
        "volume": df["volume"].to_numpy(dtype=float),
    }
    fn.input_arrays = inputs
    if kwargs:
        # 前端表單一律用 <input type="number"> + Number()，JSON 序列化又不區分「剛好是整數
        # 的浮點數」跟「整數」（2.0 跟 2 過一輪 JSON 都變成 2），到這裡就沒辦法從數字本身
        # 分辨原本該是 int 還是 float——BBANDS 的 nbdevup/nbdevdn 這種本來就要求 float 的
        # 參數，使用者輸入整數值時會被還原成 Python int，TA-Lib 嚴格檢查型別直接報錯
        # （"expected float, got int"）。這裡照 TA-Lib 自己宣告的預設值型別（fn.parameters
        # 拿到的就是每個參數本來該有的正確型別）轉換一次，不管前端送過來的數字型別是什麼。
        expected_types = fn.parameters
        kwargs = {
            k: (float(v) if isinstance(expected_types.get(k), float) else
                int(v) if isinstance(expected_types.get(k), int) else v)
            for k, v in kwargs.items()
        }
        fn.set_parameters(kwargs)
    out = fn.run()
    if isinstance(out, list):
        out = np.stack(out, axis=1)
    return _as_2d(out)


def _fetch_saved_strategy_sync(saved_id: str) -> dict:
    """同步查詢（訓練 worker 的執行緒沒有事件迴圈，用 psycopg2，跟 progress.py 同一套做法）。"""
    import os

    import psycopg2
    from dotenv import load_dotenv

    load_dotenv()
    url = os.getenv("HERMESNOTE_DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT long_entry_tree, long_exit_tree, short_entry_tree, short_exit_tree "
                "FROM quant_saved_strategies WHERE id = %s",
                (saved_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"找不到收藏的策略 {saved_id!r}")
            return {
                "long_entry_tree": row[0], "long_exit_tree": row[1],
                "short_entry_tree": row[2], "short_exit_tree": row[3],
            }
    finally:
        conn.close()


@register("quant_saved_strategy")
def quant_saved_strategy(df, params: dict) -> np.ndarray:
    """把一筆收藏的量化回測策略（條件樹）拿來當特徵：對這段資料重新算一次進出場訊號，
    輸出 0/1 訊號欄位（有幾棵樹就有幾欄）。回測有效不代表對模型預測有用，這只是把
    驗證過的規則轉成模型看得懂的數字，不是保證有用。params: {"saved_id": "..."}
    """
    from indicators.tree import evaluate_node

    strategy = _fetch_saved_strategy_sync(params["saved_id"])
    columns = []
    for key in ("long_entry_tree", "long_exit_tree", "short_entry_tree", "short_exit_tree"):
        tree = strategy.get(key)
        if tree:
            series, _ = evaluate_node(df, tree)
            columns.append(series.astype(float).to_numpy())
    if not columns:
        raise ValueError(f"收藏的策略 {params['saved_id']!r} 沒有任何條件樹可以算")
    return np.stack(columns, axis=1)


def list_available() -> list[dict]:
    """給 GET /api/model/registry/features 用，Swagger 自動文件化，Agent/MCP 查詢用。"""
    return [{"key": k, "description": (fn.__doc__ or "").strip()} for k, fn in FEATURE_REGISTRY.items()]


def compute_feature(key: str, df, params: dict | None = None) -> np.ndarray:
    if key not in FEATURE_REGISTRY:
        raise ValueError(f"未登記的 feature key: {key!r}，目前有：{sorted(FEATURE_REGISTRY)}")
    return FEATURE_REGISTRY[key](df, params or {})
