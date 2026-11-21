"""任務的流程版本、環境指紋與研究期間提示。

流程版本（PIPELINE_VERSION）：會改變結果的修正都升一版，任務建立／執行時記錄在
model_training_jobs.pipeline_version。不同流程版本的結果不互相比較；舊任務（NULL）保留、不重算。

  2026-11-r1  讀取修正：排除 session='other'、依 bar_start_ts 排序並驗證唯一、重複報錯不靜默去重；
              跨時間框架的圖暫時拒絕（M4 對齊完成前）。

研究期間：預設使用 RESEARCH_DATA_END 以前的資料。2026 年起的資料來自另一條匯入管線（rule_version
不同、有重複夜盤與缺口，尚未核對），選到時只提示並記錄，不擋任務。
"""

import importlib
import os
import platform
import sys
from datetime import date

# 2026-11-r1：讀取修正（M1）。2026-11-r2：v2 引擎（mode="v2" 的圖）、Phase 3 評估、實驗層（M2～M7）。
# 舊格式的圖在 r2 的訓練結果與 r1 逐位元組相同（contract_harness 驗證）；結果比較仍以流程版本區隔。
PIPELINE_VERSION = "2026-11-r2"
RESEARCH_DATA_END = date(2026, 1, 1)


def period_warnings(start, end) -> list[dict]:
    """依任務期間回傳提示清單（只提示，不擋）。start／end 可以是 date 或 'YYYY-MM-DD'。"""
    end_d = end if isinstance(end, date) else date.fromisoformat(str(end))
    if end_d > RESEARCH_DATA_END:
        return [{
            "code": "period_after_research_end",
            "message": (f"期間迄日 {end_d} 超過研究資料上限 {RESEARCH_DATA_END}：2026 年起的行情來自新匯入管線，"
                        "重複夜盤、1～4 月缺口與時段定義尚未核對，結果請另外標示。"),
        }]
    return []


def _version(module: str):
    try:
        return getattr(importlib.import_module(module), "__version__", "unknown")
    except Exception as e:  # noqa: BLE001 — 指紋只記錄，任何匯入失敗都不能讓任務失敗
        return f"n/a ({type(e).__name__})"


def env_fingerprint() -> dict:
    """執行環境的版本指紋（只讀取，不初始化 CUDA）。"""
    fp = {
        "pipeline_version": PIPELINE_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "image_build": os.getenv("HERMESNOTE_IMAGE_BUILD"),
    }
    for m in ("numpy", "pandas", "sklearn", "xgboost", "torch"):
        fp[m] = _version(m)
    try:
        import talib

        fp["talib"] = talib.__version__
        c = talib.__ta_version__
        fp["talib_c"] = c.decode() if isinstance(c, bytes) else str(c)
    except Exception as e:  # noqa: BLE001
        fp["talib"] = f"n/a ({type(e).__name__})"
    torch = sys.modules.get("torch")
    if torch is not None:
        fp["cuda"] = getattr(torch.version, "cuda", None)
        try:
            fp["cudnn"] = torch.backends.cudnn.version()
            if torch.cuda.is_available():
                fp["gpu"] = torch.cuda.get_device_name(0)
        except Exception as e:  # noqa: BLE001
            fp["gpu"] = f"n/a ({type(e).__name__})"
    return fp


def data_report(dfs: dict, graph_spec: dict) -> dict:
    """把各時間框架的讀取報告（services/quotes.py 放在 DataFrame.attrs）彙整成任務紀錄。"""
    return {
        "pipeline_version": PIPELINE_VERSION,
        "period": {"start": graph_spec.get("start"), "end": graph_spec.get("end")},
        "research_data_end": str(RESEARCH_DATA_END),
        "warnings": period_warnings(graph_spec["start"], graph_spec["end"]) if graph_spec.get("end") else [],
        # 目前的時間語意（M4 會改成依 bar_start_ts 對齊、bar_end_ts＋延遲判定可用）：只記錄，不改變計算
        "available_time": {"source": "bar_end_ts", "publication_delay_sec": 0, "delay_status": "assumed"},
        "timeframes": {tf: df.attrs.get("read_report") for tf, df in dfs.items()},
    }
