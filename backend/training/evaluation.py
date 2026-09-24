"""通用評估指標：訓練完成後對「整個驗證集」算一次完整報告，跟訓練迴圈裡逐輪回報的快速指標
（`on_epoch` 的 loss/accuracy 那組，是逐批累計平均，給即時進度看）分開——這裡不受批次大小、
訓練模式（dropout 開著）影響，是模型收斂後的正式評估，API 跟後台都讀得到。

盤點 V1 有、V2 目前缺或分散的評估項目，這裡補齊：
  分類：類別分布、混淆矩陣、precision／recall／F1（逐類別＋macro-F1）、balanced accuracy、
        ROC-AUC、AP（Average Precision），加一個「訓練集多數類別」基準
  回歸：MSE、RMSE、MAE、R²，加兩個基準：「預測訓練集平均值」、「預測零報酬」（適用於報酬類目標，
        分不清楚方向也不猜漲跌時最單純的參照）

**AP（Average Precision）不是「PR 曲線下面積用梯形積分算出來的 PR-AUC」**——兩者是不同的量。
`sklearn.metrics.average_precision_score` 是把 precision 在每個 recall 變化點的值加權平均
（等於用 step function 累加，不做點與點之間的線性內插），跟先取得一串 (precision, recall) 點
再用 `np.trapz` 之類的梯形公式積分出面積，數值上通常不相等（sklearn 文件本身也特別提醒這件事）。
這裡只實作 AP、明確叫它 `average_precision`，不叫 `pr_auc`，避免報告的人誤以為是同一種東西、
拿去跟別的論文用梯形積分算出來的 PR-AUC 直接比較。

任何指標算不出來（例如沒有機率輸出、驗證集裡類別數不足、樣本數太少、沒給訓練集資料），一律在
對應的 `*_unavailable_reason` 欄位明講原因，不是拿 None／0 混過去假裝算過。

sklearn 只在函式內部匯入（呼叫時才需要），這個檔案本身模組層級不依賴 sklearn／torch，
API 容器（沒裝 torch，且理論上也不一定會用到這支）匯入不會出錯。
"""

import numpy as np


def classification_evaluation(y_true, y_pred, y_proba=None, n_classes: int | None = None, y_train=None) -> dict:
    """`y_true`／`y_pred` 是類別索引（整數）；`y_proba` 是 (n, k) 的機率陣列（k=n_classes），
    沒有機率輸出（例如架構只給類別、沒有 predict_proba）就傳 None，ROC-AUC／AP 會明講原因。
    `y_train`（可選）：有提供才會算「訓練集多數類別」這個基準，多數類別由 train 集決定，
    不是驗證集，避免用到驗證集資訊反推基準（那樣會偷看答案）。
    """
    from sklearn.metrics import (
        balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = list(range(n_classes)) if n_classes is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "class_distribution": {int(c): int((y_true == c).sum()) for c in classes},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "confusion_matrix_labels": classes,
        "per_class": {
            int(c): {"precision": float(precision[i]), "recall": float(recall[i]),
                      "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(classes)
        },
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if y_proba is None:
        reason = "沒有機率輸出（架構這個輸出模式不是機率），ROC-AUC／AP 需要機率分數才能算"
        out["roc_auc"], out["roc_auc_unavailable_reason"] = None, reason
        out["average_precision"], out["average_precision_unavailable_reason"] = None, reason
    else:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_proba = np.asarray(y_proba)
        if len(classes) == 2:
            pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            try:
                out["roc_auc"] = float(roc_auc_score(y_true, pos))
                out["average_precision"] = float(average_precision_score(y_true, pos))
            except ValueError as e:
                reason = f"驗證集裡實際出現的類別不足兩種，AUC／AP 沒有定義：{e}"
                out["roc_auc"], out["roc_auc_unavailable_reason"] = None, reason
                out["average_precision"], out["average_precision_unavailable_reason"] = None, reason
        else:
            missing = [c for c in classes if not np.any(y_true == c)]
            if missing:
                # sklearn 此時不丟錯、回傳 NaN（該類別的 OvR AUC 沒有定義），明講原因，不回 NaN
                out["roc_auc"] = None
                out["roc_auc_unavailable_reason"] = f"驗證集裡沒有出現類別 {missing}，這些類別的 One-vs-Rest AUC 沒有定義，macro average 無法計算"
            else:
                try:
                    out["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes))
                except ValueError as e:
                    out["roc_auc"], out["roc_auc_unavailable_reason"] = None, f"多分類 One-vs-Rest macro average 無法計算：{e}"
            # 多分類 AP 的加總方式（micro／macro／逐類別）沒有明確指定要用哪種，這裡不替研究端
            # 決定，先明講缺口，不是漏做
            out["average_precision"] = None
            out["average_precision_unavailable_reason"] = "AP 目前只實作二元分類；多分類的加總方式（micro/macro/逐類別）未指定，未實作"

    if y_train is None or len(np.asarray(y_train)) == 0:
        out["baseline_majority_class"] = None
        out["baseline_majority_class_unavailable_reason"] = "沒有提供訓練集標籤，無法決定「訓練集多數類別」"
        return out

    y_train_arr = np.asarray(y_train).astype(int)
    counts = np.bincount(y_train_arr, minlength=(n_classes or int(y_train_arr.max()) + 1))
    majority_class = int(np.argmax(counts))
    baseline_pred = np.full_like(y_true, majority_class)
    out["baseline_majority_class"] = {
        "method": "predict_majority_class_from_train", "class": majority_class,
        "accuracy": float(np.mean(baseline_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, baseline_pred)),
        "macro_f1": float(f1_score(y_true, baseline_pred, labels=classes, average="macro", zero_division=0)),
    }
    return out


def regression_evaluation(y_true, y_pred, y_train=None) -> dict:
    """`y_train`（可選）：有提供才會算「預測訓練集平均值」這個基準；「預測零報酬」不需要
    `y_train`，直接對驗證集算（目標是報酬類數值時，0＝「猜不漲不跌」，最單純的參照點）。"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    out = {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(y_true, y_pred))}
    enough = len(y_true) > 1
    if enough:
        out["r2"] = float(r2_score(y_true, y_pred))
    else:
        out["r2"], out["r2_unavailable_reason"] = None, "樣本數 <= 1，R² 需要至少兩筆才有變異數可比較"

    zero_pred = np.zeros_like(y_true)
    z_mse = float(mean_squared_error(y_true, zero_pred))
    out["baseline_zero"] = {
        "method": "predict_zero", "mse": z_mse, "rmse": float(np.sqrt(z_mse)),
        "mae": float(mean_absolute_error(y_true, zero_pred)),
        "r2": float(r2_score(y_true, zero_pred)) if enough else None,
    }

    if y_train is None or len(np.asarray(y_train)) == 0:
        out["baseline_mean"] = None
        out["baseline_mean_unavailable_reason"] = "沒有提供訓練集目標值，無法算「預測訓練集平均值」這個基準"
        return out

    mean_pred = np.full_like(y_true, float(np.mean(np.asarray(y_train, dtype=float))))
    m_mse = float(mean_squared_error(y_true, mean_pred))
    out["baseline_mean"] = {
        "method": "predict_train_mean", "mse": m_mse, "rmse": float(np.sqrt(m_mse)),
        "mae": float(mean_absolute_error(y_true, mean_pred)),
        "r2": float(r2_score(y_true, mean_pred)) if enough else None,
    }
    return out
