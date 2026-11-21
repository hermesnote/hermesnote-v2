"""指標定義登記表：指標是可共用元件，模型引用，不各自定義。

每個指標宣告它是什麼（標籤、說明）、怎麼比較（`direction`：min＝越低越好、max＝越高越好、None＝不比較）、
單位（`unit`）與資料形狀（`shape`，前端依此選元件）：

  shape      scalar（數值；逐輪曲線、比較圖）／per_class（逐類別表）／distribution（分布）／matrix（熱圖）
  unit       loss_space（損失空間；啟用目標縮放時是縮放後尺度）／target（目標原單位，實際單位見模型
             metric_specs 的 heads[].target_unit）／target_squared／ratio（0～1 比例）／count
  format     percent／decimal／int（顯示建議）

資料契約（評估報告裡的值，依 shape）：
  scalar        數值（可為 null，並在報告附 `<key>_unavailable_reason`）
  per_class     {類別: {欄位: 數值, ...}}（欄位由資料決定，例如 precision／recall／f1／support）
  distribution  {類別: 數值}
  matrix        二維陣列；列／欄標籤放在報告的 `<key>_labels`，軸名稱由定義的 `axes` 宣告
衍生指標：定義帶 `derive={"from": 來源指標, "method": DERIVATIONS 的名稱}`，報告沒有時由來源推算。
基準：報告裡 `baseline_<名稱>` 物件（含 method 與同名指標數值），名稱與顯示標籤在 BASELINE_REGISTRY 登記。

模型用 `metric_specs(node, label_node, task_type)`（見 architectures.register 的 metric_specs）宣告
這個設定會產生哪些逐輪序列、哪些評估指標、屬於哪個輸出頭；引用的指標必須在這裡登記。
新指標＝在這裡登記＋模型引用，前端依 shape 自動用對應元件呈現。
"""

METRIC_REGISTRY: dict = {}


BASELINE_REGISTRY: dict = {}
SHAPES = ("scalar", "per_class", "distribution", "matrix")


def register(key: str, *, label: str, shape: str = "scalar", direction: str | None = None,
             unit: str = "ratio", fmt: str = "decimal", description: str = "",
             axes: dict | None = None, derive: dict | None = None):
    if shape not in SHAPES:
        raise ValueError(f"未知的 shape：{shape!r}，可用：{SHAPES}")
    METRIC_REGISTRY[key] = {"key": key, "label": label, "shape": shape, "direction": direction,
                            "unit": unit, "format": fmt, "description": description}
    if axes is not None:
        METRIC_REGISTRY[key]["axes"] = axes
    if derive is not None:
        METRIC_REGISTRY[key]["derive"] = derive


def register_baseline(raw_key: str, *, key: str, label: str, description: str = ""):
    """評估報告裡的 `baseline_*` 物件 → 評估清單的基準名稱與顯示標籤。"""
    BASELINE_REGISTRY[raw_key] = {"key": key, "label": label, "description": description}


def _matrix_diagonal_ratio(matrix):
    total = sum(sum(row) for row in matrix)
    return sum(matrix[i][i] for i in range(len(matrix))) / total if total else None


# 衍生指標的推算方法：名稱 → (來源值 → 數值)
DERIVATIONS = {"matrix_diagonal_ratio": _matrix_diagonal_ratio}


# ── 損失 ──
register("cross_entropy", label="Cross-entropy", direction="min", unit="loss_space",
         description="分類交叉熵（有 class_weight 時是加權版本）")
register("logloss", label="LogLoss", direction="min", unit="loss_space", description="二元對數損失（XGBoost）")
register("mlogloss", label="多分類 LogLoss", direction="min", unit="loss_space", description="多分類對數損失（XGBoost）")
register("joint_loss", label="聯合損失", direction="min", unit="loss_space",
         description="雙頭：loss_weights.regression × MSE ＋ loss_weights.direction × BCE")
register("bce", label="BCE", direction="min", unit="loss_space", description="方向頭的二元交叉熵")
# ── 回歸 ──
register("mse", label="MSE", direction="min", unit="target_squared", fmt="decimal", description="均方誤差")
register("rmse", label="RMSE", direction="min", unit="target", description="均方根誤差")
register("mae", label="MAE", direction="min", unit="target", description="平均絕對誤差")
register("r2", label="R²", direction="max", unit="ratio", description="決定係數")
# ── 分類 ──
register("accuracy", label="Accuracy", direction="max", unit="ratio", fmt="percent", description="預測正確比例",
         derive={"from": "confusion_matrix", "method": "matrix_diagonal_ratio"})
register("dir_acc", label="方向準確率", direction="max", unit="ratio", fmt="percent",
         description="方向頭：機率以 0.5 為界判定事件，與實際事件相符的比例")
register("balanced_accuracy", label="Balanced accuracy", direction="max", unit="ratio", fmt="percent",
         description="各類別召回率的平均")
register("macro_f1", label="Macro-F1", direction="max", unit="ratio", description="各類別 F1 的平均")
register("roc_auc", label="ROC-AUC", direction="max", unit="ratio",
         description="二元為正類機率；多分類為 One-vs-Rest macro average")
register("average_precision", label="Average Precision", direction="max", unit="ratio",
         description="AP（step 加權平均，不是梯形積分的 PR-AUC）；目前只實作二元分類")
register("per_class", label="逐類別指標", shape="per_class", unit="ratio",
         description="各類別 precision／recall／f1／support")
register("class_distribution", label="類別分布", shape="distribution", unit="count", fmt="int",
         description="評估資料集中各類別的實際樣本數")
register("confusion_matrix", label="混淆矩陣", shape="matrix", unit="count", fmt="int",
         description="列＝實際類別、欄＝預測類別", axes={"row": "實際", "col": "預測"})

# ── 基準（training/evaluation.py 的 baseline_* 物件）──
register_baseline("baseline_majority_class", key="majority_class", label="訓練集多數類別",
                  description="一律預測訓練集最多的類別")
register_baseline("baseline_mean", key="train_mean", label="預測訓練集平均")
register_baseline("baseline_zero", key="zero", label="預測零")

CLASSIFICATION_EVAL = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "average_precision",
                       "per_class", "class_distribution", "confusion_matrix"]
REGRESSION_EVAL = ["rmse", "mae", "mse", "r2"]


def definitions(keys) -> dict:
    """指定指標的定義；引用未登記的指標直接報錯（模型宣告與登記表必須一致）。"""
    missing = [k for k in keys if k not in METRIC_REGISTRY]
    if missing:
        raise KeyError(f"未登記的指標：{missing}")
    return {k: METRIC_REGISTRY[k] for k in keys}


def list_available() -> list[dict]:
    """GET /api/model/registry/metrics。"""
    return list(METRIC_REGISTRY.values())


def build_specs(*, round_unit: str, heads: list[dict], series: list[dict], evaluation: dict) -> dict:
    """組出一個模型節點的 metric_specs，並附上所有引用指標的定義（回應自足，前端不必另查）。

    heads       [{"key", "label", "task", "target_unit"?}]
    series      [{"id", "metric", "head", "train"?, "val"?, "unit"?}]：train／val 是逐輪紀錄裡的欄位名
                （固定四欄或 metrics JSONB 的鍵）；unit 可覆寫定義的單位（例：訓練用 MSE 在損失空間）
    evaluation  {head: [指標 key, ...]}：評估報告會有哪些指標
    """
    keys = {s["metric"] for s in series} | {k for ks in evaluation.values() for k in ks}
    return {"round_unit": round_unit, "heads": heads, "series": series, "evaluation": evaluation,
            "definitions": definitions(sorted(keys))}
