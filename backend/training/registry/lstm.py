"""LSTM（architecture key：`lstm`）——LSTM 家族唯一的入口。

三個互相獨立、可自由組合的軸（2 × 2 × 2 = 8 種組合），全部由 params 決定：
  方向       `bidirectional`：單向／雙向
  Attention  `attention`：null（不用）或一個登記過的元件（見 training/registry/attention.py）
  輸出頭     `heads`："single"（依 Label 分類或回歸）／"dual"（回歸頭＋方向事件頭，共享編碼器聯合訓練）

網路結構（契約）：
  輸入 (批, window, 特徵)
   → 疊層 LSTM：lstm_layers 每項一個單層 nn.LSTM（各自 units），每層輸出後接該層 dropout
     （不使用 nn.LSTM 內建的層間 dropout）
   → 序列摘要：最後一層的**真正最終 hidden state**（nn.LSTM 回傳的 h_n，同樣過該層 dropout）；雙向時是
     [正向 h_n, 反向 h_n] 串接——反向方向是從序列尾端往回跑，它看完整條序列的最終狀態
     對應序列開頭，不是 out[:, -1]（那個位置反向只看過一個時間步）
   → 有 Attention：concat([context, 摘要])；沒有：只用摘要
   → shared Dense(shared.dense) + ReLU + dropout(shared.dropout)
   → single：Linear(類別數 或 1)；dual：回歸頭 Linear(1)＋方向頭 Linear(1)（logit）
  dual 損失 = loss_weights.regression × MSE + loss_weights.direction × BCEWithLogits；
  方向事件標籤 = sign_threshold(原單位目標, direction_rule)；回歸目標可選 target_scaling（train-only fit）。

訓練控制見 training/registry/training_control.py：epochs 跑滿或早停（patience=0 關閉），
監控指標最好的一輪（best）與最後一輪（last）都保存、都完整評估。
研究端尚未確認的值（層寬、dense、lr、epochs、seed…）一律必填、沒有預設值。
torch 只在函式內匯入（API 容器沒有 torch 也能驗證參數）。
"""

import io
import math

import numpy as np

from training.evaluation import classification_evaluation, regression_evaluation
from training.output_specs import (
    DEFAULT_OUTPUT, build_event_rule, default_output_specs, event_probability_spec, label_metadata,
)
from training.registry import attention as attention_registry
from training.registry.architectures import register, register_loader
from training.registry.optimizers import build_optimizer
from training.registry.target_transforms import apply_target_transform
from training.registry.training_control import BestTracker, early_stopping_fields

KEY = "lstm"
TARGET_TRANSFORM = "sign_threshold"

MONITORS_CLS = {"val_loss": "min", "val_accuracy": "max"}
MONITORS_REG = {"val_loss": "min", "val_rmse": "min", "val_mae": "min"}
MONITORS_DUAL = {"val_joint_loss": "min", "val_mse": "min", "val_bce": "min", "val_rmse": "min",
                 "val_mae": "min", "val_dir_acc": "max"}

_DUAL = {"field": "heads", "equals": "dual"}
_REGRESSION = {"field": "$task_type", "equals": "regression"}
_SINGLE_CLS = {"all": [{"field": "heads", "equals": "single"}, {"field": "$task_type", "equals": "classification"}]}

PARAMS_SCHEMA = [
    {"name": "lstm_layers", "type": "array_of_object", "required": True, "min_items": 1, "max_items": 8,
     "label": "LSTM 疊層", "description": "依序堆疊，每層一個單層 LSTM",
     "item_schema": [
         {"name": "units", "type": "int", "required": True, "min": 1, "label": "units"},
         {"name": "dropout", "type": "float", "required": True, "min": 0, "exclusive_max": 1, "label": "dropout"},
     ]},
    {"name": "bidirectional", "type": "bool", "default": False, "label": "雙向",
     "description": "雙向時序列摘要是兩個方向各自最終 hidden state 的串接"},
    {"name": "attention", "type": "component_ref", "component_registry": "attention", "default": None,
     "label": "Attention", "description": "不選＝不用；本版最多一個"},
    {"name": "heads", "type": "enum", "required": True, "enum_values": ["single", "dual"], "label": "輸出頭",
     "enum_cases": [{"when": {"field": "$task_type", "equals": "classification"}, "values": ["single"]}],
     "description": "single＝依 Label 分類或回歸；dual＝回歸頭＋方向事件頭（需要連續目標且帶正負號，labeling_rule 用 identity；分類 Label 只能 single）"},
    {"name": "direction_rule.op", "type": "enum", "enum_values": [">", ">=", "<", "<="], "required": _DUAL,
     "visible_if": _DUAL, "label": "方向規則 op"},
    {"name": "direction_rule.threshold", "type": "float", "required": _DUAL, "visible_if": _DUAL,
     "label": "方向規則 threshold", "description": "目標原單位"},
    {"name": "direction_rule.threshold_unit", "type": "string", "derived_from": "$outcome_unit", "visible_if": _DUAL,
     "label": "threshold_unit", "description": "固定等於 Label outcome 的單位（不做單位換算）"},
    {"name": "loss_weights.regression", "type": "float", "min": 0, "required": _DUAL, "visible_if": _DUAL,
     "label": "損失權重：回歸"},
    {"name": "loss_weights.direction", "type": "float", "min": 0, "required": _DUAL, "visible_if": _DUAL,
     "label": "損失權重：方向"},
    {"name": "target_scaling", "type": "enum", "enum_values": ["none", "standardize"], "default": "none",
     "visible_if": _REGRESSION, "label": "目標縮放",
     "description": "只影響回歸目標在訓練時的尺度（train-only fit）；輸出與 RMSE／MAE 一律原單位"},
    {"name": "class_weight", "type": "enum", "enum_values": ["none", "balanced"], "default": "none",
     "visible_if": _SINGLE_CLS, "label": "類別權重", "description": "balanced＝依訓練集類別頻率反比加權"},
    {"name": "shared.dense", "type": "int", "required": True, "min": 1, "label": "共享 Dense 寬度"},
    {"name": "shared.dropout", "type": "float", "required": True, "min": 0, "exclusive_max": 1, "label": "共享 Dropout"},
    {"name": "optimizer", "type": "component_ref", "component_registry": "optimizer", "required": True,
     "label": "優化器", "description": "可共用的優化器元件（見 GET /api/model/registry/components/optimizer）"},
    {"name": "batch_size", "type": "int", "required": True, "min": 1, "label": "batch_size"},
    {"name": "epochs", "type": "int", "required": True, "min": 1, "label": "epochs",
     "description": "完整走過訓練集的輪數上限（patience=0 時跑滿）"},
    *early_stopping_fields({
        "enum_values": list(MONITORS_CLS),
        "enum_cases": [{"when": _DUAL, "values": list(MONITORS_DUAL)},
                       {"when": _REGRESSION, "values": list(MONITORS_REG)}],
    }),
    {"name": "seed", "type": "int", "required": True, "min": 0, "label": "seed"},
]


def _extra_checks(cfg: dict, context: dict) -> list:
    problems = []
    if cfg.get("heads") == "dual":
        if context.get("task_type") == "classification":
            problems.append("params.heads=dual 需要連續目標：Label 的 labeling_rule 請用 identity")
        if context.get("signed") is False:
            problems.append("params.heads=dual 的方向頭需要帶正負號的報酬型 outcome")
        w = cfg.get("loss_weights", {})
        if w and w.get("regression", 0) + w.get("direction", 0) <= 0:
            problems.append("params.loss_weights 兩個權重不能同時為 0")
        if "outcome_unit" in context:
            try:
                build_event_rule(cfg["direction_rule"], context["outcome_unit"])
            except ValueError as e:
                problems.append(f"params.direction_rule：{e}")
    return problems


def metric_specs(node: dict, label_node: dict, task_type: str) -> dict:
    """逐輪序列鍵對應 train_lstm 的 on_epoch；評估指標對應 _full_evaluation。"""
    from training.output_specs import label_metadata
    from training.registry import metrics as M

    params = node.get("params") or {}
    unit = label_metadata(label_node, task_type)["unit"]
    if params.get("heads") == "dual":
        heads = [{"key": "regression", "label": "回歸頭", "task": "regression", "target_unit": unit},
                 {"key": "direction", "label": "方向頭", "task": "classification"},
                 {"key": "joint", "label": "聯合", "task": "joint"}]
        series = [
            {"id": "joint_loss", "metric": "joint_loss", "head": "joint", "train": "joint_loss", "val": "val_joint_loss"},
            {"id": "rmse", "metric": "rmse", "head": "regression", "train": "rmse", "val": "val_rmse"},
            {"id": "mse", "metric": "mse", "head": "regression", "train": "mse", "val": "val_mse", "unit": "loss_space"},
            {"id": "dir_acc", "metric": "dir_acc", "head": "direction", "train": "dir_acc", "val": "val_dir_acc"},
            {"id": "bce", "metric": "bce", "head": "direction", "train": "bce", "val": "val_bce"},
        ]
        evaluation = {"regression": M.REGRESSION_EVAL, "direction": M.CLASSIFICATION_EVAL}
    elif task_type == "classification":
        heads = [{"key": "default", "label": "輸出", "task": "classification"}]
        series = [
            {"id": "loss", "metric": "cross_entropy", "head": "default", "train": "loss", "val": "val_loss"},
            {"id": "accuracy", "metric": "accuracy", "head": "default", "train": "accuracy", "val": "val_accuracy"},
        ]
        evaluation = {"default": M.CLASSIFICATION_EVAL}
    else:
        heads = [{"key": "default", "label": "輸出", "task": "regression", "target_unit": unit}]
        series = [
            {"id": "loss", "metric": "mse", "head": "default", "train": "loss", "val": "val_loss", "unit": "loss_space"},
            {"id": "rmse", "metric": "rmse", "head": "default", "train": "rmse", "val": "val_rmse"},
            {"id": "mae", "metric": "mae", "head": "default", "val": "val_mae"},
        ]
        evaluation = {"default": M.REGRESSION_EVAL}
    return M.build_specs(round_unit="epoch", heads=heads, series=series, evaluation=evaluation)


def describe(node: dict, label_node: dict, task_type: str) -> dict:
    params = node.get("params") or {}
    if params.get("heads") != "dual":
        return default_output_specs(node, label_node, task_type)
    if node.get("output_mode") is not None:
        raise ValueError("heads=dual 不使用 output_mode（回歸值是 default，方向機率用具名輸出 direction_probability 引用）")
    target = label_metadata(label_node, task_type)
    reg = {"kind": "regression", "columns": 1, "target": target,
           "description": "回歸頭預測值（目標原單位；啟用 target_scaling 時已還原）"}
    rule = {k: v for k, v in (params.get("direction_rule") or {}).items() if k != "threshold_unit"}
    return {DEFAULT_OUTPUT: reg, "regression": reg,
            "direction_probability": event_probability_spec(target, rule)}


def _mode(cfg: dict, task_type: str) -> str:
    if cfg["heads"] == "dual":
        return "dual"
    return "cls" if task_type == "classification" else "reg"


def sequence_summary_label(bidirectional: bool, has_attention: bool) -> str:
    head = "雙向：正反兩個方向各自最終 hidden state 串接" if bidirectional else "單向：最終 hidden state"
    return head + ("，再與 Attention context 串接" if has_attention else "，直接接共享層")


def build_net(n_features: int, cfg: dict, out_dim: int):
    """out_dim：single 模式的輸出寬度（分類＝類別數、回歸＝1）；dual 模式忽略。"""
    import torch
    from torch import nn

    layers, bidirectional, dual = cfg["lstm_layers"], bool(cfg["bidirectional"]), cfg["heads"] == "dual"
    mult = 2 if bidirectional else 1
    summary_dim = layers[-1]["units"] * mult

    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            in_sizes = [n_features] + [layers[i]["units"] * mult for i in range(len(layers) - 1)]
            self.lstms = nn.ModuleList([
                nn.LSTM(in_sizes[i], layers[i]["units"], batch_first=True, bidirectional=bidirectional)
                for i in range(len(layers))
            ])
            self.drops = nn.ModuleList([nn.Dropout(layer["dropout"]) for layer in layers])
            ctx_dim = 0
            self.attn = None
            if cfg["attention"] is not None:
                self.attn, ctx_dim = attention_registry.build_checked(cfg["attention"], summary_dim)
            self.shared = nn.Linear(summary_dim + ctx_dim, cfg["shared"]["dense"])
            self.shared_drop = nn.Dropout(cfg["shared"]["dropout"])
            if dual:
                self.reg_head = nn.Linear(cfg["shared"]["dense"], 1)
                self.dir_head = nn.Linear(cfg["shared"]["dense"], 1)
            else:
                self.head = nn.Linear(cfg["shared"]["dense"], out_dim)

        def forward(self, x, return_attention: bool = False):
            seq, h_n = x, None
            for lstm, drop in zip(self.lstms, self.drops):
                seq, (h_n, _c_n) = lstm(seq)
                seq = drop(seq)
            # 摘要也是最後一層的輸出，同樣經過該層 dropout（eval 模式下 dropout 不作用，推論不受影響）
            summary = self.drops[-1](torch.cat([h_n[0], h_n[1]], dim=-1) if bidirectional else h_n[0])
            alpha = None
            if self.attn is not None:
                ctx, alpha = self.attn(seq)
                summary = torch.cat([ctx, summary], dim=1)
            z = self.shared_drop(torch.relu(self.shared(summary)))
            out = (self.reg_head(z).squeeze(-1), self.dir_head(z).squeeze(-1)) if dual else self.head(z)
            if return_attention:
                return out, alpha
            return out

    return LSTMModel()


def _forward_batches(model, X, device, batch: int = 2048):
    """eval 模式分批前向，回傳 numpy：single → (n, out_dim)；dual → ((n,), (n,))。"""
    import torch

    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.tensor(np.asarray(X[i:i + batch], dtype=np.float32), device=device)
            out = model(xb)
            outs.append(tuple(o.cpu().numpy().astype(np.float64) for o in out) if isinstance(out, tuple)
                        else out.cpu().numpy().astype(np.float64))
    if outs and isinstance(outs[0], tuple):
        return tuple(np.concatenate([o[k] for o in outs]) for k in range(2))
    return np.concatenate(outs) if outs else np.zeros((0, 1))


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _predictors(model, mode: str, scaling: dict, device: str) -> dict:
    """訓練結束與重載推論共用同一組預測函式——兩邊邏輯一致，保存重載才會一致。"""
    mean, std = scaling["mean"], scaling["std"]
    if mode == "cls":
        def predict_proba(X):
            return _softmax(_forward_batches(model, X, device))

        return {"predict": lambda X: predict_proba(X).argmax(axis=1), "predict_proba": predict_proba}
    if mode == "reg":
        return {"predict": lambda X: _forward_batches(model, X, device)[:, 0] * std + mean}

    def predict_outputs(X):
        reg, logit = _forward_batches(model, X, device)
        return {"regression": reg * std + mean, "direction_probability": 1.0 / (1.0 + np.exp(-logit))}

    return {"predict": lambda X: predict_outputs(X)["regression"], "predict_outputs": predict_outputs}


def _scaling(method: str, y: np.ndarray) -> dict:
    if method == "standardize":
        std = float(np.std(y))
        return {"method": method, "mean": float(np.mean(y)), "std": std if std > 0 else 1.0}
    return {"method": "none", "mean": 0.0, "std": 1.0}


def _val_metrics(model, mode: str, X_val, targets: dict, scaling: dict, device: str, cfg: dict, loss_fn) -> dict:
    """驗證集完整前向，回傳不帶 val_ 前綴的指標（監控指標 val_xxx 對應這裡的 xxx）。"""
    import torch

    if mode == "cls":
        logits = _forward_batches(model, X_val, device)
        y = targets["y"]
        loss = float(loss_fn(torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.long)).item())
        return {"loss": loss, "accuracy": float(np.mean(logits.argmax(axis=1) == y))}
    if mode == "reg":
        pred_s = _forward_batches(model, X_val, device)[:, 0]
        err = pred_s * scaling["std"] + scaling["mean"] - targets["y"]
        return {"loss": float(np.mean((pred_s - targets["y_scaled"]) ** 2)),
                "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err)))}
    reg_s, logit = _forward_batches(model, X_val, device)
    w = cfg["loss_weights"]
    mse = float(np.mean((reg_s - targets["y_scaled"]) ** 2))
    bce = float(np.mean(np.logaddexp(0.0, logit) - targets["cls"] * logit))
    err = reg_s * scaling["std"] + scaling["mean"] - targets["y"]
    return {"joint_loss": w["regression"] * mse + w["direction"] * bce, "mse": mse, "bce": bce,
            "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "dir_acc": float(np.mean((logit >= 0.0) == (targets["cls"] >= 0.5)))}


def _full_evaluation(mode: str, preds: dict, X_val, targets: dict, train_targets: dict, n_classes: int) -> dict:
    if mode == "cls":
        return classification_evaluation(targets["y"], preds["predict"](X_val), y_proba=preds["predict_proba"](X_val),
                                         n_classes=n_classes, y_train=train_targets["y"])
    if mode == "reg":
        return regression_evaluation(targets["y"], preds["predict"](X_val), y_train=train_targets["y"])
    out = preds["predict_outputs"](X_val)
    proba = out["direction_probability"]
    return {
        "direction": classification_evaluation(targets["cls"].astype(int), (proba >= 0.5).astype(int),
                                               y_proba=np.stack([1 - proba, proba], axis=1), n_classes=2,
                                               y_train=train_targets["cls"].astype(int)),
        "regression": regression_evaluation(targets["y"], out["regression"], y_train=train_targets["y"]),
    }


@register(KEY, family="lstm", label="LSTM", params_schema=PARAMS_SCHEMA, extra_checks=_extra_checks,
          capabilities={"bidirectional": "configurable", "attention": "configurable", "dual_head": "configurable",
                        "optimizer": "configurable"},
          slots=[{"slot_name": "attention", "component_registry": "attention", "cardinality": "zero_or_one",
                  "accepts_output_kind": ["context_vector"]},
                 {"slot_name": "optimizer", "component_registry": "optimizer", "cardinality": "exactly_one"}],
          outputs=(DEFAULT_OUTPUT, "regression", "direction_probability"), describe_outputs=describe,
          metric_specs=metric_specs,
          lazy_windows=True)
def train_lstm(X_train, y_train, X_val, y_val, cfg: dict, on_epoch, task_type: str = "classification", *,
               preview_hook=None, n_classes: int | None = None) -> dict:
    """LSTM：方向／Attention／輸出頭三軸獨立組合（見模組說明）。cfg 是 validate() 解析後的設定。"""
    import torch
    from torch import nn

    mode = _mode(cfg, task_type)
    if mode == "dual" and task_type != "regression":
        raise ValueError("heads=dual 需要連續目標（regression）")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = int(n_classes or 2)

    def targets_for(y):
        if mode == "cls":
            return {"y": np.asarray(y).astype(int)}
        y = np.asarray(y, dtype=np.float64)
        t = {"y": y, "y_scaled": (y - scaling["mean"]) / scaling["std"]}
        if mode == "dual":
            t["cls"] = apply_target_transform(TARGET_TRANSFORM, y, cfg["direction_rule"])
        return t

    scaling = _scaling(cfg.get("target_scaling", "none"), np.asarray(y_train, dtype=np.float64)) \
        if mode != "cls" else {"method": "none", "mean": 0.0, "std": 1.0}
    tr_t, va_t = targets_for(y_train), targets_for(y_val)

    out_dim = n_classes if mode == "cls" else 1
    model = build_net(X_train.shape[-1], cfg, out_dim).to(device)
    opt = build_optimizer(model.parameters(), cfg["optimizer"])

    if mode == "cls":
        weight = None
        if cfg.get("class_weight") == "balanced":
            counts = np.bincount(tr_t["y"], minlength=n_classes).astype(float)
            counts[counts == 0] = 1.0
            weight = torch.tensor(len(tr_t["y"]) / (n_classes * counts), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weight)  # 驗證用（CPU numpy 算出的 logits）
        loss_fn_dev = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None)
        yt = {"y": torch.tensor(tr_t["y"], dtype=torch.long)}
    else:
        loss_fn = loss_fn_dev = nn.MSELoss()
        yt = {"y_scaled": torch.tensor(tr_t["y_scaled"], dtype=torch.float32)}
        if mode == "dual":
            yt["cls"] = torch.tensor(tr_t["cls"], dtype=torch.float32)
    bce_fn = nn.BCEWithLogitsLoss()

    es = cfg["early_stopping"]
    monitors = {"cls": MONITORS_CLS, "reg": MONITORS_REG, "dual": MONITORS_DUAL}[mode]
    tracker = BestTracker(monitors[es["monitor"]], es["patience"], es.get("min_delta", 0.0))
    best_state, stopped_early, epochs_run = None, False, 0
    n, bs = len(X_train), cfg["batch_size"]

    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        sums = {"loss": 0.0, "hit": 0.0, "se": 0.0, "mse": 0.0, "bce": 0.0}
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.tensor(np.asarray(X_train[idx.numpy()], dtype=np.float32), device=device)
            opt.zero_grad()
            out = model(xb)
            m = len(idx)
            if mode == "cls":
                yb = yt["y"][idx].to(device)
                loss = loss_fn_dev(out, yb)
                sums["hit"] += (out.argmax(1) == yb).sum().item()
                shown = int(out[0].argmax().item())
            elif mode == "reg":
                yb = yt["y_scaled"][idx].to(device)
                pred = out.squeeze(-1)
                loss = loss_fn_dev(pred, yb)
                sums["se"] += (((pred.detach() - yb) * scaling["std"]) ** 2).sum().item()
                shown = float(pred[0].item() * scaling["std"] + scaling["mean"])
            else:
                reg, logit = out
                yb_s, yb_c = yt["y_scaled"][idx].to(device), yt["cls"][idx].to(device)
                l_mse, l_bce = loss_fn_dev(reg, yb_s), bce_fn(logit, yb_c)
                loss = cfg["loss_weights"]["regression"] * l_mse + cfg["loss_weights"]["direction"] * l_bce
                sums["mse"] += l_mse.item() * m
                sums["bce"] += l_bce.item() * m
                sums["se"] += (((reg.detach() - yb_s) * scaling["std"]) ** 2).sum().item()
                sums["hit"] += ((logit.detach() >= 0) == (yb_c >= 0.5)).sum().item()
                shown = float(reg[0].item() * scaling["std"] + scaling["mean"])
            loss.backward()
            opt.step()
            sums["loss"] += loss.item() * m
            if preview_hook is not None:
                try:
                    preview_hook(epoch, int(idx[0].item()), shown)
                except Exception:
                    pass

        va = _val_metrics(model, mode, X_val, va_t, scaling, device, cfg, loss_fn)
        epochs_run = epoch + 1
        if on_epoch:
            if mode == "cls":
                metrics = {"loss": sums["loss"] / n, "accuracy": sums["hit"] / n,
                           "val_loss": va["loss"], "val_accuracy": va["accuracy"]}
            elif mode == "reg":
                metrics = {"loss": sums["loss"] / n, "val_loss": va["loss"],
                           "rmse": math.sqrt(sums["se"] / n), "val_rmse": va["rmse"], "val_mae": va["mae"]}
            else:
                tr = {"mse": sums["mse"] / n, "bce": sums["bce"] / n, "rmse": math.sqrt(sums["se"] / n),
                      "dir_acc": sums["hit"] / n, "joint_loss": sums["loss"] / n}
                metrics = {"loss": tr["joint_loss"], "val_loss": va["joint_loss"],
                           "accuracy": tr["dir_acc"], "val_accuracy": va["dir_acc"]}
                for k in ("joint_loss", "mse", "bce", "rmse", "dir_acc"):
                    metrics[k], metrics[f"val_{k}"] = tr[k], va[k]
            on_epoch(epoch, metrics)

        improved, stop = tracker.update(epoch, va[es["monitor"][4:]])
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stop:
            stopped_early = True
            break

    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def finalize(state):
        model.load_state_dict(state)
        buf = io.BytesIO()
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, buf)
        preds = _predictors(model, mode, scaling, device)
        va_state = _val_metrics(model, mode, X_val, va_t, scaling, device, cfg, loss_fn)
        return ({f"val_{k}": v for k, v in va_state.items()}, buf.getvalue(), preds,
                _full_evaluation(mode, preds, X_val, va_t, tr_t, n_classes))

    # 先 last 後 best：predict 閉包綁同一個 model，最後載入的是 best，回傳給呼叫端的就是 best。
    final_last, weights_last, _, eval_last = finalize(last_state)
    final_best, weights_best, preds, eval_best = finalize(best_state)

    attention_meta = None
    if cfg["attention"] is not None:
        entry = attention_registry.ATTENTION_REGISTRY[cfg["attention"]["type"]]
        attention_meta = {k: entry[k] for k in ("key", "label", "query_source", "combine", "output_kind")}

    return {
        "final_metrics": {
            **final_best, "last": final_last,
            "monitor": es["monitor"], "monitor_mode": monitors[es["monitor"]],
            "patience": es["patience"], "min_delta": es.get("min_delta", 0.0),
            "best_epoch": tracker.best_epoch, "stopped_epoch": epochs_run - 1, "epochs_run": epochs_run,
            "stopped_early": stopped_early,
        },
        "evaluation": {"best": eval_best, "last": eval_last},
        **preds, "device": device,
        "weights": weights_best, "weights_last": weights_last,
        "model_config": {
            "n_features": int(X_train.shape[-1]), "config": cfg, "mode": mode, "out_dim": out_dim,
            "n_classes": n_classes if mode == "cls" else None, "target_scaling": scaling,
            "sequence_summary": sequence_summary_label(cfg["bidirectional"], cfg["attention"] is not None),
            "attention_meta": attention_meta,
        },
    }


@register_loader(KEY)
def load_lstm(weights: bytes, model_config: dict) -> dict:
    """依保存的 model_config 重建同一個網路（同一個 build_net），只做推論（CPU）。"""
    import torch

    model = build_net(model_config["n_features"], model_config["config"], model_config["out_dim"])
    model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
    model.eval()
    return _predictors(model, model_config["mode"], model_config["target_scaling"], "cpu")
