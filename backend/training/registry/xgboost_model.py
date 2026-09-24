"""XGBoost（architecture key：`xgboost`），跟 LSTM 走同一套登記／schema／驗證／評估框架。

輸入 (n, window, 特徵) 攤平成 (n, window × 特徵)。輪數用 `n_estimators` 表達（boosting
rounds＝長幾棵樹），不跟神經網路的 epochs 混用。早停 `early_stopping.patience`：0＝關閉、
跑滿 n_estimators；正整數＝驗證集監控指標連續幾輪沒改善就停（XGBoost 原生機制）。
監控指標固定、不開放選擇：分類是驗證集 logloss（多分類 mlogloss），回歸是驗證集 rmse。

best／last：不管早停開或關，都從每輪的驗證集監控指標找出最好的一輪（best），
best 權重＝截到那一輪的 booster，last 權重＝全部長出來的 booster；兩者都保存、都完整評估。
分類一律輸出機率（二元 binary:logistic、多元 multi:softprob），predict＝機率 argmax。
"""

import numpy as np

from training.evaluation import classification_evaluation, regression_evaluation
from training.registry.architectures import register, register_loader
from training.registry.training_control import early_stopping_fields

KEY = "xgboost"

PARAMS_SCHEMA = [
    {"name": "n_estimators", "type": "int", "required": True, "min": 1, "label": "n_estimators（boosting rounds）",
     "description": "最多長幾棵樹（patience=0 時全部長完）"},
    {"name": "max_depth", "type": "int", "required": True, "min": 1, "label": "max_depth"},
    {"name": "learning_rate", "type": "float", "required": True, "exclusive_min": 0, "label": "learning_rate"},
    {"name": "subsample", "type": "float", "default": 1.0, "exclusive_min": 0, "max": 1, "label": "subsample"},
    {"name": "colsample_bytree", "type": "float", "default": 1.0, "exclusive_min": 0, "max": 1,
     "label": "colsample_bytree"},
    *early_stopping_fields(None),
    {"name": "seed", "type": "int", "required": True, "min": 0, "label": "seed"},
]


def _predictors(booster, is_cls: bool, n_classes: int) -> dict:
    """訓練結束與重載推論共用：直接用 booster（best 權重已經截好樹數），不再靠 iteration_range。"""
    import xgboost as xgb

    def raw(X):
        return booster.predict(xgb.DMatrix(np.asarray(X).reshape(len(X), -1)))

    if not is_cls:
        return {"predict": raw}

    def predict_proba(X):
        p = raw(X)
        return np.stack([1 - p, p], axis=1) if n_classes == 2 else p

    return {"predict": lambda X: predict_proba(X).argmax(axis=1), "predict_proba": predict_proba}


@register(KEY, family="xgboost", label="XGBoost", params_schema=PARAMS_SCHEMA,
          capabilities={"bidirectional": "fixed_off", "attention": "fixed_off", "dual_head": "fixed_off",
                        "optimizer": "fixed_off"})
def train_xgboost(X_train, y_train, X_val, y_val, cfg: dict, on_epoch, task_type: str = "classification", *,
                  preview_hook=None, n_classes: int | None = None) -> dict:
    """XGBoost 梯度提升樹，分類或回歸；參數見 params_schema。"""
    import xgboost as xgb

    is_cls = task_type == "classification"
    n_classes = int(n_classes or 2)
    X_train_2d = np.asarray(X_train).reshape(len(X_train), -1)
    X_val_2d = np.asarray(X_val).reshape(len(X_val), -1)
    patience = cfg["early_stopping"]["patience"]
    rng = np.random.default_rng(cfg["seed"])

    if is_cls:
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        err_metric, loss_metric = ("error", "logloss") if n_classes == 2 else ("merror", "mlogloss")
        eval_metric = [err_metric, loss_metric]  # 最後一個（logloss）是原生早停的監控指標
    else:
        objective, eval_metric, loss_metric = "reg:squarederror", ["rmse"], "rmse"

    class _Progress(xgb.callback.TrainingCallback):
        def after_iteration(self, model, epoch, evals_log):
            tr, va = evals_log.get("validation_0", {}), evals_log.get("validation_1", {})
            if on_epoch:
                if is_cls:
                    on_epoch(epoch, {"loss": tr[loss_metric][-1], "val_loss": va[loss_metric][-1],
                                     "accuracy": 1 - tr[err_metric][-1], "val_accuracy": 1 - va[err_metric][-1]})
                else:
                    on_epoch(epoch, {"loss": tr["rmse"][-1], "val_loss": va["rmse"][-1]})
            if preview_hook is not None:
                try:
                    i = int(rng.integers(0, len(X_train_2d)))
                    p = model.predict(xgb.DMatrix(X_train_2d[i:i + 1]), iteration_range=(0, epoch + 1))
                    shown = (int(np.argmax(p[0])) if n_classes > 2 else int(p[0] >= 0.5)) if is_cls else float(p[0])
                    preview_hook(epoch, i, shown)
                except Exception:
                    pass
            return False

    # 用原生 xgb.train（不用 sklearn 的 XGBClassifier）：類別數固定由 Label 的 n_classes 決定（num_class），
    # 不從 y 實際出現的值推——例如 3 類、threshold_pct=0 時「平」幾乎不會出現，sklearn 包裝會直接報錯。
    params = {"objective": objective, "eval_metric": eval_metric, "max_depth": cfg["max_depth"],
              "learning_rate": cfg["learning_rate"], "subsample": cfg["subsample"],
              "colsample_bytree": cfg["colsample_bytree"], "seed": cfg["seed"]}
    if is_cls and n_classes > 2:
        params["num_class"] = n_classes
    dtrain = xgb.DMatrix(X_train_2d, label=y_train)
    evals_result: dict = {}
    booster = xgb.train(
        params, dtrain, num_boost_round=cfg["n_estimators"],
        evals=[(dtrain, "validation_0"), (xgb.DMatrix(X_val_2d, label=y_val), "validation_1")],
        evals_result=evals_result, early_stopping_rounds=patience if patience > 0 else None,
        callbacks=[_Progress()], verbose_eval=False,
    )

    rounds = booster.num_boosted_rounds()
    history = evals_result["validation_1"][loss_metric][:rounds]
    best_iteration = int(np.argmin(history))
    best_booster = booster[: best_iteration + 1]

    def finalize(b):
        preds = _predictors(b, is_cls, n_classes)
        if is_cls:
            proba = preds["predict_proba"](X_val_2d)
            eps = 1e-15
            logloss = float(-np.mean(np.log(np.clip(proba[np.arange(len(y_val)), np.asarray(y_val, int)], eps, 1))))
            metrics = {f"val_{loss_metric}": logloss,
                       "val_accuracy": float(np.mean(proba.argmax(axis=1) == np.asarray(y_val, int)))}
            evaluation = classification_evaluation(y_val, proba.argmax(axis=1), y_proba=proba,
                                                   n_classes=n_classes, y_train=y_train)
        else:
            pred = preds["predict"](X_val_2d)
            metrics = {"val_rmse": float(np.sqrt(np.mean((pred - y_val) ** 2))),
                       "val_mae": float(np.mean(np.abs(pred - y_val)))}
            evaluation = regression_evaluation(y_val, pred, y_train=y_train)
        return metrics, bytes(b.save_raw()), preds, evaluation

    final_last, weights_last, _, eval_last = finalize(booster)
    final_best, weights_best, preds, eval_best = finalize(best_booster)

    return {
        "final_metrics": {
            **final_best, "last": final_last, "monitor": f"val_{loss_metric}", "patience": patience,
            "best_round": best_iteration, "rounds_run": rounds,
            "stopped_early": patience > 0 and rounds < cfg["n_estimators"],
        },
        "evaluation": {"best": eval_best, "last": eval_last},
        **preds, "device": "cpu",
        "weights": weights_best, "weights_last": weights_last,
        "model_config": {"config": cfg, "is_cls": is_cls, "n_classes": n_classes if is_cls else None,
                         "best_round": best_iteration, "rounds_run": rounds},
    }


@register_loader(KEY)
def load_xgboost(weights: bytes, model_config: dict) -> dict:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(bytearray(weights))
    return _predictors(booster, model_config["is_cls"], model_config.get("n_classes") or 2)
