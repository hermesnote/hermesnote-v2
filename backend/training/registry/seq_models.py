"""其他序列模型（2026-11-21）：GRU、CNN-LSTM、Transformer Encoder（architecture key：`gru`／`cnn_lstm`／`transformer`）。

三者都和 LSTM 共用同一套：v2 資料組裝與分支、正規化、訓練迴圈（training/registry/lstm.py 的 train_lstm：
批次、早停、best／last、單頭分類／回歸、雙頭）、評估、具名輸出、串接、推論與產物保存。只換「每個分支的序列編碼器」：

  分支輸入 (批, window, 特徵)
   → 編碼器（見下）→ 序列摘要向量
   → 各分支摘要 concat（last_value 分支直接取最後一根）→ shared Dense＋ReLU＋dropout → 輸出頭（同 LSTM）

  gru          疊層 GRU（gru_layers 每項一個單層 nn.GRU，各層輸出後接該層 dropout）；摘要＝最後一層 h_n
               （雙向時正反兩個方向串接），語意與 LSTM 相同
  cnn_lstm     Conv1d 疊層（conv_layers：filters、kernel_size、activation；stride 1、padding="same"）
               → MaxPool1d(pool_size)（步長＝pool_size，時間步數 floor(window/pool_size)；pool_size=1 不池化）
               → 疊層 LSTM（lstm_layers）；摘要同 LSTM
  transformer  Linear 投影到 d_model → 位置編碼（sinusoidal 固定／learned 可學）→ n_layers 層 nn.TransformerEncoderLayer
               （post-norm、ReLU、batch_first）→ pooling：last（最後一根）或 mean（全部時間步平均）。窗格內
               全部 K 棒在決策時都已收完，不加因果遮罩

本版不支援 Attention 元件（attention 只登記給 LSTM）。研究端尚未確認的值一律必填、沒有預設值。
"""

import math

from training.registry import lstm as L
from training.registry.architectures import register, register_loader

_COMMON = [f for f in L.PARAMS_SCHEMA if f["name"] not in ("lstm_layers", "bidirectional", "attention")]
_LAYER_ITEMS = [
    {"name": "units", "type": "int", "required": True, "min": 1, "label": "units"},
    {"name": "dropout", "type": "float", "required": True, "min": 0, "exclusive_max": 1, "label": "dropout"},
]
_BIDIR = {"name": "bidirectional", "type": "bool", "default": False, "label": "雙向",
          "description": "雙向時序列摘要是兩個方向各自最終 hidden state 的串接"}

SCHEMAS = {
    "gru": [
        {"name": "gru_layers", "type": "array_of_object", "required": True, "min_items": 1, "max_items": 8,
         "label": "GRU 疊層", "description": "依序堆疊，每層一個單層 GRU", "item_schema": _LAYER_ITEMS},
        _BIDIR, *_COMMON,
    ],
    "cnn_lstm": [
        {"name": "conv_layers", "type": "array_of_object", "required": True, "min_items": 1, "max_items": 8,
         "label": "卷積疊層", "description": "Conv1d，stride 1、padding same",
         "item_schema": [
             {"name": "filters", "type": "int", "required": True, "min": 1, "label": "filters"},
             {"name": "kernel_size", "type": "int", "required": True, "min": 1, "label": "kernel_size"},
             {"name": "activation", "type": "enum", "enum_values": ["relu", "none"], "default": "relu",
              "label": "啟動函數"},
         ]},
        {"name": "pool_size", "type": "int", "required": True, "min": 1, "label": "池化大小",
         "description": "MaxPool1d，步長＝池化大小；1＝不池化"},
        {"name": "lstm_layers", "type": "array_of_object", "required": True, "min_items": 1, "max_items": 8,
         "label": "LSTM 疊層", "item_schema": _LAYER_ITEMS},
        _BIDIR, *_COMMON,
    ],
    "transformer": [
        {"name": "transformer.d_model", "type": "int", "required": True, "min": 1, "label": "d_model"},
        {"name": "transformer.n_heads", "type": "int", "required": True, "min": 1, "label": "注意力頭數"},
        {"name": "transformer.n_layers", "type": "int", "required": True, "min": 1, "label": "層數"},
        {"name": "transformer.ff_dim", "type": "int", "required": True, "min": 1, "label": "前饋寬度"},
        {"name": "transformer.dropout", "type": "float", "required": True, "min": 0, "exclusive_max": 1,
         "label": "dropout"},
        {"name": "transformer.positional", "type": "enum", "enum_values": ["sinusoidal", "learned"],
         "default": "sinusoidal", "label": "位置編碼"},
        {"name": "transformer.pooling", "type": "enum", "enum_values": ["last", "mean"], "default": "last",
         "label": "序列摘要"},
        *_COMMON,
    ],
}
LABELS = {"gru": "GRU", "cnn_lstm": "CNN-LSTM", "transformer": "Transformer Encoder"}
SUMMARY = {"gru": "GRU 最終 hidden state（雙向時兩方向串接）",
           "cnn_lstm": "Conv1d → MaxPool → LSTM 最終 hidden state",
           "transformer": "Transformer Encoder 輸出依 pooling 彙總（last／mean）"}


def _extra_checks(cfg: dict, context: dict) -> list:
    problems = L._extra_checks(cfg, context)
    t = cfg.get("transformer")
    if t and t["d_model"] % t["n_heads"] != 0:
        problems.append(f"params.transformer.d_model（{t['d_model']}）必須能被 n_heads（{t['n_heads']}）整除")
    return problems


def _rnn_encoder(nn, torch, cell, f: int, layers: list, bidirectional: bool):
    mult = 2 if bidirectional else 1

    class RNNEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            in_sizes = [f] + [layers[i]["units"] * mult for i in range(len(layers) - 1)]
            self.rnns = nn.ModuleList([cell(in_sizes[i], layers[i]["units"], batch_first=True, bidirectional=bidirectional)
                                       for i in range(len(layers))])
            self.drops = nn.ModuleList([nn.Dropout(layer["dropout"]) for layer in layers])

        def forward(self, seq):
            h_n = None
            for rnn, drop in zip(self.rnns, self.drops):
                seq, h = rnn(seq)
                h_n = h[0] if isinstance(h, tuple) else h  # LSTM 回 (h, c)，GRU 回 h
                seq = drop(seq)
            return self.drops[-1](torch.cat([h_n[-2], h_n[-1]], dim=-1) if bidirectional else h_n[-1])

    return RNNEncoder(), layers[-1]["units"] * mult


def _encoder(kind: str, cfg: dict, f: int, window: int):
    import torch
    from torch import nn

    if kind == "gru":
        return _rnn_encoder(nn, torch, nn.GRU, f, cfg["gru_layers"], bool(cfg["bidirectional"]))
    if kind == "cnn_lstm":
        convs, c_in = [], f
        for c in cfg["conv_layers"]:
            convs.append(nn.Conv1d(c_in, c["filters"], c["kernel_size"], padding="same"))
            if c.get("activation", "relu") == "relu":
                convs.append(nn.ReLU())
            c_in = c["filters"]
        pool = int(cfg["pool_size"])
        if window // pool < 1:
            raise ValueError(f"cnn_lstm：分支 window={window} 小於 pool_size={pool}，池化後沒有時間步")
        rnn, dim = _rnn_encoder(nn, torch, nn.LSTM, c_in, cfg["lstm_layers"], bool(cfg["bidirectional"]))

        class CNNLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(*convs)
                self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()
                self.rnn = rnn

            def forward(self, seq):
                z = self.pool(self.conv(seq.transpose(1, 2))).transpose(1, 2)
                return self.rnn(z)

        return CNNLSTM(), dim
    if kind == "transformer":
        t = cfg["transformer"]
        d = t["d_model"]

        class TransformerEnc(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(f, d)
                if t.get("positional", "sinusoidal") == "learned":
                    self.pos = nn.Parameter(torch.zeros(1, window, d))
                else:
                    pe = torch.zeros(window, d)
                    pos = torch.arange(window, dtype=torch.float32).unsqueeze(1)
                    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
                    pe[:, 0::2] = torch.sin(pos * div)
                    pe[:, 1::2] = torch.cos(pos * div)[:, : d // 2]
                    self.register_buffer("pos", pe.unsqueeze(0))
                layer = nn.TransformerEncoderLayer(d, t["n_heads"], t["ff_dim"], t["dropout"], batch_first=True)
                self.enc = nn.TransformerEncoder(layer, t["n_layers"], enable_nested_tensor=False)

            def forward(self, seq):
                z = self.enc(self.proj(seq) + self.pos[:, -seq.shape[1]:])
                return z[:, -1] if t.get("pooling", "last") == "last" else z.mean(dim=1)

        return TransformerEnc(), d
    raise ValueError(f"未知的序列編碼器 {kind!r}")


def build_net(kind: str):
    def build(n_features: int, cfg: dict, out_dim: int, layout: dict):
        import torch
        from torch import nn

        dual = cfg["heads"] == "dual"
        w_max = layout["w_max"]

        class SeqModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.specs = layout["branches"]
                self.encoders = nn.ModuleDict()
                total = 0
                for b in self.specs:
                    f = b["cols"][1] - b["cols"][0]
                    if b["encoder"] == "last_value":
                        total += f
                        continue
                    self.encoders[b["id"]], dim = _encoder(kind, cfg, f, b["window"])
                    total += dim
                self.shared = nn.Linear(total, cfg["shared"]["dense"])
                self.shared_drop = nn.Dropout(cfg["shared"]["dropout"])
                if dual:
                    self.reg_head = nn.Linear(cfg["shared"]["dense"], 1)
                    self.dir_head = nn.Linear(cfg["shared"]["dense"], 1)
                else:
                    self.head = nn.Linear(cfg["shared"]["dense"], out_dim)

            def forward(self, x, return_attention: bool = False):
                parts = []
                for b in self.specs:
                    c0, c1 = b["cols"]
                    if b["encoder"] == "last_value":
                        parts.append(x[:, -1, c0:c1])
                    else:
                        parts.append(self.encoders[b["id"]](x[:, w_max - b["window"]:, c0:c1]))
                z = self.shared_drop(torch.relu(self.shared(torch.cat(parts, dim=1))))
                out = (self.reg_head(z).squeeze(-1), self.dir_head(z).squeeze(-1)) if dual else self.head(z)
                return (out, None) if return_attention else out

        return SeqModel()

    return build


def _single_layout(X, kind: str) -> dict:
    """單一預設分支（engine 不傳 input_layout）時補一份等價的配置，網路一律走分支結構。"""
    _n, w, f = X.shape
    return {"version": 1, "w_max": int(w), "f_total": int(f),
            "branches": [{"id": "main", "timeframe": None, "window": int(w), "encoder": kind, "cols": [0, int(f)]}]}


def _make(kind: str):
    builder = build_net(kind)

    @register(kind, family=kind, label=LABELS[kind], params_schema=SCHEMAS[kind], extra_checks=_extra_checks,
              capabilities={"bidirectional": "configurable" if kind != "transformer" else "fixed_off",
                            "attention": "fixed_off", "dual_head": "configurable", "optimizer": "configurable"},
              slots=[{"slot_name": "optimizer", "component_registry": "optimizer", "cardinality": "exactly_one"}],
              outputs=(L.DEFAULT_OUTPUT, "regression", "direction_probability"), describe_outputs=L.describe,
              metric_specs=L.metric_specs, lazy_windows=True)
    def train(X_train, y_train, X_val, y_val, cfg: dict, on_epoch, task_type: str = "classification", *,
              preview_hook=None, n_classes: int | None = None, input_layout: dict | None = None) -> dict:
        layout = input_layout or _single_layout(X_train, kind)
        return L.train_lstm(X_train, y_train, X_val, y_val, cfg, on_epoch, task_type, preview_hook=preview_hook,
                            n_classes=n_classes, input_layout=layout, _builder=builder, _arch=kind,
                            _summary=SUMMARY[kind])

    train.__doc__ = f"{LABELS[kind]}（見 training/registry/seq_models.py 模組說明）。"

    @register_loader(kind)
    def load(weights: bytes, model_config: dict) -> dict:
        import io

        import torch

        model = builder(model_config["n_features"], model_config["config"], model_config["out_dim"],
                        model_config["input_layout"])
        model.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
        model.eval()
        return L._predictors(model, model_config["mode"], model_config["target_scaling"], "cpu")

    return train


for _k in ("gru", "cnn_lstm", "transformer"):
    _make(_k)
