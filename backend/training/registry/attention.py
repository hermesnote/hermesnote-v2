"""Attention 元件登記表：Attention 是可插拔積木，不焊死在任何 architecture 裡。

architecture 在自己的 `slots` 宣告「哪個位置可以插元件、接受哪種輸出形狀」（例如 LSTM 的
`attention` 位置），這裡的每個元件宣告自己能插在哪裡、輸出什麼形狀；相容性由
`architectures.validate()` 依兩邊登記資訊實際比對，建網路時再做一次真實的形狀檢查
（`build_checked()`），不是只靠描述文字。

每個元件宣告：
  key                 唯一識別字串（存進 model_config，重載時依它找回 build）
  params_schema       這個元件自己的參數 FieldSpec（UI 選了這個型態後動態長出這些欄位）
  validate_params     (params) -> 解析後設定；最後一道把關
  build               (hidden_size, cfg) -> nn.Module，forward(seq)->(context, weights)
                        seq (批, T, hidden)；context (批, context_dim)；weights (批, T)
  context_dim         (hidden_size, cfg) -> int
  output_kind         "context_vector"（序列進、聚合成向量出）／"sequence"（序列進、序列出）
  query_source        "none"／"self"／"external"（保留給 cross-attention）
  combine             輸出與其餘表示的合併方式（目前唯一是 "concat_summary"：context 與序列摘要串接）
  slot_compatibility  [(family, slot_name), ...]：只列真的驗證過的組合

多元件組合（本版不支援，文件記錄擴充位置）：同一位置插多個元件要先定義連接方式——
兩個 "sequence" 元件可以串接（前一個的序列輸出餵下一個）；"context_vector" 元件沒有時間維度，
只能並聯（各自對同一份序列算向量後合併）。屆時擴充點是：slot 的 cardinality 改為
"zero_or_many"、payload 的欄位改成清單、combine 增加 "sequential"／"parallel_concat"，
由 architecture 的 build 依 combine 分派；本版 payload 送清單一律 400。
"""

ATTENTION_REGISTRY: dict = {}


def register(key: str, *, query_source: str, combine: str, slot_compatibility: list, label: str,
             output_kind: str = "context_vector"):
    def deco(spec_cls):
        ATTENTION_REGISTRY[key] = {
            "key": key, "label": label, "description": (spec_cls.__doc__ or "").strip(),
            "validate_params": spec_cls.validate_params, "build": spec_cls.build,
            "context_dim": spec_cls.context_dim, "params_schema": list(spec_cls.params_schema),
            "query_source": query_source, "combine": combine, "output_kind": output_kind,
            "slot_compatibility": [tuple(p) for p in slot_compatibility],
        }
        return spec_cls
    return deco


@register("additive", query_source="none", combine="concat_summary", slot_compatibility=[("lstm", "attention")],
          label="Additive（無 query，tanh 打分）")
class AdditiveNoQueryAttention:
    """Query-free additive attention：e_t = v^T tanh(W_a h_t + b_a)，α = softmax_t(e)，
    context = Σ α_t h_t；打分只看各時間步自己的輸出，不引入 query 狀態。
    params: {"dim": 打分用的隱藏維度}。"""

    params_schema = [
        {"name": "dim", "type": "int", "required": True, "min": 1, "label": "attention.dim",
         "description": "打分用的隱藏維度（W_a 的輸出寬度）"},
    ]

    @staticmethod
    def validate_params(params) -> dict:
        if not isinstance(params, dict) or set(params) - {"dim"}:
            raise ValueError(f"additive 只接受 dim 一個參數，收到 {params!r}")
        dim = params.get("dim")
        if isinstance(dim, bool) or not isinstance(dim, (int, float)) or not float(dim).is_integer() or dim < 1:
            raise ValueError("additive 的 dim 必須是 >= 1 的整數")
        return {"dim": int(dim)}

    @staticmethod
    def context_dim(hidden_size: int, cfg: dict) -> int:
        return hidden_size

    @staticmethod
    def build(hidden_size: int, cfg: dict):
        from torch import nn
        import torch

        class _Additive(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(hidden_size, cfg["dim"])
                self.v = nn.Linear(cfg["dim"], 1, bias=False)

            def forward(self, seq):
                alpha = torch.softmax(self.v(torch.tanh(self.proj(seq))), dim=1)
                return (alpha * seq).sum(dim=1), alpha.squeeze(-1)

        return _Additive()


def list_available() -> list[dict]:
    """GET /api/model/registry/components/attention。"""
    return [
        {"key": k, "label": s["label"], "description": s["description"], "params_schema": s["params_schema"],
         "query_source": s["query_source"], "combine": s["combine"], "output_kind": s["output_kind"],
         "slot_compatibility": [list(p) for p in s["slot_compatibility"]]}
        for k, s in ATTENTION_REGISTRY.items()
    ]


def build_checked(cfg: dict, hidden_size: int):
    """建立元件並做一次真實的介面檢查：用假序列跑一次 forward，確認回傳 (context, weights)
    的形狀跟登記的 context_dim 一致——宣告跟實作漂移會在建網路當下就報錯，不會等到訓練中途。"""
    import torch

    entry = ATTENTION_REGISTRY[cfg["type"]]
    module = entry["build"](hidden_size, cfg)
    expected = entry["context_dim"](hidden_size, cfg)
    with torch.no_grad():
        ctx, weights = module(torch.zeros(2, 3, hidden_size))
    if tuple(ctx.shape) != (2, expected) or tuple(weights.shape) != (2, 3):
        raise ValueError(f"attention {cfg['type']!r} 介面不符：context {tuple(ctx.shape)}（應為 (2, {expected})）、"
                         f"weights {tuple(weights.shape)}（應為 (2, 3)）")
    return module, expected
