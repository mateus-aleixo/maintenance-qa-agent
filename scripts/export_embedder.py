"""Export the query embedder to ONNX, with a parity check against the real thing.

The index in `data/index.db` holds 384-dim vectors produced by
sentence-transformers `BAAI/bge-small-en-v1.5`. To search it, the *query* has to be
embedded by the same model, which would drag torch and sentence-transformers into
the serving container: about 2 GB for a 33M-parameter encoder.

So the encoder goes to ONNX and serving runs onnxruntime, the conformal-rul and
conformal-seg pattern. Pooling (CLS, per the model's own pooling config) and L2
normalisation are baked into the graph rather than reimplemented on the Python side,
because a pooling mismatch does not raise: it silently returns a vector that points
somewhere else, and retrieval quietly gets worse against an index built the other
way. The parity check is what makes that impossible to ship.

    python scripts/export_embedder.py   ->  models/embedder/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).parent.parent
DEFAULT_OUT = ROOT / "models" / "embedder"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Long enough for a maintenance question; bge-small caps at 512 anyway.
PARITY_TEXTS = [
    "What does low oil pressure at idle indicate?",
    "torque specification for the cylinder head bolts",
    "a" * 400,
    "short",
]


class ClsNormalised(nn.Module):
    """transformer -> CLS token -> L2 normalise, exactly what SentenceTransformer
    does for this model (Transformer, Pooling(pooling_mode=cls), Normalize)."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        cls = hidden[:, 0]
        return cls / cls.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def export(out_dir: Path, model_name: str = MODEL_NAME, check: bool = True) -> Path:
    from transformers import AutoModel, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    wrapped = ClsNormalised(AutoModel.from_pretrained(model_name)).eval()

    enc = tok(PARITY_TEXTS[:2], padding=True, truncation=True, max_length=512,
              return_tensors="pt")
    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        wrapped,
        (enc["input_ids"], enc["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        # NOT "embedding": BERT has an internal embedding node and the dynamo
        # exporter reuses the name, producing a graph onnxruntime rejects with
        # "Duplicate definition of name (embedding)".
        output_names=["sentence_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "sentence_embedding": {0: "batch"},
        },
        # 18 matches the sibling repos; asking for 17 makes torch's version
        # converter throw on this graph while still emitting a loadable model.
        opset_version=18,
        verbose=False,
    )
    # The tokenizer travels with the graph: a different vocabulary is the same
    # class of silent failure as a different pooling.
    tok.save_pretrained(out_dir)

    if check:
        parity(onnx_path, out_dir, model_name)
    return onnx_path


def parity(onnx_path: Path, tok_dir: Path, model_name: str) -> float:
    """Compare ONNX output against sentence-transformers on real strings."""
    import onnxruntime as ort
    from sentence_transformers import SentenceTransformer
    from tokenizers import Tokenizer

    reference = SentenceTransformer(model_name)
    ref = np.asarray(reference.encode(PARITY_TEXTS, normalize_embeddings=True),
                     dtype=np.float32)

    tokenizer = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=512)
    tokenizer.enable_padding()
    encs = tokenizer.encode_batch(PARITY_TEXTS)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    max_diff = float(np.abs(ref - got).max())
    cos = float((ref * got).sum(axis=1).min())
    print(f"parity max |diff| = {max_diff:.2e}   min cosine = {cos:.6f}")
    if max_diff > 1e-4:
        raise SystemExit(f"parity check FAILED: {max_diff:.2e} > 1e-4")
    return max_diff


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--no-check", action="store_true")
    a = p.parse_args(argv)
    path = export(a.out, a.model, check=not a.no_check)
    size = sum(f.stat().st_size for f in a.out.rglob("*") if f.is_file())
    print(f"exported: {path}  ({size / 1e6:.0f} MB including tokenizer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
