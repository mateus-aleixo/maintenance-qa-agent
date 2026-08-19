"""Embedders behind one protocol.

- HashEmbedder: deterministic, dependency-free. Used by tests and CI so the whole
  suite runs with no model downloads. Not semantically meaningful — but stable, so
  retrieval mechanics (fusion, ranking, storage round-trips) are fully testable.
- BgeEmbedder: sentence-transformers bge-small-en-v1.5, the real thing. Optional
  extra: `pip install -e ".[embed]"`.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashEmbedder:
    """Bag-of-hashed-tokens, L2-normalised. Deterministic across runs/platforms."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _TOKEN.findall(text.lower()):
                h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=4).digest(), "big")
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        np.divide(out, norms, out=out, where=norms > 0)
        return out


class BgeEmbedder:
    """Real semantic embeddings. Import cost paid lazily and only when chosen."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # optional extra

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(list(texts), normalize_embeddings=True), dtype=np.float32
        )


class OnnxEmbedder:
    """The same vectors as BgeEmbedder, without torch.

    Serving has to embed the query with whatever built the index, and pulling
    sentence-transformers into the container costs about 2 GB for a 33M-parameter
    encoder. `scripts/export_embedder.py` bakes CLS pooling and L2 normalisation
    into the graph and checks parity against sentence-transformers before this is
    allowed to ship: measured max |diff| 2.3e-07, cosine 1.000000.

    A pooling or vocabulary mismatch here would not raise. It would return a vector
    pointing somewhere else and quietly degrade recall against an index built the
    other way, which is why the tokenizer travels with the graph.
    """

    dim = 384

    def __init__(self, model_dir: "Path | str" = "models/embedder", max_length: int = 512):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding()
        self.session = ort.InferenceSession(
            str(d / "model.onnx"), providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        out = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        return np.asarray(out, dtype=np.float32)


def get_embedder(name: str = "hash", **kwargs) -> Embedder:
    if name == "hash":
        return HashEmbedder()
    if name == "bge":
        return BgeEmbedder()
    if name == "onnx":
        return OnnxEmbedder(**kwargs)
    raise ValueError(f"unknown embedder: {name!r}")
