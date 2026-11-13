"""Deterministic, dependency-free text embedder.

A hashed bag-of-words vector (the "hashing trick") with unigrams and bigrams,
L2-normalised so that dot product == cosine similarity. It needs no network or
model download, which keeps the demo and the tests reproducible. The `Embedder`
protocol makes it easy to swap for a real embedding model later.
"""

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"\w+", re.UNICODE)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _tokens(text: str) -> list[str]:
    # str.lower() mishandles Turkish dotted/dotless I, so normalise explicitly.
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return _TOKEN.findall(text)


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return value % self.dim, sign

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        toks = _tokens(text)
        features = toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]
        for feature in features:
            idx, sign = self._bucket(feature)
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]
