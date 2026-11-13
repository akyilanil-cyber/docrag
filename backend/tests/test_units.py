import numpy as np
import pytest

from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.extract import UnsupportedFile, validate_upload


def test_chunking_overlaps_and_covers_everything():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), chunk_words=30, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].split()[-1] == "w99"
    assert all(len(c.split()) <= 30 for c in chunks)


def test_chunking_empty_and_invalid():
    assert chunk_text("   ") == []
    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_words=5, overlap=5)


def test_embeddings_are_unit_length_and_deterministic():
    e = HashingEmbedder(64)
    a, b = e.embed(["merhaba dünya"]), e.embed(["merhaba dünya"])
    assert a == b
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-9


def test_similar_text_scores_higher_than_unrelated():
    e = HashingEmbedder(256)
    q, near, far = e.embed(
        ["refund policy for returned items", "our refund policy covers returned items", "the cat sat on a warm roof"]
    )
    assert np.dot(q, near) > np.dot(q, far)


def test_turkish_dotted_i_is_normalised():
    e = HashingEmbedder(128)
    assert e.embed(["İSTANBUL"]) == e.embed(["istanbul"])


def test_validate_upload_rejects_wrong_content():
    assert validate_upload("a.pdf", b"%PDF-1.7 ...") == ".pdf"
    with pytest.raises(UnsupportedFile):
        validate_upload("a.pdf", b"MZ\x90\x00 not a pdf")
    with pytest.raises(UnsupportedFile):
        validate_upload("a.exe", b"hello")
    with pytest.raises(UnsupportedFile):
        validate_upload("a.txt", b"bin\x00ary")
