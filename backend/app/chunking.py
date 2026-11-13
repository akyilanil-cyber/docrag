def chunk_text(text: str, chunk_words: int = 180, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be >= 0 and < chunk_words")

    words = text.split()
    if not words:
        return []

    step = chunk_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]
        chunks.append(" ".join(window))
        if start + chunk_words >= len(words):
            break
    return chunks
