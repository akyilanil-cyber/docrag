from pathlib import Path

from pypdf import PdfReader

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


class UnsupportedFile(ValueError):
    pass


def validate_upload(filename: str, head: bytes) -> str:
    """Return the normalised extension or raise UnsupportedFile.

    Extension alone is not trusted: PDFs must start with the %PDF magic bytes and
    text files must be valid UTF-8 (checked on the first bytes).
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise UnsupportedFile("Only .pdf, .txt and .md files are supported")
    if ext == ".pdf":
        if not head.startswith(b"%PDF"):
            raise UnsupportedFile("File content is not a valid PDF")
    else:
        if b"\x00" in head:
            raise UnsupportedFile("File content is not text")
        try:
            head.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            # A multi-byte character may be cut at the end of the sample.
            if exc.start < len(head) - 4:
                raise UnsupportedFile("Text files must be UTF-8") from exc
    return ext


def extract_text(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(str(p))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return p.read_text(encoding="utf-8", errors="replace")
