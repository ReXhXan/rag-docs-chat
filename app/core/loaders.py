"""Document loaders: turn PDF / DOCX / TXT / MD files into (text, metadata) pages."""
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def load_document(path: str | Path) -> list[tuple[str, dict]]:
    """Return a list of (text, metadata) "pages" for one file.

    PDFs keep page numbers so answers can cite "file.pdf p.3".
    Other formats are returned as a single page.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        from pypdf import PdfReader  # lazy import keeps tests light
        reader = PdfReader(str(path))
        return [
            (page.extract_text() or "", {"source": path.name, "page": i + 1})
            for i, page in enumerate(reader.pages)
        ]
    if ext == ".docx":
        import docx
        text = "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
        return [(text, {"source": path.name})]
    return [(path.read_text(encoding="utf-8", errors="ignore"), {"source": path.name})]
