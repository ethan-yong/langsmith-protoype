import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PyPDF2 import PdfReader


class PDFSourceIndex:
    """
    In-memory index for verbatim PDF page text by source + page number.
    """

    def __init__(self, pdf_directory: str, cache_path: Optional[str] = None):
        self.pdf_directory = Path(pdf_directory)
        self.cache_path = Path(cache_path) if cache_path else None
        self._pages: Dict[Tuple[str, int], str] = {}

    def build(self) -> None:
        if not self.pdf_directory.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_directory}")

        for pdf_path in self.pdf_directory.glob("*.pdf"):
            reader = PdfReader(str(pdf_path))
            for page_index, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                self._pages[(pdf_path.name, page_index)] = text

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        data = {f"{source}::{page}": text for (source, page), text in self._pages.items()}
        self.cache_path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")

    def load_cache(self) -> bool:
        if not self.cache_path or not self.cache_path.exists():
            return False
        raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        pages: Dict[Tuple[str, int], str] = {}
        for key, text in raw.items():
            if "::" not in key:
                continue
            source, page_str = key.rsplit("::", 1)
            try:
                page = int(page_str)
            except ValueError:
                continue
            pages[(source, page)] = text
        self._pages = pages
        return True

    def get_page_text(self, source: str, page: int) -> Optional[str]:
        return self._pages.get((source, page))

    def build_context_from_sources(self, source_docs: Iterable[object]) -> str:
        """
        Build a context string using verbatim PDF page text.

        Expects items with:
          - metadata: dict containing 'source' and 'page'
        """
        parts: List[str] = []
        for doc in source_docs:
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source", "Unknown")
            page = metadata.get("page")
            if not isinstance(page, int):
                parts.append(f"Source: {source} (Page {page})\n")
                continue

            text = self.get_page_text(source, page) or ""
            parts.append(f"Source: {source} (Page {page})\n{text}")

        return "\n\n---\n\n".join(parts)
