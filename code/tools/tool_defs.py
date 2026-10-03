"""Read-only PDF retrieval tools scoped to one project directory."""

import json
import math
import re
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader
import pymupdf

from .calculator import Calculator
from .collection_index import CollectionIndex

ROOT = Path(__file__).resolve().parents[2]


class PDFTools:
    """Discover, search, and read PDFs with an independent extraction cache."""

    def __init__(self, root: Path = ROOT, ocr_data_path: str = "data/tessdata"):
        self.root = Path(root).resolve()
        self.ocr_data_path = ocr_data_path
        self._collection_index = None
        # Each tool collection owns its bounded cache; file metadata invalidates it.
        self._extract_pages = lru_cache(maxsize=4)(self._read_pages)

    @property
    def registry(self) -> dict:
        """Return bound methods for Ollama schemas and tool-call dispatch."""
        return {tool.__name__: tool for tool in (
            self.list_directory, self.list_pdfs, self.search_pdf,
            self.read_pdf_pages, self.read_pdf_content,
            self.search_collection, self.extract_pdf_tables, self.get_pdf_info,
            self.ocr_pdf_pages, self.calculate,
        )}

    @property
    def _index(self):
        database = self._project_path(".rag_cache/search.sqlite3")
        if self._collection_index is None or self._collection_index.database_path != database:
            self._collection_index = CollectionIndex(self.root, database, self._native_pages)
        return self._collection_index

    @contextmanager
    def _open_pdf(self, pdf):
        with pymupdf.open(str(pdf)) as document:
            if document.needs_pass and not document.authenticate(""):
                raise ValueError("Password-protected PDFs are not supported.")
            yield document

    def _native_pages(self, pdf):
        with self._open_pdf(pdf) as document:
            # Keyword indexing needs text; rebuilding visual layout is much slower.
            # Table tools recover layout separately when the question requires it.
            return tuple(page.get_text() for page in document)

    def _pdf_files(self, directory):
        for candidate in sorted(directory.rglob("*")):
            if (candidate.suffix.lower() == ".pdf" and candidate.is_file()
                    and self.root in candidate.resolve().parents):
                yield candidate

    def _project_path(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("Path must stay inside the project.")
        return resolved

    def _pdf_path(self, path: str) -> Path:
        resolved = self._project_path(path)
        if resolved.suffix.lower() != ".pdf" or not resolved.is_file():
            raise ValueError("Path must identify an existing PDF file.")
        return resolved

    @staticmethod
    def _bounded(value: int, maximum: int, name: str) -> None:
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer between 1 and {maximum}.")

    @staticmethod
    def _read_pages(path: str, modified: int, size: int) -> tuple:
        # File metadata in the key invalidates cached text when a PDF changes.
        with open(path, "rb") as source:
            reader = PdfReader(source)
            if reader.is_encrypted and not reader.decrypt(""):
                raise ValueError("Password-protected PDFs are not supported.")
            return tuple(page.extract_text() or "" for page in reader.pages)

    def _pages(self, path: Path) -> tuple:
        stat = path.stat()
        pages = self._extract_pages(str(path), stat.st_mtime_ns, stat.st_size)
        recognized = self._index.ocr_text(path)
        return tuple(recognized.get(number) or text for number, text in enumerate(pages, 1))

    @staticmethod
    def _listing_page(items: list, limit: int, offset: int) -> dict:
        PDFTools._bounded(limit, 200, "limit")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer.")
        page = items[offset:offset + limit]
        has_more = offset + len(page) < len(items)
        return {"offset": offset, "limit": limit, "has_more": has_more,
                "next_offset": offset + len(page) if has_more else None,
                "truncated": has_more}

    def list_directory(self, path: str = ".", limit: int = 200, offset: int = 0) -> str:
        """List files and folders inside the project.

        Args:
            path: Directory path relative to the project root.
            limit: Maximum entries returned, from 1 to 200.
            offset: Number of entries to skip, starting at 0; use next_offset to continue.
        """
        directory = self._project_path(path)

        entries = sorted(directory.iterdir(), key=lambda item: item.name)
        pagination = self._listing_page(entries, limit, offset)
        return json.dumps({
            "path": str(directory.relative_to(self.root)),
            "entries": [
                {
                    "name": item.name,
                    "type": (
                        "symlink" if item.is_symlink()
                        else "directory" if item.is_dir()
                        else "file"
                    ),
                }
                for item in entries[offset:offset + limit]
            ],
            "total_entries": len(entries),
            **pagination,
        })

    def list_pdfs(self, path: str = "data/pdfs", name_contains: str = "", limit: int = 50, offset: int = 0) -> str:
        """Search all PDF filenames recursively, then paginate the matching results.

        A total_matches of 0 means no filename matches in the searched directory,
        even if an earlier unfiltered listing was truncated. This does not search text.

        Args:
            path: Directory relative to the project root.
            name_contains: Case-insensitive filename substring, e.g. ADOBE_2022.
            limit: Maximum files returned, from 1 to 200.
            offset: Number of matching files to skip, starting at 0; use next_offset to continue.
        """
        self._bounded(limit, 200, "limit")
        directory = self._project_path(path)
        if not directory.is_dir():
            raise ValueError("Path must identify an existing directory.")
        files = []
        scanned = 0
        for candidate in self._pdf_files(directory):
            scanned += 1
            if name_contains.casefold() not in candidate.name.casefold():
                continue
            files.append({"path": str(candidate.relative_to(self.root)),
                          "filename": candidate.name})
        pagination = self._listing_page(files, limit, offset)
        return json.dumps({"path": str(directory.relative_to(self.root)),
                           "name_contains": name_contains, "files": files[offset:offset + limit],
                           "total_matches": len(files), "total_pdfs_scanned": scanned,
                           "search_scope": "recursive PDF filenames, not document contents",
                           "filename_search_complete": True, **pagination})

    def read_pdf_pages(self, path: str, start_page: int = 1, end_page: int = 1) -> str:
        """Read up to five PDF pages for context and source verification.

        Args:
            path: PDF path relative to the project root, as returned by list_pdfs.
            start_page: First physical PDF page, numbered from 1.
            end_page: Last physical PDF page, inclusive; at most five pages per call.
        """
        pdf = self._pdf_path(path)
        if type(start_page) is not int or type(end_page) is not int:
            raise ValueError("Page numbers must be integers.")
        if start_page < 1 or end_page < start_page or end_page - start_page >= 5:
            raise ValueError("Specify an ascending range of one to five pages, starting at 1.")
        pages = self._pages(pdf)
        recognized = self._index.ocr_text(pdf)
        if end_page > len(pages):
            raise ValueError(f"PDF has {len(pages)} pages.")
        return json.dumps({
            "path": str(pdf.relative_to(self.root)), "filename": pdf.name,
            "total_pages": len(pages), "page_numbering": "1-based physical PDF pages",
            "pages": [{"page": number, "text": pages[number - 1][:12000],
                       "truncated": len(pages[number - 1]) > 12000,
                       "text_source": "ocr" if recognized.get(number) else "pdf_text",
                       "has_text": bool(pages[number - 1].strip())}
                      for number in range(start_page, end_page + 1)],
        })

    def read_pdf_content(self, path: str, page: int = 1, offset: int = 0, max_chars: int = 6000) -> str:
        """Read PDF text directly, with a continuation cursor for long pages.

        Args:
            path: PDF path relative to the project root, as returned by list_pdfs.
            page: Physical PDF page to read, numbered from 1.
            offset: Character offset within that page; start at 0.
            max_chars: Maximum characters returned, from 1 to 12000.
        """
        self._bounded(max_chars, 12000, "max_chars")
        if type(page) is not int or page < 1:
            raise ValueError("page must be an integer starting at 1.")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer.")
        pdf = self._pdf_path(path)
        pages = self._pages(pdf)
        if page > len(pages):
            raise ValueError(f"PDF has {len(pages)} pages.")
        text = pages[page - 1]
        recognized = self._index.ocr_text(pdf)
        if offset > len(text):
            raise ValueError(f"Page has {len(text)} characters; offset is out of range.")
        end = min(offset + max_chars, len(text))
        continuation = None
        if end < len(text):
            continuation = {"page": page, "offset": end}
        elif page < len(pages):
            continuation = {"page": page + 1, "offset": 0}
        return json.dumps({
            "path": str(pdf.relative_to(self.root)), "filename": pdf.name,
            "page": page, "total_pages": len(pages),
            "page_numbering": "1-based physical PDF pages",
            "offset": offset, "text": text[offset:end], "page_characters": len(text),
            "truncated": end < len(text), "next": continuation,
        "has_text": bool(text.strip()),
            "text_source": "ocr" if recognized.get(page) else "pdf_text",
            "note": "Image-only pages can be processed with ocr_pdf_pages. OCR text may contain errors.",
        })

    def search_pdf(self, path: str, query: str, top_k: int = 5) -> str:
        """Search a selected PDF using BM25 keyword ranking over overlapping passages.

        Args:
            path: PDF path relative to the project root, as returned by list_pdfs.
            query: Specific keywords to retrieve; reformulate if results are inadequate.
            top_k: Maximum passages returned, from 1 to 10.
        """
        self._bounded(top_k, 10, "top_k")
        terms = set(re.findall(r"\w+", query.casefold()))
        if not terms:
            raise ValueError("Query must contain at least one word or number.")
        pdf = self._pdf_path(path)
        pages = self._pages(pdf)
        recognized = self._index.ocr_text(pdf)
        chunks = []
        for page_number, text in enumerate(pages, 1):
            words = list(re.finditer(r"\S+", text))
            for start in range(0, len(words), 210):
                stop = min(start + 250, len(words))
                passage = text[words[start].start():words[stop - 1].end()]
                counts = Counter(re.findall(r"\w+", passage.casefold()))
                chunks.append((page_number, start, passage, counts, sum(counts.values())))
                if stop == len(words):
                    break
        results = []
        if chunks:
            average_length = sum(chunk[4] for chunk in chunks) / len(chunks)
            frequencies = {term: sum(term in chunk[3] for chunk in chunks) for term in terms}
            for page, start, text, counts, length in chunks:
                score = 0.0
                for term in terms:
                    frequency = counts[term]
                    if not frequency:
                        continue
                    idf = math.log(1 + (len(chunks) - frequencies[term] + 0.5)
                                   / (frequencies[term] + 0.5))
                    score += idf * frequency * 2.5 / (
                        frequency + 1.5 * (0.25 + 0.75 * length / average_length))
                if score > 0:
                    results.append({"page": page, "chunk_id": f"p{page}-w{start}",
                                    "text": text[:6000], "truncated": len(text) > 6000,
                                    "text_source": "ocr" if recognized.get(page) else "pdf_text",
                                    "score": round(score, 6)})
        results.sort(key=lambda item: (-item["score"], item["page"], item["chunk_id"]))
        return json.dumps({
            "path": str(pdf.relative_to(self.root)), "filename": pdf.name,
            "total_pages": len(pages), "page_numbering": "1-based physical PDF pages",
            "query": query, "retrieval": "BM25 keyword search", "results": results[:top_k],
            "pages_without_text": [number for number, text in enumerate(pages, 1)
                                   if not text.strip()],
            "note": "No keyword matches does not prove absence. Try other terms. "
                    "Use ocr_pdf_pages for pages without text. Verify numbers recognized by OCR.",
        })

    def search_collection(self, query: str, path: str = "data/pdfs", company: str = "",
                          year: int = 0, top_k: int = 5, offset: int = 0) -> str:
        """Search PDF contents across a directory using a cached BM25 keyword index.

        Args:
            query: Focused keywords; matches any keyword, ranked by relevance.
            path: Directory relative to the project root, searched recursively.
            company: Optional case-insensitive company substring in filenames, not contents.
            year: Optional four-digit filename year; 0 searches all years.
            top_k: Maximum passages returned per call, from 1 to 10.
            offset: Number of ranked passages to skip; use next_offset to continue.
        """
        self._bounded(top_k, 10, "top_k")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a nonnegative integer.")
        if type(year) is not int or (year != 0 and not 1000 <= year <= 9999):
            raise ValueError("year must be 0 or a four-digit integer.")
        terms = sorted(set(re.findall(r"[^\W_]+", query.casefold())))
        if not terms:
            raise ValueError("Query must contain at least one word or number.")
        directory = self._project_path(path)
        if not directory.is_dir():
            raise ValueError("Path must identify an existing directory.")
        pdfs = [pdf for pdf in self._pdf_files(directory)
                if company.casefold() in pdf.name.casefold()
                and (not year or re.search(rf"(?<!\d){year}(?!\d)", pdf.name))]
        result = self._index.search(pdfs, terms, top_k, offset)
        has_more = offset + len(result["results"]) < result["total_matches"]
        return json.dumps({
            "path": str(directory.relative_to(self.root)), "query": query,
            "company": company, "year": year, "filter_scope": "PDF filenames",
            "total_documents": len(pdfs), "retrieval": "SQLite FTS5 BM25 keyword search",
            "page_numbering": "1-based physical PDF pages", "offset": offset,
            "limit": top_k, "has_more": has_more, "truncated": has_more,
            "next_offset": offset + len(result["results"]) if has_more else None,
            **result,
            "note": "The index refreshes changed PDFs automatically. Missing text and extraction errors "
                    "limit coverage. No keyword matches does not establish absence; try synonyms or OCR.",
        })

    def get_pdf_info(self, path: str, outline_limit: int = 100, outline_offset: int = 0) -> str:
        """Inspect PDF metadata, bookmarks, and text coverage before targeted retrieval.

        Args:
            path: PDF path relative to the project root.
            outline_limit: Maximum bookmarks returned, from 1 to 200.
            outline_offset: Number of bookmarks to skip; use next_offset to continue.
        """
        pdf = self._pdf_path(path)
        recognized = self._index.ocr_text(pdf)
        with self._open_pdf(pdf) as document:
            outline = [{"level": level, "title": title, "page": page if page > 0 else None}
                       for level, title, page in document.get_toc()]
            pagination = self._listing_page(outline, outline_limit, outline_offset)
            missing = [number for number, page in enumerate(document, 1)
                       if not page.get_text().strip()]
            remaining = [number for number in missing if not recognized.get(number)]
            languages = sorted(file.stem for file in self._project_path(self.ocr_data_path).glob("*.traineddata")
                               if self.root in file.resolve().parents)
            return json.dumps({
                "path": str(pdf.relative_to(self.root)), "filename": pdf.name,
                "total_pages": len(document), "metadata": document.metadata,
                "page_numbering": "1-based physical PDF pages",
                "outline": outline[outline_offset:outline_offset + outline_limit],
                "total_outline_entries": len(outline), **pagination,
                "pages_without_native_text": missing[:200], "total_pages_without_native_text": len(missing),
                "pages_without_text": remaining[:200], "total_pages_without_text": len(remaining),
                "coverage_lists_truncated": len(missing) > 200 or len(recognized) > 200,
                "pages_with_cached_ocr": sorted(page for page, text in recognized.items() if text.strip())[:200],
                "total_pages_with_cached_ocr": sum(bool(text.strip()) for text in recognized.values()),
                "available_ocr_languages": languages,
                "note": "Missing bookmarks do not mean missing sections. Pages without text may be blank "
                        "or scanned; use ocr_pdf_pages when needed.",
            })

    def extract_pdf_tables(self, path: str, page: int = 1, table_index: int = 0,
                           row_offset: int = 0, max_rows: int = 50, strategy: str = "lines") -> str:
        """Extract one table with headers, raw cells, context, and a continuation cursor.

        Args:
            path: PDF path relative to the project root.
            page: Physical PDF page numbered from 1; inspect one page per call.
            table_index: Zero-based table index on the selected page.
            row_offset: Number of rows to skip in that table, starting at 0.
            max_rows: Maximum rows returned, from 1 to 100.
            strategy: Table detection strategy: lines, lines_strict, or text for borderless tables.
        """
        self._bounded(max_rows, 100, "max_rows")
        if type(page) is not int or page < 1:
            raise ValueError("page must be an integer starting at 1.")
        for value, name in ((table_index, "table_index"), (row_offset, "row_offset")):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if strategy not in ("lines", "lines_strict", "text"):
            raise ValueError("strategy must be lines, lines_strict, or text.")
        pdf = self._pdf_path(path)
        with self._open_pdf(pdf) as document:
            if page > len(document):
                raise ValueError(f"PDF has {len(document)} pages.")
            selected = document[page - 1]
            tables = selected.find_tables(strategy=strategy).tables
            result = {"path": str(pdf.relative_to(self.root)), "filename": pdf.name,
                      "page": page, "total_pages": len(document),
                      "page_numbering": "1-based physical PDF pages", "strategy": strategy,
                      "total_tables": len(tables), "table_index": table_index, "row_offset": row_offset,
                      "note": "Verify detected headers and merged cells. Context preserves possible units "
                              "and footnotes without assigning meanings. Scanned tables require text/layout "
                              "reconstruction; OCR text alone does not create detectable table cells."}
            if not tables and table_index == 0 and row_offset == 0:
                return json.dumps({**result, "table": None, "truncated": False, "next": None,
                                   "hint": "Try strategy=text for borderless tables or inspect page/OCR text."})
            if table_index >= len(tables):
                raise ValueError(f"Page has {len(tables)} tables.")
            table = tables[table_index]
            rows = table.extract()
            if row_offset > len(rows):
                raise ValueError(f"Table has {len(rows)} rows; row_offset is out of range.")
            end = min(row_offset + max_rows, len(rows))
            continuation = None
            if end < len(rows):
                continuation = {"page": page, "table_index": table_index, "row_offset": end}
            elif table_index + 1 < len(tables):
                continuation = {"page": page, "table_index": table_index + 1, "row_offset": 0}
            before = selected.get_text(clip=pymupdf.Rect(0, 0, selected.rect.width, table.bbox[1]), sort=True)
            after = selected.get_text(clip=pymupdf.Rect(0, table.bbox[3], selected.rect.width,
                                                       selected.rect.height), sort=True)
            return json.dumps({**result,
                "table": {"bbox": list(table.bbox), "headers": table.header.names,
                          "header_external": table.header.external,
                          "total_rows": len(rows), "total_columns": table.col_count,
                          "rows": rows[row_offset:end]},
                "context_before": before[-3000:], "context_after": after[:3000],
                "context_truncated": len(before) > 3000 or len(after) > 3000,
                "truncated": end < len(rows), "next": continuation,
            })

    def ocr_pdf_pages(self, path: str, start_page: int = 1, end_page: int = 1,
                      language: str = "eng", dpi: int = 200) -> str:
        """Recognize scanned PDF pages locally and cache text for later reads and searches.

        Args:
            path: PDF path relative to the project root.
            start_page: First physical PDF page numbered from 1.
            end_page: Last physical PDF page inclusive; at most five pages per call.
            language: Installed Tesseract language code, e.g. eng, ita, or eng+ita.
            dpi: Rendering resolution from 72 to 300; higher values use more CPU and memory.
        """
        if (type(start_page) is not int or type(end_page) is not int or start_page < 1
                or end_page < start_page or end_page - start_page >= 5):
            raise ValueError("Specify an ascending range of one to five pages, starting at 1.")
        if type(dpi) is not int or not 72 <= dpi <= 300:
            raise ValueError("dpi must be an integer between 72 and 300.")
        if not re.fullmatch(r"[a-z]{3}(?:_[a-zA-Z0-9]+)?(?:\+[a-z]{3}(?:_[a-zA-Z0-9]+)?){0,3}", language):
            raise ValueError("Use Tesseract language codes such as eng, ita, or eng+ita.")
        tessdata = self._project_path(self.ocr_data_path)
        for code in language.split("+"):
            data_file = self._project_path(str(tessdata / f"{code}.traineddata"))
            if not data_file.is_file():
                raise ValueError(f"OCR language data is missing for {code}. "
                                 f"Run python3 scripts/setup_ocr.py --languages {code}.")
        pdf = self._pdf_path(path)
        results = []
        with self._open_pdf(pdf) as document:
            if end_page > len(document):
                raise ValueError(f"PDF has {len(document)} pages.")
            total_pages = len(document)
            for number in range(start_page, end_page + 1):
                selected = document[number - 1]
                if selected.rect.width * selected.rect.height * (dpi / 72) ** 2 > 40000000:
                    raise ValueError("Rendered page would exceed 40 million pixels; use a lower dpi.")
                textpage = selected.get_textpage_ocr(language=language, dpi=dpi, full=True,
                                                      tessdata=str(tessdata))
                text = selected.get_text(textpage=textpage, sort=True)
                self._index.save_ocr(pdf, number, text, language, dpi)
                results.append({"page": number, "offset": 0, "text": text[:12000], "text_source": "ocr",
                                "has_text": bool(text.strip()), "truncated": len(text) > 12000,
                                "next": {"page": number, "offset": 12000} if len(text) > 12000 else None})
        return json.dumps({
            "path": str(pdf.relative_to(self.root)), "filename": pdf.name,
            "total_pages": total_pages, "page_numbering": "1-based physical PDF pages",
            "language": language, "dpi": dpi, "pages": results, "cached": True,
            "note": "OCR may misread numbers or symbols. Cached text is available to search_pdf, "
                    "search_collection, and page reads. Use read_pdf_content for truncated OCR text.",
        })

    def calculate(self, expression: str) -> str:
        """Calculate arithmetic with 50-digit decimal precision and no Python execution.

        Args:
            expression: Numbers, +, -, *, /, //, %, **, parentheses, or round(value, digits).
        """
        return json.dumps(Calculator.evaluate(expression))
