"""Persistent, incremental keyword indexing and OCR text for one PDF collection."""

import json
import re
import sqlite3
from contextlib import closing


class CollectionIndex:
    """Keep extracted passages on disk so repeated searches do not reopen PDFs."""

    def __init__(self, root, database_path, page_loader):
        self.root = root
        self.database_path = database_path
        self.page_loader = page_loader

    def _connect(self):
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.database_path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    path TEXT PRIMARY KEY, modified INTEGER, size INTEGER,
                    total_pages INTEGER, blank_pages TEXT, error TEXT
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                    text, path UNINDEXED, page UNINDEXED,
                    chunk_id UNINDEXED, source UNINDEXED
                );
                CREATE TABLE IF NOT EXISTS ocr_pages (
                    path TEXT, page INTEGER, text TEXT, language TEXT, dpi INTEGER,
                    PRIMARY KEY (path, page)
                );
            """)
        except Exception:
            connection.close()
            raise
        return connection

    @staticmethod
    def _chunks(path, page, text, source):
        words = list(re.finditer(r"\S+", text))
        for start in range(0, len(words), 210):
            stop = min(start + 250, len(words))
            passage = text[words[start].start():words[stop - 1].end()]
            yield passage, path, page, f"p{page}-w{start}", source
            if stop == len(words):
                break

    def _ensure_document(self, connection, pdf):
        path = str(pdf.relative_to(self.root))
        stat = pdf.stat()
        previous = connection.execute(
            "SELECT * FROM documents WHERE path=?", (path,)).fetchone()
        if previous and (previous["modified"], previous["size"]) == (stat.st_mtime_ns, stat.st_size):
            return previous, False
        error = None
        try:
            pages = self.page_loader(pdf)
        except Exception as exc:
            pages = ()
            error = str(exc)
        with connection:
            # Drop stale passages and OCR whenever the source document changes.
            if previous:
                connection.execute("DELETE FROM passages WHERE path=?", (path,))
                connection.execute("DELETE FROM ocr_pages WHERE path=?", (path,))
            blank = [number for number, text in enumerate(pages, 1) if not text.strip()]
            connection.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                               (path, stat.st_mtime_ns, stat.st_size, len(pages), json.dumps(blank), error))
            for number, text in enumerate(pages, 1):
                connection.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?)",
                                       self._chunks(path, number, text, "pdf_text"))
        return connection.execute("SELECT * FROM documents WHERE path=?", (path,)).fetchone(), True

    def search(self, pdfs, terms, top_k, offset):
        with closing(self._connect()) as connection:
            with connection:
                for row in connection.execute("SELECT path FROM documents").fetchall():
                    candidate = self.root / row["path"]
                    if self.root not in candidate.resolve().parents or not candidate.is_file():
                        connection.execute("DELETE FROM passages WHERE path=?", (row["path"],))
                        connection.execute("DELETE FROM ocr_pages WHERE path=?", (row["path"],))
                        connection.execute("DELETE FROM documents WHERE path=?", (row["path"],))
            connection.execute("CREATE TEMP TABLE selected (path TEXT PRIMARY KEY)")
            errors = []
            blank = []
            refreshed = 0
            indexed = 0
            for pdf in pdfs:
                path = str(pdf.relative_to(self.root))
                try:
                    document, changed = self._ensure_document(connection, pdf)
                except OSError as exc:
                    errors.append({"path": path, "error": str(exc)})
                    continue
                refreshed += changed
                if document["error"]:
                    errors.append({"path": path, "error": document["error"]})
                    continue
                indexed += 1
                connection.execute("INSERT INTO selected VALUES (?)", (path,))
                recovered = {row["page"] for row in connection.execute(
                    "SELECT page, text FROM ocr_pages WHERE path=?", (path,)) if row["text"].strip()}
                missing = [page for page in json.loads(document["blank_pages"]) if page not in recovered]
                if missing:
                    blank.append({"path": path, "pages": missing[:200],
                                  "total_pages_without_text": len(missing),
                                  "truncated": len(missing) > 200})
            # Plain words are quoted individually: user text is never FTS syntax or SQL.
            query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            where = "FROM passages JOIN selected ON passages.path=selected.path WHERE passages MATCH ?"
            count = connection.execute("SELECT count(*) " + where, (query,)).fetchone()[0]
            rows = connection.execute(
                "SELECT passages.*, bm25(passages) AS rank " + where
                + " ORDER BY rank, passages.path, CAST(page AS INTEGER), chunk_id LIMIT ? OFFSET ?",
                (query, top_k, offset)).fetchall()
            results = [{"path": row["path"], "filename": row["path"].rsplit("/", 1)[-1],
                        "page": int(row["page"]), "chunk_id": row["chunk_id"],
                        "text": row["text"][:6000], "truncated": len(row["text"]) > 6000,
                        "text_source": row["source"], "score": -row["rank"]} for row in rows]
            return {"results": results, "total_matches": count,
                    "documents_searched": indexed, "documents_refreshed": refreshed,
                    "errors": errors[:50], "total_errors": len(errors),
                    "errors_truncated": len(errors) > 50,
                    "documents_with_pages_without_text": blank[:50],
                    "total_documents_with_pages_without_text": len(blank),
                    "text_coverage_truncated": len(blank) > 50,
                    "search_complete": not errors and not blank}

    def ocr_text(self, pdf):
        """Read valid cached OCR without creating an index or extracting native text."""
        if not self.database_path.exists():
            return {}
        path = str(pdf.relative_to(self.root))
        stat = pdf.stat()
        with closing(self._connect()) as connection:
            document = connection.execute("SELECT modified, size FROM documents WHERE path=?", (path,)).fetchone()
            if not document or tuple(document) != (stat.st_mtime_ns, stat.st_size):
                return {}
            return {row["page"]: row["text"] for row in connection.execute(
                "SELECT page, text FROM ocr_pages WHERE path=?", (path,)) if row["text"].strip()}

    def save_ocr(self, pdf, page, text, language, dpi):
        """Make recognized text available to later collection and document searches."""
        with closing(self._connect()) as connection:
            document, _ = self._ensure_document(connection, pdf)
            if document["error"]:
                raise ValueError(document["error"])
            path = str(pdf.relative_to(self.root))
            with connection:
                connection.execute("INSERT OR REPLACE INTO ocr_pages VALUES (?, ?, ?, ?, ?)",
                                   (path, page, text, language, dpi))
                if text.strip():
                    connection.execute("DELETE FROM passages WHERE path=? AND page=?", (path, page))
                    connection.executemany("INSERT INTO passages VALUES (?, ?, ?, ?, ?)",
                                           self._chunks(path, page, text, "ocr"))
