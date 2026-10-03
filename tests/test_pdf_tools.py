import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter
from ollama._utils import convert_function_to_tool

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tools import PDFTools


class PDFToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = PDFTools(self.root)
        (self.root / "data/pdfs").mkdir(parents=True)
        self.pdf = self.root / "data/pdfs/report.PDF"
        self.write_pdf()

    def write_pdf(self, page_count=2, password=None):
        writer = PdfWriter()
        for _ in range(page_count):
            writer.add_blank_page(width=100, height=100)
        if password:
            writer.encrypt(password)
        with self.pdf.open("wb") as output:
            writer.write(output)

    def test_discovery_returns_usable_paths_and_limits(self):
        (self.pdf.parent / "other.pdf").write_bytes(self.pdf.read_bytes())
        result = json.loads(self.tools.list_pdfs(limit=1))
        self.assertEqual(result["total_matches"], 2)
        self.assertTrue(result["truncated"])
        result = json.loads(self.tools.list_pdfs(name_contains="REPORT"))
        self.assertEqual(result["files"][0]["path"], "data/pdfs/report.PDF")

    def test_bound_methods_generate_ollama_tool_schemas(self):
        expected = {
            "list_directory": {"path", "limit", "offset"},
            "list_pdfs": {"path", "name_contains", "limit", "offset"},
            "search_pdf": {"path", "query", "top_k"},
            "read_pdf_pages": {"path", "start_page", "end_page"},
            "read_pdf_content": {"path", "page", "offset", "max_chars"},
            "search_collection": {"query", "path", "company", "year", "top_k", "offset"},
            "extract_pdf_tables": {"path", "page", "table_index", "row_offset", "max_rows", "strategy"},
            "get_pdf_info": {"path", "outline_limit", "outline_offset"},
            "ocr_pdf_pages": {"path", "start_page", "end_page", "language", "dpi"},
            "calculate": {"expression"},
        }
        self.assertEqual(set(self.tools.registry), set(expected))
        for name, method in self.tools.registry.items():
            with self.subTest(tool=name):
                definition = convert_function_to_tool(method).function
                self.assertEqual(definition.name, name)
                self.assertEqual(set(definition.parameters.properties), expected[name])
                self.assertTrue(definition.description)

    def test_instances_use_their_own_project_roots_and_caches(self):
        other_root = self.root / "other-project"
        other_pdf = other_root / "data/pdfs/report.PDF"
        other_pdf.parent.mkdir(parents=True)
        writer = PdfWriter()
        for _ in range(3):
            writer.add_blank_page(width=100, height=100)
        with other_pdf.open("wb") as output:
            writer.write(output)
        other = PDFTools(other_root)
        path = "data/pdfs/report.PDF"
        self.assertEqual(json.loads(self.tools.read_pdf_content(path))["total_pages"], 2)
        self.assertEqual(json.loads(other.read_pdf_content(path))["total_pages"], 3)
        self.tools._extract_pages.cache_clear()
        self.assertEqual(other._extract_pages.cache_info().currsize, 1)

    def test_paths_cannot_escape_project(self):
        with self.assertRaises(ValueError):
            self.tools.list_directory("..")
        (self.root / "outside").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.tools.list_pdfs("outside")

    def test_pagination_reaches_files_after_first_two_hundred(self):
        for index in range(205):
            (self.pdf.parent / f"company-{index:03d}.pdf").touch()
        first = json.loads(self.tools.list_pdfs(limit=200))
        second = json.loads(self.tools.list_pdfs(limit=200, offset=first["next_offset"]))
        paths = [item["path"] for page in (first, second) for item in page["files"]]
        self.assertEqual(len(paths), 206)
        self.assertEqual(len(set(paths)), 206)
        self.assertEqual(paths, sorted(paths))
        self.assertEqual(first["next_offset"], 200)
        self.assertFalse(second["has_more"])
        self.assertIsNone(second["next_offset"])

    def test_offset_applies_after_filename_filter(self):
        for name in ("ADOBE_2020.pdf", "ADOBE_2021.pdf", "COCACOLA.pdf"):
            (self.pdf.parent / name).touch()
        result = json.loads(self.tools.list_pdfs(name_contains="adobe", limit=1, offset=1))
        self.assertEqual(result["total_matches"], 2)
        self.assertEqual(result["files"][0]["filename"], "ADOBE_2021.pdf")
        self.assertIsNone(result["next_offset"])

    def test_filtered_search_is_complete_even_when_unfiltered_list_is_truncated(self):
        for index in range(205):
            (self.pdf.parent / f"company-{index:03d}.pdf").touch()
        self.assertTrue(json.loads(self.tools.list_pdfs(limit=200))["truncated"])
        result = json.loads(self.tools.list_pdfs(name_contains="Fincantieri"))
        self.assertEqual(result["total_matches"], 0)
        self.assertEqual(result["total_pdfs_scanned"], 206)
        self.assertTrue(result["filename_search_complete"])
        self.assertFalse(result["has_more"])
        self.assertFalse(result["truncated"])
        self.assertIsNone(result["next_offset"])

    def test_directory_pagination_and_offsets_beyond_end(self):
        (self.pdf.parent / "other.pdf").touch()
        first = json.loads(self.tools.list_directory("data/pdfs", limit=1))
        second = json.loads(self.tools.list_directory("data/pdfs", limit=1, offset=first["next_offset"]))
        self.assertEqual(first["total_entries"], 2)
        self.assertNotEqual(first["entries"], second["entries"])
        self.assertIsNone(second["next_offset"])
        for tool, field in [(self.tools.list_directory, "entries"), (self.tools.list_pdfs, "files")]:
            result = json.loads(tool("data/pdfs", offset=1000))
            self.assertEqual(result[field], [])
            self.assertFalse(result["has_more"])
            self.assertIsNone(result["next_offset"])

    def test_listing_rejects_invalid_offsets_and_limits(self):
        for tool in (self.tools.list_directory, self.tools.list_pdfs):
            for arguments in [{"offset": -1}, {"offset": 1.5}, {"offset": True},
                              {"limit": 0}, {"limit": 201}]:
                with self.assertRaises(ValueError):
                    tool("data/pdfs", **arguments)

    def test_blank_pages_are_explicit(self):
        result = json.loads(self.tools.read_pdf_pages("data/pdfs/report.PDF", 2, 2))
        self.assertEqual(result["pages"][0]["page"], 2)
        self.assertFalse(result["pages"][0]["has_text"])
        search = json.loads(self.tools.search_pdf("data/pdfs/report.PDF", "revenue"))
        self.assertEqual(search["results"], [])
        self.assertEqual(search["pages_without_text"], [1, 2])

    def test_invalid_ranges_and_limits(self):
        for start, end in [(0, 1), (2, 1), (1, 6), (1, 3)]:
            with self.assertRaises(ValueError):
                self.tools.read_pdf_pages("data/pdfs/report.PDF", start, end)
        with self.assertRaises(ValueError):
            self.tools.search_pdf("data/pdfs/report.PDF", "revenue", top_k=11)
        with self.assertRaises(ValueError):
            self.tools.search_pdf("data/pdfs/report.PDF", "!!!")

    def test_search_returns_ranked_citable_passages(self):
        pages = ("Company headquarters and employees.",
                 "Subscription revenue increased in 2022.", "")
        with patch.object(self.tools, "_pages", return_value=pages):
            result = json.loads(self.tools.search_pdf("data/pdfs/report.PDF", "subscription revenue"))
            self.assertEqual(len(result["results"]), 1)
            self.assertEqual(result["results"][0]["page"], 2)
            self.assertIn("Subscription revenue", result["results"][0]["text"])
            self.assertEqual(result["pages_without_text"], [3])
            self.assertEqual(json.loads(self.tools.search_pdf(
                "data/pdfs/report.PDF", "unfindable"))["results"], [])

    def test_cache_refreshes_when_pdf_changes(self):
        self.assertEqual(len(self.tools._pages(self.pdf)), 2)
        self.write_pdf(page_count=3)
        self.assertEqual(len(self.tools._pages(self.pdf)), 3)

    def test_password_protected_pdf_is_rejected(self):
        self.write_pdf(password="secret")
        with self.assertRaisesRegex(ValueError, "Password-protected"):
            self.tools.read_pdf_pages("data/pdfs/report.PDF")

    def test_content_cursor_reads_all_text_without_gaps(self):
        pages = ("abcdefghijk", "second page")
        cursor = {"page": 1, "offset": 0}
        pieces = {1: [], 2: []}
        with patch.object(self.tools, "_pages", return_value=pages):
            while cursor:
                result = json.loads(self.tools.read_pdf_content(
                    "data/pdfs/report.PDF", max_chars=4, **cursor))
                pieces[result["page"]].append(result["text"])
                cursor = result["next"]
        self.assertEqual("".join(pieces[1]), pages[0])
        self.assertEqual("".join(pieces[2]), pages[1])

    def test_content_blank_page_and_invalid_cursor(self):
        result = json.loads(self.tools.read_pdf_content("data/pdfs/report.PDF"))
        self.assertFalse(result["has_text"])
        self.assertEqual(result["next"], {"page": 2, "offset": 0})
        for arguments in [{"page": 0}, {"page": 3}, {"offset": -1},
                          {"offset": 1}, {"max_chars": 12001}]:
            with self.assertRaises(ValueError):
                self.tools.read_pdf_content("data/pdfs/report.PDF", **arguments)


if __name__ == "__main__":
    unittest.main()
