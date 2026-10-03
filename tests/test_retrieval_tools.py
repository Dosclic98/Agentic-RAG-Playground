import json
import shutil
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from tools import PDFTools


class RetrievalToolsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / "data/pdfs"
        self.folder.mkdir(parents=True)
        self.tools = PDFTools(self.root)

    def write_pdf(self, name, texts, bookmarks=None):
        path = self.folder / name
        with pymupdf.open() as document:
            for text in texts:
                page = document.new_page(width=500, height=400)
                if text:
                    page.insert_textbox(pymupdf.Rect(30, 30, 470, 370), text, fontsize=12)
            document.set_metadata({"title": "Example report", "author": "Test author"})
            if bookmarks:
                document.set_toc(bookmarks)
            document.save(str(path))
        return path

    def table_pdf(self, name="TABLE_2022.pdf", bordered=True):
        path = self.folder / name
        with pymupdf.open() as document:
            page = document.new_page(width=400, height=300)
            page.insert_text((40, 40), "Amounts in USD millions")
            if bordered:
                for x in (40, 180, 280):
                    page.draw_line((x, 70), (x, 190))
                for y in (70, 100, 130, 160, 190):
                    page.draw_line((40, y), (280, y))
            cells = (("Year", "Revenue"), ("2021", "100.00"),
                     ("2022", "123.45"), ("2023", "150.50"))
            for row, values in enumerate(cells):
                for column, value in enumerate(values):
                    page.insert_text((50 if column == 0 else 190, 90 + row * 30), value)
            page.insert_text((40, 220), "Footnote: consolidated revenue.")
            document.save(str(path))
        return path

    def test_collection_ranks_citable_passages_across_documents(self):
        self.write_pdf("ALPHA_2022.pdf", ["Employees and offices.", "Subscription revenue increased."])
        self.write_pdf("BETA_2023.pdf", ["Revenue fell in the quarter."])
        result = json.loads(self.tools.search_collection("subscription revenue"))
        self.assertEqual(result["total_documents"], 2)
        self.assertEqual(result["documents_searched"], 2)
        self.assertTrue(result["search_complete"])
        self.assertEqual(result["results"][0]["path"], "data/pdfs/ALPHA_2022.pdf")
        self.assertEqual(result["results"][0]["page"], 2)
        self.assertEqual({row["path"] for row in result["results"]},
                         {"data/pdfs/ALPHA_2022.pdf", "data/pdfs/BETA_2023.pdf"})

    def test_collection_filename_filters_and_pagination(self):
        self.write_pdf("ALPHA_2022.pdf", ["Revenue first.", "Revenue second."])
        self.write_pdf("ALPHA_2023.pdf", ["Revenue in 2023."])
        self.write_pdf("BETA_2022.pdf", ["Mentions ALPHA revenue."])
        first = json.loads(self.tools.search_collection("revenue", company="alpha", year=2022, top_k=1))
        second = json.loads(self.tools.search_collection("revenue", company="alpha", year=2022,
                                                         top_k=1, offset=first["next_offset"]))
        self.assertEqual(first["total_documents"], 1)
        self.assertEqual(first["total_matches"], 2)
        self.assertEqual({first["results"][0]["page"], second["results"][0]["page"]}, {1, 2})
        self.assertIsNone(second["next_offset"])
        empty = json.loads(self.tools.search_collection("revenue", company="missing"))
        self.assertEqual(empty["total_matches"], 0)
        self.assertEqual(empty["total_documents"], 0)
        past_end = json.loads(self.tools.search_collection("revenue", offset=999))
        self.assertEqual(past_end["results"], [])
        self.assertFalse(past_end["has_more"])

    def test_collection_cache_is_persistent_and_refreshes_changed_files(self):
        path = self.write_pdf("ALPHA_2022.pdf", ["Original revenue information."])
        self.tools.search_collection("revenue")
        second = PDFTools(self.root)
        with patch.object(second._index, "page_loader", side_effect=AssertionError("Should use saved index")):
            result = json.loads(second.search_collection("revenue"))
        self.assertEqual(result["documents_refreshed"], 0)
        path.unlink()
        self.write_pdf("ALPHA_2022.pdf", ["Updated dividend information, with different length."])
        self.assertEqual(json.loads(second.search_collection("revenue"))["results"], [])
        self.assertTrue(json.loads(second.search_collection("dividend"))["results"])
        path.unlink()
        self.assertEqual(json.loads(second.search_collection("dividend"))["results"], [])

    def test_collection_reports_unreadable_and_image_only_documents(self):
        self.write_pdf("VALID.pdf", ["Revenue increased."])
        self.write_pdf("BLANK.pdf", [""])
        (self.folder / "INVALID.pdf").write_bytes(b"not a PDF")
        result = json.loads(self.tools.search_collection("revenue"))
        self.assertEqual(result["total_documents"], 3)
        self.assertEqual(result["total_errors"], 1)
        self.assertEqual(result["errors"][0]["path"], "data/pdfs/INVALID.pdf")
        self.assertEqual(result["documents_with_pages_without_text"][0]["pages"], [1])
        self.assertFalse(result["search_complete"])
        self.assertEqual(len(result["results"]), 1)

    def test_collection_rejects_invalid_queries_and_arguments(self):
        for arguments in ({"query": "!!!"}, {"query": "x", "offset": -1},
                          {"query": "x", "year": True}, {"query": "x", "year": 22},
                          {"query": "x", "top_k": 11}, {"query": "x", "path": ".."}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.tools.search_collection(**arguments)

    def test_collection_excludes_external_symlinks_and_fts_syntax(self):
        self.write_pdf("VALID.pdf", ["Revenue increased."])
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "external.pdf"
            target.write_bytes((self.folder / "VALID.pdf").read_bytes())
            (self.folder / "OUTSIDE.pdf").symlink_to(target)
            result = json.loads(self.tools.search_collection('"revenue" OR path:*'))
        self.assertEqual(result["total_documents"], 1)
        self.assertEqual(len(result["results"]), 1)

    def test_cache_cannot_be_written_outside_project(self):
        self.write_pdf("VALID.pdf", ["Revenue increased."])
        (self.root / ".rag_cache").symlink_to(self.root.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.tools.search_collection("revenue")

    def test_pdf_info_returns_metadata_outline_and_text_coverage(self):
        self.write_pdf("INFO.pdf", ["Introduction.", "", "Financial statements."],
                       bookmarks=[[1, "Introduction", 1], [1, "Financial statements", 3]])
        info = json.loads(self.tools.get_pdf_info("data/pdfs/INFO.pdf", outline_limit=1))
        self.assertEqual(info["total_pages"], 3)
        self.assertEqual(info["metadata"]["title"], "Example report")
        self.assertEqual(info["pages_without_text"], [2])
        self.assertEqual(info["outline"][0]["page"], 1)
        other = json.loads(self.tools.get_pdf_info("data/pdfs/INFO.pdf", outline_limit=1,
                                                  outline_offset=info["next_offset"]))
        self.assertEqual(other["outline"][0]["page"], 3)
        self.assertIsNone(other["next_offset"])

    def test_table_extracts_headers_units_footnotes_and_all_rows_with_cursor(self):
        self.table_pdf()
        first = json.loads(self.tools.extract_pdf_tables("data/pdfs/TABLE_2022.pdf", max_rows=2))
        self.assertEqual(first["table"]["headers"], ["Year", "Revenue"])
        self.assertIn("USD millions", first["context_before"])
        self.assertIn("consolidated revenue", first["context_after"])
        second = json.loads(self.tools.extract_pdf_tables("data/pdfs/TABLE_2022.pdf",
                                                           max_rows=2, **first["next"]))
        self.assertEqual(first["table"]["rows"] + second["table"]["rows"],
                         [["Year", "Revenue"], ["2021", "100.00"],
                          ["2022", "123.45"], ["2023", "150.50"]])
        self.assertIsNone(second["next"])

    def test_borderless_table_can_use_text_strategy(self):
        self.table_pdf(bordered=False)
        default = json.loads(self.tools.extract_pdf_tables("data/pdfs/TABLE_2022.pdf"))
        self.assertIsNone(default["table"])
        detected = json.loads(self.tools.extract_pdf_tables("data/pdfs/TABLE_2022.pdf", strategy="text"))
        self.assertGreater(detected["total_tables"], 0)
        self.assertIn("123.45", str(detected["table"]["rows"]))

    def test_table_validation_and_missing_tables(self):
        self.write_pdf("TEXT.pdf", ["Text with no tables."])
        result = json.loads(self.tools.extract_pdf_tables("data/pdfs/TEXT.pdf"))
        self.assertEqual(result["total_tables"], 0)
        self.assertIsNone(result["next"])
        for arguments in ({"page": 0}, {"page": 2}, {"table_index": -1}, {"row_offset": True},
                          {"max_rows": 101}, {"strategy": "unknown"}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.tools.extract_pdf_tables("data/pdfs/TEXT.pdf", **arguments)

    def test_new_document_tools_reject_protected_pdfs(self):
        path = self.write_pdf("PROTECTED.pdf", ["Secret revenue."])
        with pymupdf.open(str(path)) as document:
            encrypted = self.folder / "encrypted.pdf"
            document.save(str(encrypted), encryption=pymupdf.PDF_ENCRYPT_AES_256,
                          owner_pw="secret", user_pw="secret")
        for tool in (self.tools.get_pdf_info, self.tools.extract_pdf_tables):
            with self.subTest(tool=tool.__name__), self.assertRaisesRegex(ValueError, "Password-protected"):
                tool("data/pdfs/encrypted.pdf")
        result = json.loads(self.tools.search_collection("revenue"))
        self.assertEqual(result["total_errors"], 1)

    def test_ocr_missing_language_and_invalid_parameters_are_actionable(self):
        self.write_pdf("SCAN.pdf", [""])
        with self.assertRaisesRegex(ValueError, "setup_ocr.py"):
            self.tools.ocr_pdf_pages("data/pdfs/SCAN.pdf")
        for arguments in ({"language": "../eng"}, {"dpi": 301}, {"dpi": True},
                          {"start_page": 0}, {"end_page": 6}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.tools.ocr_pdf_pages("data/pdfs/SCAN.pdf", **arguments)

    def test_saved_ocr_is_used_in_document_search_collection_and_page_reads(self):
        pdf = self.write_pdf("SCAN.pdf", [""])
        self.tools._index.save_ocr(pdf, 1, "Recognized revenue is 123.45.", "eng", 200)
        tools = PDFTools(self.root)
        self.assertEqual(json.loads(tools.search_pdf("data/pdfs/SCAN.pdf", "revenue"))
                         ["results"][0]["text_source"], "ocr")
        collection = json.loads(tools.search_collection("revenue"))
        self.assertTrue(collection["search_complete"])
        self.assertEqual(collection["results"][0]["text_source"], "ocr")
        read = json.loads(tools.read_pdf_content("data/pdfs/SCAN.pdf"))
        self.assertIn("123.45", read["text"])
        self.assertEqual(read["text_source"], "ocr")
        info = json.loads(tools.get_pdf_info("data/pdfs/SCAN.pdf"))
        self.assertEqual(info["pages_without_native_text"], [1])
        self.assertEqual(info["pages_without_text"], [])

    def test_changed_pdf_invalidates_saved_ocr(self):
        pdf = self.write_pdf("SCAN.pdf", [""])
        self.tools._index.save_ocr(pdf, 1, "Old recognized revenue.", "eng", 200)
        pdf.unlink()
        self.write_pdf("SCAN.pdf", ["New dividend details in this replacement PDF."])
        self.assertNotIn("Old recognized", json.loads(self.tools.read_pdf_content("data/pdfs/SCAN.pdf"))["text"])
        self.assertEqual(json.loads(self.tools.search_collection("recognized"))["results"], [])

    def test_empty_ocr_does_not_erase_native_evidence(self):
        pdf = self.write_pdf("NATIVE.pdf", ["Revenue information."])
        for text in ("", " \n\t"):
            with self.subTest(text=text):
                self.tools._index.save_ocr(pdf, 1, text, "eng", 200)
                self.assertTrue(json.loads(self.tools.search_collection("revenue"))["results"])
                self.assertIn("Revenue", json.loads(self.tools.read_pdf_content("data/pdfs/NATIVE.pdf"))["text"])

    def test_actual_ocr_on_an_image_only_pdf(self):
        language = Path(__file__).resolve().parents[1] / "data/tessdata/eng.traineddata"
        if not language.is_file():
            self.skipTest("Run scripts/setup_ocr.py to enable the real OCR integration test.")
        destination = self.root / "data/tessdata"
        destination.mkdir()
        shutil.copyfile(language, destination / language.name)
        pdf = self.write_pdf("NATIVE.pdf", ["Revenue increased to 123.45 million."])
        with pymupdf.open(str(pdf)) as source:
            image = source[0].get_pixmap(dpi=150).tobytes("png")
        with pymupdf.open() as document:
            page = document.new_page(width=500, height=400)
            page.insert_image(page.rect, stream=image)
            document.save(str(self.folder / "SCAN.pdf"))
        result = json.loads(self.tools.ocr_pdf_pages("data/pdfs/SCAN.pdf", dpi=150))
        self.assertEqual(result["total_pages"], 1)
        self.assertIn("Revenue", result["pages"][0]["text"])
        self.assertIn("123.45", result["pages"][0]["text"])
        self.assertEqual(json.loads(self.tools.search_collection("revenue", company="SCAN"))
                         ["results"][0]["text_source"], "ocr")


class CalculatorTests(unittest.TestCase):
    def setUp(self):
        self.tools = PDFTools()

    def test_decimal_arithmetic_growth_and_rounding(self):
        for expression, expected in (("0.1 + 0.2", "0.3"), ("(150 - 100) / 100 * 100", "50"),
                                     ("round(123.456, 2)", "123.46"), ("round(2.5)", "2"),
                                     ("-5 // 2", "-3"), ("-5 % 2", "1"), ("2 ** 3", "8")):
            with self.subTest(expression=expression):
                value = json.loads(self.tools.calculate(expression))
                self.assertEqual(Decimal(value["result"]), Decimal(expected))

    def test_calculator_rejects_code_and_unbounded_arithmetic(self):
        for expression in ("__import__('os').system('id')", "(1).__class__", "[1, 2]", "True + 1",
                           "sum([1, 2])", "1 / 0", "10 ** 1000000", "1e1000", "round(1, 200)",
                           "round(1, 1.5)", "", "1+" * 300):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                self.tools.calculate(expression)


if __name__ == "__main__":
    unittest.main()
