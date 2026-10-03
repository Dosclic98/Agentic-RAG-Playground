import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from citations import CitationFormatter


class CitationTests(unittest.TestCase):
    def test_sources_are_numbered_deduplicated_and_keep_page_ranges(self):
        formatter = CitationFormatter()
        text = "A [data/pdfs/report.pdf, p. 12]; B [data/pdfs/report.pdf, pp. 14–16]; C [data/pdfs/report.pdf, p. 12]."
        self.assertEqual(formatter.format(text), "A [1]; B [2]; C [1].")
        self.assertEqual(list(formatter.sources), [("data/pdfs/report.pdf", "p. 12"),
                                                 ("data/pdfs/report.pdf", "pp. 14–16")])
        self.assertIn("data/pdfs/report.pdf", text)

    def test_partial_citation_is_retained_until_complete(self):
        formatter = CitationFormatter()
        partial = "Evidence [data/pdfs/report.pdf, p. "
        self.assertEqual(formatter.format(partial), partial)
        self.assertFalse(formatter.sources)
        self.assertEqual(formatter.format(partial + "12]"), "Evidence [1]")

    def test_code_examples_and_unrecognized_brackets_are_preserved(self):
        formatter = CitationFormatter()
        text = ("`[data/pdfs/report.pdf, p. 12]`\n\n"
                "```python\nprint('[data/pdfs/report.pdf, p. 12]')\n```\n\n"
                "[unknown] and [data/pdfs/report.pdf, p. 12]")
        formatted = formatter.format(text)
        self.assertIn("`[data/pdfs/report.pdf, p. 12]`", formatted)
        self.assertIn("print('[data/pdfs/report.pdf, p. 12]')", formatted)
        self.assertTrue(formatted.endswith("[unknown] and [1]"))

    def test_filenames_with_spaces_and_commas_are_preserved(self):
        formatter = CitationFormatter()
        self.assertEqual(formatter.format("Claim [data/pdfs/Report, annual 2022.PDF, pp. 1, 3]."), "Claim [1].")
        self.assertEqual(list(formatter.sources), [("data/pdfs/Report, annual 2022.PDF", "pp. 1, 3")])


if __name__ == "__main__":
    unittest.main()
