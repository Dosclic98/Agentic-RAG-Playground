# Agentic-RAG-Playground

A terminal question-answering agent for local PDFs, using Ollama and keyword retrieval.
Place PDFs in `data/pdfs/`, start Ollama with the configured model available, and run:

```bash
python3 main.py
```

Use `/clear` to reset the conversation, `/exit` to quit, or Ctrl+C to interrupt.
Use `/verbose` to toggle tool arguments and generation statistics; `/verbose on`
and `/verbose off` set the mode explicitly. Warnings and errors appear in both modes.
The agent can continue making tool calls until it finishes or is interrupted.

Interactive terminals render streamed Markdown with headings, bold text, lists,
aligned tables, and highlighted code. Completed blocks stay in scrollback, while
the current block updates live. For a very long unfinished table or paragraph,
the live view shows the latest lines; the complete block is printed when it finishes.
The temporary thinking line remains grey and is erased when an answer or another
message is printed. Tool calls appear as brief progress messages by default.

PDF citations display as `[1]`, `[2]`, etc., with full paths and page numbers in a
source list under each response. The model's saved conversation retains its full
inline citations. Redirected output keeps raw Markdown and full citations, without
terminal colors or live redraw controls.

The code is organized by responsibility:

| Class | File | Responsibility |
| --- | --- | --- |
| `AgentConfig` | `code/config.py` | Model, Ollama host, context window, output length, and project root. |
| `PDFChatAgent` | `code/run_agent.py` | Streaming, tool dispatch, compatibility retries, and the terminal loop. |
| `Conversation` | `code/conversation.py` | Chat history, resets, failure recovery, and evidence compaction. |
| `PDFTools` | `code/tools/tool_defs.py` | PDF discovery, search, reading, metadata, tables, OCR, and calculations. |
| `CollectionIndex` | `code/tools/collection_index.py` | Persistent keyword index and recognized page text. |
| `Calculator` | `code/tools/calculator.py` | Restricted arithmetic with decimal precision. |
| `TerminalOutput` | `code/terminal_output.py` | Streamed answers and transient thinking display. |
| `CitationFormatter` | `code/citations.py` | Compact display references with complete source details. |

Edit the defaults in `code/config.py` to change model settings. The default model
is `qwen3.8:27b-q8_0`, with a 65,536-token context and an 8,192-token output limit.
The PDF question-answering instructions live in `code/prompts.py`.
Set `output_width` to change the maximum rendered width (100 columns by default),
or set `verbose=True` to start with detailed logs. Rendering also fits the terminal's
actual width. These settings are in code; no environment variables are required.

For use from Python, create an agent with its own configuration and conversation:

```python
from code.config import AgentConfig
from code.run_agent import PDFChatAgent

agent = PDFChatAgent(AgentConfig(context_length=65536))
agent.run()
```

Each agent owns its conversation and tool registry. Each `PDFTools` instance owns
its extraction cache and confines paths to its project root. Its tools return JSON
with source paths and physical PDF page numbers.

The model has these retrieval tools available:

| Tool | Behavior |
| --- | --- |
| `list_directory`, `list_pdfs` | Discover files with offset pagination and filename filters. |
| `search_pdf` | Rank keyword passages within one PDF. |
| `read_pdf_pages`, `read_pdf_content` | Read physical pages, including cached OCR text, with truncation markers and continuation cursors. |
| `search_collection` | Search all PDF contents recursively, optionally filtering company and year in filenames. Results contain source paths and pages; `next_offset` retrieves further ranked passages. |
| `get_pdf_info` | Return metadata, paginated bookmarks, text coverage, and available OCR languages. |
| `extract_pdf_tables` | Extract raw table rows and detected headers, with surrounding text for units and footnotes. Follow the `next` cursor across rows and tables on one page; use `strategy="text"` for borderless tables. |
| `ocr_pdf_pages` | Recognize up to five scanned pages locally and cache the text for document search, collection search, and page reading. |
| `calculate` | Evaluate arithmetic using 50-digit decimal precision. Supports `+ - * / // % **` and `round(value, digits)`; returns the result as a decimal string. |

Collection search builds an index under `.rag_cache/` on first use. Later searches
reuse it, refresh modified documents, and discard entries for deleted files.
Errors and pages without text are reported explicitly; `search_complete=false`
means the searched content may be incomplete. Company and year filters inspect
filenames, so omit them when searching for mentions inside other companies' PDFs.
Index scores rank passages; they do not count every mention of a subject.

Install the PDF dependency and English/Italian OCR data with:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/setup_ocr.py
```

OCR uses PyMuPDF's integrated Tesseract support and local language files;
no environment variables are required. The setup script downloads files from
[Tesseract's official tessdata_fast repository](https://github.com/tesseract-ocr/tessdata_fast).
For additional languages, run `python3 scripts/setup_ocr.py --languages deu`, for
example. Set `ocr_data_path` in `code/config.py` if you use a different folder inside
the project, and pass the same folder to the setup script with `--directory`.
See [PyMuPDF's OCR setup documentation](https://pymupdf.readthedocs.io/en/latest/installation.html#enabling-integrated-ocr-support).

OCR can misread figures, and table extraction can misidentify merged headers.
The tools label OCR text and preserve table context so answers can state uncertainty.
Recognized text does not reconstruct scanned table geometry. The original PDFs
are never modified; the index and downloaded language files are ignored by Git.
Removing `.rag_cache/` resets the index and cached OCR text.

Install the test dependencies and run the checks with:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m unittest discover -s tests -v
```

The terminal tests use `pyte` to check scrollback, table rows, and cursor restoration.
