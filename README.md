# Agentic-RAG-Playground

A terminal question-answering agent for local PDFs, using Ollama and keyword retrieval,
with optional web search for external context.

## Setup (Linux / Bash)

Start from a checkout of this repository. You need Python with `pip` and `venv`
support; the project has been tested with Python 3.9.25. For NVIDIA GPU inference,
the NVIDIA driver must also be installed; `nvidia-smi` should show your GPU.
Package installation, model downloads, and OCR language downloads require internet access.

### 1. Create and activate the virtual environment

Run these commands from the project directory, adjusting the path if needed:

```bash
cd ~/Agentic-RAG-Playground
python3 -m venv .venv
source .venv/bin/activate
```

Create `.venv` once. In each new terminal, run `source .venv/bin/activate` from
the project directory again. The following `python` commands use that environment.

### 2. Install the Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`pip check` should report no broken requirements. If an existing environment has
dependency conflicts, update the dependencies together and check again:

```bash
python -m pip install --upgrade -r requirements.txt
python -m pip check
```

### 3. Install Ollama and download the model

Ollama runs as a separate server. The Python `ollama` dependency is its client;
install the server using the [official Linux installation instructions](https://docs.ollama.com/linux)
if it is not already available:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

If Ollama is not already running as a service, start it in a second terminal and
leave that terminal open:

```bash
ollama serve
```

Back in the project terminal, download the model configured in `code/config.py`
and check the available models:

```bash
ollama pull qwen3.8:27b-q8_0
ollama ls
```

The default is the [Qwen3.8 27B Q8_0 model](https://ollama.com/library/qwen3.8:27b-q8_0).
To use another model, download its tag and set `AgentConfig.model` to that exact
tag in `code/config.py`. It must support tool calling and thinking, which the
agent requests. The default server address is `http://127.0.0.1:11434`;
edit `AgentConfig.host` if your server uses a different address.
See the [Ollama CLI reference](https://docs.ollama.com/cli) for server and model commands.

### 4. Add PDFs and optional OCR language data

```bash
mkdir -p data/pdfs
```

Copy your PDFs into `data/pdfs/`; subdirectories are supported. For scanned PDFs,
also install the English and Italian OCR language data:

```bash
python scripts/setup_ocr.py
```

Text-based PDFs work without this OCR setup. Additional languages and OCR settings
are described below.

### 5. Start the chat

With `.venv` activated and Ollama running, launch the agent from the project root:

```bash
python main.py
```

For example, ask: `Summarize the main results in the PDFs and cite the relevant pages.`
When finished, exit the chat with `/exit`, then run `deactivate` to leave the virtual
environment. A manually started `ollama serve` can be stopped with Ctrl+C in its terminal.

## Terminal usage

Use `/clear` to reset the conversation, `/exit` to quit, or Ctrl+C to interrupt.
Use `/verbose` to toggle tool arguments and generation statistics; `/verbose on`
and `/verbose off` set the mode explicitly. Warnings and errors appear in both modes.
The agent can continue making tool calls until it finishes or is interrupted.

Interactive terminals render streamed Markdown with headings, bold text, lists,
aligned tables, and highlighted code. Completed blocks stay in scrollback, while
the current block updates live. For a very long unfinished table or paragraph,
the live view shows the latest lines; the complete block is printed when it finishes.
The temporary thinking line remains grey and is erased when an answer or another
message is printed.

At startup, a compact panel shows the model, configured context window, shortened
project path, and web-access badge. Green user prompts and cyan assistant separators
make conversation turns easier to scan. One transient activity line shows the
current model or tool operation and elapsed time, including while a PDF tool is
busy. `/verbose` retains the detailed tool-call log. After each turn, a subdued
footer shows elapsed time and tool-call count; generated tokens appear only when
Ollama returned counts for all completed model rounds. The footer labels this sum
as the **turn total**: it includes earlier generations that requested tools and
the final model call. The verbose `Final model call` line reports only that last
generation. These are generated-token counts, including thinking, rather than
input/context-token counts.

Interactive input supports Tab completion for commands and Up/Down question
history through Python's optional `readline` module. End a line with a single
backslash (`\`) and press Enter to continue a multiline question; submit it by
ending the final line without a backslash. With GNU readline, bracketed multiline
pastes remain one editable question until Enter. Input history holds up to 100
entries in memory for the current session and is never written to a history file.

PDF citations display as `[1]`, `[2]`, etc., with full paths and page numbers in a
source list under each response. The model's saved conversation retains its full
inline citations. Redirected output keeps raw Markdown and full citations, without
terminal colors or live redraw controls.

## Optional web retrieval

Web access defaults to **off**. To enable it, get a [Tavily API key](https://app.tavily.com)
and create the local credentials file from the supplied example:

```bash
cp .web_credentials.example.json .web_credentials.json
chmod 600 .web_credentials.json
```

Edit `.web_credentials.json` to set your key:

```json
{
  "tavily_api_key": "your-tavily-api-key"
}
```

The local credentials file is ignored by Git. Set `web_enabled=True` in
`code/config.py`, then restart `python main.py`. No environment variables or
additional Python packages are needed. Enabled mode checks credentials before
starting the chat. It adds exactly two tools:

| Tool | Behavior |
| --- | --- |
| `web_search(query, max_results=5)` | Return up to 10 public source URLs, titles, publication dates when available, and excerpts of up to 2,000 characters per result. |
| `read_web_page(url, offset=0, max_chars=6000)` | Extract page text through Tavily, with up to 12,000 characters per call. Follow the returned `next` cursor to read further text. |

The agent searches PDFs first for document questions. Web sources can supplement
broader questions or provide current information; requests restricted to the PDFs
remain scoped to those PDFs. The prompt tells the model to verify web claims by
reading the pages, prefer original sources, preserve reporting dates, and label
external context. Page extraction can fail, and publication dates can be unknown.

Web claims use `[https://the-returned-page-url]` in saved messages and redirected
output. Interactive output gives them compact citation numbers alongside PDF
citations, with `Web:` labels and full URLs in the source list.

Search queries and requested page URLs are sent to Tavily and use your account's
quota. The tools do not upload PDF files; the model is instructed to exclude private
PDF passages and credentials from search queries. Extracted pages are cached in
memory for 15 minutes, up to eight pages, so pagination reuses one extraction.
Network calls use `web_timeout_seconds` (25 seconds by default). Set
`web_credentials_path` to change the credentials file location inside the project;
keep any custom credentials file out of Git as well.
See the [search](https://docs.tavily.com/documentation/api-reference/endpoint/search)
and [page extraction](https://docs.tavily.com/documentation/api-reference/endpoint/extract)
API documentation.

Tool descriptions consume context and can increase model processing and selection
time. With web access disabled, neither these tool definitions nor their extra
instructions are sent to the model. When enabled, bounded results and cached page
reads reduce repeated network work and keep evidence sizes manageable.

## Code and configuration

The code is organized by responsibility:

| Class | File | Responsibility |
| --- | --- | --- |
| `AgentConfig` | `code/config.py` | Model, Ollama host, context window, output length, and project root. |
| `PDFChatAgent` | `code/run_agent.py` | Streaming, tool dispatch, compatibility retries, and the terminal loop. |
| `Conversation` | `code/conversation.py` | Chat history, resets, failure recovery, and evidence compaction. |
| `PDFTools` | `code/tools/tool_defs.py` | PDF discovery, search, reading, metadata, tables, OCR, and calculations. |
| `WebTools` | `code/tools/web_tools.py` | Optional public web search, page extraction, and an in-memory page cache. |
| `CollectionIndex` | `code/tools/collection_index.py` | Persistent keyword index and recognized page text. |
| `Calculator` | `code/tools/calculator.py` | Restricted arithmetic with decimal precision. |
| `TerminalOutput` | `code/terminal_output.py` | Startup panel, streamed answers, transient thinking/activity, and turn statistics. |
| `TerminalInput` | `code/terminal_input.py` | Command completion, private session history, and multiline input. |
| `CitationFormatter` | `code/citations.py` | Compact PDF and web references with complete source details. |

Edit the defaults in `code/config.py` to change model settings. The default model
is `qwen3.8:27b-q8_0`, with a 131,072-token context and a 16,384-token output limit.
The PDF question-answering instructions live in `code/prompts.py`.
Set `output_width` to change the maximum rendered width (100 columns by default),
or set `verbose=True` to start with detailed logs. Rendering also fits the terminal's
actual width. These settings are in code; no environment variables are required.

For use from Python, create an agent with its own configuration and conversation:

```python
from code.config import AgentConfig
from code.run_agent import PDFChatAgent

agent = PDFChatAgent(AgentConfig(context_length=131072))
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

The optional OCR setup from the installation steps downloads English/Italian data:

```bash
python scripts/setup_ocr.py
```

OCR uses PyMuPDF's integrated Tesseract support and local language files;
no environment variables are required. The setup script downloads files from
[Tesseract's official tessdata_fast repository](https://github.com/tesseract-ocr/tessdata_fast).
For additional languages, run `python scripts/setup_ocr.py --languages deu`, for
example. Set `ocr_data_path` in `code/config.py` if you use a different folder inside
the project, and pass the same folder to the setup script with `--directory`.
See [PyMuPDF's OCR setup documentation](https://pymupdf.readthedocs.io/en/latest/installation.html#enabling-integrated-ocr-support).

OCR can misread figures, and table extraction can misidentify merged headers.
The tools label OCR text and preserve table context so answers can state uncertainty.
Recognized text does not reconstruct scanned table geometry. The original PDFs
are never modified; the index and downloaded language files are ignored by Git.
Removing `.rag_cache/` resets the index and cached OCR text.

For development, activate `.venv`, install the test dependencies, and run the checks:

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

The terminal tests use `pyte` to check scrollback, table rows, and cursor restoration.
