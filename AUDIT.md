# Repository audit

Audit date: 2026-10-03 (Europe/Rome). Reviewed commit: `9138afd`.

The harness has useful foundations: tools are registered explicitly, project paths
are checked, the calculator parses restricted arithmetic, extraction reports source
paths and physical pages, and web access is disabled by default. However, passing
unit tests currently does not establish reliable context management or consistent
retrieved evidence. The highest priority is making request preparation and evidence
handling dependable before adding more capabilities.

This report records findings and proposed changes. Application fixes remain
outstanding. The audit did not change the application, dependencies, PDFs, or the
existing retrieval database.

## Scope and verification

Reviewed all application modules, setup scripts, dependency files, README, and the
six test modules. Checks included:

- All **94 existing tests passed** in the installed Python 3.9.25 environment.
- `compileall` and `git diff --check` passed.
- `pip check` failed with the dependency conflict described below.
- Generated actual tool schemas with the installed Ollama client, rather than
  assuming mocked chat requests reflect its serialization.
- Reproduced defects with temporary PDFs/databases, mock providers, and in-memory
  terminal streams. Concurrent indexing was tested with separate connections.
- Measured extraction of the existing 183-page Coca-Cola 2022 PDF without changing it.
- Verified selected external API/runtime facts against official documentation.

No live Ollama generation, authenticated Tavily calls, or exhaustive semantic
validation of the 368 PDFs was performed. Native-parser vulnerabilities and the
full transitive dependency vulnerability inventory were not assessed. Findings
about retained thinking concern the request payload; whether the model's chat
template incorporates that field into its prompt remains unverified.

Priorities: **P1** = address first for reliable operation; **P2** = correctness,
security, integration, or substantial performance defect; **P3** = narrower defect
or maintenance issue. Architecture gaps and optional optimizations are identified
separately from reproduced bugs.

## Findings

### 1. P1 — Normal model requests have no context budget

Location: [run_agent.py:127](/home/davide.savarro/Agentic-RAG-Playground/code/run_agent.py:127),
[conversation.py:110](/home/davide.savarro/Agentic-RAG-Playground/code/conversation.py:110).

Normal requests send the entire stored history. Compaction runs only after the
specific HTTP 500 error containing `no user query found`. Answers, thinking, and
tool evidence accumulate in application memory and request payloads without a
budget. A mock request with a 1,024-token context forwarded about 488 KB of history.
Increasing the configured context postpones this problem; it does not bound growth.

Prepare bounded request history on every round. Reserve space for system
instructions, tool definitions, formatting overhead, and generation. Preserve the
active task and complete assistant/tool exchanges; retain full evidence separately
from the bounded model request. Decide explicitly when prior thinking is useful.
Use Ollama's returned `prompt_eval_count` for diagnostics/calibration, while making
clear any preflight token estimate is approximate. The field is documented in the
[Ollama chat API](https://docs.ollama.com/api/chat).

### 2. P2 — Compaction can lose the task behind “Continue”

Location: [conversation.py:120](/home/davide.savarro/Agentic-RAG-Playground/code/conversation.py:120),
[conversation.py:138](/home/davide.savarro/Agentic-RAG-Playground/code/conversation.py:138).

The reducer selects recent records independently and pins only the latest user
message. Under budget pressure, the original substantive question and its
constraints can be dropped while `Latest question: Continue` survives. This is
conditional: short original questions often happen to fit.

Reproduction used a valid 16,384-token context, 8,192-token output reserve, a
757-character original question, twenty 1,600-character page results, then
`Continue`. The compact request lost the original task. Another reproduction lost
both the company name and a `PDFs only` restriction. The latest question also
bypasses the byte allowance: a 100 KB question creates a roughly 103 KB compact
request despite an 8 KB record budget.

Store the active task and its constraints explicitly, reserve their budget first,
and bound the entire serialized request. Account for the system prompt, schemas,
wrapper text, and quoted/escaped JSON. Reject an oversized essential request
clearly instead of silently losing its meaning.

### 3. P2 — Table compaction removes columns without a usable continuation

Location: [conversation.py:40](/home/davide.savarro/Agentic-RAG-Playground/code/conversation.py:40),
[conversation.py:83](/home/davide.savarro/Agentic-RAG-Playground/code/conversation.py:83).

The generic reducer limits every nested list, including table headers and each
row's cells. Cursor repair only detects omitted rows. A one-row, twelve-column
table becomes an eight-column table while retaining `total_columns=12`,
`truncated=false`, and `next=null`. A `context_compacted` marker is present, but
neither the advertised truncation flag nor row pagination explains or recovers the
missing columns. Repeating the same tool call in compatibility mode clips them again.

Use tool-specific reducers. Preserve complete rows and aligned headers; reduce the
number of rows instead of the number of cells. Preserve fixed-size vectors and
source metadata. Add tests that cover wide tables, not just many rows.

### 4. P2 — Replacing successful OCR with empty OCR leaves stale search evidence

Location: [collection_index.py:153](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:153).

`save_ocr` replaces the OCR row regardless of its contents, but only replaces FTS
passages when the new text is nonempty. Native page text followed by successful
OCR and then empty/whitespace OCR produces contradictory tools: page reading
returns native text, while collection search still returns the old recognized text
and no longer finds the original native keywords.

Update OCR storage and searchable passages atomically for every replacement.
Either preserve the last successful OCR deliberately, or discard it and restore
native passages. Test the successful-OCR-to-empty-OCR transition; the existing
empty-OCR test begins with native text and misses this case.

### 5. P2 — Hybrid scanned pages are reported as fully covered

Location: [collection_index.py:69](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:69),
[collection_index.py:130](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:130),
[tool_defs.py:344](/home/davide.savarro/Agentic-RAG-Playground/code/tools/tool_defs.py:344).

Coverage detection marks only pages with completely empty extracted text as
missing. A full-page scanned report image containing revenue figures, plus a
selectable `Page 1` footer, produces no revenue matches, no missing-text pages, and
`search_complete=true`. The prompt principally directs OCR toward pages without
extractable text, so this scan can be missed.

Distinguish successful search of extracted text from verified coverage of the
document. Report image-backed/low-text pages as potentially incomplete and suggest
selective OCR. Avoid automatically classifying every illustrated page as scanned;
coverage should remain qualified until inspected.

### 6. P2 — Reading one page extracts the entire PDF

Location: [tool_defs.py:84](/home/davide.savarro/Agentic-RAG-Playground/code/tools/tool_defs.py:84),
[tool_defs.py:182](/home/davide.savarro/Agentic-RAG-Playground/code/tools/tool_defs.py:182),
[tool_defs.py:211](/home/davide.savarro/Agentic-RAG-Playground/code/tools/tool_defs.py:211).

Both page-reading tools first extract every page with pypdf. Saved OCR does not
bypass that extraction. The in-memory cache holds only four whole documents,
which can thrash during an eight-year comparison.

One cold extraction measurement on the existing 183-page Coca-Cola 2022 report:

| Operation | Elapsed |
| --- | ---: |
| Whole-document pypdf extraction used by page reads | 12.507 s |
| Requested first page with pypdf | 0.035 s |
| Whole-document PyMuPDF extraction | 0.826 s |

These are measurements on this workspace and one document, not a universal speed
guarantee. Extract requested pages lazily and cache by document metadata plus page
number. Consider using PyMuPDF consistently for native reading and indexing;
maintain a separate search-passage cache. This has more direct latency benefit
than adding more retrieval tools.

### 7. P2 — Concurrent initial indexing permanently duplicates passages

Location: [collection_index.py:54](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:54),
[collection_index.py:64](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:64).

The metadata lookup and extraction occur before the write transaction. Two agents
or processes can both see a missing document, then both insert its passages.
Deletion is conditional on that earlier lookup having found an existing document.
Two synchronized loaders with separate connections indexed a one-passage PDF;
the first search returned one match, the second two, and later searches kept both.

Recheck document state inside an explicit write transaction and replace passages
atomically. Enforce a unique document/page/chunk identity through a normal passage
table or equivalent scheme. Do not assume separate `PDFTools` instances have
separate databases when they share a project root.

### 8. P2 — PDF tool schemas contradict their Python defaults

Location: [run_agent.py:59](/home/davide.savarro/Agentic-RAG-Playground/code/run_agent.py:59),
[test_pdf_tools.py:42](/home/davide.savarro/Agentic-RAG-Playground/tests/test_pdf_tools.py:42).

The installed Ollama 0.5.1 converter marks default-valued PDF parameters required:

```text
list_pdfs: path, name_contains, limit, offset
search_collection: query, path, company, year, top_k, offset
read_pdf_content: path, page, offset, max_chars
```

This contradicts the documented defaults and instructions to omit unnecessary
filters. The existing PDF schema test does invoke the real converter, but checks
names, properties, and descriptions rather than required/default semantics.
The web tools currently work around this converter with nullable annotations.

Construct explicit schemas from signatures, marking only parameters with no
default as required, and cache them for repeated requests. Test required fields,
defaults, and bounds for every registered tool. Avoid declaring `Optional` on
parameters whose implementations reject `None` just to satisfy this SDK version.

### 9. P2 — Calculator floor division and modulo can silently be wrong

Location: [calculator.py:42](/home/davide.savarro/Agentic-RAG-Playground/code/tools/calculator.py:42).

The quotient is rounded to fifty significant digits before it is floored. Allowed
inputs whose integer quotient needs more precision can produce an invalid remainder:

```text
1e51 % 3   -> 10    (exact result: 1)
-1e51 % 3  -> -10   (floor-modulo result: 2)
1e51 // 3  -> ...330 (exact integer quotient ends in ...333)
```

The modulo results even violate the expected range for a positive divisor of 3.
The inputs are inside the stated supported magnitude range. Use exact scaled-integer
arithmetic for `//` and `%`, or reject operations whose quotient exceeds the
supported precision. Test large quotients and sign/range invariants. Ordinary
fifty-digit rounding for `/` is a separate, documented precision choice.

### 10. P2 — Untrusted answers can emit terminal control sequences

Location: [terminal_output.py:66](/home/davide.savarro/Agentic-RAG-Playground/code/terminal_output.py:66),
[terminal_output.py:152](/home/davide.savarro/Agentic-RAG-Playground/code/terminal_output.py:152).

Thinking is filtered with `isprintable`, but answers, progress values, and error
strings are not comparably sanitized. Both raw output and Rich output preserve
model-supplied CSI clear-screen sequences and OSC 52 clipboard sequences. Safe
testing captured these strings in memory rather than sending them to a terminal.
Effects depend on terminal support and settings; arbitrary code execution was not
demonstrated.

Sanitize untrusted visible text before rendering, including escape sequences split
across streamed chunks. Preserve ordinary newlines/tabs and generate trusted Rich
styling afterward. Apply the same policy to filenames, tool arguments, URLs, and
exception text used in progress/error messages.

### 11. P2 — The application package collides with stdlib `code`

Location: [main.py:1](/home/davide.savarro/Agentic-RAG-Playground/main.py:1),
[README.md:201](/home/davide.savarro/Agentic-RAG-Playground/README.md:201).

From the project directory, `from code import InteractiveConsole` imports this
application package and fails. Conversely, if Python's standard-library `code`
module is already loaded, the documented `from code.run_agent import PDFChatAgent`
fails because `code` is not a package. This affects embedded use and environments
that use the standard console module even though the standalone CLI starts.

Rename the package to an application-specific name and update entrypoints and
documentation. Test package imports in subprocesses without `sys.path` injection,
including a process that imports stdlib `code` first. Prefer one package entrypoint
over maintaining package/direct-script import branches everywhere.

### 12. P2 — Unused dependencies leave the installed environment inconsistent

Location: [requirements.txt:1](/home/davide.savarro/Agentic-RAG-Playground/requirements.txt:1).

`langchain` and `langchain-ollama` are declared but are not imported by application
or test code. The existing environment contains Ollama 0.5.1 and langchain-ollama
0.3.10. `pip check` exits 1 because the latter requires `ollama>=0.5.3,<1.0.0`.
This conflict predates the audit. Direct Ollama and framework requirements are
also unbounded, making a fresh install different from the tested environment.

Remove unused frameworks from direct dependencies, define a tested Ollama client
range, and provide reproducible dependency versions. Resolve the actual application
dependencies together rather than upgrading everything to satisfy an unused adapter.

The current interpreter is Python 3.9.25, which reached end of life on 2025-10-31.
Move the supported baseline and CI to a maintained interpreter, such as Python
3.12 or newer, and document an explicit minimum. The support dates are listed by
the [Python Developer's Guide](https://devguide.python.org/versions/). The README's
statement that 3.9 was tested is true, but it should not be the only runtime guidance.

### 13. P2 — Internal PDF aliases have inconsistent identities

Location: [tool_defs.py:60](/home/davide.savarro/Agentic-RAG-Playground/code/tools/tool_defs.py:60),
[collection_index.py:52](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:52).

Discovery indexes lexical symlink paths while page reads and OCR resolve canonical
paths. An internal `scan_alias.pdf` can therefore read canonical cached OCR but
fail to find that same text in collection search filtered by the alias filename.
The same physical PDF can also be indexed more than once under different names.

Canonicalize and deduplicate document identities consistently. Preserve aliases
as separate discovery/filtering metadata if they are useful; keep paths returned
for citations consistent with the chosen identity policy.

### 14. P2 — Configuration accepts incompatible or nonsensical values

Location: [config.py:8](/home/davide.savarro/Agentic-RAG-Playground/code/config.py:8),
[conversation.py:119](/home/davide.savarro/Agentic-RAG-Playground/code/conversation.py:119).

Dataclass annotations do not validate runtime values. Agent construction accepted
a negative context, a negative output limit, zero display width, and an output
budget larger than the context. The compaction allowance then applies an 8 KB
floor even when the configured budget has no room for a response.

Validate model/host values, positive widths and token counts, and compatible
input/output budgets before starting a chat. Validate actual model capabilities
when selecting another model instead of relying only on the README; thinking is
currently always requested. The current default model supports the requested
features, so this finding concerns invalid edits and alternate configurations.

### 15. P3 — Transient extraction errors are cached indefinitely

Location: [collection_index.py:56](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:56).

An unchanged document record is reused even when it contains an extraction error.
A simulated one-time loader failure remained in subsequent searches after the
loader recovered. A valid unchanged file remains unavailable until its metadata
changes or the cache is deleted.

Retry errored records with backoff or add an explicit refresh/repair operation.
Distinguish permanent document errors from temporary resource failures.

### 16. P3 — Public URL validation is only syntactic

Location: [web_tools.py:75](/home/davide.savarro/Agentic-RAG-Playground/code/tools/web_tools.py:75).

The validator accepts `http://127.1/report`, `http://0177.0.0.1/report`,
`http://127.0.0.1.nip.io/report`, and a hostname containing a NUL character.
Literal conventional private IPs are rejected, but these accepted forms show that
the validation does not establish a public destination.

Extraction is delegated to Tavily, so this is not demonstrated local-harness SSRF.
Reject control characters and ambiguous numeric hosts, tighten URL syntax, and
document that destination/redirect enforcement depends on the provider. If direct
page fetching is ever added, enforce public destinations after resolution and at
each redirect rather than reusing this check as a sufficient network boundary.

### 17. P3 — Streaming can leave unused citations in the source list

Location: [citations.py:30](/home/davide.savarro/Agentic-RAG-Playground/code/citations.py:30),
[terminal_output.py:85](/home/davide.savarro/Agentic-RAG-Playground/code/terminal_output.py:85).

Live rendering mutates the source registry before Markdown is finalized. Streaming
an unfinished inline-code span containing a citation registers it as a source;
closing the backtick later correctly renders code but leaves the unused source
in the final bibliography.

Compute provisional references without committing them to the permanent registry,
or reconcile sources from finalized Markdown. Add tests where streamed Markdown
changes the interpretation of a citation.

## Guardrail and operational gaps

These are design gaps rather than claims that a new exploit or incorrect answer
was observed against a live model.

- **Grounding and citations:** the prompt asks for evidence-backed answers, but
  no independent validator checks that a cited path/page/URL was retrieved or that
  a claim is supported. Citation numbering validates presentation only. Track an
  evidence ledger, validate references against it, and distinguish reference
  existence checks from the harder task of checking claim support.
- **Private data in web queries:** the Python search tool forwards any valid
  short string. A harmless synthetic private-looking query was forwarded unchanged
  in a mocked request. Web access is disabled by default and privacy is requested
  in the prompt, but confidentiality is not enforced. Before using private PDFs
  with web access, consider a public-source allowlist and query filtering or a
  deliberate public-search boundary appropriate to the corpus. PDF files are not
  uploaded by these tools; text placed in a query still leaves the local process.
- **Stop diagnostics:** [run_agent.py:112](/home/davide.savarro/Agentic-RAG-Playground/code/run_agent.py:112)
  handles `done_reason` only when no tools are returned. A mocked `length` response
  containing a tool call ran without a warning. Verbose logs show generated tokens
  but omit prompt count, requested budget, and answer/thinking sizes. Report stop
  reasons consistently and expose those counts before treating every `length`
  stop as a simple output-cap issue. Automatic continuation is a separate feature.
- **Failed-turn recovery:** [conversation.py:25](/home/davide.savarro/Agentic-RAG-Playground/code/conversation.py:25)
  discards all evidence retrieved during a failed turn. Preserve completed
  exchanges so a late transient failure does not require repeating successful work.
- **Repeated tool calls:** unlimited tool rounds were explicitly requested and
  should remain supported. Detect identical repeated calls and lack of progress,
  cache reusable results, and make interruption/usage visible; a fixed round cap
  would be a different behavior requiring a deliberate decision.
- **Session history:** current chat history is process-local memory. `/clear`
  removes it and restart loses it. Optional JSONL records of requests, evidence,
  responses, and timing would help reproduce failures; define retention and
  handling of private text before making logging the default.

## Additional efficiency and maintenance improvements

- **FTS updates scan the passage table.** `path` and `page` are FTS `UNINDEXED`
  columns at [collection_index.py:27](/home/davide.savarro/Agentic-RAG-Playground/code/tools/collection_index.py:27).
  `EXPLAIN QUERY PLAN` for deletion by path confirms a virtual-table scan.
  Use an ordinary passage table with indexed document/page identifiers and a
  row-ID-linked FTS index when changing the index schema. Keep both synchronized;
  SQLite describes the relevant [external-content FTS5 design](https://www.sqlite.org/fts5.html#external_content_tables).
- **Rendering long blocks repeatedly is expensive.**
  [terminal_output.py:30](/home/davide.savarro/Agentic-RAG-Playground/code/terminal_output.py:30)
  renders an entire unfinished block before cropping its visible tail. On an
  in-memory terminal, redraws measured about 11 ms for 50 table rows, 93 ms for
  500, and 264 ms for 1,500. Use adaptive refresh or a bounded live preview and
  render the complete block once when finalized. Preserve the existing scrollback
  and cursor-restoration behavior.
- **Web client lifecycle and deadlines.**
  [web_tools.py:111](/home/davide.savarro/Agentic-RAG-Playground/code/tools/web_tools.py:111)
  constructs a fresh opener for each call. A reusable client can pool connections.
  The socket timeout is not an overall elapsed-time deadline; a response that
  keeps supplying data can take longer. Add an overall deadline if predictable
  tool latency is required. The timeout's blocking-operation semantics are
  described by [Python's urllib documentation](https://docs.python.org/3/library/urllib.request.html).
- **OCR asset reproducibility.** The setup script downloads from a moving branch
  and trusts any nonempty existing language file. Pin a revision/checksum and
  support explicit repair or re-download. Current atomic replacement, path checks,
  and language-code validation are useful and should be retained.
- **Corpus distribution.** The repository tracks 368 PDFs; `data/pdfs` occupies
  approximately 674 MB and `.git` 559 MB in this workspace. Consider a documented
  dataset-fetch step or Git LFS if fast clones and reusable source distribution
  matter. This is a distribution choice, not evidence of private data exposure.
- **Test isolation and CI.** Most tests inject `code/` into `sys.path`, hiding
  package-import problems. Add subprocess entrypoint/import tests, required/default
  schema assertions, and regression cases for the defects above. Run a supported
  Python version, `pip check`, and the test suite in CI. Live model/provider checks
  should be opt-in; unit tests should remain usable without API keys or a GPU.
- **Retrieval quality.** The keyword search intentionally matches any query term.
  Add a small evaluated question/evidence set before choosing phrase search,
  reranking, or embeddings. Measure whether the correct page appears in the top
  results and whether the final answer cites it; additional tools alone do not
  establish better retrieval quality.

## Recommended implementation order

1. Bound model requests and pin the active task; make compaction preserve complete
   evidence structures and source metadata.
2. Repair OCR/index consistency and concurrent writes; qualify scanned-page coverage.
3. Sanitize terminal output and correct the calculator's integer quotient/remainder.
4. Extract requested pages lazily, then measure the same eight-year report workflow.
5. Correct and cache tool schemas; rename the package; simplify and reproduce dependencies.
6. Add evidence-reference validation and enforced public-search handling suitable
   for the documents, then expand evaluation/observability.

The first four steps address existing failure modes and latency directly. New
features should follow evidence-integrity fixes and representative retrieval tests.
