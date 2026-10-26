# Using the tool from an LLM / agent

## Command

```
"%USERPROFILE%\.pdf-ocr\venv\Scripts\python.exe" "<path>\pdf_ocr.py" "<file-or-folder>" [options]
```

- Exit code `0`: success (files that were already processed are skipped unless `--overwrite`); `1`: at least one file failed; `2`: no PDF found.
- Progress and errors go to the console; the result is plain files next to the PDF (or in `-o DIR`).
- Useful options: `--lang ru`, `--dpi 300` (small print), `--pages 1-3`, `--no-pdf` (Markdown only), `--json`, `--no-fix`.

## Markdown output (`<name>.md`)

```
---
source: <file name>.pdf
pages: 11
ocr_engine: PaddleOCR
ocr_pages: 11
native_text_pages: 0
dpi: 200
mean_confidence: 0.933
text_fixes: 41
low_confidence_pages: [11]
generated: 2026-10-22T13:24:19
---

<!-- page 1 -->

First paragraph …

Second paragraph …

<!-- page 2 -->
…
```

- `<!-- page N -->` marks each PDF page (1-based); use it for citations.
- `low_confidence_pages`: pages with mean line confidence < 0.80. Treat their text as unreliable.
- `text_fixes`: number of rule-based corrections applied (see ARCHITECTURE.md).
- Layout is flattened: tables and multi-column pages come out in reading order, not as Markdown tables.

## JSON output (`<name>.ocr.json`, with `--json`)

```
{ "source": "...", "engine": "PaddleOCR", "dpi": 200,
  "pages": [ { "page": 1, "source": "ocr" | "native", "image_size": [w, h],
               "mean_confidence": 0.96,
               "lines": [ { "text": "...", "score": 0.99, "box": [x0, y0, x1, y1], "raw": "..." } ] } ] }
```

`box` is in pixels of the rendered page image; `raw` appears only when the cleanup changed the line.

## Guidance for agents

- Do not trust numbers, dates, names or amounts from low-confidence pages or from lines with a low `score`; check them against the scan.
- Do not "correct" OCR text by guessing; quote it as recognised and flag doubtful spots.
