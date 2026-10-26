# Architecture

Single-file tool, `pdf_ocr.py`, built from small, separable parts.

```
PDF ──► page loop (PyMuPDF)
          │  page has ≥30 chars of real text and no --force-ocr?
          ├─ yes ─► native text blocks ───────────────┐
          └─ no ──► render page to RGB (dpi) ──► Engine.recognize() ──► [Line(text, score, box)]
                                                         │
                                        fix_text() (cleanup, unless --no-fix)
                                                         │
                                        lines_to_markdown() (rows → paragraphs)
                                                         │
                           ┌─────────────────────────────┴───────────┐
                           ▼                                         ▼
                    <name>.md (+ YAML header)           add_text_layer() → <name>.ocr.pdf
                                                        (render mode 3, invisible)
                           optional: <name>.ocr.json (lines, boxes, scores, raw text)
```

## Components

| Part | Role |
|---|---|
| `Engine` protocol | `recognize(image_rgb) -> list[Line]`. `PaddleEngine` is the implementation; `tests/pipeline_test.py` plugs in other engines, so the pipeline is testable without models. |
| `PaddleEngine` | Wraps `paddleocr.PaddleOCR` (detection + recognition; orientation and unwarping off by default). Chooses models from `--lang`, or uses local model folders. |
| `resolve_device()` | `auto` → `gpu:0` if this PaddlePaddle build is compiled with CUDA and sees a device, else `cpu`. GPU errors at start-up or during inference fall back to CPU once. |
| `fix_text()` | Conservative cleanup, see below. |
| `lines_to_markdown()` | Groups boxes into visual rows, then into paragraphs. |
| `add_text_layer()` | Writes each line as invisible text, scaled horizontally to its box, using a Unicode TTF (Arial on Windows, DejaVu/Liberation on Linux, or `--font`). |

## Models

For `--lang ru` PaddleOCR 3.7 selects `PP-OCRv5_server_det` (detection) and `eslav_PP-OCRv5_mobile_rec` (recognition). "eslav" is the East Slavic recognition model shared by `ru`, `uk` and `be`; PaddleOCR offers no larger Russian recognition model. The only other Cyrillic PP-OCRv5 option is the generic `cyrillic_PP-OCRv5_mobile_rec`; it can be tried with `--rec-model-dir`.

Model lookup order: `--det-model-dir` / `--rec-model-dir` → `models\<name>` or `models\<name>_infer` next to `pdf_ocr.py` → PaddleOCR's own download into `~/.paddlex`. The model name is read from `inference.yml` in the folder.

## Layout heuristics (`lines_to_markdown`)

- Boxes whose vertical centres are within 0.6 × line height share a row; rows are ordered top to bottom, boxes left to right.
- A new paragraph starts when: the vertical gap is > 0.7 × median line height; the previous line ends well before the right margin (< 82% of the text width); the line is indented by more than 1.5 × line height; or it starts like a list item (`1.1.`, `-`, `•`, `а)`).
- Multi-column pages and tables are flattened into reading order. Headings are not inferred.

## Text cleanup (`fix_text`)

Derived from errors seen on a real Russian scan. Rule-based, never adds words, counted as `text_fixes` in the header; the raw text stays in the JSON.

1. Mis-read thousands group after a number: `86 о00` → `86 000` (only a 3-character token containing a digit and an o/о, directly after a 1–3 digit number; `ООО` is never touched).
2. Latin look-alike letters inside words that are mostly Cyrillic: `Моckва` → `Москва`. Lines containing URLs or e-mail addresses are skipped.
3. Missing spaces after `,` `;` `)`, before `(` after a Cyrillic letter, and after a sentence-ending `.` before a capital letter.

Not fixed: words glued together without punctuation, garbled URLs, handwriting.

## Output format

See [LLM_USAGE.md](LLM_USAGE.md).

## Tests

`python tests/pipeline_test.py` — needs `pymupdf` and the `tesseract` command line tool (English data). It checks Markdown generation, the invisible and searchable text layer (English via Tesseract; Cyrillic via a stub engine) and the native-text path. It does not test PaddleOCR itself.
