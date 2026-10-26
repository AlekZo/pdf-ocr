# Changelog

## 1.1.1 — 2026-10-26
- Right-click menu: the parent "OCR (PaddleOCR)" entry now has an empty `SubCommands` value. Without it Explorer showed "This file does not have an app associated with it" on click. Re-run `install_context_menu.bat` to apply.

## 1.1 — 2026-10-26
- `--structure`: PP-StructureV3 layout analysis, headings and real Markdown tables (colspan/rowspan handled); text inside pictures is kept as a `>` quote.
- `--textline-orient` and `--unwarp` options.
- Right-click menu (`install_context_menu.bat`) with presets; a multi-selection is queued and OCR'd in one run (`ocr_queue.ps1`).
- Batch runs: `[n/N]` progress, summary with failed files and low-confidence files; duplicates, `*.ocr.pdf` results and non-PDFs are skipped.
- Cleanup also fixes words like `Мockвa` (Cyrillic letters mixed with Latin look-alikes only).
- `setup.bat` / `requirements.txt` install `paddlex[ocr]` for `--structure`.

## 1.0 — 2026-10-22
- PDF → Markdown with PaddleOCR (PP-OCRv5): page markers, rebuilt paragraphs, YAML header with confidence.
- Optional searchable PDF (invisible text layer), optional JSON with lines, boxes and confidences.
- Pages that already have a text layer are copied, not OCR'd.
- `--device auto`: NVIDIA GPU when the GPU build of PaddlePaddle is installed, otherwise CPU; automatic fallback to CPU on GPU errors.
- Local model folders in `models\` (also accepts the `*_infer` folders that the official archives unpack to).
- Rule-based text cleanup for Russian scans (`--no-fix` to disable).
- Windows helpers: `setup.bat` (environment in `%USERPROFILE%\.pdf-ocr`, GPU build if an NVIDIA driver is found), `ocr.bat` (drag and drop).
