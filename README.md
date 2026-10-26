# pdf-ocr — PDF → Markdown (+ searchable PDF) with PaddleOCR

Reusable command-line tool. Give it PDFs (or folders of PDFs); it gives you, next to each PDF:

| File | What it is |
|---|---|
| `name.md` | **Recognised text in Markdown** — page markers `<!-- page N -->`, paragraphs rebuilt, YAML header with engine, DPI and confidence. This is the file for your LLM projects. |
| `name.ocr.pdf` | Copy of the PDF with an **invisible text layer** (Ctrl+F and copy/paste work; the page looks identical). Skip with `--no-pdf`. |
| `name.ocr.json` | Optional (`--json`): every line with text, confidence and box coordinates. |

Pages that already contain real text are copied as-is, not OCR'd (override with `--force-ocr`), so the same tool is safe on mixed or born-digital PDFs.

## Install (Windows, once)

1. Python 3.9–3.13 with the `py` launcher (python.org installer; 3.12 is a good choice). PaddlePaddle 3.2 has no wheels for 3.14 yet.
2. Double-click **`setup.bat`**. It creates the environment in `%USERPROFILE%\.pdf-ocr\venv` (deliberately *not* in this folder, so Nextcloud doesn't sync ~1 GB) and installs PaddleOCR.
3. **NVIDIA GPU:** if `nvidia-smi` is found, `setup.bat` installs the CUDA 12.9 GPU build of PaddlePaddle (`paddlepaddle-gpu` 3.2.2 from the official PaddlePaddle index, needed for RTX 50-series) and falls back to the CPU build if that fails. Run `setup.bat cpu` to force the CPU build. Not yet verified on an RTX 5060 — the console prints `device: gpu:0` or `device: cpu` so you can see what is used.
4. The first OCR run downloads the models (for Russian: `PP-OCRv5_server_det` + `eslav_PP-OCRv5_mobile_rec`) into `%USERPROFILE%\.paddlex`. After that it works offline.

   **Manual / offline models:** put the two model folders into `models\` next to `pdf_ocr.py`. Either extract the downloaded `.tar` archives there (they unpack to `PP-OCRv5_server_det_infer\` and `eslav_PP-OCRv5_mobile_rec_infer\`, which the tool accepts), or use folders named exactly like the model:
   `models\PP-OCRv5_server_det\` and `models\eslav_PP-OCRv5_mobile_rec\` (the folder must directly contain the model files, e.g. `inference.*` and `config.yml`). The tool picks them up automatically and prints which ones it uses. Download, e.g.:
   `pip install huggingface_hub`, then
   `hf download PaddlePaddle/PP-OCRv5_server_det --local-dir models\PP-OCRv5_server_det` and
   `hf download PaddlePaddle/eslav_PP-OCRv5_mobile_rec --local-dir models\eslav_PP-OCRv5_mobile_rec`.

## Use

- **Drag and drop:** drop PDF files or a folder onto **`ocr.bat`**.
- **Right-click menu:** double-click **`install_context_menu.bat`** once (current user, no admin). Then right-click a PDF or a folder → **OCR (PaddleOCR)** → pick a preset (Russian, **tables and headings**, high quality 300 dpi, fast text-only, redo existing, fix rotated pages, photo / curved pages, English, Ukrainian, force CPU). On Windows 11 it is under **Show more options** (or Shift+right-click). Edit the `$presets` list in `context_menu.ps1` to change the entries and re-run the installer; `uninstall_context_menu.bat` removes it. If you move this folder, run the installer again. **Several files:** select as many PDFs / folders as you like and pick one preset: they are collected and processed in **one run** in one console (models load once, one summary at the end).
- **Command line:** `ocr.bat "D:\Scans\contract.pdf"` or `ocr.bat D:\Scans --lang en --dpi 250`

Useful options (`ocr.bat` with no arguments prints the basics; `python pdf_ocr.py -h` lists everything):

| Option | Meaning |
|---|---|
| `-l ru` | language: `ru` (default, also reads Latin text/digits), `en`, `ch`, `uk`, … |
| `--dpi 200` | render resolution; try 250–300 for small print |
| `--pages 1-3,5` | only these pages |
| `-o DIR` | write outputs to DIR instead of next to the PDF |
| `--no-pdf` | Markdown only (faster, no extra file) |
| `--json` | also write the JSON with boxes and confidences |
| `--no-fix` | skip the conservative text cleanup (see below) |
| `--force-ocr` | OCR even pages that have a text layer |
| `--overwrite` | redo files whose `.md` already exists (otherwise they are skipped) |
| `--auto-rotate` | fix sideways/upside-down pages (slower) |
| `--structure` | layout analysis (PP-StructureV3): `#`/`##` headings and **real Markdown tables**, text in pictures/stamps marked with `>`. Slower (a few seconds per page on GPU); needs the extras `setup.bat` installs |
| `--textline-orient` | fix upside-down / rotated single text lines |
| `--unwarp` | flatten curved or photographed pages (Markdown only: the page geometry changes, so no searchable PDF) |
| `--fast` | lighter detection model |
| `--no-mkldnn` | use if you get a CPU/oneDNN runtime error (the tool also retries by itself once) |
| `--det-model-dir`, `--rec-model-dir` | use manually downloaded model folders (offline) |
| `--device auto` | default: uses the NVIDIA GPU if the GPU build of PaddlePaddle is installed (`setup.bat` does this when it finds an NVIDIA driver), otherwise CPU; also `cpu` or `gpu:0`. If the GPU fails, the tool falls back to CPU by itself |

## Using it from an LLM / agent

Give the agent this instruction (it works the same in Claude Code, Codex, Cursor, etc.):

> To OCR a PDF run `"%USERPROFILE%\.pdf-ocr\venv\Scripts\python.exe" "<path>\pdf-ocr\pdf_ocr.py" "<file.pdf>" --lang ru`. It writes `<file>.md` next to the PDF; read that file. Pages listed in `low_confidence_pages` in its header need a manual check. Exit code 0 = success.

Everything is plain files and a plain exit code, so no special integration is needed. To process a whole folder, pass the folder path; already-processed files are skipped.

For RAG / LLM projects use the `.md` files; the `<!-- page N -->` markers let you cite pages.

## Text cleanup (on by default, `--no-fix` to disable)

PaddleOCR makes a few systematic mistakes on Russian scans, so the tool applies small, rule-based fixes and counts them (`text_fixes:` in the Markdown header). The raw text stays in the `.ocr.json` (`raw`).
- spaces after `,` `;` `)` and before `(` when the OCR glued words together;
- Latin look-alike letters inside otherwise Cyrillic words (`Моckва` → `Москва`); URLs and e-mails are left alone;
- a mis-read thousands group after a number (`86 о00` → `86 000`).

It never adds words. Words that the OCR glued together without punctuation (`ОБЯЗАННОСТИСТОРОН`) cannot be fixed safely and stay as recognised.

## Limits (honest list)

- OCR is not perfect (on the sample lease: amounts were read as `о00`, some spaces were lost, a hand-written/stamped last page was mostly unreadable): check numbers, dates, names and amounts in important documents. Pages with mean confidence below 0.80 are listed in the Markdown header and in the console.
- By default tables and multi-column layouts are flattened into reading order; use `--structure` for Markdown tables and headings. Handwriting is poorly supported.
- Paragraph breaks are inferred from geometry (gaps, short lines, numbered items like `1.1.`); headings are not guessed.
- The searchable PDF needs a Unicode font; on Windows `arial.ttf` is used automatically (or pass `--font`). Rotated pages (`/Rotate`) get no text layer.
- Speed on CPU is a few seconds per page.

## Documentation

- [docs/LLM_USAGE.md](docs/LLM_USAGE.md) — command, exit codes and output formats for agents
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline, models, layout heuristics, text cleanup
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) — common problems
- [CHANGELOG.md](CHANGELOG.md)

## Files

`pdf_ocr.py` (the tool) · `requirements.txt` · `setup.bat` · `ocr.bat` · `context_menu.ps1` · `ocr_queue.ps1` · `install_context_menu.bat` · `uninstall_context_menu.bat` · `tests/pipeline_test.py` (pipeline test without models) · `docs/`.
