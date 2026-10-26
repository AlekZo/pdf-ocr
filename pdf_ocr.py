#!/usr/bin/env python3
"""
pdf_ocr.py - OCR for PDF files with PaddleOCR.

For every PDF it produces (next to the PDF, or in --out):
    <name>.md        Markdown with the recognised text (always)
    <name>.ocr.pdf   copy of the PDF with an invisible, searchable text layer (skip with --no-pdf)
    <name>.ocr.json  per-line text, confidence and boxes (only with --json)

Pages that already contain real text are copied as-is (no OCR) unless --force-ocr is given.

Examples
    python pdf_ocr.py contract.pdf
    python pdf_ocr.py D:\\Scans --lang ru --dpi 250
    python pdf_ocr.py contract.pdf --pages 1-3 --no-pdf
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

# Silence PaddleX's connectivity check; must be set before paddle is imported.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import numpy as np  # noqa: E402
import pymupdf  # noqa: E402

__version__ = "1.1"

NATIVE_TEXT_MIN_CHARS = 30  # a page with at least this much real text is not OCR'd
LOW_CONFIDENCE = 0.80  # pages whose mean confidence is below this are flagged


# --------------------------------------------------------------------------- data


@dataclass
class Line:
    text: str
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in image pixels
    raw: str | None = None  # original OCR text when a cleanup changed it

    @property
    def h(self) -> float:
        return max(self.box[3] - self.box[1], 1.0)

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2


@dataclass
class PageResult:
    number: int  # 1-based
    source: str  # "ocr" | "native"
    text: str  # markdown body of the page
    lines: list[Line]
    img_size: tuple[int, int] | None  # (w, h) of the rendered image, OCR pages only

    @property
    def mean_score(self) -> float | None:
        if not self.lines:
            return None
        return sum(l.score for l in self.lines) / len(self.lines)


class Engine(Protocol):
    name: str

    # returns (lines, layout blocks or None); an engine may implement just recognize() -> lines
    def recognize_page(self, image_rgb: np.ndarray) -> tuple[list[Line], list[dict] | None]: ...


# --------------------------------------------------------------------------- engine


def resolve_device(requested: str) -> str:
    """'auto' -> 'gpu:0' when this PaddlePaddle build can use an NVIDIA GPU, else 'cpu'."""
    if requested != "auto":
        return requested
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


def _model_dir_name(d: str) -> str:
    """Real model name of a downloaded model folder (from inference.yml), else the folder name."""
    try:
        txt = (Path(d) / "inference.yml").read_text(encoding="utf-8")
        m = re.search(r"^\s*model_name:\s*(\S+)", txt, re.M)
        if m:
            return m.group(1)
    except OSError:
        pass
    return Path(d).name


def _find_local_model(base: Path, name: str | None) -> str | None:
    if not name:
        return None
    for cand in (base / name, base / f"{name}_infer"):
        if cand.is_dir():
            return str(cand)
    return None


class PaddleEngine:
    """PaddleOCR 3.x wrapper (text detection + recognition)."""

    name = "PaddleOCR"

    def __init__(
        self,
        lang: str = "ru",
        device: str = "auto",
        auto_rotate: bool = False,
        fast: bool = False,
        det_model_dir: str | None = None,
        rec_model_dir: str | None = None,
        mkldnn: bool = True,
        textline_orient: bool = False,
        unwarp: bool = False,
    ):
        self.lang = lang
        self.device = resolve_device(device)
        self.auto_rotate = auto_rotate
        self.fast = fast
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self.mkldnn = mkldnn
        self.textline_orient = textline_orient
        self.unwarp = unwarp
        print(f"  device: {self.device}" + ("" if self.device.startswith("gpu") else " (no usable GPU build found; see README for GPU setup)" if device == "auto" else ""))
        try:
            self._ocr = self._build(mkldnn)
        except Exception as e:
            if not self.device.startswith("gpu"):
                raise
            print(f"  ! GPU init failed ({type(e).__name__}: {e}); falling back to CPU", file=sys.stderr)
            self.device = "cpu"
            self._ocr = self._build(mkldnn)

    def _build(self, mkldnn: bool):
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "PaddleOCR is not installed. Run setup.bat (Windows) or "
                "`pip install -r requirements.txt`.\n" + str(e)
            )
        kwargs = dict(
            use_doc_orientation_classify=self.auto_rotate,
            use_doc_unwarping=self.unwarp,
            use_textline_orientation=self.textline_orient,
            device=self.device,
            enable_mkldnn=mkldnn,
        )
        kwargs.update(self._model_kwargs())
        return PaddleOCR(**kwargs)

    def _model_kwargs(self) -> dict:
        """Detection/recognition model arguments shared by the plain and the structure pipeline."""
        kwargs: dict = {}
        try:
            det_name, rec_name = self._default_names()
        except Exception:
            det_name = rec_name = None
        if self.fast and det_name:
            det_name = det_name.replace("server", "mobile").replace("medium", "mobile")

        # Manually downloaded models: explicit --*-model-dir wins, otherwise a folder
        # named exactly like the model inside <this script's folder>\models is used.
        local = Path(__file__).resolve().parent / "models"
        det_dir = self.det_model_dir or _find_local_model(local, det_name)
        rec_dir = self.rec_model_dir or _find_local_model(local, rec_name)

        if det_dir or rec_dir or self.fast:
            # Explicit model names/dirs: PaddleOCR then ignores `lang`.
            kwargs["text_detection_model_name"] = _model_dir_name(det_dir) if det_dir else det_name
            kwargs["text_recognition_model_name"] = _model_dir_name(rec_dir) if rec_dir else rec_name
            if det_dir:
                kwargs["text_detection_model_dir"] = det_dir
                print(f"  using local detection model: {det_dir}")
            if rec_dir:
                kwargs["text_recognition_model_dir"] = rec_dir
                print(f"  using local recognition model: {rec_dir}")
        else:
            kwargs["lang"] = self.lang
        return kwargs

    def _default_names(self) -> tuple[str, str]:
        from paddleocr import PaddleOCR

        names = PaddleOCR._get_ocr_model_names(PaddleOCR.__new__(PaddleOCR), self.lang, None)
        return names

    def _predict(self, image_rgb: np.ndarray):
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])  # PaddleOCR expects BGR
        try:
            return list(self._ocr.predict(bgr))
        except Exception as e:
            if self.device.startswith("gpu"):
                print(f"  ! GPU inference failed ({type(e).__name__}); falling back to CPU", file=sys.stderr)
                self.device = "cpu"
                self._ocr = self._build(self.mkldnn)
                return list(self._ocr.predict(bgr))
            # Known oneDNN/MKLDNN incompatibilities on some CPUs/paddle builds:
            # rebuild once without MKLDNN and retry.
            if self.mkldnn:
                print(f"  ! predict failed ({type(e).__name__}); retrying without MKLDNN", file=sys.stderr)
                self.mkldnn = False
                self._ocr = self._build(False)
                return list(self._ocr.predict(bgr))
            raise

    @staticmethod
    def _parse_lines(res) -> list[Line]:
        lines: list[Line] = []
        texts = res["rec_texts"]
        scores = res["rec_scores"]
        boxes = res["rec_boxes"] if "rec_boxes" in res else None
        polys = res["rec_polys"] if "rec_polys" in res else None
        for i, (t, s) in enumerate(zip(texts, scores)):
            t = (t or "").strip()
            if not t:
                continue
            if boxes is not None and len(boxes) > i:
                x0, y0, x1, y1 = [float(v) for v in boxes[i]]
            else:
                p = np.asarray(polys[i], dtype=float)
                x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
            lines.append(Line(t, float(s), (x0, y0, x1, y1)))
        return lines

    def recognize(self, image_rgb: np.ndarray) -> list[Line]:
        lines: list[Line] = []
        for res in self._predict(image_rgb):
            lines += self._parse_lines(res)
        return lines

    def recognize_page(self, image_rgb: np.ndarray) -> tuple[list[Line], list[dict] | None]:
        """Lines plus, for the structure engine, the layout blocks (None = plain reading-order rebuild)."""
        return self.recognize(image_rgb), None


class StructureEngine(PaddleEngine):
    """PP-StructureV3: layout blocks, headings and real tables (slower, ~5-8 s per page on GPU)."""

    name = "PaddleOCR PP-StructureV3"

    def _build(self, mkldnn: bool):
        try:
            from paddleocr import PPStructureV3
        except ImportError as e:  # pragma: no cover
            raise SystemExit("PaddleOCR is not installed. Run setup.bat.\n" + str(e))
        kwargs = dict(
            # the document preprocessor is switched on by the orientation flag only, so unwarping needs it too
            use_doc_orientation_classify=self.auto_rotate or self.unwarp,
            use_doc_unwarping=self.unwarp,
            use_textline_orientation=self.textline_orient,
            use_table_recognition=True,
            use_seal_recognition=False,
            use_formula_recognition=False,
            use_chart_recognition=False,
            device=self.device,
            enable_mkldnn=mkldnn,
        )
        kwargs.update(self._model_kwargs())
        try:
            return PPStructureV3(**kwargs)
        except RuntimeError as e:
            if "dependenc" in str(e).lower():
                raise SystemExit(
                    "--structure needs extra packages. Run setup.bat again, or:\n"
                    '  "%USERPROFILE%\\.pdf-ocr\\venv\\Scripts\\python.exe" -m pip install "paddlex[ocr]==3.7.2"'
                )
            raise

    def recognize_page(self, image_rgb: np.ndarray) -> tuple[list[Line], list[dict] | None]:
        lines: list[Line] = []
        blocks: list[dict] = []
        for res in self._predict(image_rgb):
            lines += self._parse_lines(res["overall_ocr_res"])
            for b in res["parsing_res_list"]:
                x0, y0, x1, y1 = [float(v) for v in b.bbox]
                block = {"label": b.label, "bbox": (x0, y0, x1, y1)}
                if b.label == "table" and isinstance(b.content, str) and "<table" in b.content:
                    block["html"] = b.content
                blocks.append(block)
        return lines, blocks


# --------------------------------------------------------------------------- text cleanup
# Conservative fixes for systematic PaddleOCR mistakes on Russian text. They never add words:
#  - spaces after , ; ) and before ( when the OCR glued words together,
#  - Latin look-alike letters inside otherwise Cyrillic words (Mockва -> Москва),
#  - a mis-read thousands group after a number (86 o00 -> 86 000).
# Original text is kept in the JSON ("raw"). Disable with --no-fix.

_LATIN_TO_CYR = str.maketrans(
    "AaBCcEeHKkMmOoPpTXxyY",
    "\u0410\u0430\u0412\u0421\u0441\u0415\u0435\u041d\u041a\u043a\u041c\u043c\u041e\u043e\u0420\u0440\u0422\u0425\u0445\u0443\u0423",
)
_LOOKALIKES = frozenset("AaBCcEeHKkMmOoPpTXxyY")
_CYR = re.compile(r"[\u0400-\u04ff]")
_LAT = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[A-Za-z\u0400-\u04ff]{3,}")
_ZERO_TOK = re.compile(r"^([OoО\u043e0-9]{3})(\W*)$")


def fix_text(text: str) -> tuple[str, int]:
    n = 0
    out = text

    # a) thousands group mis-read as letters: "86 o00", "5 оо0"
    toks = out.split(" ")
    for i in range(1, len(toks)):
        m = _ZERO_TOK.match(toks[i])
        if (
            m
            and re.fullmatch(r"\d{1,3}", toks[i - 1])
            and any(c.isdigit() for c in m.group(1))
            and re.search(r"[OoО\u043e]", m.group(1))
        ):
            toks[i] = re.sub(r"[OoО\u043e]", "0", m.group(1)) + m.group(2)
            n += 1
    out = " ".join(toks)

    # b) Latin look-alikes inside words that are mostly Cyrillic
    def _word(m: re.Match) -> str:
        w = m.group(0)
        cyr, lat = len(_CYR.findall(w)), len(_LAT.findall(w))
        only_lookalikes = all(c in _LOOKALIKES for c in _LAT.findall(w))
        if cyr and lat and (cyr > lat or (cyr >= 2 and only_lookalikes)):  # "Мockвa": 2 Cyrillic, 4 look-alikes
            fixed = w.translate(_LATIN_TO_CYR)
            if fixed != w:
                nonlocal n
                n += 1
                return fixed
        return w

    if "://" not in out and "@" not in out:  # leave URLs and e-mails alone
        out = _WORD.sub(_word, out)

    # c) missing spaces next to punctuation
    for pat, rep in (
        (r"(?<=[,;])(?=[A-Za-z\u0400-\u04ff])", " "),
        (r"(?<=\))(?=[A-Za-z\u0400-\u04ff])", " "),
        (r"(?<=[\u0430-\u044f\u0451])\((?=[\u0400-\u04ff0-9])", " ("),
        (r"(?<=[\u0430-\u044f\u0451]{2}\.)(?=[\u0410-\u042f\u0401])", " "),
    ):
        if "://" in out or "@" in out:
            break
        out, k = re.subn(pat, rep, out)
        n += k
    return out, n


# --------------------------------------------------------------------------- layout -> markdown

_ITEM_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?|[-\u2013\u2014\u2022*]|[A-Za-z\u0410-\u044f]\))\s"
)


def lines_to_markdown(lines: list[Line]) -> str:
    """Rebuild reading order and paragraphs from OCR lines (one page)."""
    if not lines:
        return ""

    # 1) group boxes into visual rows
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l: l.cy):
        if rows:
            ref = rows[-1]
            row_cy = sum(l.cy for l in ref) / len(ref)
            row_h = statistics.median(l.h for l in ref)
            if abs(ln.cy - row_cy) < 0.6 * min(ln.h, row_h):
                ref.append(ln)
                continue
        rows.append([ln])

    # 2) row geometry
    @dataclass
    class Row:
        text: str
        x0: float
        x1: float
        y0: float
        y1: float
        h: float

    built: list[Row] = []
    for r in rows:
        r.sort(key=lambda l: l.box[0])
        text = r[0].text
        for prev, cur in zip(r, r[1:]):
            gap = cur.box[0] - prev.box[2]
            text += ("   " if gap > 3 * prev.h else " ") + cur.text
        built.append(
            Row(
                text,
                min(l.box[0] for l in r),
                max(l.box[2] for l in r),
                min(l.box[1] for l in r),
                max(l.box[3] for l in r),
                statistics.median(l.h for l in r),
            )
        )

    med_h = statistics.median(r.h for r in built)
    left = statistics.median(sorted(r.x0 for r in built)[: max(3, len(built) // 3)])
    rights = sorted(r.x1 for r in built)
    right = rights[int(0.9 * (len(rights) - 1))]  # robust right margin

    # 3) paragraphs
    paragraphs: list[list[str]] = []
    prev: Row | None = None
    for r in built:
        new_par = prev is None
        if prev is not None:
            gap_y = r.y0 - prev.y1
            if gap_y > 0.7 * med_h:
                new_par = True
            elif prev.x1 < left + 0.82 * (right - left):  # previous line was short
                new_par = True
            elif r.x0 > left + 1.5 * med_h:  # first-line indent
                new_par = True
            elif _ITEM_RE.match(r.text):  # numbered / bulleted item
                new_par = True
        if new_par:
            paragraphs.append([r.text])
        else:
            paragraphs[-1].append(r.text)
        prev = r
    return "\n\n".join(" ".join(p) for p in paragraphs)


def html_table_to_markdown(html: str) -> str:
    """HTML <table> from the table recogniser -> Markdown table (raw HTML if bs4 is missing)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html
    table = BeautifulSoup(html, "html.parser").find("table")
    if table is None:
        return ""
    grid: list[list[str]] = []
    pending: dict[int, int] = {}  # column -> rows still covered by a rowspan from above
    for tr in table.find_all("tr"):
        row: list[str] = []
        col = 0

        def fill_spans():
            nonlocal col
            while pending.get(col, 0) > 0:
                pending[col] -= 1
                row.append("")
                col += 1

        for td in tr.find_all(["td", "th"]):
            fill_spans()
            txt = " ".join(td.get_text(" ").split()).replace("|", "\\|")
            cs = int(td.get("colspan") or 1) if str(td.get("colspan") or "1").isdigit() else 1
            rs = int(td.get("rowspan") or 1) if str(td.get("rowspan") or "1").isdigit() else 1
            for k in range(cs):
                row.append(txt if k == 0 else "")
                if rs > 1:
                    pending[col] = rs - 1
                col += 1
        fill_spans()
        grid.append(row)
    if not grid:
        return ""
    width = max(len(r) for r in grid)
    grid = [r + [""] * (width - len(r)) for r in grid]
    out = ["| " + " | ".join(grid[0]) + " |", "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in grid[1:]]
    return "\n".join(out)


_TITLE_PREFIX = {"doc_title": "# ", "title": "# ", "paragraph_title": "## "}
_IMAGE_LABELS = {"image", "figure", "chart", "seal", "header_image", "footer_image", "formula_number"}


def structure_to_markdown(blocks: list[dict], lines: list[Line]) -> str:
    """Markdown from layout blocks (reading order) with the text of the OCR lines inside each block."""
    owner: dict[int, int] = {}
    for ln in lines:
        cx = (ln.box[0] + ln.box[2]) / 2
        best: tuple[float, int] | None = None
        for i, b in enumerate(blocks):
            x0, y0, x1, y1 = b["bbox"]
            if x0 <= cx <= x1 and y0 <= ln.cy <= y1:
                area = (x1 - x0) * (y1 - y0)
                if best is None or area < best[0]:  # nested blocks: the smallest one wins
                    best = (area, i)
        if best:
            owner[id(ln)] = best[1]

    parts: list[str] = []
    for i, b in enumerate(blocks):
        own = [l for l in lines if owner.get(id(l)) == i]
        label = b["label"]
        if label == "table" and b.get("html"):
            md = html_table_to_markdown(b["html"])
        elif not own:
            continue
        elif label in _TITLE_PREFIX:
            md = _TITLE_PREFIX[label] + " ".join(lines_to_markdown(own).split())
        elif label in _IMAGE_LABELS:  # text found inside a picture / stamp: keep it, but mark it
            md = "\n".join("> " + p.replace("\n", " ") for p in lines_to_markdown(own).split("\n\n"))
        else:
            md = lines_to_markdown(own)
        if md.strip():
            parts.append(md)
    orphans = [l for l in lines if id(l) not in owner]
    if orphans:
        parts.append(lines_to_markdown(orphans))
    return "\n\n".join(parts)


def native_page_markdown(page: pymupdf.Page) -> str:
    blocks = page.get_text("blocks", sort=True)
    out = []
    for b in blocks:
        if b[6] != 0:  # not a text block
            continue
        txt = re.sub(r"\s*\n\s*", " ", b[4]).strip()
        if txt:
            out.append(txt)
    return "\n\n".join(out)


# --------------------------------------------------------------------------- searchable PDF

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def find_font(user_font: str | None) -> str | None:
    for p in ([user_font] if user_font else []) + _FONT_CANDIDATES:
        if p and Path(p).is_file():
            return p
    return None


def add_text_layer(page: pymupdf.Page, res: PageResult, font: pymupdf.Font) -> int:
    """Write invisible (render mode 3) text over the scan so it is searchable/selectable."""
    if not res.lines or not res.img_size:
        return 0
    scale = page.rect.width / res.img_size[0]
    n = 0
    for ln in res.lines:
        x0, y0, x1, y1 = (v * scale for v in ln.box)
        h = y1 - y0
        w = x1 - x0
        if h <= 0 or w <= 0:
            continue
        fs = max(h * 0.85, 2.0)
        tl = font.text_length(ln.text, fontsize=fs)
        if tl <= 0:
            continue
        origin = pymupdf.Point(x0, y1 - 0.2 * h)
        tw = pymupdf.TextWriter(page.rect)
        tw.append(origin, ln.text, font=font, fontsize=fs)
        tw.write_text(page, morph=(origin, pymupdf.Matrix(w / tl, 1)), render_mode=3)
        n += 1
    return n


# --------------------------------------------------------------------------- core


def parse_pages(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a) if a else 1
            b = int(b) if b else total
            out.update(range(a - 1, min(b, total)))
        else:
            out.add(int(part) - 1)
    return sorted(p for p in out if 0 <= p < total)


def render_page(page: pymupdf.Page, dpi: int) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def process_pdf(
    pdf: Path,
    engine: Engine,
    out_dir: Path | None = None,
    dpi: int = 200,
    pages: str | None = None,
    make_pdf: bool = True,
    make_json: bool = False,
    force_ocr: bool = False,
    font_path: str | None = None,
    fix: bool = True,
    log=print,
) -> dict:
    t0 = time.time()
    out_dir = out_dir or pdf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{pdf.stem}.md"
    pdf_path = out_dir / f"{pdf.stem}.ocr.pdf"
    json_path = out_dir / f"{pdf.stem}.ocr.json"

    doc = pymupdf.open(pdf)
    if doc.needs_pass:
        raise RuntimeError("PDF is password-protected")
    total = len(doc)
    selected = parse_pages(pages, total)
    results: list[PageResult] = []
    fixes = 0

    for idx in selected:
        page = doc[idx]
        native = "" if force_ocr else page.get_text("text").strip()
        if len(native) >= NATIVE_TEXT_MIN_CHARS:
            results.append(PageResult(idx + 1, "native", native_page_markdown(page), [], None))
            log(f"  page {idx + 1}/{total}: text layer found, copied")
            continue
        ts = time.time()
        img = render_page(page, dpi)
        if hasattr(engine, "recognize_page"):
            lines, layout = engine.recognize_page(img)
        else:  # simple engines (tests) only implement recognize()
            lines, layout = engine.recognize(img), None
        if fix:
            for ln in lines:
                new, k = fix_text(ln.text)
                if k and new != ln.text:
                    ln.raw, ln.text = ln.text, new
                    fixes += k
        text = structure_to_markdown(layout, lines) if layout is not None else lines_to_markdown(lines)
        res = PageResult(idx + 1, "ocr", text, lines, (img.shape[1], img.shape[0]))
        results.append(res)
        ms = res.mean_score
        info = f"{len(lines)} lines, confidence {ms:.2f}" if ms is not None else "no text found"
        log(f"  page {idx + 1}/{total}: {info} ({time.time() - ts:.1f}s)")

    # ---- markdown
    scores = [l.score for r in results for l in r.lines]
    mean = sum(scores) / len(scores) if scores else None
    low_pages = [r.number for r in results if r.mean_score is not None and r.mean_score < LOW_CONFIDENCE]
    front = [
        "---",
        f"source: {pdf.name}",
        f"pages: {total}" + (f" (processed: {len(results)})" if len(results) != total else ""),
        f"ocr_engine: {engine.name}",
        f"ocr_pages: {sum(1 for r in results if r.source == 'ocr')}",
        f"native_text_pages: {sum(1 for r in results if r.source == 'native')}",
        f"dpi: {dpi}",
        f"mean_confidence: {mean:.3f}" if mean is not None else "mean_confidence: n/a",
        f"text_fixes: {fixes}",
        f"low_confidence_pages: [{', '.join(map(str, low_pages))}]",
        f"generated: {_dt.datetime.now().isoformat(timespec='seconds')}",
        "---",
        "",
    ]
    body = []
    for r in results:
        body.append(f"<!-- page {r.number} -->\n\n{r.text}".rstrip())
    md_path.write_text("\n".join(front) + "\n" + "\n\n".join(body) + "\n", encoding="utf-8")

    # ---- searchable pdf
    pdf_note = None
    if make_pdf:
        fpath = find_font(font_path)
        if not fpath:
            pdf_note = "no Unicode font found (use --font); searchable PDF skipped"
        else:
            font = pymupdf.Font(fontfile=fpath)
            added = 0
            for r in results:
                if r.source != "ocr" or not r.lines:
                    continue
                page = doc[r.number - 1]
                if page.rotation:
                    log(f"  ! page {r.number} is rotated; text layer skipped for this page")
                    continue
                added += add_text_layer(page, r, font)
            try:
                doc.subset_fonts()
            except Exception:
                pass
            doc.save(pdf_path, garbage=3, deflate=True)
    doc.close()

    # ---- json
    if make_json:
        json_path.write_text(
            json.dumps(
                {
                    "source": pdf.name,
                    "engine": engine.name,
                    "dpi": dpi,
                    "pages": [
                        {
                            "page": r.number,
                            "source": r.source,
                            "image_size": r.img_size,
                            "mean_confidence": r.mean_score,
                            "lines": [
                                {"text": l.text, "score": round(l.score, 4), "box": [round(v, 1) for v in l.box], **({"raw": l.raw} if l.raw else {})}
                                for l in r.lines
                            ],
                        }
                        for r in results
                    ],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )

    return {
        "pdf": str(pdf),
        "markdown": str(md_path),
        "searchable_pdf": str(pdf_path) if make_pdf and not pdf_note else None,
        "json": str(json_path) if make_json else None,
        "pages": total,
        "mean_confidence": mean,
        "low_confidence_pages": low_pages,
        "text_fixes": fixes,
        "seconds": round(time.time() - t0, 1),
        "note": pdf_note,
    }


# --------------------------------------------------------------------------- cli


def collect_pdfs(inputs: Iterable[str]) -> list[Path]:
    """PDFs from files and folders: no duplicates, our own *.ocr.pdf outputs are never OCR'd again."""
    found: list[Path] = []
    seen: set[str] = set()

    def add(q: Path) -> None:
        key = os.path.normcase(str(q.resolve()))
        if key in seen:
            return
        seen.add(key)
        found.append(q)

    for item in inputs:
        p = Path(item)
        if p.is_dir():
            for q in sorted(p.rglob("*.pdf")):
                if not q.name.lower().endswith(".ocr.pdf"):
                    add(q)
        elif p.is_file():
            if p.name.lower().endswith(".ocr.pdf"):
                print(f"skip (already an OCR result): {p}")
            elif p.suffix.lower() != ".pdf":
                print(f"skip (not a PDF): {p}")
            else:
                add(p)
        else:
            print(f"! not found: {item}", file=sys.stderr)
    return found


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="OCR PDF files with PaddleOCR -> Markdown (+ searchable PDF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="PDF files and/or folders (folders are searched recursively)")
    ap.add_argument("-o", "--out", help="output folder (default: next to each PDF)")
    ap.add_argument("-l", "--lang", default="ru", help="PaddleOCR language code: ru, en, ch, uk, ...")
    ap.add_argument("--dpi", type=int, default=200, help="rendering resolution for OCR")
    ap.add_argument("--pages", help="page selection, e.g. 1-3,5,8-")
    ap.add_argument("--no-pdf", action="store_true", help="do not create the searchable PDF")
    ap.add_argument("--json", action="store_true", help="also write <name>.ocr.json with lines, boxes, confidence")
    ap.add_argument("--force-ocr", action="store_true", help="OCR even pages that already have a text layer")
    ap.add_argument("--overwrite", action="store_true", help="redo files whose .md already exists")
    ap.add_argument("--no-fix", action="store_true", help="skip the conservative text cleanup (spaces, look-alike letters, 86 o00 -> 86 000)")
    ap.add_argument("--auto-rotate", action="store_true", help="detect and fix page orientation (slower)")
    ap.add_argument("--fast", action="store_true", help="use the lighter mobile detection model")
    ap.add_argument("--structure", action="store_true", help="layout analysis: headings and real Markdown tables (PP-StructureV3, slower)")
    ap.add_argument("--textline-orient", action="store_true", help="fix upside-down / rotated single text lines")
    ap.add_argument("--unwarp", action="store_true", help="flatten curved or photographed pages (no searchable PDF is made then)")
    ap.add_argument("--device", default="auto", help="auto (GPU if available, else CPU), cpu, or gpu:0")
    ap.add_argument("--no-mkldnn", action="store_true", help="disable oneDNN (use if you get CPU/runtime errors)")
    ap.add_argument("--det-model-dir", help="folder with an already downloaded detection model")
    ap.add_argument("--rec-model-dir", help="folder with an already downloaded recognition model")
    ap.add_argument("--font", help="TTF font with Cyrillic/Unicode support for the searchable PDF")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    pdfs = collect_pdfs(args.inputs)
    if not pdfs:
        print("No PDF files found.", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else None
    todo = []
    for p in pdfs:
        md = (out_dir or p.parent) / f"{p.stem}.md"
        if md.exists() and not args.overwrite:
            print(f"skip (already done, use --overwrite): {p}")
            continue
        todo.append(p)
    if not todo:
        return 0

    if args.unwarp and not args.no_pdf:
        print("note: --unwarp changes the page geometry, so no searchable PDF is written (Markdown only)")
        args.no_pdf = True

    print(f"Loading PaddleOCR (lang={args.lang}); the first run downloads models, please wait...")
    engine = (StructureEngine if args.structure else PaddleEngine)(
        lang=args.lang,
        device=args.device,
        auto_rotate=args.auto_rotate,
        fast=args.fast,
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        mkldnn=not args.no_mkldnn,
        textline_orient=args.textline_orient,
        unwarp=args.unwarp,
    )

    failed_files: list[Path] = []
    low_files: list[Path] = []
    done = 0
    t_all = time.time()
    for n, p in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] {p}")
        try:
            r = process_pdf(
                p,
                engine,
                out_dir=out_dir,
                dpi=args.dpi,
                pages=args.pages,
                make_pdf=not args.no_pdf,
                make_json=args.json,
                force_ocr=args.force_ocr,
                font_path=args.font,
                fix=not args.no_fix,
            )
        except Exception as e:  # keep going with the other files
            failed_files.append(p)
            print(f"FAILED: {p}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        done += 1
        print(f"  -> {r['markdown']}")
        if r["searchable_pdf"]:
            print(f"  -> {r['searchable_pdf']}")
        if r["note"]:
            print(f"  ! {r['note']}")
        if r["low_confidence_pages"]:
            low_files.append(p)
            print(f"  ! low confidence on pages: {r['low_confidence_pages']} - check them manually")
        print(f"  done in {r['seconds']}s")

    if len(todo) > 1 or failed_files:
        print(f"\nSummary: {done} done, {len(failed_files)} failed, {len(pdfs) - len(todo)} skipped, {time.time() - t_all:.0f}s total")
        for p in low_files:
            print(f"  check manually (low confidence): {p}")
        for p in failed_files:
            print(f"  FAILED: {p}")
    return 1 if failed_files else 0


if __name__ == "__main__":
    sys.exit(main())
