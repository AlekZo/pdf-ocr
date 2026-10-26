"""Pipeline test without PaddleOCR models: uses Tesseract (English) and a stub engine.
Run: python tests/pipeline_test.py   (needs pytesseract-free: calls the tesseract CLI)"""
import hashlib, subprocess, sys, tempfile, csv, io
from pathlib import Path
import numpy as np, pymupdf
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pdf_ocr as P

class TesseractEngine:
    name = "tesseract-test"
    def recognize(self, rgb):
        with tempfile.TemporaryDirectory() as d:
            png = Path(d) / "p.png"
            pymupdf.Pixmap(pymupdf.csRGB, rgb.shape[1], rgb.shape[0], rgb.tobytes(), False).save(png)
            out = subprocess.run(["tesseract", str(png), "-", "--psm", "4", "tsv"], capture_output=True, text=True).stdout
        lines = {}
        for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
            if r["level"] == "5" and r["text"].strip():
                k = (r["block_num"], r["par_num"], r["line_num"])
                x, y, w, h = (int(r[c]) for c in ("left", "top", "width", "height"))
                lines.setdefault(k, []).append((r["text"], float(r["conf"]) / 100, x, y, x + w, y + h))
        res = []
        for ws in lines.values():
            res.append(P.Line(" ".join(w[0] for w in ws), sum(w[1] for w in ws) / len(ws),
                (min(w[2] for w in ws), min(w[3] for w in ws), max(w[4] for w in ws), max(w[5] for w in ws))))
        return res

class StubCyrillic:
    name = "stub"
    def recognize(self, rgb):
        W = rgb.shape[1]
        return [P.Line("Договор аренды квартиры", .99, (W*.2, 200, W*.8, 260)),
                P.Line("1.1. Арендодатель передаёт Арендатору квартиру", .95, (W*.1, 400, W*.9, 450)),
                P.Line("во временное пользование.", .95, (W*.1, 460, W*.5, 510))]

def make_scan(path, text_lines):
    d = pymupdf.open(); pg = d.new_page()
    y = 80
    for t in text_lines:
        pg.insert_text((60, y), t, fontsize=14); y += 22
    pix = pg.get_pixmap(dpi=200)
    s = pymupdf.open(); sp = s.new_page(width=595, height=842)
    sp.insert_image(sp.rect, pixmap=pix); s.save(path)

tmp = Path(tempfile.mkdtemp())
src = tmp / "scan test.pdf"
make_scan(src, ["This lease agreement is made between the Landlord", "and the Tenant for the flat at Mira avenue 135A.", "", "Payment is due on the fifth day of each month."])
before = pymupdf.open(src)[0].get_pixmap(dpi=72).samples

import shutil
if shutil.which("tesseract"):
    r = P.process_pdf(src, TesseractEngine(), out_dir=tmp/"out", make_json=True)
    print(r)
    md = Path(r["markdown"]).read_text(encoding="utf-8"); print(md)
    assert "Landlord" in md and "Tenant" in md
    out = pymupdf.open(r["searchable_pdf"])
    assert out[0].search_for("Landlord"), "searchable text layer missing"
    assert out[0].get_pixmap(dpi=72).samples == before, "text layer is visible!"
    print("English pipeline OK, text layer invisible, size", Path(r["searchable_pdf"]).stat().st_size)
else:
    print("tesseract not installed: English pipeline test skipped")

r = P.process_pdf(src, StubCyrillic(), out_dir=tmp/"out2")
md = Path(r["markdown"]).read_text(encoding="utf-8"); print(md)
assert "Арендатору квартиру во временное пользование." in md
out = pymupdf.open(r["searchable_pdf"])
assert out[0].search_for("Арендодатель"), "cyrillic text layer missing"
assert "Договор аренды квартиры" in out[0].get_text().replace(chr(160), " ")  # PyMuPDF 2.0 extracts spaces as NBSP
print("Cyrillic text layer OK, size", Path(r["searchable_pdf"]).stat().st_size)

# native-text page is not OCR'd
d = pymupdf.open(); d.new_page().insert_text((50, 50), "Born digital page with enough real text to skip OCR entirely.")
d.save(tmp/"native.pdf")
r = P.process_pdf(tmp/"native.pdf", StubCyrillic(), out_dir=tmp/"out3")
assert "native_text_pages: 1" in Path(r["markdown"]).read_text(encoding="utf-8")
print("native page OK")

# structure: headings, table -> Markdown table with colspan/rowspan, text inside an image is marked
html = ("<html><body><table><tr><td>Name</td><td colspan=2>Amount | RUB</td></tr>"
        "<tr><td rowspan=2>Rent</td><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table></body></html>")
md = P.html_table_to_markdown(html)
assert md.splitlines()[0] == "| Name | Amount \\| RUB |  |", md
assert md.splitlines()[3] == "|  | 3 | 4 |", md
blocks = [{"label": "doc_title", "bbox": (0, 0, 1000, 100)},
          {"label": "text", "bbox": (0, 120, 1000, 300)},
          {"label": "table", "bbox": (0, 320, 1000, 500), "html": html},
          {"label": "image", "bbox": (0, 520, 1000, 700)}]
ls = [P.Line("Договор", .9, (100, 20, 300, 60)),
      P.Line("Первая строка абзаца", .9, (10, 130, 900, 170)),
      P.Line("вторая строка.", .9, (10, 180, 400, 220)),
      P.Line("Name", .9, (10, 330, 100, 360)),
      P.Line("ШТАМП", .9, (10, 540, 200, 580))]
md = P.structure_to_markdown(blocks, ls)
assert md.startswith("# Договор\n\nПервая строка абзаца вторая строка.\n\n| Name |"), md
assert md.endswith("> ШТАМП"), md
assert "Name" not in md.split("| Name")[0], "table text must not be repeated as a paragraph"
print("structure markdown OK")

# batch input: duplicates, results of earlier runs and non-PDFs are filtered
(tmp / "b").mkdir(); (tmp / "b" / "x.pdf").write_bytes(b""); (tmp / "b" / "x.ocr.pdf").write_bytes(b""); (tmp / "b" / "n.txt").write_text("x")
got = P.collect_pdfs([str(tmp / "b"), str(tmp / "b" / "x.pdf"), str(tmp / "b" / "x.ocr.pdf"), str(tmp / "b" / "n.txt")])
assert [p.name for p in got] == ["x.pdf"], got
print("batch input OK\nALL PASSED")
