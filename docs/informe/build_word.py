"""Genera docs/Informe_HealthRAG.docx a partir de docs/informe/informe.html.

Requisitos: Windows con Microsoft Word, Microsoft Edge (para pasar el diagrama SVG a PNG),
y los paquetes pywin32 y beautifulsoup4 (vienen con Anaconda base).

Uso (después de `python docs/informe/build_informe.py`):
    python docs/informe/build_word.py [--pdf ruta.pdf]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import win32com.client as win32
from bs4 import BeautifulSoup, NavigableString, Tag

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HTML = HERE / "informe.html"
OUT = ROOT / "docs" / "Informe_HealthRAG.docx"

# Constantes de Word
H1, H2, NORMAL, BULLET, CAPTION, TITLE = -2, -3, -1, -49, -35, -63
FMT_DOCX, FMT_PDF = 12, 17
BODY_FONT, CODE_FONT = "Calibri", "Consolas"


def bgr(hex_: str) -> int:
    r, g, b = int(hex_[1:3], 16), int(hex_[3:5], 16), int(hex_[5:7], 16)
    return r + (g << 8) + (b << 16)


INK, INK2, LINE, SOFT, ACCENT, ACCENT_SOFT = (bgr(c) for c in
    ("#0b0b0b", "#52514e", "#cfcec9", "#f4f3f0", "#2a78d6", "#e8f0fb"))


# ----------------------------------------------------------------------------- HTML → runs
def runs_of(node: Tag) -> list[tuple[str, dict]]:
    """Texto con formato de un elemento: [(texto, {bold, italic, code})]."""
    out: list[tuple[str, dict]] = []

    def walk(n, fmt):
        if isinstance(n, NavigableString):
            if n.__class__.__name__ == "Comment":
                return
            out.append((str(n), dict(fmt)))
            return
        if not isinstance(n, Tag):
            return
        if n.name == "br":
            out.append(("\v", dict(fmt)))
            return
        f = dict(fmt)
        if n.name in ("strong", "b", "th"):
            f["bold"] = True
        if n.name in ("em", "i"):
            f["italic"] = True
        if n.name == "code":
            f["code"] = True
        for c in n.children:
            walk(c, f)

    walk(node, {})
    # Normalizar espacios como lo hace el navegador
    clean: list[tuple[str, dict]] = []
    for t, f in out:
        t = t.replace("\u2011", "-")
        if t != "\v":
            t = re.sub(r"\s+", " ", t)
        if clean and (clean[-1][0].endswith((" ", "\v")) or not clean) and t.startswith(" "):
            t = t.lstrip(" ")
        if t:
            clean.append((t, f))
    if clean:
        clean[0] = (clean[0][0].lstrip(), clean[0][1])
        clean[-1] = (clean[-1][0].rstrip(), clean[-1][1])
    return [(t, f) for t, f in clean if t]


# ----------------------------------------------------------------------------- Word
class Doc:
    def __init__(self):
        self.word = win32.DispatchEx("Word.Application")
        self.word.Visible = False
        self.word.DisplayAlerts = 0
        try:
            self.doc = self.word.Documents.Add()
            self._setup()
        except Exception:
            self.word.Quit()
            raise

    def _setup(self):
        d = self.doc
        ps = d.PageSetup
        ps.PaperSize = 7  # A4
        for side in ("TopMargin", "BottomMargin", "LeftMargin", "RightMargin"):
            setattr(ps, side, 2.0 * 72 / 2.54)  # 2 cm
        ps.DifferentFirstPageHeaderFooter = True
        self.width = ps.PageWidth - ps.LeftMargin - ps.RightMargin

        st = d.Styles(NORMAL)
        st.Font.Name, st.Font.Size, st.Font.Color = BODY_FONT, 10.5, INK
        st.LanguageID = 11274  # español (Argentina)
        st.ParagraphFormat.SpaceBefore, st.ParagraphFormat.SpaceAfter = 0, 6
        st.ParagraphFormat.LineSpacingRule = 5  # múltiple
        st.ParagraphFormat.LineSpacing = 12 * 1.15
        for sid, size, before, after in ((H1, 16, 18, 6), (H2, 12.5, 12, 4)):
            s = d.Styles(sid)
            s.Font.Name, s.Font.Size, s.Font.Bold, s.Font.Color = BODY_FONT, size, True, INK
            s.Font.Italic = False
            pf = s.ParagraphFormat
            pf.SpaceBefore, pf.SpaceAfter, pf.KeepWithNext = before, after, True
        b = d.Styles(H1).ParagraphFormat.Borders(-3)
        b.LineStyle, b.LineWidth, b.Color = 1, 12, INK
        cap = d.Styles(CAPTION)
        cap.Font.Name, cap.Font.Size, cap.Font.Italic, cap.Font.Bold, cap.Font.Color = BODY_FONT, 8.5, False, False, INK2
        cap.ParagraphFormat.SpaceAfter = 10
        bl = d.Styles(BULLET)
        bl.Font.Name, bl.Font.Size = BODY_FONT, 10.5
        bl.ParagraphFormat.SpaceAfter = 3
        self.pending_break = False

        footer = d.Sections(1).Footers(1)  # principal (no la portada)
        footer.PageNumbers.Add(2, False)    # derecha, sin número en la primera página
        fr = footer.Range
        fr.Font.Size, fr.Font.Color = 8.5, INK2
        hdr = d.Sections(1).Headers(1).Range
        hdr.Text = "HealthRAG · Informe técnico"
        hdr.Font.Size, hdr.Font.Color = 8.5, INK2
        hdr.ParagraphFormat.Alignment = 2
        # Plantillas de lista propias (las de la galería dependen de cada instalación de Word)
        self.tpl_bullet = self._list_template(bullet=True)
        self.tpl_num = self._list_template(bullet=False)

    def _list_template(self, bullet: bool):
        tpl = self.doc.ListTemplates.Add(False)
        lv = tpl.ListLevels(1)
        if bullet:
            lv.NumberFormat = "•"  # viñeta
            lv.TrailingCharacter = 0
            lv.NumberStyle = 23
            lv.Font.Name = BODY_FONT
        else:
            lv.NumberFormat = "%1."
            lv.TrailingCharacter = 0
            lv.NumberStyle = 0
        lv.NumberPosition, lv.TextPosition, lv.TabPosition = 8, 22, 22
        lv.Alignment = 0
        lv.StartAt = 1
        return tpl

    def listify(self, rng, ordered=False, continue_previous=True):
        rng.Paragraphs(1).Range.ListFormat.ApplyListTemplate(
            self.tpl_num if ordered else self.tpl_bullet, continue_previous, 0)

    # --- básicos
    @property
    def end(self) -> int:
        return self.doc.Content.End - 1

    def par(self, runs, style=NORMAL, *, size=None, color=None, align=None, space_after=None,
            shade=None, border_left=None, keep_next=False):
        runs = [(t, f) for t, f in runs if t]
        if not runs:
            return None
        start = self.end
        text = "".join(t for t, _ in runs)
        self.doc.Range(start, start).InsertAfter(text)
        rng = self.doc.Range(start, start + len(text))
        p = rng.Paragraphs(1)
        p.Style = style
        p.Range.Font.Reset()
        p.Range.HighlightColorIndex = 0
        p.Range.ParagraphFormat.Borders.Enable = False
        p.Range.Shading.BackgroundPatternColor = -16777216  # automático
        pos = start
        for t, f in runs:
            r = self.doc.Range(pos, pos + len(t))
            if f.get("bold"):
                r.Font.Bold = True
            if f.get("italic"):
                r.Font.Italic = True
            if f.get("code"):
                r.Font.Name, r.Font.Size = CODE_FONT, 9
            if "[Completar" in t:
                r.HighlightColorIndex = 7  # amarillo
            pos += len(t)
        pf = p.Range.ParagraphFormat
        if size:
            rng.Font.Size = size
        if color is not None:
            rng.Font.Color = color
        if align is not None:
            pf.Alignment = align
        if space_after is not None:
            pf.SpaceAfter = space_after
        if keep_next or text.rstrip().endswith(":"):
            pf.KeepWithNext = True
        if self.pending_break:
            pf.PageBreakBefore = True
            self.pending_break = False
        if shade is not None:
            pf.Shading.BackgroundPatternColor = shade
        if border_left is not None:
            bd = pf.Borders(-2)
            bd.LineStyle, bd.LineWidth, bd.Color = 1, 18, border_left
            pf.Borders.DistanceFromLeft = 6
            pf.LeftIndent = 4
            pf.RightIndent = 4
        self.doc.Range(start + len(text), start + len(text)).InsertParagraphAfter()
        self._reset_tail()
        return self.doc.Range(start, start + len(text))

    def _reset_tail(self):
        """El párrafo vacío del final hereda estilo y formato del anterior: se limpia."""
        tail = self.doc.Range(self.end, self.end).Paragraphs(1)
        tail.Style = NORMAL
        tail.Range.ListFormat.RemoveNumbers()
        tail.Range.Font.Reset()
        tail.Range.HighlightColorIndex = 0
        tail.Range.ParagraphFormat.Borders.Enable = False
        tail.Range.ParagraphFormat.Shading.BackgroundPatternColor = -16777216

    def page_break(self):
        """El próximo párrafo empieza en una página nueva (evita páginas en blanco)."""
        self.pending_break = True

    def heading(self, text, level, page_break=False):
        if page_break:
            self.page_break()
        self.par([(text, {})], H1 if level == 1 else H2)

    def bullets(self, items, ordered=False, **kw):
        first = True
        for li in items:
            r = self.par(runs_of(li), BULLET, **kw)
            if r is None:
                continue
            self.listify(r, ordered, continue_previous=not first)
            first = False

    def table(self, el: Tag):
        rows = el.find_all("tr")
        cells = [tr.find_all(["th", "td"]) for tr in rows]
        data = [[runs_of(c) for c in r] for r in cells]
        left = [[("text-align:left" in (c.get("style") or "").replace(" ", "")) for c in r] for r in cells]
        header = bool(rows and rows[0].find("th"))
        ncols = max(len(r) for r in data)
        if ncols >= 10:
            # Demasiado ancha para A4 vertical: métricas en filas, experimentos en columnas
            for r, l in zip(data, left):
                r += [[]] * (ncols - len(r))
                l += [False] * (ncols - len(l))
            data = [list(c) for c in zip(*data)]
            left = [list(c) for c in zip(*left)]
            ncols = len(data[0])
        start = self.end
        t = self.doc.Tables.Add(self.doc.Range(start, start), len(data), ncols)
        t.Range.Font.Size = 7.5 if ncols >= 10 else 8 if "small" in el.get("class", []) else 9
        t.Range.HighlightColorIndex = 0
        t.Range.ParagraphFormat.SpaceAfter = 0
        t.Range.ParagraphFormat.SpaceBefore = 0
        t.Range.ParagraphFormat.LineSpacingRule = 0
        t.TopPadding = t.BottomPadding = 2
        numeric = "num" in el.get("class", [])
        for i, row in enumerate(data, 1):
            for j in range(1, ncols + 1):
                cell = t.Cell(i, j)
                runs = row[j - 1] if j - 1 < len(row) else []
                text = "".join(x for x, _ in runs)
                cell.Range.Text = text
                pos = cell.Range.Start
                for x, f in runs:
                    r = self.doc.Range(pos, pos + len(x))
                    if f.get("bold"):
                        r.Font.Bold = True
                    if f.get("italic"):
                        r.Font.Italic = True
                    if f.get("code"):
                        r.Font.Name = CODE_FONT
                    pos += len(x)
                is_left = j - 1 < len(left[i - 1]) and left[i - 1][j - 1]
                if numeric and j > 1 and not is_left:
                    cell.Range.ParagraphFormat.Alignment = 2
        t.Borders.Enable = True
        for idx in (-1, -2, -3, -4, -5, -6):
            bd = t.Borders(idx)
            bd.LineStyle, bd.LineWidth, bd.Color = 1, 4, LINE
        for idx in (-2, -4, -6):  # sin bordes verticales
            t.Borders(idx).LineStyle = 0
        if header:
            hr = t.Rows(1)
            hr.HeadingFormat = True
            hr.Range.Font.Bold = True
            hr.Range.Font.Color = INK2
            hr.Shading.BackgroundPatternColor = SOFT
            hb = hr.Borders(-3)
            hb.LineStyle, hb.LineWidth, hb.Color = 1, 12, INK2
        t.Rows.AllowBreakAcrossPages = False
        t.AutoFitBehavior(1)
        t.AutoFitBehavior(2)
        # Mantener las filas juntas cuando la tabla es corta
        if len(data) <= 18:
            for i in range(1, len(data)):
                t.Rows(i).Range.ParagraphFormat.KeepWithNext = True
        after = self.doc.Range(self.end, self.end)
        after.ParagraphFormat.SpaceBefore = 6

    def image(self, path: Path, width_frac=1.0):
        start = self.end
        rng = self.doc.Range(start, start)
        shp = self.doc.InlineShapes.AddPicture(str(path), False, True, rng)
        shp.LockAspectRatio = True
        shp.Width = self.width * width_frac
        p = self.doc.Range(start, start + 1).Paragraphs(1)
        p.Style = NORMAL
        p.Format.Alignment = 1
        p.Format.KeepWithNext = True
        p.Format.SpaceAfter = 2
        self.doc.Range(self.end, self.end).InsertParagraphAfter()
        self._reset_tail()

    def toc(self):
        start = self.end
        self.doc.TablesOfContents.Add(self.doc.Range(start, start), True, 1, 2)
        self.doc.Range(self.end, self.end).InsertParagraphAfter()

    def save(self, out: Path, pdf: Path | None):
        for toc in self.doc.TablesOfContents:
            toc.Update()
        props = self.doc.BuiltInDocumentProperties
        props("Title").Value = "Informe técnico HealthRAG"
        props("Subject").Value = "Procesamiento del Lenguaje Natural III"
        self.doc.SaveAs2(str(out), FMT_DOCX)
        if pdf:
            self.doc.ExportAsFixedFormat(str(pdf), FMT_PDF)

    def close(self):
        try:
            self.doc.Close(False)
        finally:
            self.word.Quit()


# ----------------------------------------------------------------------------- SVG → PNG
def svg_to_png(svg: Tag, out: Path) -> Path:
    vb = [float(x) for x in (svg.get("viewBox") or svg["viewbox"]).split()]  # html.parser pasa los atributos a minúscula
    w, h = int(vb[2]), int(vb[3])
    page = out.with_suffix(".html")
    page.write_text(
        "<!doctype html><meta charset='utf-8'><style>html,body{margin:0;background:#fff}"
        "svg{display:block;width:%dpx;height:%dpx} svg text{font-family:'Segoe UI',Arial,sans-serif}</style>%s"
        % (w, h, str(svg)), encoding="utf-8")
    edge = next(p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).exists())
    subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=2", f"--window-size={w},{h}",
                    f"--screenshot={out}", page.as_uri()], check=True, timeout=120,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out


# ----------------------------------------------------------------------------- armado
def build(pdf: Path | None):
    soup = BeautifulSoup(HTML.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main")
    tmp = Path(tempfile.mkdtemp(prefix="healthrag_word_"))
    d = Doc()
    try:
        for el in main.children:
            if not isinstance(el, Tag):
                continue
            cls = el.get("class", [])
            if "cover" in cls:
                d.par([("", {})])
                for _ in range(6):
                    d.doc.Range(d.end, d.end).InsertParagraphAfter()
                d.par([("HealthRAG", {"bold": True})], NORMAL, size=34, space_after=8)
                d.par(runs_of(el.find(class_="sub")), NORMAL, size=15, color=INK2, space_after=120)
                meta = el.find(class_="meta")
                for line in re.split(r"<br\s*/?>", meta.decode_contents()):
                    frag = BeautifulSoup(line, "html.parser")
                    d.par(runs_of(frag), NORMAL, size=11, space_after=2)
                d.page_break()
            elif "toc" in cls:
                d.par([("Contenido", {"bold": True})], NORMAL, size=16, space_after=10)
                note = el.find(class_="note")
                if note:
                    d.par(runs_of(note), NORMAL, size=9.5, shade=SOFT, border_left=INK2, space_after=12)
                d.toc()
                d.page_break()
            elif el.name == "h2":
                d.heading(el.get_text(" ", strip=True), 1, page_break="page-break" in cls)
            elif el.name == "h3":
                d.heading(el.get_text(" ", strip=True), 2)
            elif el.name == "p":
                if "note" in cls:
                    d.par(runs_of(el), NORMAL, size=9.5, shade=SOFT, border_left=INK2)
                else:
                    d.par(runs_of(el))
            elif el.name in ("ul", "ol"):
                d.bullets(el.find_all("li", recursive=False), ordered=el.name == "ol")
            elif el.name == "table":
                d.table(el)
            elif el.name == "figure":
                svg = el.find("svg")
                img = el.find("img")
                if svg is not None:
                    d.image(svg_to_png(svg, tmp / f"diagrama_{len(list(tmp.glob('diagrama_*.png')))}.png"))
                elif img is not None:
                    d.image(HERE / img["src"], 0.92)
                cap = el.find("figcaption")
                if cap:
                    d.par(runs_of(cap), CAPTION)
            elif el.name == "div" and "key" in cls:
                for li in el.find_all("li"):
                    r = d.par(runs_of(li), BULLET, shade=ACCENT_SOFT)
                    if r is not None:
                        d.listify(r)
            elif el.name == "div" and "ex" in cls:
                q, a, m = (el.find(class_=c) for c in ("q", "a", "m"))
                d.par(runs_of(q), NORMAL, keep_next=True, space_after=2, shade=SOFT)
                d.par(runs_of(a), NORMAL, keep_next=True, space_after=2, shade=SOFT)
                d.par(runs_of(m), NORMAL, size=8.5, color=INK2, space_after=10, shade=SOFT)
        d.save(OUT, pdf)
    finally:
        d.close()
    print("DOCX:", OUT)
    if pdf:
        print("PDF:", pdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", type=Path, help="exporta además una copia en PDF desde Word")
    build(ap.parse_args().pdf)
