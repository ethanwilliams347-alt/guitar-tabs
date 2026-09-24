"""Stage 6: the tab sheet (PDF), tab.json and the report.

tab.json holds everything the PDF and the metrics are computed from: the video ID, the
tab-area width, the canvas row (origin, extent, s, line y, bar lines measured from the
origin) and every note that isn't ignored, with its string, x, fret, status and each
reader's result. A flagged note no reader could read has fret null and is drawn as "?".

Scale: one factor for the whole song. The tab-area width in the video maps to the width
inside the page margins, so the horizontal spacing of the numbers matches the video.
Vertical spacing is a free choice (RENDER_STRING_GAP_PT); only horizontal spacing is measured.

Rows: the canvas is cut into PDF rows at bar lines, each cut at the last bar line that
still fits one tab-area width. A measure wider than that is split at the gap between two
notes, and the split is flagged in red and listed in the report.

Writes: out/<video id>.pdf (or -o), and tab.json and report.html in render/. Debug:
cuts.png, the tab band with each row cut marked (green at a bar line, red at a split).
"""
import base64
import html
import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas as pdfcanvas

from src import Refused, config
from src.cache import ROOT

STAGE_VERSION = 1

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "tab.schema.json"
OUT_DIR = ROOT / "out"
PAPERS = {"letter": letter, "a4": A4}
FLAG_RGB = (0.85, 0.0, 0.0)
FLAG_BGR = (0, 0, 220)
OK_BGR = (0, 150, 0)


def output_path(ctx):
    return Path(ctx.output) if ctx.output else OUT_DIR / f"{ctx.video_id}.pdf"


def inputs(ctx):
    return [ctx.dir("ingest") / "meta.json", ctx.dir("canvas") / "canvas.png", ctx.dir("canvas") / "pages.json",
            ctx.dir("rows") / "rows.json", ctx.dir("notes") / "notes.json", ctx.dir("read") / "read.json",
            ctx.dir("read") / "cost.json"]


def params(ctx):
    names = ["RENDER_MARGIN_PT", "RENDER_STRING_GAP_PT", "RENDER_ROW_GAP_PT", "RENDER_FRET_FONT_PT",
             "RENDER_TITLE_FONT_PT", "RENDER_LINE_WIDTH_PT", "REPORT_PAD_FRAC", "REVIEW_FRET_FONT_SCALE"]
    p = {n: getattr(config, n) for n in names}
    p.update(paper=ctx.paper, output=str(output_path(ctx)))
    return p


# --- tab.json ------------------------------------------------------------------------------

def build_tab(video_id, tab_width_px, row, read_notes):
    """The tab.json dict, checked against src/schema/tab.schema.json. Ignored marks are left out."""
    origin = row["origin_x"]
    notes = []
    for n in read_notes:
        if n["status"] == "ignored":
            continue
        reread = n.get("reread") or {}
        notes.append({"id": n["id"], "string": n["string"], "x": n["x"], "fret": n["fret"],
                      "status": "flagged" if n["status"] == "flagged" else "verified",
                      "readers": {"model": n["model"]["fret"], "template": n["template"]["fret"],
                                  "reread": reread.get("fret")}})
    tab = {"video_id": video_id, "tab_width_px": tab_width_px,
           "row": {"origin_x": origin, "extent": row["extent"], "s": row["s"], "line_y": row["line_y"],
                   "bar_x": [b - origin for b in row["bar_x"]]},
           "notes": notes}
    jsonschema.validate(tab, json.loads(SCHEMA_PATH.read_text()))
    return tab


# --- rows ----------------------------------------------------------------------------------

def cut_rows(bar_x, note_x, end_x, width):
    """[(x0, x1, split)] covering the row from its start to end_x, all measured from the origin.

    Each row ends at the last bar line within width of its start. If there is none, the
    measure is split at the midpoint of the last gap between two notes that fits, and split
    is True. The first row starts at the origin, or at the first note if one lies before it.
    """
    x0 = min([0.0] + list(note_x))
    bars = sorted(bar_x)
    xs = sorted(set(note_x))
    rows = []
    while end_x - x0 > width:
        fit = [b for b in bars if x0 < b <= x0 + width]
        if fit:
            rows.append((x0, fit[-1], False))
        else:
            gaps = [(a + b) / 2 for a, b in zip(xs, xs[1:]) if a > x0 and (a + b) / 2 <= x0 + width]
            rows.append((x0, gaps[-1] if gaps else x0 + width, True))
        x0 = rows[-1][1]
    rows.append((x0, end_x, False))
    return rows


def notes_in(notes, x0, x1, last):
    return [n for n in notes if x0 <= n["x"] < x1 or (last and n["x"] == x1)]


# --- PDF -----------------------------------------------------------------------------------

def scale(tab, paper):
    """Points per canvas pixel: the tab-area width maps to the page width inside the margins."""
    return (PAPERS[paper][0] - 2 * config.RENDER_MARGIN_PT) / tab["tab_width_px"]


def pdf_x(x, x0, k):
    """A note's PDF x in its row, for canvas x measured from the origin and the row starting at x0."""
    return config.RENDER_MARGIN_PT + (x - x0) * k


def fret_text(fret):
    return "?" if fret is None else str(fret)


def header_lines(tab):
    verified = sum(n["status"] == "verified" for n in tab["notes"])
    return tab["video_id"], (f"{verified} of {len(tab['notes'])} numbers verified. "
                             "Red numbers are flagged: check them against the video.")


def layout(tab, rows, paper):
    """Where everything goes, in PDF points: one dict per row with its page, the six line ys (string 1
    first), the left and right ends, the bar-line xs, whether it ends at a split, and each number as
    (note id, text, x, y, flagged). A row that doesn't fit below the previous one starts a new page."""
    page_h = PAPERS[paper][1]
    k = scale(tab, paper)
    m, gap, font = config.RENDER_MARGIN_PT, config.RENDER_STRING_GAP_PT, config.RENDER_FRET_FONT_PT
    first_y = page_h - m - 1.6 * font - 3 * gap          # below the title and the verified line
    page, y, placed = 0, first_y, []
    for r, (x0, x1, split) in enumerate(rows):
        if y - 5 * gap < m:
            page, y = page + 1, first_y
        ys = [y - s * gap for s in range(6)]
        numbers = [(n["id"], fret_text(n["fret"]), pdf_x(n["x"], x0, k), ys[n["string"] - 1], n["status"] == "flagged")
                   for n in notes_in(tab["notes"], x0, x1, r == len(rows) - 1)]
        placed.append({"page": page, "ys": ys, "left": pdf_x(x0, x0, k), "right": pdf_x(x1, x0, k),
                       "bars": [pdf_x(b, x0, k) for b in tab["row"]["bar_x"] if x0 <= b <= x1],
                       "split": split, "numbers": numbers})
        y -= 5 * gap + config.RENDER_ROW_GAP_PT
    return placed


def draw_pdf(tab, rows, path, paper):
    """The tab sheet: six lines, bar lines and each number centred on its string over a white box
    that blanks the line, flagged ones in red. Returns the page count."""
    page_w, page_h = PAPERS[paper]
    m, font = config.RENDER_MARGIN_PT, config.RENDER_FRET_FONT_PT
    title, subtitle = header_lines(tab)
    c = pdfcanvas.Canvas(str(path), pagesize=(page_w, page_h))
    c.setTitle(tab["video_id"])
    placed = layout(tab, rows, paper)
    for page in range(placed[-1]["page"] + 1):
        if page:
            c.showPage()
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", config.RENDER_TITLE_FONT_PT)
        c.drawString(m, page_h - m, title)
        c.setFont("Helvetica", font)
        c.drawString(m, page_h - m - 1.6 * font, subtitle)
        for row in (r for r in placed if r["page"] == page):
            ys = row["ys"]
            c.setLineWidth(config.RENDER_LINE_WIDTH_PT)
            c.setStrokeColorRGB(0, 0, 0)
            for yy in ys:
                c.line(row["left"], yy, row["right"], yy)
            for bx in row["bars"]:
                c.line(bx, ys[0], bx, ys[-1])
            if row["split"]:
                c.setStrokeColorRGB(*FLAG_RGB)
                c.setDash(1.5, 1.5)
                c.line(row["right"], ys[0], row["right"], ys[-1])
                c.setDash()
            c.setFont("Helvetica", font)
            for _, text, x, yy, flagged in row["numbers"]:
                w = c.stringWidth(text, "Helvetica", font)
                c.setFillColorRGB(1, 1, 1)
                c.rect(x - w / 2 - 0.5, yy - 0.4 * font, w + 1, 0.8 * font, fill=1, stroke=0)
                c.setFillColorRGB(*(FLAG_RGB if flagged else (0, 0, 0)))
                c.drawCentredString(x, yy - 0.35 * font, text)
    c.save()
    return placed[-1]["page"] + 1


def row_end(row):
    """The row's right end from the origin: the last bar line if it lies within s of the extent's end
    (a final bar), else the extent's end. This mirrors the origin, which snaps to a first bar line."""
    end = row["extent"][1] - row["origin_x"]
    last = max(row["bar_x"], default=None)
    return last - row["origin_x"] if last is not None and end - (last - row["origin_x"]) <= row["s"] else end


# --- images for the report and the debug overlay -------------------------------------------

def band(canvas, tab):
    """(top, bottom) canvas rows of the tab band: the six lines plus REPORT_PAD_FRAC * s each side."""
    s, ly = tab["row"]["s"], tab["row"]["line_y"]
    pad = int(config.REPORT_PAD_FRAC * s)
    return max(int(ly[0]) - pad, 0), min(int(ly[-1]) + pad, canvas.shape[0])


def row_images(canvas, tab, boxes, x0, x1, notes):
    """The video's strip for one PDF row with its notes boxed (red flagged, green verified), stacked
    above the row redrawn from tab.json at the same pixel scale, so positions can be compared."""
    origin = tab["row"]["origin_x"]
    top, bottom = band(canvas, tab)
    c0, c1 = int(np.floor(origin + x0)), int(np.ceil(origin + x1)) + 1
    video = cv2.cvtColor(canvas[top:bottom, c0:c1], cv2.COLOR_GRAY2BGR)
    drawn = np.full_like(video, 255)
    for ly in tab["row"]["line_y"]:
        cv2.line(drawn, (0, int(round(ly - top))), (drawn.shape[1] - 1, int(round(ly - top))), (0, 0, 0), 1)
    for b in tab["row"]["bar_x"]:
        if x0 <= b <= x1:
            bx = int(round(origin + b - c0))
            cv2.line(drawn, (bx, int(tab["row"]["line_y"][0] - top)), (bx, int(tab["row"]["line_y"][-1] - top)),
                     (0, 0, 0), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for n in notes:
        colour = FLAG_BGR if n["status"] == "flagged" else OK_BGR
        bx0, by0, bx1, by1 = boxes[n["id"]]
        cv2.rectangle(video, (bx0 - c0 - 1, by0 - top - 1), (bx1 - c0 + 1, by1 - top + 1), colour, 1)
        text = fret_text(n["fret"])
        (tw, th), _ = cv2.getTextSize(text, font, config.REVIEW_FRET_FONT_SCALE, 2)
        x = int(round(origin + n["x"] - c0))
        y = int(round(tab["row"]["line_y"][n["string"] - 1] - top))
        cv2.rectangle(drawn, (x - tw // 2 - 1, y - th // 2 - 1), (x + tw // 2 + 1, y + th // 2 + 1), (255, 255, 255), -1)
        cv2.putText(drawn, text, (x - tw // 2, y + th // 2), font, config.REVIEW_FRET_FONT_SCALE,
                    FLAG_BGR if n["status"] == "flagged" else (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([video, np.full((6, video.shape[1], 3), 255, np.uint8), drawn])


def draw_cuts(canvas, tab, rows):
    """Debug: the tab band with each row cut marked, green at a bar line and red at a split."""
    top, bottom = band(canvas, tab)
    img = cv2.cvtColor(canvas[top:bottom], cv2.COLOR_GRAY2BGR)
    origin = tab["row"]["origin_x"]
    for x0, x1, split in rows:
        for x, colour in ((x0, OK_BGR), (x1, FLAG_BGR if split else OK_BGR)):
            cx = int(round(origin + x))
            cv2.line(img, (cx, 0), (cx, img.shape[0] - 1), colour, 2)
    return img


def png_b64(img):
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode("ascii")


# --- report --------------------------------------------------------------------------------

def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def fmt(v, digits=3):
    if v is None:
        return "none"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def write_report(path, tab, meta, pages, row, notes_data, read_data, cost, rows, pdf, canvas, boxes, paper):
    notes = tab["notes"]
    flagged = [n for n in notes if n["status"] == "flagged"]
    ignored = sum(n["status"] == "ignored" for n in read_data["notes"])
    by_id = {n["id"]: n for n in read_data["notes"]}
    dropped = notes_data["dropped"]
    k = scale(tab, paper)
    row_notes = [notes_in(notes, x0, x1, r == len(rows) - 1) for r, (x0, x1, _) in enumerate(rows)]
    parts = [f"<h1>{html.escape(tab['video_id'])}</h1>",
             f"<p class='headline'><b>{len(notes) - len(flagged)} of {len(notes)} numbers verified.</b> "
             f"{len(flagged)} flagged, {ignored} marks ignored. Model cost ${cost['total_usd']:.4f} over "
             f"{len(cost['calls'])} calls (at paid-tier prices).</p>",
             f"<p>PDF: {html.escape(str(pdf))}, {len(rows)} rows. Scale {k:.4f} pt per video px, one factor for "
             f"the whole song ({tab['tab_width_px']} px of video = the width inside the margins).</p>",
             "<h2>Automatic decisions</h2>",
             "<p>The run completed, so nothing was refused. A refused run stops with its reason and writes no report.</p>",
             "<h3>Ingest</h3>",
             table(["video", "fps", "frames", "frame size", "crop (x, y, w, h)", "samples"],
                   [[meta["source"], fmt(meta["fps"], 2), meta["frame_count"], f"{meta['width']} × {meta['height']}",
                     ", ".join(map(str, meta["crop"])), len(meta.get("sample_times", []))]]),
             "<h3>Pages</h3>",
             table(["page", "time ranges (s)", "frames in median", "line y", "s (px)", "bar lines"],
                   [[i + 1, "; ".join(f"{a:.2f}–{b:.2f}" for a, b in p["ranges"]), len(p["frames"]),
                     f"{p['row']['line_y'][0]:.1f}–{p['row']['line_y'][-1]:.1f}", fmt(p["row"]["s"], 2),
                     len(p["row"]["bar_x"])] for i, p in enumerate(pages["pages"])]),
             "<h3>Stitching</h3>",
             table(["pages", "offset (px)", "overlap (px)", "NCC", "runner-up offset", "runner-up NCC",
                    "bar-line residual (px)"],
                   [["–".join(map(str, p["pages"])), p["offset"], p["overlap"], fmt(p["ncc"]), p["runner_up_offset"],
                     fmt(p["runner_up_ncc"]), fmt(p["bar_residual"], 1)] for p in pages["pairs"]]),
             f"<p>Canvas {pages['canvas_size'][0]} × {pages['canvas_size'][1]} px.</p>",
             "<h3>Row</h3>",
             table(["line y", "s (px)", "extent", "origin", "bar lines"],
                   [[f"{row['line_y'][0]:.1f}–{row['line_y'][-1]:.1f}", fmt(row["s"], 2),
                     f"{row['extent'][0]}–{row['extent'][1]}", fmt(row["origin_x"], 1), len(row["bar_x"])]]),
             "<h3>Notes</h3>",
             table(["notes", "two or more digits", "dropped: short", "dropped: tall", "dropped: off line"],
                   [[len(notes_data["notes"]), sum(len(n["digits"]) > 1 for n in notes_data["notes"]),
                     *[sum(d["reason"] == r for d in dropped) for r in ("short", "tall", "off line")]]]),
             "<h3>Reading</h3>",
             table(["model (read)", "model (re-read)", "grids", "invalid grids", "templates (crops per digit)",
                    "verified", "verified after re-read", "flagged", "ignored"],
                   [[read_data["models"]["read"], read_data["models"]["reread"], len(read_data["grids"]),
                     sum(not g["valid"] for g in read_data["grids"]),
                     " ".join(f"{d}:{c}" for d, c in read_data["templates"].items()),
                     *[sum(n["status"] == s for n in read_data["notes"])
                       for s in ("verified", "verified_reread", "flagged", "ignored")]]]),
             "<h3>PDF rows</h3>",
             table(["row", "from x (px)", "to x (px)", "ends at", "notes", "flagged"],
                   [[r + 1, fmt(x0, 1), fmt(x1, 1), "split between notes (flagged)" if split else "bar line",
                     len(ns), sum(n["status"] == "flagged" for n in ns)]
                    for r, ((x0, x1, split), ns) in enumerate(zip(rows, row_notes))])]
    parts.append("<h2>Rows with a flag</h2><p>Each image is the video's strip for one PDF row, notes boxed "
                 "(red flagged, green verified), above the row redrawn from tab.json at the same scale.</p>")
    any_flag = False
    for r, ((x0, x1, split), ns) in enumerate(zip(rows, row_notes)):
        fl = [n for n in ns if n["status"] == "flagged"]
        if not fl and not split:
            continue
        any_flag = True
        parts.append(f"<h3>Row {r + 1}{' (measure split here)' if split else ''}</h3>")
        parts.append(f"<img src='data:image/png;base64,{png_b64(row_images(canvas, tab, boxes, x0, x1, ns))}'>")
        parts.append(table(["note", "string", "x (px)", "shown", "model", "template", "re-read", "reason"],
                           [[n["id"], n["string"], fmt(n["x"], 1), fret_text(n["fret"]),
                             *[fmt(n["readers"][reader]) for reader in ("model", "template", "reread")],
                             by_id[n["id"]]["reason"]] for n in fl]))
    if not any_flag:
        parts.append("<p>No row has a flag.</p>")
    style = ("body{font-family:system-ui,sans-serif;margin:24px;max-width:1400px;color:#1a1a1a;background:#fff}"
             "table{border-collapse:collapse;margin:6px 0 14px}td,th{border:1px solid #ccc;padding:3px 8px;"
             "text-align:left;font-size:13px}th{background:#f2f2f2}img{max-width:100%;border:1px solid #ddd}"
             ".headline{font-size:17px}")
    path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(tab['video_id'])} report"
                    f"</title><style>{style}</style></head><body>{''.join(parts)}</body></html>", encoding="utf-8")


# --- stage ---------------------------------------------------------------------------------

def run(ctx):
    out = ctx.dir("render")
    meta = json.loads((ctx.dir("ingest") / "meta.json").read_text())
    pages = json.loads((ctx.dir("canvas") / "pages.json").read_text())
    row = json.loads((ctx.dir("rows") / "rows.json").read_text())
    notes_data = json.loads((ctx.dir("notes") / "notes.json").read_text())
    read_data = json.loads((ctx.dir("read") / "read.json").read_text())
    cost = json.loads((ctx.dir("read") / "cost.json").read_text())
    canvas = cv2.imread(str(ctx.dir("canvas") / "canvas.png"), cv2.IMREAD_GRAYSCALE)

    tab = build_tab(ctx.video_id, meta["crop"][2], row, read_data["notes"])
    if not tab["notes"]:
        raise Refused("no numbers to draw: every mark was ignored")
    rows = cut_rows(tab["row"]["bar_x"], [n["x"] for n in tab["notes"]], row_end(row), tab["tab_width_px"])
    (out / "tab.json").write_text(json.dumps(tab, indent=1))

    pdf = output_path(ctx)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    n_pages = draw_pdf(tab, rows, pdf, ctx.paper)
    boxes = {n["id"]: n["box"] for n in notes_data["notes"]}
    write_report(out / "report.html", tab, meta, pages, row, notes_data, read_data, cost, rows, pdf, canvas, boxes,
                 ctx.paper)
    if ctx.debug:
        cv2.imwrite(str(out / "debug" / "cuts.png"), draw_cuts(canvas, tab, rows))
    flagged = sum(n["status"] == "flagged" for n in tab["notes"])
    print(f"[render] {pdf}: {len(rows)} rows on {n_pages} page(s), {len(tab['notes']) - flagged} of "
          f"{len(tab['notes'])} numbers verified, {flagged} flagged in red, "
          f"{sum(split for _, _, split in rows)} split measures")
    print(f"[render] report: {out / 'report.html'}")
