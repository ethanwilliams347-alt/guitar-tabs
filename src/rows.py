"""Stage 3: find the tab row, its extent, bar lines and origin.

Stage 2 calls analyze() on single pages; the stage itself runs it on the canvas.
All x and y values are image pixels; the row's x-positions downstream are measured
from its origin.

Writes: rows.json (line y-positions and spans, s, extent, origin, bar-line x-positions and spans).
Debug: rows.png (lines, extent, bar lines and origin over the canvas).
"""
import json

import cv2
import numpy as np

from src import Refused, config

STAGE_VERSION = 2


def binarize(gray):
    """Ink mask (True = ink) by Otsu's threshold."""
    _, b = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b.astype(bool)


def _runs(mask):
    """(start, end) inclusive index pairs of the True runs in a 1-D mask."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def find_lines(ink):
    """Horizontal lines: runs of rows in which ink covers at least LINE_MIN_COVER_FRAC of the width.

    No morphological opening: fret numbers blank the line behind them, so tab lines are
    broken into pieces much shorter than the width. Each line's y is the coverage-weighted
    centroid of its rows.
    """
    cover = ink.mean(axis=1)
    lines = []
    for y0, y1 in _runs(cover >= config.LINE_MIN_COVER_FRAC):
        ys = np.arange(y0, y1 + 1)
        lines.append({"y": float(np.average(ys, weights=cover[y0:y1 + 1])), "y0": int(y0), "y1": int(y1)})
    return lines


def find_rows(lines):
    """Groups of exactly 6 consecutive, evenly spaced lines.

    Every gap in the group is within LINE_SPACING_TOL_FRAC of the group's mean gap, and the
    gaps to the lines just outside the group are not, so a 7-line group is not a row.
    """
    tol = config.LINE_SPACING_TOL_FRAC
    rows, i = [], 0
    while i + 6 <= len(lines):
        group = lines[i:i + 6]
        ys = [ln["y"] for ln in group]
        gaps = np.diff(ys)
        s = float(gaps.mean())
        even = np.all(np.abs(gaps - s) <= tol * s)
        before = i > 0 and abs(ys[0] - lines[i - 1]["y"] - s) <= tol * s
        after = i + 6 < len(lines) and abs(lines[i + 6]["y"] - ys[-1] - s) <= tol * s
        if even and not before and not after:
            rows.append({"lines": group, "line_y": ys, "s": (ys[-1] - ys[0]) / 5})
            i += 6
        else:
            i += 1
    return rows


def line_presence(ink, row):
    """Columns where all six lines have ink."""
    return np.all([ink[ln["y0"]:ln["y1"] + 1].any(axis=0) for ln in row["lines"]], axis=0)


def row_extent(ink, row):
    """First and last columns of the runs, at least EXTENT_MIN_RUN_FRAC * s long, where all six lines are present.

    The minimum run keeps a narrow vertical stroke crossing all six line rows (such as a
    bracket) from being taken for the row's end. Returns None if there is no such run.
    """
    min_len = config.EXTENT_MIN_RUN_FRAC * row["s"]
    runs = [(a, b) for a, b in _runs(line_presence(ink, row)) if b - a + 1 >= min_len]
    if not runs:
        return None
    return int(runs[0][0]), int(runs[-1][1])


def barline_cover(ink, row):
    """Fraction of each column that is ink, over the span from the top line to the bottom line."""
    top, bottom = row["lines"][0]["y0"], row["lines"][-1]["y1"]
    return ink[top:bottom + 1].mean(axis=0)


def find_barlines(ink, row, extent):
    """Column spans [first, last] of bar lines inside the extent.

    A bar line is a run of columns covering at least BARLINE_MIN_COVER_FRAC of the span,
    with sharp edges: both columns beside the run cover less than BARLINE_EDGE_MAX_COVER_FRAC.
    An arpeggio's thick wavy stroke can fill its centre column, but its coverage ramps up and
    down, so its edges fail. Runs within BARLINE_MERGE_FRAC * s of each other merge into one
    bar line spanning the whole group (a double bar line becomes one). Its x is the span's centre.
    """
    cover = barline_cover(ink, row)
    left, right = extent
    w = len(cover)
    strokes = []
    for a, b in _runs(cover >= config.BARLINE_MIN_COVER_FRAC):
        if b < left or a > right:
            continue
        edge_l = cover[a - 1] if a > 0 else 0.0
        edge_r = cover[b + 1] if b + 1 < w else 0.0
        if max(edge_l, edge_r) < config.BARLINE_EDGE_MAX_COVER_FRAC:
            strokes.append([int(a), int(b)])
    merged = []
    for a, b in strokes:
        if merged and a - merged[-1][1] <= config.BARLINE_MERGE_FRAC * row["s"]:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def find_origin(bar_x, extent, s):
    """The first bar line if it lies within s of the row's left end, or else the left end."""
    left = extent[0]
    if bar_x and bar_x[0] - left <= s:
        return bar_x[0]
    return float(left)


def analyze(gray):
    """All rows in a grayscale image, each with line y-positions, s, extent, bar lines and origin."""
    ink = binarize(gray)
    rows = []
    for row in find_rows(find_lines(ink)):
        extent = row_extent(ink, row)
        if extent is None:
            continue
        spans = find_barlines(ink, row, extent)
        bars = [(a + b) / 2 for a, b in spans]
        rows.append({
            "line_y": row["line_y"],
            "line_spans": [[ln["y0"], ln["y1"]] for ln in row["lines"]],
            "s": row["s"],
            "extent": list(extent),
            "bar_x": bars,
            "bar_spans": spans,
            "origin_x": find_origin(bars, extent, row["s"]),
        })
    return rows


def draw_overlay(gray, rows):
    """Lines green, extent ends blue, bar lines red, origin magenta."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = (img * 0.6 + 100).astype(np.uint8)          # fade the page so marks stand out
    for row in rows:
        left, right = row["extent"]
        top, bottom = row["line_spans"][0][0], row["line_spans"][-1][1]
        for y in row["line_y"]:
            cv2.line(img, (left, int(round(y))), (right, int(round(y))), (0, 170, 0), 1)
        for x in (left, right):
            cv2.line(img, (x, top - 14), (x, bottom + 14), (220, 120, 0), 2)
        for x in row["bar_x"]:
            xi = int(round(x))
            cv2.line(img, (xi, top - 8), (xi, bottom + 8), (0, 0, 230), 2)
        ox = int(round(row["origin_x"]))
        cv2.drawMarker(img, (ox, top - 22), (200, 0, 200), cv2.MARKER_TRIANGLE_DOWN, 14, 2)
        cv2.putText(img, f"s={row['s']:.2f} bars={len(row['bar_x'])} origin={row['origin_x']:.1f}",
                    (max(left, 5) + 12, top - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 200), 1, cv2.LINE_AA)
    return img


def inputs(ctx):
    return [ctx.dir("canvas") / "canvas.png"]


def params(ctx):
    names = ["LINE_MIN_COVER_FRAC", "LINE_SPACING_TOL_FRAC", "BARLINE_MIN_COVER_FRAC", "BARLINE_MERGE_FRAC",
             "BARLINE_EDGE_MAX_COVER_FRAC", "EXTENT_MIN_RUN_FRAC"]
    return {n: getattr(config, n) for n in names}


def run(ctx):
    out = ctx.dir("rows")
    canvas = cv2.imread(str(ctx.dir("canvas") / "canvas.png"), cv2.IMREAD_GRAYSCALE)
    found = analyze(canvas)
    if ctx.debug:
        cv2.imwrite(str(out / "debug" / "rows.png"), draw_overlay(canvas, found))
    if len(found) != 1:
        raise Refused(f"the canvas has {len(found)} tab rows; the paged-strip layout needs exactly one")
    row = found[0]
    (out / "rows.json").write_text(json.dumps(row, indent=1))
    print(f"[rows] 1 row: s = {row['s']:.2f} px, extent {row['extent'][0]}-{row['extent'][1]}, "
          f"origin {row['origin_x']:.1f}, {len(row['bar_x'])} bar lines")
