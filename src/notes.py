"""Stage 4: find the notes on the canvas row and measure where they are.

Line-free image: the string lines and bar lines are cleared, except where a digit crosses
them. Marks are its connected components near the tab row; a mark of digit height centred
on a string line belongs to that string. Marks on one string that sit side by side are
joined into one note ("10", "12"). A note's x is its box centre, measured from the origin.
This stage decides where notes are, never what they say.

Writes: notes.json (ID, string with 1 = high e, x, box, digit boxes, and the dropped marks
with their reasons), line_free.png. Debug: notes.png (kept marks boxed in their string's
colour, dropped marks grey, joins bracketed).
"""
import json

import cv2
import numpy as np

from src import Refused, config, rows

STAGE_VERSION = 1

# BGR colour per string, 1 (high e) to 6
STRING_COLOURS = [(0, 0, 220), (0, 140, 255), (0, 170, 0), (200, 150, 0), (220, 0, 160), (120, 60, 0)]
DROPPED_COLOUR = (170, 170, 170)


def inputs(ctx):
    return [ctx.dir("canvas") / "canvas.png", ctx.dir("rows") / "rows.json"]


def params(ctx):
    names = ["BARLINE_CLEAR_PAD_PX", "MARK_MIN_H_FRAC", "MARK_MAX_H_FRAC", "MARK_MAX_DY_FRAC", "DIGIT_JOIN_OVERLAP_FRAC",
             "DIGIT_JOIN_GAP_FRAC"]
    return {n: getattr(config, n) for n in names}


def line_free(ink, row):
    """The ink mask with string lines and bar lines cleared.

    A line's pixels are kept only where the pixels just outside it on both sides are ink,
    because a digit crosses the line there. A bar line is cleared with BARLINE_CLEAR_PAD_PX
    columns on each side: its anti-aliased edge column is partly ink and would otherwise be
    left as a thin sliver of digit height.
    """
    out = ink.copy()
    h, w = ink.shape
    for y0, y1 in row["line_spans"]:
        above = ink[y0 - 1] if y0 > 0 else np.zeros(w, bool)
        below = ink[y1 + 1] if y1 + 1 < h else np.zeros(w, bool)
        out[y0:y1 + 1, ~(above & below)] = False
    pad = config.BARLINE_CLEAR_PAD_PX
    for x0, x1 in row["bar_spans"]:
        x0, x1 = max(x0 - pad, 0), min(x1 + pad, w - 1)
        left = ink[:, x0 - 1] if x0 > 0 else np.zeros(h, bool)
        right = ink[:, x1 + 1] if x1 + 1 < w else np.zeros(h, bool)
        out[~(left & right), x0:x1 + 1] = False
    return out


def find_marks(free, row):
    """Connected components whose box meets the tab band (top line - s to bottom line + s).

    Returns (kept, dropped). A kept mark has a string (1 = top line); a dropped one a reason.
    Boxes are [x0, y0, x1, y1], inclusive.
    """
    s = row["s"]
    band_top, band_bottom = row["line_y"][0] - s, row["line_y"][-1] + s
    n, _, stats, _ = cv2.connectedComponentsWithStats(free.astype(np.uint8), connectivity=8)
    kept, dropped = [], []
    for x, y, w, h, _ in stats[1:]:
        box = [int(x), int(y), int(x + w - 1), int(y + h - 1)]
        if box[3] < band_top or box[1] > band_bottom:
            continue
        cy = (box[1] + box[3]) / 2
        dy = [abs(cy - ly) for ly in row["line_y"]]
        string = int(np.argmin(dy)) + 1
        if h < config.MARK_MIN_H_FRAC * s:
            dropped.append({"box": box, "reason": "short"})
        elif h > config.MARK_MAX_H_FRAC * s:
            dropped.append({"box": box, "reason": "tall"})
        elif min(dy) > config.MARK_MAX_DY_FRAC * s:
            dropped.append({"box": box, "reason": "off line"})
        else:
            kept.append({"box": box, "string": string})
    return kept, dropped


def join_marks(marks):
    """Notes: marks on the same string that sit side by side are one note.

    Two neighbouring marks join when their vertical extents overlap by at least
    DIGIT_JOIN_OVERLAP_FRAC of the smaller height and the gap between them is under
    DIGIT_JOIN_GAP_FRAC times the median mark width.
    """
    if not marks:
        return []
    widths = [m["box"][2] - m["box"][0] + 1 for m in marks]
    max_gap = config.DIGIT_JOIN_GAP_FRAC * float(np.median(widths))
    notes = []
    for string in range(1, 7):
        on = sorted((m for m in marks if m["string"] == string), key=lambda m: m["box"][0])
        for m in on:
            if notes and notes[-1]["string"] == string:
                last = notes[-1]["digits"][-1]
                h = min(last[3] - last[1], m["box"][3] - m["box"][1]) + 1
                overlap = min(last[3], m["box"][3]) - max(last[1], m["box"][1]) + 1
                gap = m["box"][0] - last[2] - 1
                if overlap >= config.DIGIT_JOIN_OVERLAP_FRAC * h and gap < max_gap:
                    notes[-1]["digits"].append(m["box"])
                    continue
            notes.append({"string": string, "digits": [m["box"]]})
    for note in notes:
        d = np.array(note["digits"])
        note["box"] = [int(d[:, 0].min()), int(d[:, 1].min()), int(d[:, 2].max()), int(d[:, 3].max())]
    return notes


def find_notes(canvas, row):
    """Notes sorted by x then string, with IDs from 0, and the dropped marks and line-free mask."""
    free = line_free(rows.binarize(canvas), row)
    kept, dropped = find_marks(free, row)
    notes = join_marks(kept)
    for note in notes:
        note["x"] = (note["box"][0] + note["box"][2]) / 2 - row["origin_x"]
    notes.sort(key=lambda n: (n["x"], n["string"]))
    notes = [{"id": k, "string": n["string"], "x": n["x"], "box": n["box"], "digits": n["digits"]}
             for k, n in enumerate(notes)]
    return notes, dropped, free


def draw_overlay(canvas, row, notes, dropped, pad_frac=2.0):
    """The tab band with kept marks boxed in their string's colour, dropped marks grey and joins bracketed."""
    img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    img = (img * 0.6 + 100).astype(np.uint8)
    for d in dropped:
        x0, y0, x1, y1 = d["box"]
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), DROPPED_COLOUR, 1)
    for n in notes:
        colour = STRING_COLOURS[n["string"] - 1]
        for x0, y0, x1, y1 in n["digits"]:
            cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), colour, 1)
        if len(n["digits"]) > 1:
            x0, y0, x1, _ = n["box"]
            cv2.line(img, (x0, y0 - 3), (x1, y0 - 3), colour, 1)
            cv2.line(img, (x0, y0 - 3), (x0, y0 - 1), colour, 1)
            cv2.line(img, (x1, y0 - 3), (x1, y0 - 1), colour, 1)
    pad = int(pad_frac * row["s"])
    top = max(int(row["line_y"][0]) - pad, 0)
    bottom = min(int(row["line_y"][-1]) + pad, img.shape[0])
    return img[top:bottom]


def run(ctx):
    out = ctx.dir("notes")
    canvas = cv2.imread(str(ctx.dir("canvas") / "canvas.png"), cv2.IMREAD_GRAYSCALE)
    row = json.loads((ctx.dir("rows") / "rows.json").read_text())
    notes, dropped, free = find_notes(canvas, row)
    if not notes:
        raise Refused("no notes found on the tab row")
    cv2.imwrite(str(out / "line_free.png"), np.where(free, 0, 255).astype(np.uint8))
    if ctx.debug:
        cv2.imwrite(str(out / "debug" / "notes.png"), draw_overlay(canvas, row, notes, dropped))
    (out / "notes.json").write_text(json.dumps({"origin_x": row["origin_x"], "notes": notes, "dropped": dropped},
                                               indent=1))
    joined = sum(len(n["digits"]) > 1 for n in notes)
    reasons = {r: sum(d["reason"] == r for d in dropped) for r in ("short", "tall", "off line")}
    print(f"[notes] {len(notes)} notes ({joined} of several marks); dropped marks: "
          + ", ".join(f"{v} {k}" for k, v in reasons.items()))
