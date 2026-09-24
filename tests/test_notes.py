import json

import cv2
import numpy as np
import pytest

from src import Refused, config, notes, rows
from src.cache import Context
from tests import synth

YS = synth.tab_line_ys()


def row_of(img):
    return rows.analyze(img)[0]


def plain_row(w=1200, bars=(), start_line=False):
    img = synth.blank(w=w)
    synth.draw_tab_lines(img, 60, w - 60, YS)
    for x in bars:
        synth.draw_bar(img, x, YS)
    if start_line:
        synth.draw_bar(img, 60, YS)
    return img


def blob(img, x0, y0, x1, y1):
    """A solid mark with a white margin that blanks the line behind it, like the video's fret numbers."""
    img[y0 - 2:y1 + 3, x0 - 3:x1 + 4] = 255
    img[y0:y1 + 1, x0:x1 + 1] = 0


def test_line_erasure_keeps_digits_that_cross_a_line():
    img = plain_row()
    for k, text in enumerate(["8", "4", "7", "5", "0", "3"]):
        synth.draw_digit(img, 200 + 120 * k, YS[k], text)
    row = row_of(img)
    free = notes.line_free(rows.binarize(img), row)
    found, dropped, _ = notes.find_notes(img, row)
    assert [n["string"] for n in found] == [1, 2, 3, 4, 5, 6]
    assert all(len(n["digits"]) == 1 for n in found)             # no digit split by its line
    assert not free[YS[0]:YS[0] + 2, 70:170].any()                # the line itself is gone
    assert all(d["reason"] == "short" for d in dropped)


def test_two_digit_frets_join_and_chords_do_not():
    img = plain_row()
    synth.draw_digit(img, 300, YS[2], "10")
    synth.draw_digit(img, 500, YS[4], "12")
    for k in range(6):                                            # a six-note chord, one per string
        synth.draw_digit(img, 800, YS[k], str(k + 2))
    found, _, _ = notes.find_notes(img, row_of(img))
    two = [n for n in found if len(n["digits"]) == 2]
    assert [(n["string"]) for n in two] == [3, 5]
    chord = [n for n in found if abs(n["x"] - (800 - 60)) < 10]
    assert sorted(n["string"] for n in chord) == [1, 2, 3, 4, 5, 6] and len(found) == 8


def test_x_positions_exact():
    img = plain_row(start_line=True)
    boxes = [(200, YS[0] - 6, 207, YS[0] + 7, 1), (401, YS[3] - 6, 410, YS[3] + 7, 4), (700, YS[5] - 6, 705, YS[5] + 7, 6)]
    for x0, y0, x1, y1, _ in boxes:
        blob(img, x0, y0, x1, y1)
    row = row_of(img)
    assert row["origin_x"] == 60.5
    found, _, _ = notes.find_notes(img, row)
    assert [(n["string"], n["x"]) for n in found] == [(s, (x0 + x1) / 2 - 60.5) for x0, _, x1, _, s in boxes]
    assert [n["box"] for n in found] == [[x0, y0, x1, y1] for x0, y0, x1, y1, _ in boxes]


def test_arpeggio_and_between_string_marks_dropped():
    img = plain_row()
    synth.draw_arpeggio(img, 400, YS)
    blob(img, 600, YS[1] + 5, 606, YS[1] + 16)                    # centred between strings 2 and 3
    found, dropped, _ = notes.find_notes(img, row_of(img))
    assert found == []
    assert sorted(d["reason"] for d in dropped if d["reason"] != "short") == ["off line", "tall"]


def test_anti_aliased_bar_edge_is_cleared():
    img = plain_row(bars=(500,))
    # a partly inked edge column: in the video a 17 px run of it, centred near string 5, survived as a "note"
    img[YS[4] - 8:YS[4] + 9, 502] = 100
    row = row_of(img)
    assert row["bar_spans"] == [[500, 501]]
    found, _, _ = notes.find_notes(img, row)
    assert found == []


def test_stage_writes_notes_and_overlay(tmp_path):
    img = plain_row()
    synth.draw_digit(img, 300, YS[2], "10")
    ctx = Context(source="v.mp4", video_id="v", root=tmp_path, debug=True)
    cv2.imwrite(str(ctx.dir("canvas") / "canvas.png"), img)
    (ctx.dir("rows") / "rows.json").write_text(json.dumps(row_of(img)))
    (ctx.dir("notes") / "debug").mkdir()
    notes.run(ctx)
    out = json.loads((ctx.dir("notes") / "notes.json").read_text())
    assert [(n["id"], n["string"], len(n["digits"])) for n in out["notes"]] == [(0, 3, 2)]
    assert (ctx.dir("notes") / "line_free.png").exists() and (ctx.dir("notes") / "debug" / "notes.png").exists()


def test_stage_refuses_row_without_notes(tmp_path):
    img = plain_row()
    ctx = Context(source="v.mp4", video_id="v", root=tmp_path)
    cv2.imwrite(str(ctx.dir("canvas") / "canvas.png"), img)
    (ctx.dir("rows") / "rows.json").write_text(json.dumps(row_of(img)))
    with pytest.raises(Refused, match="no notes"):
        notes.run(ctx)
