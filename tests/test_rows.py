import json

import cv2
import pytest

from src import Refused, config, rows
from src.cache import Context
from tests import synth

# Line y is a coverage-weighted centroid. Digit pixels that land on a line's rows shift it
# by a few thousandths of a pixel, so "exact" here means within 0.01 px.
Y_TOL_PX = 0.01


def longest_run(mask):
    return max((b - a + 1 for a, b in rows._runs(mask)), default=0)


def test_row_found_next_to_staff_and_chord_grids():
    img, truth = synth.tab_page()
    found = rows.analyze(img)
    assert len(found) == 1                       # the 5-line staff and chord grids are not rows
    row = found[0]
    assert row["line_y"] == pytest.approx(truth["line_y"], abs=Y_TOL_PX)
    assert row["s"] == pytest.approx(truth["s"], abs=Y_TOL_PX)


def test_lines_found_although_digits_break_them():
    img, truth = synth.tab_page(digits_every=40)
    ink = rows.binarize(img)
    for y in truth["line_y"]:
        # the plan's first design (opening with a kernel half the width long) would miss these
        assert longest_run(ink[int(y)]) < config.LINE_MIN_COVER_FRAC * img.shape[1]
    assert rows.analyze(img)[0]["line_y"] == pytest.approx(truth["line_y"], abs=Y_TOL_PX)


def test_seven_even_lines_are_not_a_row():
    img = synth.blank()
    synth.draw_tab_lines(img, 100, 1800, synth.tab_line_ys() + [synth.TAB_TOP_Y + 6 * synth.TAB_GAP])
    assert rows.find_rows(rows.find_lines(rows.binarize(img))) == []


def test_uneven_lines_are_not_a_row():
    img = synth.blank()
    ys = synth.tab_line_ys()
    ys[3] += 6                                   # one gap 24, one 12: over 15% off the mean
    synth.draw_tab_lines(img, 100, 1800, ys)
    assert rows.find_rows(rows.find_lines(rows.binarize(img))) == []


def test_barlines_found_arpeggio_not():
    img, truth = synth.tab_page()
    row = rows.analyze(img)[0]
    assert row["bar_x"] == pytest.approx(truth["bar_x"])
    # the synthetic arpeggio really does fill its centre column, like the real ones
    ink = rows.binarize(img)
    r = rows.find_rows(rows.find_lines(ink))[0]
    cover = rows.barline_cover(ink, r)
    assert cover[470] >= config.BARLINE_MIN_COVER_FRAC and cover[1047] >= config.BARLINE_MIN_COVER_FRAC


def test_double_bar_merges_into_one():
    img, truth = synth.tab_page(line_x1=1830, final_double=1810)
    row = rows.analyze(img)[0]
    assert row["bar_x"][-1] == pytest.approx((1810 + 1817 + 5) / 2)
    assert row["bar_x"] == pytest.approx(truth["bar_x"])
    assert row["extent"][1] == 1829


def test_bracket_is_not_the_left_end():
    img, truth = synth.tab_page()
    row = rows.analyze(img)[0]
    assert row["extent"] == [truth["left"], truth["right"]]
    assert all(x > truth["left"] for x in row["bar_x"])      # the bracket is not a bar line


def test_origin_on_start_line_when_present():
    img, truth = synth.tab_page(start_line=True)
    assert rows.analyze(img)[0]["origin_x"] == pytest.approx(truth["left"] + 0.5)


def test_origin_on_left_end_without_bar():
    img, truth = synth.tab_page(start_line=False, bracket_x=None)
    row = rows.analyze(img)[0]
    assert row["origin_x"] == truth["left"]
    assert rows.find_origin([truth["left"] + 30.0], (truth["left"], 1800), 18) == truth["left"]
    assert rows.find_origin([truth["left"] + 10.0], (truth["left"], 1800), 18) == truth["left"] + 10


def test_overlay_draws():
    img, _ = synth.tab_page()
    out = rows.draw_overlay(img, rows.analyze(img))
    assert out.shape == img.shape + (3,)


def run_stage(tmp_path, canvas_img):
    ctx = Context(source="v.mp4", video_id="v", root=tmp_path, debug=True)
    cv2.imwrite(str(ctx.dir("canvas") / "canvas.png"), canvas_img)
    (ctx.dir("rows") / "debug").mkdir()
    rows.run(ctx)
    return ctx.dir("rows")


def test_stage_writes_one_row_of_a_long_canvas(tmp_path):
    img, truth = synth.tab_page(w=5230, line_x1=5140, bars=(600, 1306, 1891, 3426, 4756),
                                final_double=5120)
    out = run_stage(tmp_path, img)
    row = json.loads((out / "rows.json").read_text())
    assert row["line_y"] == pytest.approx(truth["line_y"], abs=Y_TOL_PX)
    assert row["origin_x"] == pytest.approx(truth["left"] + 0.5)
    assert row["bar_x"] == pytest.approx(truth["bar_x"])      # each once, the final double bar as one
    assert (out / "debug" / "rows.png").exists()


def test_stage_refuses_canvas_without_one_row(tmp_path):
    with pytest.raises(Refused, match="0 tab rows"):
        run_stage(tmp_path, synth.blank(w=3000))
