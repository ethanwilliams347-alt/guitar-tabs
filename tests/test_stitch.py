import numpy as np
import pytest

from src import Refused, canvas, config, rows
from tests import synth

W = 960
FADE_PX = 45                      # like the video's 80-100 px on 1920, scaled to this width


def long_strip(w=2400, bars=(500, 830, 1190, 1610, 2050), digits_every=47):
    img, _ = synth.tab_page(w=w, line_x0=60, line_x1=w - 60, bars=bars, arpeggios=(900,),
                            digits_every=digits_every, bracket_x=None, final_double=w - 90)
    return img


def stitch(pages):
    return canvas.stitch(pages, [rows.analyze(p)[0] for p in pages])


def test_exact_offsets_and_clean_composite():
    strip = long_strip()
    xs = [0, 700, 1440]                           # overlaps of 260 and 220 px
    pages = synth.strip_pages(strip, xs, W, FADE_PX)
    got_xs, pairs, img = stitch(pages)
    assert got_xs == xs
    assert [p["offset"] for p in pairs] == [700, 740]
    assert img.shape == (strip.shape[0], xs[-1] + W)
    # every column has at least one unfaded copy, so the darker pixel is the strip itself
    assert np.abs(img.astype(int) - strip[:, :img.shape[1]]).max() <= 1
    for p in pairs:
        assert p["ncc"] >= config.STRIP_NCC_MIN
        assert p["ncc"] - p["runner_up_ncc"] >= config.STRIP_NCC_MARGIN
        assert p["bar_residual"] is not None and p["bar_residual"] <= config.STRIP_MAX_RESIDUAL_PX


def test_no_bar_line_in_overlap_is_accepted_on_ncc():
    strip = long_strip(bars=(300, 1300, 2050))
    pages = synth.strip_pages(strip, [0, 700, 1440], W, FADE_PX)
    xs, pairs, _ = stitch(pages)
    assert xs == [0, 700, 1440]
    assert pairs[0]["bar_residual"] is None and pairs[0]["bar_x"] == pairs[0]["bar_x_next"] == []


def test_overlap_too_small_is_refused():
    strip = long_strip()
    pages = synth.strip_pages(strip, [0, 960 - 40], W)    # 40 px overlap, under 8% of 960
    with pytest.raises(Refused, match="pages 1 and 2"):
        stitch(pages)


def test_repeating_pattern_is_refused_on_runner_up():
    img = synth.blank(w=2000)
    synth.draw_tab_lines(img, 0, 2000)
    for x in range(20, 2000, 60):                  # the same chord every 60 px, no bar lines
        for y in synth.tab_line_ys()[1::2]:
            synth.draw_digit(img, x, y, "7")
    pages = synth.strip_pages(img, [0, 700], W)
    with pytest.raises(Refused, match="ambiguous"):
        stitch(pages)


def test_bar_lines_that_disagree_are_refused():
    strip = long_strip()
    pages = synth.strip_pages(strip, [0, 700], W, FADE_PX)
    moved = pages[1].copy()
    ys = synth.tab_line_ys()
    bar = 830 - 700                               # page 2's copy of the bar line in the overlap
    moved[ys[0]:ys[-1] + synth.LINE_THICK, bar:bar + 2] = 255
    synth.draw_bar(moved, bar + 4, ys)            # redrawn 4 px to the right
    synth.draw_tab_lines(moved, bar, bar + 2, ys)
    with pytest.raises(Refused, match="don't line up"):
        stitch([pages[0], moved])


def test_bar_residual_rules():
    assert canvas.bar_residual([], []) is None
    assert canvas.bar_residual([100.0], []) == float("inf")
    assert canvas.bar_residual([100.0, 300.0], [101.0, 299.5]) == 1.0
