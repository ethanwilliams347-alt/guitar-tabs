import json

import cv2
import numpy as np
import pytest

from src import Refused, canvas, config, ingest
from src.cache import Context
from tests import synth

FPS = 30
W, H = 960, 484


PAGE_STEP_PX = 700                                         # pages overlap by 260 px


def _strip():
    img, _ = synth.tab_page(w=W + 2 * PAGE_STEP_PX, h=H, line_x0=60, line_x1=W + 2 * PAGE_STEP_PX - 60,
                            bars=(300, 640, 830, 1190, 1610, 2050), arpeggios=(), digits_every=47,
                            bracket_x=None, start_line=False)
    return img


STRIP = _strip()


def page(k):
    """Three windows of one long strip, each with one tab row."""
    return synth.strip_pages(STRIP, [0, PAGE_STEP_PX, 2 * PAGE_STEP_PX], W)[k]


def playhead(gray, frac):
    """BGR frame with a thin red line at frac of the width, over the page."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    x = int(frac * (W - 1))
    bgr[150:470, x:x + 3] = (0, 0, 255)
    return bgr


def write_video(path, frames):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    assert vw.isOpened()
    for f in frames:
        vw.write(f)
    vw.release()


def still(gray, seconds, sweep=True):
    n = int(round(seconds * FPS))
    return [playhead(gray, i / n) if sweep else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) for i in range(n)]


def fade(a, b, seconds):
    n = int(round(seconds * FPS))
    return [cv2.cvtColor(cv2.addWeighted(a, 1 - (i + 1) / (n + 1), b, (i + 1) / (n + 1), 0), cv2.COLOR_GRAY2BGR)
            for i in range(n)]


def run_stages(tmp_path, frames):
    video = tmp_path / "v.avi"
    write_video(video, frames)
    ctx = Context(source=str(video), video_id="v", crop=(0, 0, W, H), root=tmp_path / "cache", debug=True)
    for stage in ("ingest", "canvas"):
        (ctx.dir(stage) / "debug").mkdir(exist_ok=True)
    ingest.run(ctx)
    canvas.run(ctx)
    out = ctx.dir("canvas")
    return json.loads((out / "pages.json").read_text())["pages"], out


def test_three_pages_with_crossfades_and_playhead(tmp_path):
    p = [page(k) for k in range(3)]
    frames = still(p[0], 3) + fade(p[0], p[1], 0.15) + still(p[1], 3) + fade(p[1], p[2], 0.15) + still(p[2], 3)
    pages, out = run_stages(tmp_path, frames)
    assert len(pages) == 3
    starts = [0, 3.15, 6.3]
    for pg, t0 in zip(pages, starts):
        assert abs(pg["t_start"] - t0) <= 0.25 and abs(pg["t_end"] - (t0 + 3)) <= 0.25
        assert len(pg["frames"]) == config.MEDIAN_MAX_FRAMES
    for k in range(3):
        img = cv2.imread(str(out / f"page_{k + 1}.png"), cv2.IMREAD_GRAYSCALE)
        # no playhead left: almost no pixel differs much from the clean page
        assert (np.abs(img.astype(int) - p[k]) > 40).mean() < 0.001
        # and no column is dark over the playhead's height outside the drawn page
        playhead_cols = ((img[150:470] < 128) & (p[k][150:470] > 128)).mean(axis=0)
        assert playhead_cols.max() < 0.2
    assert (out / "debug" / "change.png").exists() and (out / "debug" / "page_1_rows.png").exists()
    # stitched back into the strip
    assert [pg["canvas_x"] for pg in pages] == [0, PAGE_STEP_PX, 2 * PAGE_STEP_PX]
    img = cv2.imread(str(out / "canvas.png"), cv2.IMREAD_GRAYSCALE)
    assert img.shape == STRIP.shape
    assert (np.abs(img.astype(int) - STRIP) > 40).mean() < 0.001
    for n in (1, 2):
        assert (out / "debug" / f"ncc_{n}_{n + 1}.png").exists() and (out / "debug" / f"overlap_{n}_{n + 1}.png").exists()


def test_slow_drift_is_not_a_page(tmp_path):
    a = page(0)
    b = a.copy()
    b[160:470, 100:400] = 0                      # a big change, faded in slowly
    frames = still(a, 2, sweep=False) + fade(a, b, 3) + still(b, 2, sweep=False)
    video = tmp_path / "v.avi"
    write_video(video, frames)
    ctx = Context(source=str(video), video_id="v", crop=(0, 0, W, H), root=tmp_path / "cache")
    ingest.run(ctx)
    motion = np.load(ctx.dir("ingest") / "motion.npy")
    times = json.loads((ctx.dir("ingest") / "meta.json").read_text())["sample_times"]
    change = canvas.change_series(motion)
    assert change.max() < config.STILL_FRAME_FRAC       # frame to frame, the fade looks still
    pages, _ = canvas.page_runs(motion, times)
    assert len(pages) == 2
    assert pages[0]["t_start"] == 0 and pages[0]["t_end"] < 2.5
    assert pages[1]["t_start"] > 4.5


def test_matching_pages_merge(tmp_path):
    a = page(0)
    black = [np.zeros((H, W, 3), np.uint8)] * 3            # a 0.1 s flash
    pages, _ = run_stages(tmp_path, still(a, 2) + black + still(a, 2))
    assert len(pages) == 1
    assert len(pages[0]["ranges"]) == 2 and pages[0]["merged_ncc"][0] >= config.PAGE_MERGE_NCC
    assert len(pages[0]["frames"]) <= config.MEDIAN_MAX_FRAMES


def test_median_frame_indices_trim_and_limit():
    f = canvas.median_frame_indices(1.0, 5.0, 30, 1000)
    assert f[0] == int(np.ceil(1.25 * 30)) and f[-1] == int(np.floor(4.75 * 30))
    assert len(f) == config.MEDIAN_MAX_FRAMES
    assert len(canvas.median_frame_indices(1.0, 1.4, 30, 1000)) == 0


def test_refuses_short_page_too_few_frames():
    pg = [{"t_start": 0.0, "t_end": 0.6, "ranges": [[0.0, 0.6]]}]
    with pytest.raises(Refused):
        canvas.build_page_images(pg, "unused.avi", 10, 100, (0, 0, 10, 10))


def test_refuses_dark_page_and_wrong_row_count():
    good = page(0)
    with pytest.raises(Refused, match="dark"):
        canvas.check_pages([255 - good])
    with pytest.raises(Refused, match="0 tab rows"):
        canvas.check_pages([synth.blank(H, W)])
    two = synth.blank(H, W)
    synth.draw_tab_lines(two, 60, 930, synth.tab_line_ys(top=60))
    synth.draw_tab_lines(two, 60, 930, synth.tab_line_ys(top=300))
    with pytest.raises(Refused, match="2 tab rows"):
        canvas.check_pages([two])


def test_refuses_moved_lines():
    a = page(0)
    b = np.full_like(a, 255)
    b[3:] = a[:-3]                                         # tab 3 px lower
    with pytest.raises(Refused, match="must not move"):
        canvas.check_pages([a, b])
    assert len(canvas.check_pages([a, a])) == 2
