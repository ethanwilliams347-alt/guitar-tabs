"""Stage 2: find the still pages, make one clean image of each and stitch them into one canvas.

Pages: runs of samples that stay still (change from the previous sample and drift from
the run's first sample both small). Each page image is the pixel-wise median of up to
MEDIAN_MAX_FRAMES full-resolution frames from the trimmed run, which removes a moving
playhead and notes that are highlighted only while they play.

Stitching: each page n+1 is placed at the shift over page n with the best NCC over the
tab-row band, if that shift clearly beats every other and the bar lines in the overlap
agree. Where pages overlap, the canvas keeps the darker pixel, so the clear copy wins
over the faded one.

Writes: page_<n>.png, canvas.png, pages.json. Debug: change.png (change plot with pages
marked), page_<n>_rows.png (each page with its row overlay), ncc_<n>_<n+1>.png (NCC by
shift) and overlap_<n>_<n+1>.png (both pages' overlaps and the composite, with bar lines).
"""
import json

import cv2
import numpy as np

from src import Refused, config, ingest, rows

STAGE_VERSION = 2


def inputs(ctx):
    d = ctx.dir("ingest")
    return [d / "meta.json", d / "motion.npy"]


def params(ctx):
    names = ["DIFF_PIXEL_LEVEL", "STILL_FRAME_FRAC", "STILL_DRIFT_FRAC", "PAGE_MIN_STABLE_S", "PAGE_TRIM_S",
             "MEDIAN_MAX_FRAMES", "MEDIAN_MIN_FRAMES", "PAGE_MERGE_NCC", "LIGHT_BG_MIN_LEVEL", "LINE_Y_MATCH_PX",
             "LINE_MIN_COVER_FRAC", "LINE_SPACING_TOL_FRAC", "BARLINE_MIN_COVER_FRAC", "BARLINE_MERGE_FRAC",
             "BARLINE_EDGE_MAX_COVER_FRAC", "EXTENT_MIN_RUN_FRAC", "STRIP_MIN_OVERLAP_FRAC", "STRIP_NCC_MIN",
             "STRIP_NCC_MARGIN", "STRIP_RUNNER_UP_MIN_DIST_PX", "STRIP_MAX_RESIDUAL_PX"]
    return {n: getattr(config, n) for n in names} | {"rows_version": rows.STAGE_VERSION}


def diff_frac(a, b):
    """Fraction of pixels whose absolute difference is over DIFF_PIXEL_LEVEL."""
    return float((np.abs(a.astype(np.int16) - b.astype(np.int16)) > config.DIFF_PIXEL_LEVEL).mean())


def change_series(motion):
    """change[k] = diff_frac(sample k-1, sample k); change[0] = 0."""
    return np.array([0.0] + [diff_frac(motion[k - 1], motion[k]) for k in range(1, len(motion))])


def still_runs(motion, change):
    """Maximal runs [first, last] of samples that are still.

    A sample joins the current run if its change from the previous sample is under
    STILL_FRAME_FRAC and it differs from the run's first sample in fewer than
    STILL_DRIFT_FRAC of pixels. Otherwise it starts a new run.
    """
    runs, start = [], 0
    for k in range(1, len(motion)):
        still = change[k] < config.STILL_FRAME_FRAC
        if still and diff_frac(motion[start], motion[k]) < config.STILL_DRIFT_FRAC:
            continue
        runs.append((start, k - 1))
        start = k
    runs.append((start, len(motion) - 1))
    return runs


def page_runs(motion, times):
    """Still runs at least PAGE_MIN_STABLE_S long, as dicts with their sample range and times."""
    change = change_series(motion)
    pages = []
    for a, b in still_runs(motion, change):
        if times[b] - times[a] >= config.PAGE_MIN_STABLE_S:
            pages.append({"samples": [a, b], "t_start": times[a], "t_end": times[b]})
    return pages, change


def median_frame_indices(t_start, t_end, fps, frame_count):
    """Up to MEDIAN_MAX_FRAMES evenly spaced video frames inside the trimmed run."""
    lo = int(np.ceil((t_start + config.PAGE_TRIM_S) * fps))
    hi = min(int(np.floor((t_end - config.PAGE_TRIM_S) * fps)), frame_count - 1)
    if hi < lo:
        return []
    n = min(config.MEDIAN_MAX_FRAMES, hi - lo + 1)
    return sorted(set(int(round(v)) for v in np.linspace(lo, hi, n)))


def ncc(a, b):
    """Zero-mean normalized cross-correlation of two equal-size images."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def build_page_images(pages, video, fps, frame_count, crop):
    """Median image per page. Decodes the video once for all pages."""
    for p in pages:
        p["frames"] = [f for t0, t1 in p["ranges"] for f in median_frame_indices(t0, t1, fps, frame_count)]
        if len(p["frames"]) > config.MEDIAN_MAX_FRAMES:
            keep = np.linspace(0, len(p["frames"]) - 1, config.MEDIAN_MAX_FRAMES).round().astype(int)
            p["frames"] = [p["frames"][i] for i in keep]
        if len(p["frames"]) < config.MEDIAN_MIN_FRAMES:
            raise Refused(f"page at {p['t_start']:.2f}-{p['t_end']:.2f} s has only {len(p['frames'])} frames "
                          f"after trimming; at least {config.MEDIAN_MIN_FRAMES} are needed")
    wanted = sorted({f for p in pages for f in p["frames"]})
    stacks = {f: None for f in wanted}
    for f, frame in ingest.read_frames(video, wanted):
        stacks[f] = ingest.to_gray_crop(frame, crop)
    images = []
    for p in pages:
        images.append(np.median(np.stack([stacks[f] for f in p["frames"]]), axis=0).astype(np.uint8))
    return images


def find_pages(motion, times, video, fps, frame_count, crop):
    """Page dicts and their median images, with consecutive matching pages merged."""
    pages, change = page_runs(motion, times)
    if not pages:
        raise Refused(f"no still page of at least {config.PAGE_MIN_STABLE_S} s")
    for p in pages:
        p["ranges"] = [[p["t_start"], p["t_end"]]]
    images = build_page_images(pages, video, fps, frame_count, crop)
    k = 0
    while k + 1 < len(pages):
        score = ncc(images[k], images[k + 1])
        if score >= config.PAGE_MERGE_NCC:
            a, b = pages[k], pages.pop(k + 1)
            a["merged_ncc"] = a.get("merged_ncc", []) + [score]
            a["ranges"] += b["ranges"]
            a["samples"][1], a["t_end"] = b["samples"][1], b["t_end"]
            images.pop(k + 1)
            images[k] = build_page_images([a], video, fps, frame_count, crop)[0]
        else:
            k += 1
    return pages, images, change


def check_pages(images):
    """Rows found on each page. Refuses dark pages, pages without exactly one row, and moved lines."""
    found = []
    for n, img in enumerate(images, start=1):
        level = float(np.median(img))
        if level < config.LIGHT_BG_MIN_LEVEL:
            raise Refused(f"page {n} has a dark background (median gray {level:.0f} < "
                          f"{config.LIGHT_BG_MIN_LEVEL}); dark backgrounds aren't supported yet")
        page_rows = rows.analyze(img)
        if len(page_rows) != 1:
            raise Refused(f"page {n} has {len(page_rows)} tab rows; the paged-strip layout needs exactly one")
        row = page_rows[0]
        if found:
            dy = np.abs(np.array(row["line_y"]) - np.array(found[0]["line_y"])).max()
            if dy > config.LINE_Y_MATCH_PX:
                raise Refused(f"page {n}'s tab lines are {dy:.1f} px from page 1's "
                              f"(more than {config.LINE_Y_MATCH_PX} px); the tab must not move between pages")
        found.append(row)
    return found


def draw_change_plot(change, times, pages, height=300):
    """Change per sample (log scale), threshold dashed, pages shaded with their trimmed ranges."""
    width = len(change) * 2
    img = np.full((height, width, 3), 255, np.uint8)
    lo, hi = 1e-4, 1.0

    def y_of(v):
        v = min(max(v, lo), hi)
        return int(round((height - 20) * (1 - (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))))) + 10

    def x_of(t):
        return int(round(t / times[-1] * (width - 1)))

    for p in pages:
        for t0, t1 in p["ranges"]:
            cv2.rectangle(img, (x_of(t0), 0), (x_of(t1), height - 1), (225, 245, 225), -1)
            cv2.rectangle(img, (x_of(t0 + config.PAGE_TRIM_S), height - 8), (x_of(t1 - config.PAGE_TRIM_S), height - 1),
                          (0, 150, 0), -1)
    ys = y_of(config.STILL_FRAME_FRAC)
    for x in range(0, width, 12):
        cv2.line(img, (x, ys), (x + 6, ys), (0, 0, 220), 1)
    pts = np.array([[x_of(t), y_of(c)] for t, c in zip(times, change)], np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 40), 1, cv2.LINE_AA)
    for n, p in enumerate(pages, start=1):
        cv2.putText(img, f"page {n}: {p['t_start']:.2f}-{p['t_end']:.2f} s", (x_of(p["t_start"]) + 4, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1, cv2.LINE_AA)
    cv2.putText(img, f"STILL_FRAME_FRAC {config.STILL_FRAME_FRAC}", (4, ys - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 220), 1, cv2.LINE_AA)
    return img


def band_rows(row, height):
    """Pixel rows of the tab-row band: top line - s to bottom line + s."""
    y0 = max(int(np.floor(row["line_y"][0] - row["s"])), 0)
    y1 = min(int(np.ceil(row["line_y"][-1] + row["s"])) + 1, height)
    return y0, y1


def shift_scores(a, b, row):
    """NCC over the tab-row band for every shift dx (page b's x = 0 at page a's x = dx).

    Shifts run from 0 to the last one that leaves an overlap of STRIP_MIN_OVERLAP_FRAC of
    the width. The whole overlap is compared, faded edges included: skipping the fades
    leaves windows of plain tab lines, which match at almost any shift.
    """
    y0, y1 = band_rows(row, a.shape[0])
    w = a.shape[1]
    min_overlap = int(np.ceil(config.STRIP_MIN_OVERLAP_FRAC * w))
    return np.array([ncc(a[y0:y1, dx:], b[y0:y1, :w - dx]) for dx in range(w - min_overlap + 1)])


def best_shift(scores):
    """The best shift and the best one more than STRIP_RUNNER_UP_MIN_DIST_PX away from it."""
    best = int(np.argmax(scores))
    far = np.abs(np.arange(len(scores)) - best) > config.STRIP_RUNNER_UP_MIN_DIST_PX
    runner_up = int(np.flatnonzero(far)[np.argmax(scores[far])]) if far.any() else None
    return best, runner_up


def overlap_bars(row_a, row_b, dx):
    """Bar lines of both pages inside both rows' extents, in page a's x.

    A bar line can only be found where the row's lines are, so bar lines outside either
    extent (in a fade that one page's row doesn't reach) have nothing to be compared with.
    """
    lo = max(row_a["extent"][0], row_b["extent"][0] + dx)
    hi = min(row_a["extent"][1], row_b["extent"][1] + dx)
    bars_a = [x for x in row_a["bar_x"] if lo <= x <= hi]
    bars_b = [x + dx for x in row_b["bar_x"] if lo <= x + dx <= hi]
    return bars_a, bars_b


def bar_residual(bars_a, bars_b):
    """Largest distance from a bar line on one page to the nearest on the other.

    None when neither page has a bar line in the overlap, infinity when only one does.
    """
    if not bars_a and not bars_b:
        return None
    if not bars_a or not bars_b:
        return float("inf")
    near = [min(abs(x - y) for y in bars_b) for x in bars_a] + [min(abs(x - y) for y in bars_a) for x in bars_b]
    return float(max(near))


def stitch_pair(a, b, row_a, row_b, n):
    """Offset of page n+1 on page n with its scores. Refuses a shift that isn't clearly right."""
    scores = shift_scores(a, b, row_a)
    best, runner_up = best_shift(scores)
    bars_a, bars_b = overlap_bars(row_a, row_b, best)
    pair = {
        "pages": [n, n + 1],
        "offset": best,
        "overlap": a.shape[1] - best,
        "ncc": float(scores[best]),
        "runner_up_offset": runner_up,
        "runner_up_ncc": float(scores[runner_up]) if runner_up is not None else None,
        "bar_x": bars_a,
        "bar_x_next": bars_b,
        "bar_residual": bar_residual(bars_a, bars_b),
    }
    where = (f"pages {n} and {n + 1}: best shift {best} px (overlap {pair['overlap']} px) has NCC "
             f"{pair['ncc']:.3f}")
    if pair["ncc"] < config.STRIP_NCC_MIN:
        raise Refused(f"{where}, under {config.STRIP_NCC_MIN}; the pages don't overlap enough to be stitched")
    if runner_up is not None and pair["ncc"] - pair["runner_up_ncc"] < config.STRIP_NCC_MARGIN:
        raise Refused(f"{where}, but shift {runner_up} px scores {pair['runner_up_ncc']:.3f}, within "
                      f"{config.STRIP_NCC_MARGIN}; the overlap is ambiguous")
    if pair["bar_residual"] is not None and pair["bar_residual"] > config.STRIP_MAX_RESIDUAL_PX:
        raise Refused(f"{where}, but its bar lines don't line up (page {n}: {bars_a}, page {n + 1}: {bars_b}; "
                      f"residual {pair['bar_residual']:.1f} px > {config.STRIP_MAX_RESIDUAL_PX})")
    return pair, scores


def composite(images, xs):
    """All pages on one white canvas at their x-positions, keeping the darker pixel where they overlap."""
    h, w = images[0].shape
    out = np.full((h, xs[-1] + w), 255, np.uint8)
    for img, x in zip(images, xs):
        np.minimum(out[:, x:x + w], img, out=out[:, x:x + w])
    return out


def draw_ncc_plot(scores, pair, height=300):
    """NCC by shift, with STRIP_NCC_MIN dashed, the best shift green and the runner-up orange."""
    width = len(scores)
    img = np.full((height, width, 3), 255, np.uint8)
    lo, hi = -0.2, 1.0

    def y_of(v):
        return int(round((height - 20) * (hi - min(max(v, lo), hi)) / (hi - lo))) + 10

    for v in (0.0, 0.5, 1.0):
        cv2.line(img, (0, y_of(v)), (width - 1, y_of(v)), (225, 225, 225), 1)
    ym = y_of(config.STRIP_NCC_MIN)
    for x in range(0, width, 12):
        cv2.line(img, (x, ym), (x + 6, ym), (0, 0, 220), 1)
    pts = np.array([[x, y_of(v)] for x, v in enumerate(scores)], np.int32)
    cv2.polylines(img, [pts], False, (40, 40, 40), 1, cv2.LINE_AA)
    marks = [(pair["offset"], pair["ncc"], (0, 150, 0), "best", -10)]
    if pair["runner_up_offset"] is not None:
        marks.append((pair["runner_up_offset"], pair["runner_up_ncc"], (0, 140, 255), "runner-up", 18))
    for x, v, colour, name, dy in marks:
        cv2.circle(img, (x, y_of(v)), 5, colour, 2)
        tx = min(x + 8, width - 170)
        cv2.putText(img, f"{name} {x} px: {v:.3f}", (tx, max(y_of(v) + dy, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colour, 1, cv2.LINE_AA)
    n, m = pair["pages"]
    cv2.putText(img, f"pages {n}-{m}: NCC by shift (page {m} x=0 at page {n} x=shift)", (4, height - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_overlap(a, b, row, pair, scale=2):
    """Page n's overlap, page n+1's overlap and the composite, stacked, around the tab band.

    Page n's bar lines are red and page n+1's blue, drawn on its own panel and both on the composite.
    """
    dx, w = pair["offset"], a.shape[1]
    y0, y1 = band_rows(row, a.shape[0])
    y0, y1 = max(y0 - 2 * int(row["s"]), 0), min(y1 + 2 * int(row["s"]), a.shape[0])
    left, right = a[y0:y1, dx:], b[y0:y1, :w - dx]
    both = np.minimum(left, right)
    red, blue = (0, 0, 230), (230, 90, 0)
    panels = []
    for img, bars in ((left, [(pair["bar_x"], red)]), (right, [(pair["bar_x_next"], blue)]),
                      (both, [(pair["bar_x"], red), (pair["bar_x_next"], blue)])):
        p = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                         cv2.COLOR_GRAY2BGR)
        for xs, colour in bars:
            for x in xs:
                xi = int(round((x - dx) * scale))
                cv2.line(p, (xi, 0), (xi, 10), colour, 2)
                cv2.line(p, (xi, p.shape[0] - 11), (xi, p.shape[0] - 1), colour, 2)
        panels.append(p)
        panels.append(np.full((6, p.shape[1], 3), (0, 200, 255), np.uint8))
    res = "none" if pair["bar_residual"] is None else f"{pair['bar_residual']:.1f} px"
    n, m = pair["pages"]
    label = np.full((40, panels[0].shape[1], 3), 255, np.uint8)
    cv2.putText(label, f"p{n} / p{m} / composite", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                cv2.LINE_AA)
    cv2.putText(label, f"shift {dx}, residual {res}", (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
                cv2.LINE_AA)
    return np.vstack([label] + panels[:-1])


def stitch(images, page_rows, debug_dir=None):
    """Each page's canvas x, each pair's scores and the composite canvas."""
    xs, pairs = [0], []
    for k in range(len(images) - 1):
        pair, scores = stitch_pair(images[k], images[k + 1], page_rows[k], page_rows[k + 1], k + 1)
        if debug_dir is not None:
            n = k + 1
            cv2.imwrite(str(debug_dir / f"ncc_{n}_{n + 1}.png"), draw_ncc_plot(scores, pair))
            cv2.imwrite(str(debug_dir / f"overlap_{n}_{n + 1}.png"),
                        draw_overlap(images[k], images[k + 1], page_rows[k], pair))
        xs.append(xs[-1] + pair["offset"])
        pairs.append(pair)
    return xs, pairs, composite(images, xs)


def run(ctx):
    out = ctx.dir("canvas")
    meta = json.loads((ctx.dir("ingest") / "meta.json").read_text())
    motion = np.load(ctx.dir("ingest") / "motion.npy")
    times = meta["sample_times"]
    pages, images, change = find_pages(motion, times, meta["video"], meta["fps"], meta["frame_count"],
                                       tuple(meta["crop"]))
    page_rows = check_pages(images)
    for old in out.glob("page_*.png"):
        old.unlink()
    for n, (p, img, row) in enumerate(zip(pages, images, page_rows), start=1):
        cv2.imwrite(str(out / f"page_{n}.png"), img)
        p["row"] = row
        if ctx.debug:
            cv2.imwrite(str(out / "debug" / f"page_{n}_rows.png"), rows.draw_overlay(img, [row]))
    if ctx.debug:
        cv2.imwrite(str(out / "debug" / "change.png"), draw_change_plot(change, times, pages))
    for n, p in enumerate(pages, start=1):
        print(f"[canvas] page {n}: {p['t_start']:.2f}-{p['t_end']:.2f} s, median of {len(p['frames'])} frames")
    xs, pairs, canvas_img = stitch(images, page_rows, out / "debug" if ctx.debug else None)
    for p, x in zip(pages, xs):
        p["canvas_x"] = x
    cv2.imwrite(str(out / "canvas.png"), canvas_img)
    (out / "pages.json").write_text(json.dumps({"pages": pages, "pairs": pairs,
                                                "canvas_size": [canvas_img.shape[1], canvas_img.shape[0]]},
                                               indent=1))
    for pr in pairs:
        res = "none in overlap" if pr["bar_residual"] is None else f"{pr['bar_residual']:.1f} px"
        ru = "none" if pr["runner_up_offset"] is None else f"{pr['runner_up_ncc']:.3f} at {pr['runner_up_offset']} px"
        print(f"[canvas] pages {pr['pages'][0]}-{pr['pages'][1]}: offset {pr['offset']} px, overlap "
              f"{pr['overlap']} px, NCC {pr['ncc']:.3f}, runner-up {ru}, bar residual {res}")
    print(f"[canvas] canvas {canvas_img.shape[1]} x {canvas_img.shape[0]} px")
