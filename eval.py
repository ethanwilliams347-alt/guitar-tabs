"""Evaluate the pipeline against hand-checked ground truth.

python eval.py                  run every video in tests/truth/ and print its metrics
python eval.py --review <id>    write review images (and a truth draft if there is none)

A video is evaluated when tests/truth/<id>.args and tests/truth/<id>.json both exist.
Results go to tests/results/<date>.json.
"""
import argparse
import datetime
import json
import shlex
import statistics
import sys
from pathlib import Path

import cv2
import jsonschema
import numpy as np

from src import config
from src.cache import CACHE_ROOT, ROOT, VIDEOS_DIR

TRUTH_DIR = ROOT / "tests" / "truth"
RESULTS_DIR = ROOT / "tests" / "results"
SCHEMA_PATH = ROOT / "src" / "schema" / "tab.schema.json"

# (metric, "min" or "max", target)
TARGETS = [
    ("recall", "min", config.EVAL_MIN_RECALL_FRAC),
    ("precision", "min", config.EVAL_MIN_PRECISION_FRAC),
    ("fret_accuracy", "min", config.EVAL_MIN_FRET_ACCURACY_FRAC),
    ("unflagged_error_frac", "max", config.EVAL_MAX_UNFLAGGED_ERROR_FRAC),
    ("x_err_median_frac", "max", config.EVAL_MAX_X_ERR_MEDIAN_FRAC),
    ("x_err_max_frac", "max", config.EVAL_MAX_X_ERR_MAX_FRAC),
    ("flag_rate", "max", config.EVAL_MAX_FLAG_RATE_FRAC),
]


def load_tab(path):
    with open(path) as f:
        tab = json.load(f)
    jsonschema.validate(tab, json.loads(SCHEMA_PATH.read_text()))
    return tab


def match_notes(true_notes, pred_notes, tol_px):
    """One-to-one matching, nearest pairs first. Pairs must share a string and be within tol_px.

    Returns a list of (true index, pred index, |dx|).
    """
    candidates = []
    for i, t in enumerate(true_notes):
        for j, p in enumerate(pred_notes):
            dx = abs(t["x"] - p["x"])
            if t["string"] == p["string"] and dx <= tol_px:
                candidates.append((dx, i, j))
    candidates.sort()
    used_t, used_p, pairs = set(), set(), []
    for dx, i, j in candidates:
        if i not in used_t and j not in used_p:
            used_t.add(i)
            used_p.add(j)
            pairs.append((i, j, dx))
    return pairs


def _frac(num, den):
    return num / den if den else None


def compute_metrics(truth, pred):
    width = truth["tab_width_px"]
    if pred["tab_width_px"] != width:
        raise ValueError(f"tab widths differ: truth {width}, prediction {pred['tab_width_px']}")
    t_notes, p_notes = truth["notes"], pred["notes"]
    for n in p_notes:
        if "status" not in n:
            raise ValueError(f"predicted note {n} has no status")
    pairs = match_notes(t_notes, p_notes, config.MATCH_X_TOL_FRAC * width)

    matched_p = {j for _, j, _ in pairs}
    matched_t = {i for i, _, _ in pairs}
    wrong_fret = {j for i, j, _ in pairs if t_notes[i]["fret"] != p_notes[j]["fret"]}
    extra = set(range(len(p_notes))) - matched_p
    missing = set(range(len(t_notes))) - matched_t
    flagged = {j for j, n in enumerate(p_notes) if n["status"] == "flagged"}
    real_errors = wrong_fret | extra

    unflagged = len(real_errors - flagged) + len(missing)
    x_errs = [dx / width for _, _, dx in pairs]
    flags_real = len(flagged & real_errors)
    return {
        "true_notes": len(t_notes),
        "found_notes": len(p_notes),
        "matched": len(pairs),
        "missing": len(missing),
        "extra": len(extra),
        "wrong_fret": len(wrong_fret),
        "flagged": len(flagged),
        "flags_real_errors": flags_real,
        "unflagged_errors": unflagged,
        "recall": _frac(len(pairs), len(t_notes)),
        "precision": _frac(len(pairs), len(p_notes)),
        "fret_accuracy": _frac(len(pairs) - len(wrong_fret), len(pairs)),
        "unflagged_error_frac": _frac(unflagged, len(t_notes)),
        "x_err_median_frac": statistics.median(x_errs) if x_errs else None,
        "x_err_max_frac": max(x_errs) if x_errs else None,
        "flag_rate": _frac(len(flagged), len(p_notes)),
        "flag_precision": _frac(flags_real, len(flagged)),
    }


def check_targets(metrics):
    """Names of the targets a video misses. A metric that can't be computed misses its target."""
    failed = []
    for name, kind, target in TARGETS:
        v = metrics[name]
        if v is None or (v < target if kind == "min" else v > target):
            failed.append(name)
    return failed


def format_table(vid, metrics, failed):
    lines = [f"== {vid} ==  {'PASS' if not failed else 'FAIL: ' + ', '.join(failed)}"]
    targets = {name: (kind, target) for name, kind, target in TARGETS}
    for name, v in metrics.items():
        shown = "n/a" if v is None else (f"{v:.4%}" if isinstance(v, float) else str(v))
        goal = ""
        if name in targets:
            kind, target = targets[name]
            goal = f"{'>=' if kind == 'min' else '<='} {target:.2%}"
        lines.append(f"  {name:<22}{shown:>12}  {goal}")
    return "\n".join(lines)


def tab_path(vid, cache_root=CACHE_ROOT):
    return Path(cache_root) / vid / "render" / "tab.json"


def video_source(vid):
    local = VIDEOS_DIR / f"{vid}.mp4"
    return str(local) if local.exists() else f"https://www.youtube.com/watch?v={vid}"


def run_pipeline(vid):
    import main
    argv = [video_source(vid)] + shlex.split((TRUTH_DIR / f"{vid}.args").read_text())
    return main.main(argv)


def evaluate_all():
    results, any_failed = {}, False
    for args_file in sorted(TRUTH_DIR.glob("*.args")):
        vid = args_file.stem
        truth_file = TRUTH_DIR / f"{vid}.json"
        if not truth_file.exists():
            print(f"== {vid} ==  skipped: no ground truth yet")
            continue
        code = run_pipeline(vid)
        if code != 0:
            print(f"== {vid} ==  FAIL: pipeline exited with {code}")
            results[vid] = {"pipeline_exit": code}
            any_failed = True
            continue
        metrics = compute_metrics(load_tab(truth_file), load_tab(tab_path(vid)))
        failed = check_targets(metrics)
        any_failed |= bool(failed)
        print(format_table(vid, metrics, failed))
        results[vid] = {"metrics": metrics, "failed": failed}
    if not results:
        print("no videos with ground truth to evaluate")
        return 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    out = RESULTS_DIR / f"{now.date().isoformat()}.json"
    out.write_text(json.dumps({"time": now.isoformat(timespec="seconds"), "videos": results}, indent=2))
    print(f"results written to {out}")
    return 1 if any_failed else 0


def draw_review(canvas_gray, tab):
    """One image per tab-area width of canvas, with the predicted numbers drawn under it."""
    width = int(tab["tab_width_px"])
    origin = tab["row"]["origin_x"]
    h, w = canvas_gray.shape
    band_h = 2 * config.REVIEW_MARGIN_PX + 5 * config.REVIEW_STRING_GAP_PX
    font = cv2.FONT_HERSHEY_SIMPLEX
    parts = []
    for start in range(0, w, width):
        img = np.full((h + band_h, width, 3), 255, np.uint8)
        piece = canvas_gray[:, start:start + width]
        img[:h, :piece.shape[1]] = cv2.cvtColor(piece, cv2.COLOR_GRAY2BGR)
        ys = [h + config.REVIEW_MARGIN_PX + k * config.REVIEW_STRING_GAP_PX for k in range(6)]
        for y in ys:
            cv2.line(img, (0, y), (width - 1, y), (200, 200, 200), 1)
        for bx in tab["row"]["bar_x"]:
            x = int(round(origin + bx - start))
            if 0 <= x < width:
                cv2.line(img, (x, ys[0]), (x, ys[-1]), (160, 160, 160), 1)
        for n in tab["notes"]:
            x = int(round(origin + n["x"] - start))
            if not 0 <= x < width:
                continue
            color = (0, 0, 220) if n.get("status") == "flagged" else (0, 0, 0)
            y = ys[n["string"] - 1]
            text = str(n["fret"])
            (tw, th), _ = cv2.getTextSize(text, font, config.REVIEW_FRET_FONT_SCALE, 2)
            cv2.rectangle(img, (x - tw // 2 - 1, y - th // 2 - 1), (x + tw // 2 + 1, y + th // 2 + 1),
                          (255, 255, 255), -1)
            cv2.putText(img, text, (x - tw // 2, y + th // 2), font, config.REVIEW_FRET_FONT_SCALE,
                        color, 2, cv2.LINE_AA)
            if "id" in n:
                label = str(n["id"])
                (iw, _), _ = cv2.getTextSize(label, font, config.REVIEW_ID_FONT_SCALE, 1)
                cv2.putText(img, label, (x - iw // 2, y + th // 2 + 12), font,
                            config.REVIEW_ID_FONT_SCALE, (150, 90, 0), 1, cv2.LINE_AA)
        parts.append(img)
    return parts


def review(vid, cache_root=CACHE_ROOT, truth_dir=TRUTH_DIR):
    """Write cache/<id>/review/part_<n>.png, plus tests/truth/<id>.draft.json if there's no truth yet.

    The draft is never the truth file itself: it becomes tests/truth/<id>.json only after
    every note has been checked against the review images and the file is confirmed.
    """
    tab = load_tab(tab_path(vid, cache_root))
    canvas = cv2.imread(str(Path(cache_root) / vid / "canvas" / "canvas.png"), cv2.IMREAD_GRAYSCALE)
    if canvas is None:
        raise FileNotFoundError(f"no canvas image for {vid}; run the pipeline first")
    out_dir = Path(cache_root) / vid / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("part_*.png"):
        old.unlink()
    written = []
    for n, img in enumerate(draw_review(canvas, tab), start=1):
        p = out_dir / f"part_{n}.png"
        cv2.imwrite(str(p), img)
        written.append(p)
    truth_dir = Path(truth_dir)
    if not (truth_dir / f"{vid}.json").exists():
        draft = {"video_id": vid, "tab_width_px": tab["tab_width_px"], "row": tab["row"],
                 "notes": [{"string": n["string"], "x": n["x"], "fret": n["fret"]} for n in tab["notes"]]}
        draft_path = truth_dir / f"{vid}.draft.json"
        draft_path.write_text(json.dumps(draft, indent=1))
        written.append(draft_path)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--review", metavar="VIDEO_ID", help="write review images for one video")
    args = p.parse_args(argv)
    if args.review:
        for path in review(args.review):
            print(path)
        return 0
    return evaluate_all()


if __name__ == "__main__":
    sys.exit(main())
