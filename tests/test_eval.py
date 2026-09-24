import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

import eval as ev

FIX = Path(__file__).parent / "fixtures" / "eval"


def tab(notes, width=1000, **extra):
    return {"video_id": "v", "tab_width_px": width, "notes": notes, **extra}


def n(string, x, fret, status=None, id=None):
    d = {"string": string, "x": x, "fret": fret}
    if status:
        d["status"] = status
    if id is not None:
        d["id"] = id
    return d


def test_handmade_fixture_metrics():
    # tab width 1000, so notes match within 10 px. Cases in pred.json:
    #   ids 1-3, 9: right (dx 2, 0, 5, 0)       id 4: wrong fret, not flagged
    #   id 5: right but flagged (dx 1)          id 6: wrong fret, flagged
    #   id 7: on string 5 where truth is string 6: one missing + one extra, not flagged
    #   id 8: 15 px from its true note: one missing + one extra, not flagged
    #   truth at string 2, x 700: missing       id 10: extra, flagged
    m = ev.compute_metrics(ev.load_tab(FIX / "truth.json"), ev.load_tab(FIX / "pred.json"))
    assert m["matched"] == 7
    assert (m["missing"], m["extra"], m["wrong_fret"]) == (3, 3, 2)
    assert m["recall"] == pytest.approx(0.7)
    assert m["precision"] == pytest.approx(0.7)
    assert m["fret_accuracy"] == pytest.approx(5 / 7)
    assert m["unflagged_errors"] == 6          # ids 4, 7, 8 and the 3 missing notes
    assert m["unflagged_error_frac"] == pytest.approx(0.6)
    assert m["x_err_median_frac"] == pytest.approx(0.0)
    assert m["x_err_max_frac"] == pytest.approx(0.005)
    assert m["flagged"] == 3 and m["flags_real_errors"] == 2
    assert m["flag_rate"] == pytest.approx(0.3)
    assert m["flag_precision"] == pytest.approx(2 / 3)


def test_perfect_prediction_passes():
    truth = tab([n(1, 100, 0), n(6, 300, 12)])
    pred = tab([n(1, 101, 0, "verified", 0), n(6, 300, 12, "verified", 1)])
    m = ev.compute_metrics(truth, pred)
    assert (m["recall"], m["precision"], m["fret_accuracy"], m["unflagged_errors"]) == (1, 1, 1, 0)
    assert m["flag_precision"] is None
    assert ev.check_targets(m) == []


def test_missing_note():
    m = ev.compute_metrics(tab([n(1, 100, 0), n(2, 200, 3)]), tab([n(1, 100, 0, "verified")]))
    assert (m["missing"], m["extra"], m["recall"], m["precision"]) == (1, 0, 0.5, 1.0)
    assert m["unflagged_errors"] == 1
    assert "recall" in ev.check_targets(m)


def test_extra_note_flagged_and_unflagged():
    truth = tab([n(1, 100, 0)])
    for status, unflagged in (("verified", 1), ("flagged", 0)):
        m = ev.compute_metrics(truth, tab([n(1, 100, 0, "verified"), n(3, 500, 5, status)]))
        assert (m["extra"], m["precision"], m["unflagged_errors"]) == (1, 0.5, unflagged)


def test_wrong_fret():
    m = ev.compute_metrics(tab([n(4, 100, 7)]), tab([n(4, 100, 9, "verified")]))
    assert (m["matched"], m["wrong_fret"], m["fret_accuracy"], m["unflagged_errors"]) == (1, 1, 0.0, 1)
    m = ev.compute_metrics(tab([n(4, 100, 7)]), tab([n(4, 100, 9, "flagged")]))
    assert (m["unflagged_errors"], m["flags_real_errors"], m["flag_precision"]) == (0, 1, 1.0)


def test_wrong_string_is_missing_plus_extra():
    m = ev.compute_metrics(tab([n(2, 100, 5)]), tab([n(3, 100, 5, "verified")]))
    assert (m["matched"], m["missing"], m["extra"], m["unflagged_errors"]) == (0, 1, 1, 2)


def test_x_tolerance_edge():
    truth = tab([n(1, 100, 0)])
    assert ev.compute_metrics(truth, tab([n(1, 110, 0, "verified")]))["matched"] == 1
    assert ev.compute_metrics(truth, tab([n(1, 110.5, 0, "verified")]))["matched"] == 0


def test_matching_is_one_to_one_nearest_first():
    # A greedy pass in truth order would give A-P (6 px) and leave B-Q at 10 px.
    # Nearest first pairs A-Q (2 px) and B-P (2 px).
    truth = [n(1, 100, 0), n(1, 108, 0)]
    pred = [n(1, 106, 0), n(1, 98, 0)]
    pairs = ev.match_notes(truth, pred, 10)
    assert sorted((i, j) for i, j, _ in pairs) == [(0, 1), (1, 0)]
    # Two predictions near one true note: only one matches.
    assert len(ev.match_notes([n(1, 100, 0)], [n(1, 99, 0), n(1, 101, 0)], 10)) == 1


def test_prediction_needs_status_and_same_width():
    with pytest.raises(ValueError):
        ev.compute_metrics(tab([n(1, 1, 0)]), tab([n(1, 1, 0)]))
    with pytest.raises(ValueError):
        ev.compute_metrics(tab([n(1, 1, 0)]), tab([n(1, 1, 0, "verified")], width=999))


def test_schema_rejects_bad_notes(tmp_path):
    for bad in (n(7, 1, 0), n(1, 1, 25), n(1, 1, -1), {**n(1, 1, 0), "tempo": 90}, {"string": 1, "x": 1}):
        p = tmp_path / "t.json"
        p.write_text(json.dumps(tab([bad])))
        with pytest.raises(jsonschema.ValidationError):
            ev.load_tab(p)


ROW = {"origin_x": 50, "extent": [40, 2400], "s": 18, "line_y": [10, 28, 46, 64, 82, 100], "bar_x": [0, 900]}


def write_run(root, vid, notes):
    d = root / vid / "render"
    d.mkdir(parents=True)
    (d / "tab.json").write_text(json.dumps({"video_id": vid, "tab_width_px": 1000, "row": ROW, "notes": notes}))
    c = root / vid / "canvas"
    c.mkdir(parents=True)
    cv2.imwrite(str(c / "canvas.png"), np.full((120, 2400), 255, np.uint8))


def test_review_images_and_draft(tmp_path):
    notes = [n(1, 10, 0, "verified", 0), n(6, 1500, 12, "flagged", 1)]
    write_run(tmp_path, "vid", notes)
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    written = ev.review("vid", cache_root=tmp_path, truth_dir=truth_dir)
    parts = sorted((tmp_path / "vid" / "review").glob("part_*.png"))
    assert [p.name for p in parts] == ["part_1.png", "part_2.png", "part_3.png"]   # 2400 px / 1000
    img = cv2.imread(str(parts[1]))
    assert img.shape[1] == 1000 and img.shape[0] > 120
    # the flagged note (canvas x 1550) is drawn in red in part 2 at x ~550, under the canvas
    band = img[120:, 520:580]
    assert ((band[:, :, 2] > 180) & (band[:, :, 0] < 60) & (band[:, :, 1] < 60)).any()
    draft = json.loads((truth_dir / "vid.draft.json").read_text())
    assert draft["notes"] == [{"string": 1, "x": 10, "fret": 0}, {"string": 6, "x": 1500, "fret": 12}]
    assert truth_dir / "vid.draft.json" in written
    assert not (truth_dir / "vid.json").exists()
    # with a truth file present, no draft is written
    (truth_dir / "vid.draft.json").unlink()
    (truth_dir / "vid.json").write_text("{}")
    ev.review("vid", cache_root=tmp_path, truth_dir=truth_dir)
    assert not (truth_dir / "vid.draft.json").exists()


def test_evaluate_all_writes_results(tmp_path, monkeypatch):
    truth_dir, results_dir = tmp_path / "truth", tmp_path / "results"
    truth_dir.mkdir()
    (truth_dir / "vid.args").write_text("--crop 0,0,10,10 --layout paged-strip")
    (truth_dir / "vid.json").write_text(json.dumps({"video_id": "vid", "tab_width_px": 1000,
                                                     "notes": [n(1, 10, 0)]}))
    (truth_dir / "notruth.args").write_text("")
    write_run(tmp_path, "vid", [n(1, 10, 0, "verified", 0)])
    monkeypatch.setattr(ev, "TRUTH_DIR", truth_dir)
    monkeypatch.setattr(ev, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(ev, "run_pipeline", lambda vid: 0)
    monkeypatch.setattr(ev, "tab_path", lambda vid: tmp_path / vid / "render" / "tab.json")
    assert ev.evaluate_all() == 0
    out = json.loads(next(results_dir.glob("*.json")).read_text())
    assert list(out["videos"]) == ["vid"] and out["videos"]["vid"]["failed"] == []
    monkeypatch.setattr(ev, "run_pipeline", lambda vid: 2)
    assert ev.evaluate_all() == 1
