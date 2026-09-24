"""Render: tab.json, row cuts, the one scale factor and where every number is drawn."""
import json

import cv2
import jsonschema
import numpy as np
import pytest

from src import config, notes, render, rows
from src.cache import Context
from tests import synth

YS = synth.tab_line_ys()


def note(i, string, x, fret, status="verified"):
    return {"id": i, "string": string, "x": x, "fret": fret, "status": status,
            "readers": {"model": fret, "template": fret, "reread": None}}


def tab_of(notes_, bar_x=(0.0,), width=1920):
    return {"video_id": "v", "tab_width_px": width, "notes": notes_,
            "row": {"origin_x": 100.0, "extent": [100, 4100], "s": 18.0, "line_y": [375.5 + 18 * k for k in range(6)],
                    "bar_x": list(bar_x)}}


# --- scale and placement -------------------------------------------------------------------

@pytest.mark.parametrize("paper", ["letter", "a4"])
def test_one_scale_factor_maps_the_tab_width_to_the_printable_width(paper):
    tab = tab_of([])
    k = render.scale(tab, paper)
    printable = render.PAPERS[paper][0] - 2 * config.RENDER_MARGIN_PT
    assert k * 1920 == pytest.approx(printable)
    # the same canvas distance is the same PDF distance in any row
    assert render.pdf_x(960, 0, k) - render.pdf_x(0, 0, k) == pytest.approx(printable / 2)
    assert render.pdf_x(2860, 1900, k) - render.pdf_x(1900, 1900, k) == pytest.approx(printable / 2)


def test_numbers_are_placed_on_their_string_at_their_scaled_x():
    ns = [note(0, 1, 50.0, 3), note(1, 6, 50.0, 12), note(2, 4, 1500.0, 7, "flagged"), note(3, 2, 2100.0, 5)]
    tab = tab_of(ns, bar_x=[0.0, 1900.0, 3000.0])
    cut = render.cut_rows(tab["row"]["bar_x"], [n["x"] for n in ns], 3000.0, 1920)
    placed = render.layout(tab, cut, "letter")
    k = render.scale(tab, "letter")
    m = config.RENDER_MARGIN_PT
    assert [r["left"] for r in placed] == [m, m]                          # every row starts at the margin
    by_id = {num[0]: (num, r) for r in placed for num in r["numbers"]}
    for n in ns:
        (_, text, x, y, flagged), r = by_id[n["id"]]
        x0 = 0.0 if n["x"] < 1900 else 1900.0
        assert x == pytest.approx(m + (n["x"] - x0) * k)
        assert y == r["ys"][n["string"] - 1]
        assert text == str(n["fret"]) and flagged == (n["status"] == "flagged")
    ys = placed[0]["ys"]
    assert ys[0] > ys[5] and all(a - b == pytest.approx(config.RENDER_STRING_GAP_PT) for a, b in zip(ys, ys[1:]))


def test_rows_that_do_not_fit_go_on_a_new_page():
    tab = tab_of([note(k, 1, 100.0 + 1900 * k, 3) for k in range(40)], bar_x=[1900.0 * k for k in range(41)])
    cut = render.cut_rows(tab["row"]["bar_x"], [n["x"] for n in tab["notes"]], 76000.0, 1920)
    placed = render.layout(tab, cut, "letter")
    assert placed[-1]["page"] >= 1
    assert all(r["ys"][-1] >= config.RENDER_MARGIN_PT for r in placed)


# --- rows ----------------------------------------------------------------------------------

def test_rows_are_cut_at_the_last_bar_line_that_fits():
    assert render.cut_rows([0, 500, 1000, 1900, 2500], [100, 2700], 3000, 1920) == \
        [(0.0, 1900, False), (1900, 3000, False)]
    assert render.cut_rows([0, 1920], [10], 2000, 1920) == [(0.0, 1920, False), (1920, 2000, False)]
    assert render.cut_rows([0, 800], [10], 900, 1920) == [(0.0, 900, False)]


def test_a_measure_wider_than_a_row_is_split_between_notes():
    got = render.cut_rows([0, 2600], [100, 900, 1800, 2000, 2400], 2600, 1920)
    assert got == [(0.0, 1900.0, True), (1900.0, 2600, False)]            # midpoint of the 1800-2000 gap


def test_row_end_snaps_to_a_final_bar_near_the_extent_end():
    row = {"origin_x": 100.0, "extent": [100, 5132], "s": 17.6, "bar_x": [100.0, 5126.0]}
    assert render.row_end(row) == 5026.0
    row["bar_x"] = [100.0, 5000.0]
    assert render.row_end(row) == 5032


# --- tab.json ------------------------------------------------------------------------------

def read_note(i, status, fret, model=None, template=None, reread=None):
    return {"id": i, "string": 1, "x": 10.0 * i, "status": status, "fret": fret, "reason": None,
            "model": {"fret": model, "valid": True}, "template": {"fret": template},
            "reread": None if reread is None else {"fret": reread, "valid": True}}


def test_tab_json_keeps_statuses_and_readers_and_drops_ignored_marks():
    row = {"origin_x": 100.0, "extent": [100, 900], "s": 18.0, "line_y": [1, 2, 3, 4, 5, 6], "bar_x": [100.0, 600.0]}
    rn = [read_note(0, "verified", 3, 3, 3), read_note(1, "verified_reread", 5, 9, 5, 5),
          read_note(2, "flagged", 7, 7, None, 7), read_note(3, "ignored", None),
          read_note(4, "flagged", None, None, None, None)]
    tab = render.build_tab("v", 1920, row, rn)
    assert [(n["id"], n["status"], n["fret"]) for n in tab["notes"]] == \
        [(0, "verified", 3), (1, "verified", 5), (2, "flagged", 7), (4, "flagged", None)]
    assert tab["notes"][1]["readers"] == {"model": 9, "template": 5, "reread": 5}
    assert tab["row"]["bar_x"] == [0.0, 500.0]                            # measured from the origin


def test_schema_allows_a_null_fret_only_on_a_flagged_note():
    schema = json.loads(render.SCHEMA_PATH.read_text())
    base = {"video_id": "v", "tab_width_px": 1920}
    jsonschema.validate({**base, "notes": [{"string": 1, "x": 0, "fret": None, "status": "flagged"}]}, schema)
    for bad in ({"string": 1, "x": 0, "fret": None, "status": "verified"}, {"string": 1, "x": 0, "fret": None}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**base, "notes": [bad]}, schema)


# --- stage ---------------------------------------------------------------------------------

def stage_ctx(tmp_path):
    """A synthetic video's cache up to the read stage: one row, 3 bar lines, 12 notes, 2 of them flagged."""
    img = synth.blank(w=3000)
    synth.draw_tab_lines(img, 60, 2940, YS)
    for bx in (60, 1500, 2938):
        synth.draw_bar(img, bx, YS)
    for k in range(12):
        synth.draw_digit(img, 200 + 220 * k, YS[k % 6], str(3 + k % 7))
    row = rows.analyze(img)[0]
    found, dropped, _ = notes.find_notes(img, row)
    ctx = Context(source="v.mp4", video_id="v", root=tmp_path, debug=True, output=str(tmp_path / "out" / "v.pdf"))
    cv2.imwrite(str(ctx.dir("canvas") / "canvas.png"), img)
    (ctx.dir("ingest") / "meta.json").write_text(json.dumps(
        {"source": "v.mp4", "fps": 30.0, "frame_count": 900, "width": 1920, "height": 1080,
         "crop": [0, 0, 1920, 484], "sample_times": [0.0, 0.1]}))
    (ctx.dir("canvas") / "pages.json").write_text(json.dumps({
        "pages": [{"ranges": [[0.0, 10.0]], "frames": list(range(31)), "row": row},
                  {"ranges": [[10.25, 20.0]], "frames": list(range(31)), "row": row}],
        "pairs": [{"pages": [1, 2], "offset": 1234, "overlap": 686, "ncc": 0.9, "runner_up_offset": 999,
                   "runner_up_ncc": 0.5, "bar_x": [], "bar_x_next": [], "bar_residual": None}],
        "canvas_size": [3000, 484]}))
    (ctx.dir("rows") / "rows.json").write_text(json.dumps(row))
    (ctx.dir("notes") / "notes.json").write_text(json.dumps({"origin_x": row["origin_x"], "notes": found,
                                                             "dropped": dropped}))
    flagged = {4, 9}
    rn = [{"id": n["id"], "string": n["string"], "x": n["x"],
           "status": "flagged" if n["id"] in flagged else "verified", "fret": 3 + n["id"] % 7,
           "reason": "no template reading" if n["id"] in flagged else None,
           "model": {"fret": 3 + n["id"] % 7, "valid": True},
           "template": {"fret": None if n["id"] in flagged else 3 + n["id"] % 7},
           "reread": {"fret": 3 + n["id"] % 7, "valid": True} if n["id"] in flagged else None} for n in found]
    (ctx.dir("read") / "read.json").write_text(json.dumps({
        "models": {"read": "m", "reread": "m"}, "grids": [{"ids": [n["id"] for n in found], "valid": True}],
        "templates": {"3": 2}, "notes": rn}))
    (ctx.dir("read") / "cost.json").write_text(json.dumps({"calls": [{}], "total_usd": 0.0123}))
    (ctx.dir("render") / "debug").mkdir()
    return ctx, row, found


def test_stage_writes_pdf_tab_json_and_report(tmp_path):
    ctx, row, found = stage_ctx(tmp_path)
    assert len(found) == 12
    render.run(ctx)
    pdf = tmp_path / "out" / "v.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    tab = json.loads((ctx.dir("render") / "tab.json").read_text())
    jsonschema.validate(tab, json.loads(render.SCHEMA_PATH.read_text()))
    assert len(tab["notes"]) == 12 and sum(n["status"] == "flagged" for n in tab["notes"]) == 2
    report = (ctx.dir("render") / "report.html").read_text(encoding="utf-8")
    assert "10 of 12 numbers verified." in report
    assert "<td>1234</td>" in report and "<td>0.900</td>" in report      # the stitch decision is listed
    assert report.count("data:image/png;base64,") == 2                    # both rows have a flag
    assert (ctx.dir("render") / "debug" / "cuts.png").exists()


def test_stage_rows_start_at_bar_lines(tmp_path):
    ctx, row, _ = stage_ctx(tmp_path)
    render.run(ctx)
    tab = json.loads((ctx.dir("render") / "tab.json").read_text())
    cut = render.cut_rows(tab["row"]["bar_x"], [n["x"] for n in tab["notes"]], render.row_end(row), 1920)
    assert [x0 for x0, _, _ in cut] == [0.0, pytest.approx(1500 + 1 - row["origin_x"], abs=0.6)]
    assert not any(split for _, _, split in cut)
