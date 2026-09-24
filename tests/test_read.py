"""Model reader. No test calls the API: a fake client returns constructed responses."""
import json
import re
from dataclasses import replace

import cv2
import numpy as np
import pytest

from src import config, notes, read, rows
from src.cache import Context
from tests import synth

YS = synth.tab_line_ys()
PROMPT = read.load_prompt()


def response(entries, finish="STOP", extra_parts=()):
    """A response dict shaped like GenerateContentResponse.model_dump(mode="json", exclude_none=True)."""
    return {"candidates": [{"content": {"role": "model",
                                        "parts": [*extra_parts, {"text": json.dumps({"notes": entries})}]},
                            "finish_reason": finish}],
            "usage_metadata": {"prompt_token_count": 1000, "candidates_token_count": 150,
                               "thoughts_token_count": 50}}


def ok(ids, fret=5):
    return response([{"id": i, "fret": fret} for i in ids])


class FakeClient:
    """Stands in for genai.Client: answers each call with reply(ids, call_number) and counts the calls."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.models = self

    def generate_content(self, **kw):
        ids = [int(v) for v in re.findall(r"\d+", kw["contents"][1].text)]
        self.calls.append(kw)
        body = self.reply(ids, len(self.calls))
        return type("Response", (), {"model_dump": lambda self_, **_: body})()


# --- validation --------------------------------------------------------------------------

@pytest.mark.parametrize("entries, reason", [
    ([{"id": 0, "fret": 3}], "missing \\[1\\]"),
    ([{"id": 0, "fret": 3}, {"id": 1, "fret": 4}, {"id": 2, "fret": 5}], "extra \\[2\\]"),
    ([{"id": 0, "fret": 3}, {"id": 0, "fret": 3}, {"id": 1, "fret": 4}], "duplicate \\[0\\]"),
    ([{"id": 0, "fret": 25}, {"id": 1, "fret": 4}], "schema"),
    ([{"id": 0, "fret": -1}, {"id": 1, "fret": 4}], "schema"),
    ([{"id": 0, "fret": "3"}, {"id": 1, "fret": 4}], "schema"),
    ([{"id": 0, "fret": True}, {"id": 1, "fret": 4}], "schema"),
    ([{"id": 0, "fret": 3, "x": 10}, {"id": 1, "fret": 4}], "schema"),
    ([{"id": 0}, {"id": 1, "fret": 4}], "schema"),
])
def test_bad_responses_are_rejected(entries, reason):
    with pytest.raises(read.InvalidResponse, match=reason):
        read.parse_response(response(entries), [0, 1], PROMPT.schema)


def test_refusal_cut_off_and_non_json_are_rejected():
    good = [{"id": 0, "fret": 3}]
    for finish in ("SAFETY", "PROHIBITED_CONTENT", "MAX_TOKENS", "OTHER"):
        with pytest.raises(read.InvalidResponse, match=finish):
            read.parse_response(response(good, finish=finish), [0], PROMPT.schema)
    with pytest.raises(read.InvalidResponse, match="prompt blocked"):
        read.parse_response({"prompt_feedback": {"block_reason": "SAFETY"}}, [0], PROMPT.schema)
    with pytest.raises(read.InvalidResponse, match="0 candidates"):
        read.parse_response({"candidates": []}, [0], PROMPT.schema)
    bad = response(good)
    bad["candidates"][0]["content"]["parts"][0]["text"] = '{"notes": [{"id": 0, "fret": 3}'
    with pytest.raises(read.InvalidResponse, match="not JSON"):
        read.parse_response(bad, [0], PROMPT.schema)


def test_valid_response_gives_frets_and_nulls_and_skips_thoughts():
    thought = {"text": '{"notes": [{"id": 7, "fret": 1}]}', "thought": True}
    r = response([{"id": 7, "fret": 0}, {"id": 8, "fret": None}, {"id": 9, "fret": 24}], extra_parts=[thought])
    assert read.parse_response(r, [7, 8, 9], PROMPT.schema) == {7: 0, 8: None, 9: 24}


# --- cache keys --------------------------------------------------------------------------

def test_cache_key_changes_with_image_prompt_model_thinking_and_attempt(monkeypatch):
    base = read.call_key(b"png", "read", PROMPT, 0)
    assert read.call_key(b"png", "read", PROMPT, 0) == base
    assert read.call_key(b"png2", "read", PROMPT, 0) != base
    assert read.call_key(b"png", "read", PROMPT, 1) != base
    assert read.call_key(b"png", "read", replace(PROMPT, version=2), 0) != base
    assert read.call_key(b"png", "read", replace(PROMPT, text=PROMPT.text + " "), 0) != base
    assert read.call_key(b"png", "read", replace(PROMPT, schema={}), 0) != base
    assert read.call_key(b"png", "reread", PROMPT, 0) != base
    monkeypatch.setitem(config.MODELS, "read", "other-model")
    assert read.call_key(b"png", "read", PROMPT, 0) != base
    monkeypatch.undo()
    assert read.call_key(b"png", "read", PROMPT, 0) == base
    monkeypatch.setitem(config.READ_THINKING_LEVEL, "read", "LOW")
    assert read.call_key(b"png", "read", PROMPT, 0) != base


def test_request_has_schema_thinking_level_and_no_temperature():
    client = FakeClient(lambda ids, n: ok(ids))
    read.request(client, b"\x89PNG", [3, 4], "read", PROMPT)
    kw = client.calls[0]
    cfg = kw["config"]
    assert kw["model"] == config.MODELS["read"] and cfg.temperature is None
    assert cfg.system_instruction == PROMPT.text and cfg.max_output_tokens == config.READ_MAX_TOKENS
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == PROMPT.schema
    assert cfg.thinking_config.thinking_level.value == config.READ_THINKING_LEVEL["read"]
    assert kw["contents"][0].inline_data.data == b"\x89PNG" and kw["contents"][0].inline_data.mime_type == "image/png"
    assert kw["contents"][1].text == "IDs in this grid: 3, 4"


# --- grids -------------------------------------------------------------------------------

def test_crop_is_padded_and_scaled_to_cell_height():
    canvas = synth.blank(h=100, w=100)
    canvas[40:60, 45:55] = 0                                       # box [45, 40, 54, 59]
    s = 20.0
    pad = round(config.CROP_PAD_FRAC * s)
    c = read.crop_note(canvas, [45, 40, 54, 59], s)
    scale = config.GRID_CELL_H_PX / (20 + 2 * pad)
    assert c.shape == (config.GRID_CELL_H_PX, round((10 + 2 * pad) * scale))
    assert c[c.shape[0] // 2, c.shape[1] // 2] < 50 and c[0, 0] == 255


def test_grid_puts_each_crop_in_its_own_cell():
    crops = [np.zeros((config.GRID_CELL_H_PX, 40 + 5 * k), np.uint8) for k in range(11)]
    img = read.make_grid(crops, list(range(100, 111)))
    cw, ch, cols, origins = read.grid_cells(11, 90)
    assert cols == config.GRID_COLS and img.shape[:2] == (2 * ch, cols * cw)
    for crop, (x, y) in zip(crops, origins):
        x += (90 - crop.shape[1]) // 2
        assert (img[y:y + crop.shape[0], x:x + crop.shape[1]] == 0).all()
    blue = (img[..., 0] > 150) & (img[..., 2] < 80)
    assert blue.any()                                              # ID labels
    for x, y in origins:                                           # labels stay out of the crops
        assert not blue[y:y + config.GRID_CELL_H_PX, x:x + 90].any()


def test_ids_are_split_into_even_grids():
    assert read.split_ids(list(range(163))) == [list(range(a, b)) for a, b in
                                                [(0, 33), (33, 66), (66, 99), (99, 131), (131, 163)]]
    assert read.split_ids(list(range(40))) == [list(range(40))]
    assert read.split_ids([]) == []


# --- stage -------------------------------------------------------------------------------

def stage_ctx(tmp_path, **kw):
    img = synth.blank(w=1200)
    synth.draw_tab_lines(img, 60, 1140, YS)
    for k in range(45):                                            # 45 notes: two grids
        synth.draw_digit(img, 120 + 22 * k, YS[k % 6], str(k % 10))
    row = rows.analyze(img)[0]
    found, dropped, _ = notes.find_notes(img, row)
    ctx = Context(source="v.mp4", video_id="v", root=tmp_path, **kw)
    cv2.imwrite(str(ctx.dir("canvas") / "canvas.png"), img)
    (ctx.dir("rows") / "rows.json").write_text(json.dumps(row))
    (ctx.dir("notes") / "notes.json").write_text(json.dumps({"origin_x": row["origin_x"], "notes": found,
                                                             "dropped": dropped}))
    return ctx, found


def test_stage_reads_every_note_and_a_rerun_makes_no_calls(tmp_path):
    ctx, found = stage_ctx(tmp_path)
    assert len(found) == 45
    client = FakeClient(lambda ids, n: ok(ids, fret=7))
    read.run(ctx, client)
    assert len(client.calls) == 2
    out = json.loads((ctx.dir("read") / "read.json").read_text())
    assert [n["id"] for n in out["notes"]] == list(range(45))
    assert all(n["model"] == {"fret": 7, "valid": True} for n in out["notes"])
    assert [len(g["ids"]) for g in out["grids"]] == [23, 22]
    assert all((ctx.dir("read") / g["file"]).exists() for g in out["grids"])
    cost = json.loads((ctx.dir("read") / "cost.json").read_text())
    price = config.PRICES_USD_PER_MTOK[config.MODELS["read"]]
    one = (1000 * price["input"] + (150 + 50) * price["output"]) / 1e6        # thinking is billed as output
    assert cost["total_usd"] == pytest.approx(2 * one) and cost["new_usd"] == pytest.approx(2 * one)

    read.run(ctx, FakeClient(lambda ids, n: pytest.fail("a cached call was repeated")))
    cost = json.loads((ctx.dir("read") / "cost.json").read_text())
    assert cost["total_usd"] == pytest.approx(2 * one) and cost["new_usd"] == 0
    assert json.loads((ctx.dir("read") / "read.json").read_text()) == out


def test_invalid_response_is_requested_once_more(tmp_path):
    ctx, _ = stage_ctx(tmp_path)
    # the first grid's first answer drops an ID; its second answer is valid
    client = FakeClient(lambda ids, n: ok(ids[1:]) if n == 1 else ok(ids))
    read.run(ctx, client)
    assert len(client.calls) == 3
    out = json.loads((ctx.dir("read") / "read.json").read_text())
    assert all(n["model"]["valid"] for n in out["notes"])
    assert out["grids"][0]["errors"] == ["IDs: duplicate [], missing [0], extra []"]
    read.run(ctx, FakeClient(lambda ids, n: pytest.fail("a cached call was repeated")))


def test_grid_invalid_twice_leaves_its_notes_unread(tmp_path):
    ctx, _ = stage_ctx(tmp_path)
    client = FakeClient(lambda ids, n: response([], finish="SAFETY") if ids[0] == 0 else ok(ids))
    read.run(ctx, client)
    assert len(client.calls) == 3
    out = json.loads((ctx.dir("read") / "read.json").read_text())
    first = set(out["grids"][0]["ids"])
    assert not out["grids"][0]["valid"] and len(out["grids"][0]["errors"]) == 2
    assert all(n["model"] == {"fret": None, "valid": False} for n in out["notes"] if n["id"] in first)
    assert all(n["model"]["valid"] for n in out["notes"] if n["id"] not in first)


def test_record_fixtures_copies_the_calls_used(tmp_path, monkeypatch):
    monkeypatch.setattr(read, "FIXTURES_DIR", tmp_path / "fixtures")
    ctx, _ = stage_ctx(tmp_path, record_fixtures=True)
    read.run(ctx, FakeClient(lambda ids, n: ok(ids)))
    recorded = sorted(p.name for p in (tmp_path / "fixtures" / "v").iterdir())
    assert recorded == sorted(p.name for p in (ctx.dir("read") / "calls").iterdir()) and len(recorded) == 2
    assert read.params(ctx)["record_fixtures"] and not read.params(replace(ctx, record_fixtures=False))["record_fixtures"]


# --- recorded responses ------------------------------------------------------------------

RECORDED = sorted(read.FIXTURES_DIR.glob("*/*.json"))


def test_recorded_responses_parse_and_have_usage():
    """Real responses saved with --record-fixtures: each parses to exactly its grid's IDs, or is rejected
    with a reason (never a crash), and has token counts for the cost log."""
    assert RECORDED, "no recorded responses in tests/fixtures/read/"
    parsed = 0
    for path in RECORDED:
        rec = json.loads(path.read_text())
        try:
            frets = read.parse_response(rec["response"], rec["ids"], PROMPT.schema)
        except read.InvalidResponse:
            continue
        assert sorted(frets) == sorted(rec["ids"])
        assert all(f is None or 0 <= f <= 24 for f in frets.values())
        n_in, n_answer, _ = read.usage_tokens(rec["response"])
        assert n_in > 0 and n_answer > 0
        parsed += 1
    assert parsed > 0
