"""Stage 5: read what each note says. The model reader (iteration 1, build step 9).

Each note is cropped from the canvas with CROP_PAD_FRAC * s of padding, scaled to
GRID_CELL_H_PX high and laid out in grids of up to GRID_MAX_CELLS cells, each with its ID
in a blue label above a grey frame. Each grid goes to MODELS["read"] with the prompt and
JSON schema in src/prompts/read_v<READ_PROMPT_VERSION>. Every response is checked against
the schema and the grid's ID set. An invalid response (a refusal, cut off, bad JSON, schema
error, or a missing, extra or duplicate ID; a blocked prompt or any finish reason but STOP
counts as a refusal) is requested once more. If it fails again,
every note in that grid is left without a model reading. The model never outputs or
changes positions.

Calls go to the Gemini API (google-genai, models.generate_content, which keeps nothing
server-side). Every call is cached in read/calls/<key>.json, keyed by the grid PNG, prompt,
schema, model, thinking level and attempt, so a rerun never repeats a call. With --record-fixtures each
call used is also copied to tests/fixtures/read/<video id>/.

Writes: read.json (each note's model reading, each grid's calls), cost.json (tokens and
cost per call) and grids/grid_<role>_<n>.png, the images sent.
"""
import base64
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types
from jsonschema import Draft202012Validator

from src import config
from src.cache import ROOT

STAGE_VERSION = 1

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FIXTURES_DIR = ROOT / "tests" / "fixtures" / "read"
LABEL_COLOUR = (200, 60, 0)      # BGR blue: an ID label never looks like a fret number
FRAME_COLOUR = (150, 150, 150)


class InvalidResponse(Exception):
    """A model response that can't be used. The message is the reason."""


@dataclass(frozen=True)
class Prompt:
    version: int
    text: str
    schema: dict

    def digest(self):
        h = hashlib.sha256(self.text.encode())
        h.update(json.dumps(self.schema, sort_keys=True).encode())
        return h.hexdigest()


def prompt_paths(version):
    return PROMPTS_DIR / f"read_v{version}.md", PROMPTS_DIR / f"read_v{version}.schema.json"


def load_prompt(version=None):
    version = config.READ_PROMPT_VERSION if version is None else version
    text_path, schema_path = prompt_paths(version)
    return Prompt(version, text_path.read_text(encoding="utf-8"), json.loads(schema_path.read_text(encoding="utf-8")))


def inputs(ctx):
    return [ctx.dir("canvas") / "canvas.png", ctx.dir("rows") / "rows.json", ctx.dir("notes") / "notes.json",
            *prompt_paths(config.READ_PROMPT_VERSION)]


def params(ctx):
    names = ["CROP_PAD_FRAC", "GRID_MAX_CELLS", "GRID_CELL_H_PX", "GRID_COLS", "GRID_LABEL_H_PX", "GRID_GAP_PX",
             "GRID_LABEL_FONT_SCALE", "READ_PROMPT_VERSION", "READ_MAX_TOKENS", "READ_MAX_ATTEMPTS", "MODELS",
             "READ_THINKING_LEVEL", "PRICES_USD_PER_MTOK"]
    p = {n: getattr(config, n) for n in names}
    p["record_fixtures"] = ctx.record_fixtures
    return p


# --- grids -------------------------------------------------------------------------------

def crop_note(canvas, box, s):
    """The note's box plus CROP_PAD_FRAC * s on each side, scaled to GRID_CELL_H_PX high."""
    pad = int(round(config.CROP_PAD_FRAC * s))
    h, w = canvas.shape
    x0, y0, x1, y1 = box
    c = canvas[max(y0 - pad, 0):min(y1 + pad + 1, h), max(x0 - pad, 0):min(x1 + pad + 1, w)]
    scale = config.GRID_CELL_H_PX / c.shape[0]
    return cv2.resize(c, (max(1, round(c.shape[1] * scale)), config.GRID_CELL_H_PX), interpolation=cv2.INTER_CUBIC)


def grid_cells(n_cells, crop_w):
    """Cell size and the top-left of each crop, in reading order, GRID_COLS to a row."""
    gap, label = config.GRID_GAP_PX, config.GRID_LABEL_H_PX
    cw = crop_w + 2 * gap
    ch = label + config.GRID_CELL_H_PX + 2 * gap
    cols = min(config.GRID_COLS, n_cells)
    return cw, ch, cols, [((k % cols) * cw + gap, (k // cols) * ch + label + gap) for k in range(n_cells)]


def make_grid(crops, ids):
    """A BGR image with each crop framed in grey, centred in its cell, under a blue "#<id>" label."""
    crop_w = max(c.shape[1] for c in crops)
    cw, ch, cols, origins = grid_cells(len(crops), crop_w)
    rows_n = -(-len(crops) // cols)
    img = np.full((rows_n * ch, cols * cw, 3), 255, np.uint8)
    for crop, note_id, (x, y) in zip(crops, ids, origins):
        h, w = crop.shape
        x += (crop_w - w) // 2
        img[y:y + h, x:x + w] = crop[..., None]
        cv2.rectangle(img, (x - 2, y - 2), (x + w + 1, y + h + 1), FRAME_COLOUR, 1)
        cv2.putText(img, f"#{note_id}", (x - 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, config.GRID_LABEL_FONT_SCALE,
                    LABEL_COLOUR, 1, cv2.LINE_AA)
    return img


def split_ids(ids):
    """Consecutive IDs in as few grids as GRID_MAX_CELLS allows, with sizes as even as possible."""
    n = -(-len(ids) // config.GRID_MAX_CELLS)
    return [list(map(int, part)) for part in np.array_split(ids, n)] if ids else []


# --- responses ---------------------------------------------------------------------------

def parse_response(response, ids, schema):
    """{id: fret or None} from a response dict, or InvalidResponse with the reason."""
    block = (response.get("prompt_feedback") or {}).get("block_reason")
    if block:
        raise InvalidResponse(f"prompt blocked: {block}")
    candidates = response.get("candidates") or []
    if len(candidates) != 1:
        raise InvalidResponse(f"{len(candidates)} candidates, expected 1")
    finish = candidates[0].get("finish_reason")
    if finish != "STOP":
        raise InvalidResponse(f"finish reason {finish!r}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
    if not text:
        raise InvalidResponse("no answer text")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise InvalidResponse(f"not JSON: {e}")
    return validate(data, ids, schema)


def validate(data, ids, schema):
    """Check the schema, then that the IDs are exactly the grid's, none missing, extra or repeated."""
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        raise InvalidResponse(f"schema: {'/'.join(map(str, e.path)) or '<root>'}: {e.message}")
    got = [e["id"] for e in data["notes"]]
    dup = sorted({i for i in got if got.count(i) > 1})
    missing = sorted(set(ids) - set(got))
    extra = sorted(set(got) - set(ids))
    if dup or missing or extra:
        raise InvalidResponse(f"IDs: duplicate {dup}, missing {missing}, extra {extra}")
    return {int(e["id"]): e["fret"] for e in data["notes"]}


# --- calls -------------------------------------------------------------------------------

def call_key(png, role, prompt, attempt):
    """Hash of the grid PNG bytes, prompt version and text, schema, model, thinking level, max tokens and attempt."""
    h = hashlib.sha256(png)
    h.update(json.dumps({"prompt_version": prompt.version, "prompt": prompt.digest(), "model": config.MODELS[role],
                         "thinking_level": config.READ_THINKING_LEVEL[role], "max_tokens": config.READ_MAX_TOKENS,
                         "attempt": attempt}, sort_keys=True).encode())
    return h.hexdigest()[:32]


def request(client, png, ids, role, prompt):
    """One call, returned as a plain dict. No temperature: repeatable results come from the cache."""
    response = client.models.generate_content(
        model=config.MODELS[role],
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"),
                  types.Part.from_text(text="IDs in this grid: " + ", ".join(map(str, ids)))],
        config=types.GenerateContentConfig(
            system_instruction=prompt.text,
            max_output_tokens=config.READ_MAX_TOKENS,
            response_mime_type="application/json",
            response_json_schema=prompt.schema,
            thinking_config=types.ThinkingConfig(thinking_level=config.READ_THINKING_LEVEL[role]),
        ),
    )
    return response.model_dump(mode="json", exclude_none=True,
                               exclude={"sdk_http_response", "automatic_function_calling_history", "parsed"})


def usage_tokens(response):
    """(input, answer, thinking) tokens. Thinking is billed as output."""
    u = response.get("usage_metadata") or {}
    return u.get("prompt_token_count", 0), u.get("candidates_token_count", 0), u.get("thoughts_token_count", 0)


def call_cost(model, input_tokens, output_tokens):
    price = config.PRICES_USD_PER_MTOK[model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1e6


class Caller:
    """Makes and caches calls. The API client is created only when a call isn't cached."""

    def __init__(self, calls_dir, client=None, fixtures_dir=None):
        self.calls_dir = Path(calls_dir)
        self.calls_dir.mkdir(parents=True, exist_ok=True)
        self._client = client
        self.fixtures_dir = fixtures_dir
        self.log = []

    def client(self):
        if self._client is None:
            retry = types.HttpRetryOptions(attempts=config.API_RETRY_ATTEMPTS,
                                           initial_delay=config.API_RETRY_INITIAL_DELAY_S)
            self._client = genai.Client(http_options=types.HttpOptions(retry_options=retry))
        return self._client

    def read_grid(self, png, ids, role, prompt, grid):
        """{id: fret or None}, or None if every attempt was invalid. Each call is logged."""
        errors = []
        for attempt in range(config.READ_MAX_ATTEMPTS):
            key = call_key(png, role, prompt, attempt)
            path = self.calls_dir / f"{key}.json"
            new = not path.exists()
            if new:
                response = request(self.client(), png, ids, role, prompt)
                path.write_text(json.dumps({"role": role, "model": config.MODELS[role],
                                            "thinking_level": config.READ_THINKING_LEVEL[role], "prompt_version": prompt.version,
                                            "attempt": attempt, "ids": ids, "response": response}, indent=1))
            rec = json.loads(path.read_text())
            if self.fixtures_dir:
                self.fixtures_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, self.fixtures_dir / path.name)
            n_in, n_answer, n_thinking = usage_tokens(rec["response"])
            entry = {"key": key, "grid": grid, "role": role, "model": rec["model"], "attempt": attempt, "new": new,
                     "input_tokens": n_in, "output_tokens": n_answer + n_thinking, "thinking_tokens": n_thinking,
                     "cost_usd": call_cost(rec["model"], n_in, n_answer + n_thinking)}
            self.log.append(entry)
            try:
                return parse_response(rec["response"], ids, prompt.schema), errors
            except InvalidResponse as e:
                entry["error"] = str(e)
                errors.append(str(e))
        return None, errors

    def summary(self):
        return {"calls": self.log,
                "input_tokens": sum(c["input_tokens"] for c in self.log),
                "output_tokens": sum(c["output_tokens"] for c in self.log),
                "total_usd": sum(c["cost_usd"] for c in self.log),
                "new_usd": sum(c["cost_usd"] for c in self.log if c["new"])}


# --- stage -------------------------------------------------------------------------------

def read_notes(canvas, notes, s, caller, prompt, grids_dir, role="read"):
    """The model's reading of every note, and a record of each grid."""
    by_id = {n["id"]: n for n in notes}
    readings, grids = {}, []
    for g, ids in enumerate(split_ids([n["id"] for n in notes])):
        img = make_grid([crop_note(canvas, by_id[i]["box"], s) for i in ids], ids)
        png = cv2.imencode(".png", img)[1].tobytes()
        name = f"grid_{role}_{g}.png"
        (grids_dir / name).write_bytes(png)
        frets, errors = caller.read_grid(png, ids, role, prompt, g)
        grids.append({"file": f"grids/{name}", "ids": ids, "valid": frets is not None, "errors": errors})
        for i in ids:
            readings[i] = {"fret": frets[i], "valid": True} if frets else {"fret": None, "valid": False}
    return readings, grids


def run(ctx, client=None):
    out = ctx.dir("read")
    canvas = cv2.imread(str(ctx.dir("canvas") / "canvas.png"), cv2.IMREAD_GRAYSCALE)
    notes_data = json.loads((ctx.dir("notes") / "notes.json").read_text())
    row = json.loads((ctx.dir("rows") / "rows.json").read_text())
    notes = notes_data["notes"]
    grids_dir = out / "grids"
    if grids_dir.exists():
        shutil.rmtree(grids_dir)
    grids_dir.mkdir()
    prompt = load_prompt()
    caller = Caller(out / "calls", client, FIXTURES_DIR / ctx.video_id if ctx.record_fixtures else None)
    readings, grids = read_notes(canvas, notes, row["s"], caller, prompt, grids_dir)
    (out / "read.json").write_text(json.dumps({
        "prompt_version": prompt.version, "model": config.MODELS["read"], "grids": grids,
        "notes": [{"id": n["id"], "string": n["string"], "x": n["x"], "model": readings[n["id"]]} for n in notes],
    }, indent=1))
    cost = caller.summary()
    (out / "cost.json").write_text(json.dumps(cost, indent=1))
    frets = [r["fret"] for r in readings.values() if r["valid"]]
    invalid = sum(not r["valid"] for r in readings.values())
    new = sum(c["new"] for c in cost["calls"])
    print(f"[read] {len(notes)} notes in {len(grids)} grids: {len(frets) - frets.count(None)} numbers, "
          f"{frets.count(None)} not a number, {invalid} without a valid response")
    print(f"[read] {len(cost['calls'])} calls ({new} new): {cost['input_tokens']} input and "
          f"{cost['output_tokens']} output tokens (thinking included), ${cost['new_usd']:.4f} spent this run")
