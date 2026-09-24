"""Stage 5: read what each note says, with two independent readers and a re-read.

Model reader: each note is cropped from the canvas with CROP_PAD_FRAC * s of padding,
scaled to GRID_CELL_H_PX high and laid out in grids of up to GRID_MAX_CELLS cells, each
with its ID in a blue label above a grey frame. Each grid goes to MODELS["read"] with the
prompt and JSON schema in src/prompts/read_v<READ_PROMPT_VERSION>. Every response is
checked against the schema and the grid's ID set. An invalid response (a blocked prompt, a
finish reason other than STOP, bad JSON, a schema error, or a missing, extra or duplicate
ID) is requested once more. If it fails again, the grid's notes have no model reading.

Template reader (local): each digit box of the line-free mask, fitted into a
TEMPLATE_SIZE_PX square, is matched against per-digit templates. A template is the median
of the crops the model read as that digit, from single-digit notes and, by position, from
multi-digit ones, leaving out crops that match their own class's leave-one-out median
poorly. A crop is never matched against a template that includes itself.

Decision: verified when both readers agree; otherwise the note is re-read by
MODELS["reread"] and verified only if the re-read agrees with the template reader, else
flagged. Model "not a number" with no template match is ignored. Neither reader ever
outputs or changes positions.

Calls go to the Gemini API (google-genai, models.generate_content, which keeps nothing
server-side). Every call is cached in read/calls/<key>.json, keyed by the grid PNG, prompt,
schema, model, thinking level and attempt, so a rerun never repeats a call. With
--record-fixtures each call used is also copied to tests/fixtures/read/<video id>/.

Writes: read.json (per note: status, final fret, the flag reason and all three readings;
per grid: IDs, validity and errors; template sizes), cost.json (tokens and cost per call)
and grids/grid_<role>_<n>.png, the images sent. Debug: templates.png (each digit's
template and crops, outliers in red) and status.png (notes coloured by status).
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

STAGE_VERSION = 2

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
            ctx.dir("notes") / "line_free.png", *prompt_paths(config.READ_PROMPT_VERSION)]


def params(ctx):
    names = ["CROP_PAD_FRAC", "GRID_MAX_CELLS", "GRID_CELL_H_PX", "GRID_COLS", "GRID_LABEL_H_PX", "GRID_GAP_PX",
             "GRID_LABEL_FONT_SCALE", "READ_PROMPT_VERSION", "READ_MAX_TOKENS", "READ_MAX_ATTEMPTS", "MODELS",
             "READ_THINKING_LEVEL", "PRICES_USD_PER_MTOK", "TEMPLATE_SIZE_PX", "TEMPLATE_MIN_SAMPLES",
             "TEMPLATE_CLASS_MIN_NCC", "TEMPLATE_MATCH_MIN_NCC", "TEMPLATE_MARGIN_NCC"]
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


# --- template reader ---------------------------------------------------------------------

def digit_image(free, box):
    """A digit's pixels in the line-free mask, fitted into a TEMPLATE_SIZE_PX square with its aspect
    ratio kept and centred, as floats from 0 to 1. Stretching would turn a narrow "1" into a block."""
    x0, y0, x1, y1 = box
    m = free[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    size = config.TEMPLATE_SIZE_PX
    scale = size / max(m.shape)
    w, h = max(1, round(m.shape[1] * scale)), max(1, round(m.shape[0] * scale))
    out = np.zeros((size, size), np.float32)
    oy, ox = (size - h) // 2, (size - w) // 2
    out[oy:oy + h, ox:ox + w] = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return out


def ncc(a, b):
    """Zero-mean normalized cross-correlation of two same-size images; 0 if either is flat."""
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum()) / d if d > 0 else 0.0


def digit_labels(notes, readings):
    """{(note id, digit index): digit} from the model's valid readings whose digit count matches the
    note's digit boxes. A two-digit reading labels each box by position: "12" is 1 then 2."""
    labels = {}
    for n in notes:
        r = readings[n["id"]]
        if r["valid"] and r["fret"] is not None and len(str(r["fret"])) == len(n["digits"]):
            for k, d in enumerate(str(r["fret"])):
                labels[(n["id"], k)] = d
    return labels


class Templates:
    """Per-digit templates: the pixel-wise median of the crops the model read as that digit.

    Each labelled crop is compared with its own class's median rebuilt without it (leave one
    out). A crop under TEMPLATE_CLASS_MIN_NCC is an outlier: it is left out of the template, so a
    systematic model error spreads less. A class needs TEMPLATE_MIN_SAMPLES crops after that.
    """

    def __init__(self, images, labels):
        self.images = images
        self.classes = {}                              # digit -> every crop labelled with it, outliers included
        for key, digit in sorted(labels.items()):
            self.classes.setdefault(digit, []).append(key)
        self.class_ncc = {}
        for digit, keys in self.classes.items():
            stack = np.stack([images[k] for k in keys])
            for j, k in enumerate(keys):
                self.class_ncc[k] = ncc(images[k], np.median(np.delete(stack, j, axis=0), axis=0)) if len(keys) > 1 else None
        self.outliers = {k for k, c in self.class_ncc.items() if c is not None and c < config.TEMPLATE_CLASS_MIN_NCC}
        self.members = {d: [k for k in keys if k not in self.outliers] for d, keys in self.classes.items()}
        self.members = {d: ks for d, ks in self.members.items() if len(ks) >= config.TEMPLATE_MIN_SAMPLES}
        self.templates = {d: np.median(np.stack([images[k] for k in ks]), axis=0) for d, ks in self.members.items()}

    def template_for(self, digit, key):
        """The digit's template, rebuilt without this crop if the crop is one of its members."""
        ks = self.members[digit]
        if key not in ks:
            return self.templates[digit]
        return np.median(np.stack([self.images[k] for k in ks if k != key]), axis=0)

    def read_digit(self, key):
        """(digit or None, best NCC, margin over the second best). None unless the best match scores at
        least TEMPLATE_MATCH_MIN_NCC and beats the second best by TEMPLATE_MARGIN_NCC."""
        scores = sorted(((ncc(self.images[key], self.template_for(d, key)), d) for d in self.templates), reverse=True)
        if not scores:
            return None, None, None
        best, digit = scores[0]
        margin = best - scores[1][0] if len(scores) > 1 else None
        ok = best >= config.TEMPLATE_MATCH_MIN_NCC and (margin is None or margin >= config.TEMPLATE_MARGIN_NCC)
        return (digit if ok else None), best, margin

    def read_note(self, note):
        """{"fret": int or None, "digits": [...]}: the note read digit by digit. A fret needs every digit
        read, no leading zero and a value of at most 24."""
        digits = []
        for k in range(len(note["digits"])):
            d, best, margin = self.read_digit((note["id"], k))
            digits.append({"digit": d, "ncc": best, "margin": margin,
                           "class_ncc": self.class_ncc.get((note["id"], k)),
                           "outlier": (note["id"], k) in self.outliers})
        text = "".join(d["digit"] or "?" for d in digits)
        fret = int(text) if text.isdigit() and not (len(text) > 1 and text[0] == "0") and int(text) <= 24 else None
        return {"fret": fret, "digits": digits}


def draw_template_sheet(tpl):
    """Debug: one row per digit, its template enlarged, then every crop labelled that digit, with
    outliers boxed in red and each crop's leave-one-out NCC printed under it."""
    size, zoom = config.TEMPLATE_SIZE_PX, 2
    cell = size * zoom
    rows_ = []
    for digit in sorted(tpl.classes):
        keys = tpl.classes[digit]
        row = np.full((cell + 16, (len(keys) + 2) * (cell + 4), 3), 255, np.uint8)
        cv2.putText(row, digit, (8, cell - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 60, 0), 2, cv2.LINE_AA)
        images = ([tpl.templates[digit]] if digit in tpl.templates else [np.zeros((size, size))]) + [tpl.images[k] for k in keys]
        for j, img in enumerate(images):
            x = (j + 1) * (cell + 4)
            g = (255 - cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
            row[0:cell, x:x + cell] = g[..., None]
            if j == 0:
                cv2.rectangle(row, (x, 0), (x + cell - 1, cell - 1), (200, 60, 0), 2)
                continue
            key = keys[j - 1]
            colour = (0, 0, 220) if key in tpl.outliers else (150, 150, 150)
            cv2.rectangle(row, (x, 0), (x + cell - 1, cell - 1), colour, 1)
            c = tpl.class_ncc.get(key)
            cv2.putText(row, "-" if c is None else f"{c:.2f}", (x + 2, cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        colour, 1, cv2.LINE_AA)
        rows_.append(row)
    width = max(r.shape[1] for r in rows_)
    return np.vstack([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows_])


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


# --- decision ----------------------------------------------------------------------------

STATUS_COLOURS = {"verified": (0, 160, 0), "verified_reread": (0, 160, 0), "flagged": (0, 0, 230),
                  "ignored": (170, 170, 170)}   # BGR


def first_decision(note, model, template):
    """"verified" when both readers give the same fret with as many digits as the note has digit boxes,
    "ignored" when the model says "not a number" and no template matches, else "reread"."""
    fret = model["fret"]
    if model["valid"] and fret is not None and fret == template["fret"] and len(str(fret)) == len(note["digits"]):
        return "verified"
    if model["valid"] and fret is None and template["fret"] is None:
        return "ignored"
    return "reread"


def final_decision(model, template, reread):
    """(status, fret, reason) after the re-read. Verified only if the re-read agrees with the template
    reader; otherwise flagged, showing the re-read, else the template's reading, else the first read."""
    if reread["valid"] and reread["fret"] is not None and reread["fret"] == template["fret"]:
        return "verified_reread", reread["fret"], None
    if template["fret"] is None:
        reason = "no template reading"
    elif not reread["valid"]:
        reason = "re-read invalid"
    else:
        reason = "re-read disagrees with the template"
    shown = next((f for f in (reread["fret"], template["fret"], model["fret"]) if f is not None), None)
    return "flagged", shown, reason


def draw_status(canvas, row, notes, decided, pad_frac=2.0):
    """Debug: the tab band with each note boxed in its status colour and its final fret written above it."""
    img = (cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR) * 0.6 + 100).astype(np.uint8)
    for n in notes:
        d = decided[n["id"]]
        colour = STATUS_COLOURS[d["status"]]
        x0, y0, x1, y1 = n["box"]
        cv2.rectangle(img, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), colour, 1)
        label = "-" if d["fret"] is None else str(d["fret"])
        cv2.putText(img, label, (x0, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1, cv2.LINE_AA)
    pad = int(pad_frac * row["s"])
    return img[max(int(row["line_y"][0]) - pad, 0):min(int(row["line_y"][-1]) + pad, img.shape[0])]


def run(ctx, client=None):
    out = ctx.dir("read")
    canvas = cv2.imread(str(ctx.dir("canvas") / "canvas.png"), cv2.IMREAD_GRAYSCALE)
    free = cv2.imread(str(ctx.dir("notes") / "line_free.png"), cv2.IMREAD_GRAYSCALE) == 0
    notes = json.loads((ctx.dir("notes") / "notes.json").read_text())["notes"]
    row = json.loads((ctx.dir("rows") / "rows.json").read_text())
    by_id = {n["id"]: n for n in notes}
    grids_dir = out / "grids"
    if grids_dir.exists():
        shutil.rmtree(grids_dir)
    grids_dir.mkdir()
    prompt = load_prompt()
    caller = Caller(out / "calls", client, FIXTURES_DIR / ctx.video_id if ctx.record_fixtures else None)

    model, grids = read_notes(canvas, notes, row["s"], caller, prompt, grids_dir)

    images = {(n["id"], k): digit_image(free, box) for n in notes for k, box in enumerate(n["digits"])}
    tpl = Templates(images, digit_labels(notes, model))
    template = {n["id"]: tpl.read_note(n) for n in notes}

    first = {n["id"]: first_decision(n, model[n["id"]], template[n["id"]]) for n in notes}
    reread_ids = [i for i in sorted(first) if first[i] == "reread"]
    reread, reread_grids = ({}, [])
    if reread_ids:
        reread, reread_grids = read_notes(canvas, [by_id[i] for i in reread_ids], row["s"], caller, prompt,
                                          grids_dir, role="reread")
    decided = {}
    for n in notes:
        i = n["id"]
        if first[i] == "verified":
            decided[i] = {"status": "verified", "fret": model[i]["fret"], "reason": None}
        elif first[i] == "ignored":
            decided[i] = {"status": "ignored", "fret": None, "reason": "not a number to either reader"}
        else:
            status, fret, reason = final_decision(model[i], template[i], reread[i])
            decided[i] = {"status": status, "fret": fret, "reason": reason}

    if ctx.debug:
        cv2.imwrite(str(out / "debug" / "templates.png"), draw_template_sheet(tpl))
        cv2.imwrite(str(out / "debug" / "status.png"), draw_status(canvas, row, notes, decided))
    (out / "read.json").write_text(json.dumps({
        "prompt_version": prompt.version, "models": {r: config.MODELS[r] for r in ("read", "reread")},
        "grids": grids + reread_grids,
        "templates": {d: len(ks) for d, ks in sorted(tpl.members.items())},
        "notes": [{"id": n["id"], "string": n["string"], "x": n["x"], **decided[n["id"]],
                   "model": model[n["id"]], "template": template[n["id"]], "reread": reread.get(n["id"])}
                  for n in notes],
    }, indent=1))
    cost = caller.summary()
    (out / "cost.json").write_text(json.dumps(cost, indent=1))

    count = {s: sum(d["status"] == s for d in decided.values()) for s in STATUS_COLOURS}
    frets = [r["fret"] for r in model.values() if r["valid"]]
    new = sum(c["new"] for c in cost["calls"])
    print(f"[read] model: {len(notes)} notes in {len(grids)} grids, {len(frets) - frets.count(None)} numbers, "
          f"{frets.count(None)} not a number, {sum(not r['valid'] for r in model.values())} without a valid response")
    print(f"[read] templates for digits {''.join(sorted(tpl.templates))}, {len(tpl.outliers)} outlier crops; "
          f"template readings for {sum(t['fret'] is not None for t in template.values())} notes")
    print(f"[read] {count['verified'] + count['verified_reread']} of {len(notes)} verified "
          f"({count['verified_reread']} after the re-read of {len(reread_ids)}), {count['flagged']} flagged, "
          f"{count['ignored']} ignored")
    print(f"[read] {len(cost['calls'])} calls ({new} new): {cost['input_tokens']} input and "
          f"{cost['output_tokens']} output tokens (thinking included), ${cost['new_usd']:.4f} spent this run")
