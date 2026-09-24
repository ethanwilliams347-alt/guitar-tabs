# video_to_sheet

Turns YouTube guitar tab videos into printable PDF tab sheets. [PLAN.md](PLAN.md) has the design and [SUPPORTED_VIDEOS.md](SUPPORTED_VIDEOS.md) says which videos work.

## What counts as correct

Only two things matter:
1. Every fret number is on the right string.
2. Horizontal spacing matches the video, scaled by one factor for the whole song.

Rhythm, techniques, notation, tuning, tempo and MusicXML are out of scope. Don't add them, and judge every change by its effect on these two things.

## How the work is split

- **Image processing decides where things are:** rows, string y-positions, bar lines, blob positions. It is deterministic and unit-tested on synthetic images.
- **The model and a local template reader decide what a note says:** a fret number from 0 to 24, or "not a number". Neither ever outputs or changes positions.
- **Flag, don't guess.** If the two readers disagree, the number is drawn in red and listed in the report. Never quietly fill in a value.

## Rules for model calls

- Models: the Gemini API, through the `google-genai` SDK. `gemini-3.8-flash` reads every note, then `gemini-3.1-pro-preview` re-reads only the notes where the model and the local template reader disagree. Keep model IDs in one config constant, not scattered through the code.
  - **Temporary:** the reader uses `gemini-3.5-flash` because `gemini-3.8-flash` has been overloaded, rejecting every request with a 503. Switch back to `gemini-3.8-flash` once it answers reliably. PLAN.md (stage 5, "Temporary model choice") says how to check and what to rerun.
- Use structured output (a JSON schema or tool use) and validate every response against it. Reject any response with an extra, missing or duplicate blob ID.
- Don't set temperature: leave it at the model's default, and rely on the call cache for repeatable results. Set the thinking level from a config constant. The prompt text and schema live in versioned files under `src/prompts/`.
- Cache every call in `cache/<video id>/`, keyed by a hash of the image, prompt version, model and schema. A rerun must never repeat a call that already succeeded.
- Log tokens and cost per video. Print the total at the end of each run.
- Tests never call the API. They use recorded responses in `tests/fixtures/`. Re-record fixtures only on purpose, and say so in the change.
- Passing the checks is not the same as being correct. Don't call a number "verified" unless an independent signal agrees, for example a second read or a template match.

## Evaluation comes first

- Before tuning a threshold, a prompt or a model choice, measure it on the test set against the hand-checked `tab.json` ground truth.
- Report these metrics: fret accuracy (right fret on the right string), note recall and precision, x-position error as a fraction of row width, flag rate, and how many flags were real errors.
- Any change to a prompt, model or threshold needs before and after numbers. Don't merge a regression on any metric without saying so.
- Test videos are stored locally, not in git. Ground truth and metrics results are committed.

## One video at a time

- Work targets one video at a time (the current iteration in PLAN.md). Build only what that video needs.
- Never add per-video code paths. A video's only special inputs are CLI flags saved in `tests/truth/<video id>.args`.
- Every video that already passes is a regression test. Before a change lands, rerun `eval.py` on all of them. If the new video's fix breaks an old one, look for a more general rule instead of a special case.

## Code conventions

- Python. Stages match PLAN.md: `src/ingest.py`, `canvas.py`, `rows.py`, `notes.py`, `read.py`, `render.py`. Each stage reads and writes files in `cache/<video id>/`, so any stage can be rerun on its own.
- Thresholds are named constants in `src/config.py`, with their units in the name (`PAGE_MIN_STABLE_S`, `BARLINE_MERGE_FRAC`). No magic numbers inside functions.
- `--debug` must produce an overlay image for every decision a stage makes. When you add a decision, add its overlay too.
- Refuse videos you can't handle, and give the reason. Don't produce a doubtful PDF.

## Keep SUPPORTED_VIDEOS.md current

Update [SUPPORTED_VIDEOS.md](SUPPORTED_VIDEOS.md) in the same change whenever any of these change:
- a pipeline stage is implemented or changed
- a threshold changes (page stability time, line counts, digit-size warning, overlap length, and so on)
- a video type gains or loses support
- a real test video is run. Add its result to the "Tested videos" table.

Also update the Status and Last updated lines at the top. Move items between sections as testing confirms them or rules them out.

## Keep PLAN.md honest

If the implementation differs from PLAN.md, update the plan in the same change. Don't leave the design and the code disagreeing.
