# video_to_sheet — Plan

## Goal

Give the tool a guitar tab video. It produces a PDF tab sheet in which:
- every fret number is on the right string, and
- the horizontal spacing of the numbers matches the video, scaled by one factor for the whole song.

Nothing else matters. Rhythm, techniques, notation, tuning, tempo and MusicXML are out of scope.

Three principles:
- **Image processing decides where, the readers decide what.** Image processing measures where each number sits. A model and a local template reader each say which number it is. The model never outputs positions.
- **Verified means two independent readers agree.** Passing sanity checks only shows a reading is possible, not that it is correct.
- **Flag, don't guess.** A number that isn't verified is drawn in red and listed in the report. So is any automatic decision that could delete or change content.

## Strategy: one video at a time

The tool is built by getting a complete, correct tab sheet out of one real video, then out of the next, and so on. Each video is one **iteration**. The aim is for the code to converge on a general solution.

**Rules for every iteration:**
- **One codebase.** Every iteration extends the same `src/` pipeline. No per-video scripts, no `if video_id == ...` code paths.
- **Build only what the current video needs.** New behaviour is designed when a video needs it, not before. Building a part includes its config constants, its `--debug` overlay and its unit test.
- **Per-video inputs are CLI flags only,** such as `--crop` and `--layout`, which any user could supply. They are saved as one line in `tests/truth/<video id>.args`, so `eval.py` runs every video the same way.
- **Earlier videos are the regression set.** Once a video passes, it stays in the `eval.py` set. A change can land only if every earlier video still meets the pass bar. If a fix for the new video breaks an old one, the answer is a more general rule, not a special case.
- **Refuse what isn't handled.** A video that breaks an assumption in this plan is refused with the reason. It is never turned into a doubtful PDF. SUPPORTED_VIDEOS.md says what is refused.
- **Let each new video differ in one main way** from the ones that already pass, where possible. A failure then points at one cause.

**Steps of an iteration:**
1. Pick the video and add its URL to `tests/videos.md`. Examine it, and record what it needs under "Current iteration".
2. Design the new parts in this plan, then build them with unit tests on synthetic images.
3. Run it end to end with `--debug`, and check every overlay against the video.
4. Correct the first `tab.json` by hand into ground truth.
5. Measure with `eval.py` and tune until the video meets the pass bar. After every change, rerun all earlier videos.
6. Record the result in SUPPORTED_VIDEOS.md and the Iterations table, and make this plan match the code.

An iteration is done when its video meets the pass bar and every earlier video still does. If it can't meet the bar, record why, decide what the tool should refuse, and move on.

## Current iteration: 1, `mhmDGhkUZt4`

URL: https://www.youtube.com/watch?v=mhmDGhkUZt4. 61 s at 29.97 fps, downloaded at 1920×1080 in H.264.

**What's on screen** (measured on the downloaded file):
- A white strip across the top 484 px of the frame holds, from top to bottom:
  - chord diagrams
  - a notation staff (5 lines, y ≈ 225–280)
  - one tab row (6 lines at y ≈ 375, 393, 410, 428, 446 and 463, so line spacing `s` ≈ 17.7 px)

  Below the strip is live camera video of the guitarist, which changes in every frame.
- **Layout: paged strip.** The song is one long line of tab, shown one screen at a time on 3 pages: 0–17.3 s, 17.45–36.9 s and 37.0–61 s. Pages change with a crossfade of about 0.15 s. Every page draws the music at the same scale and height. Each page starts a little before the previous one ended: page 2 is page 1 shifted left by 1692 px (228 px of overlap), and page 3 is page 2 shifted left by 1618 px (302 px of overlap). Where the music continues past an edge, the outer 80–100 px of the tab fades out.
- An orange-red playhead line sweeps across each page, and the notes under it turn red while they play. Within a page, the median change per frame in the strip is 0.3% of pixels, and it never reaches 2%. A median over a page's frames removes both the playhead and the red highlights.
- Frets from 0 to 13, including two-digit ones. Chords of up to 5 notes.
- Arpeggio marks: a wavy vertical line with an arrowhead, crossing several strings.
- The first page starts with a bracket, a clef and a "TAB" label, with no bar line at the left. The last page ends with a final double bar line.

**Run with:** `--crop 0,0,1920,484 --layout paged-strip` (in `tests/truth/mhmDGhkUZt4.args`). The crop keeps the camera video out.

**Refused in this iteration** (each with a reason): any `--layout` other than `paged-strip`, a dark background, and anything other than exactly one tab row per page.

## Pipeline

```
ingest → canvas → rows → notes → read → render
  1        2        3      4       5       6
```

- Each stage is one module in `src/` and one directory in `cache/<video id>/<stage>/`. Each stage can be rerun on its own with `python main.py <input> --stage <name>`.
- A stage's cache key is a hash of its input files, the config values it uses and its `STAGE_VERSION`. A changed key reruns that stage and everything after it.
- Model calls have their own cache (stage 5), so a successful call is never repeated.
- All images are grayscale from stage 2 on.

### 1. Ingest: `src/ingest.py`
- **Input:** a YouTube URL or a local file. The video ID is the YouTube ID, or the file name without its extension.
- **Download:** `yt-dlp`, video only, the best H.264 stream up to 1080p (`bv*[vcodec^=avc1][height<=1080]`), saved to `videos/<video id>.mp4`. H.264 is chosen because OpenCV can decode it without ffmpeg.
- **Sampling:** take the frame nearest each multiple of 1/`SAMPLE_FPS` s, crop it to `--crop`, and convert it to grayscale. Only a copy downscaled to `MOTION_WIDTH_PX` wide is stored. Stage 2 rereads the full-resolution frames it needs straight from the video.
- **Writes:** `meta.json` (source, fps, frame count, crop, sample times) and `motion.npy` (the downscaled frames).
- **Refuses:** a video that can't be decoded, or a crop outside the frame.

### 2. Canvas: `src/canvas.py`
**Pages.**
- `change(t)` is the fraction of pixels whose absolute difference from the previous sample is over `DIFF_PIXEL_LEVEL`.
- A page is a maximal run of samples in which:
  - `change(t) < STILL_FRAME_FRAC` for every sample, and
  - every sample differs from the run's first sample in fewer than `STILL_DRIFT_FRAC` of pixels. This catches slow crossfades.
- A run shorter than `PAGE_MIN_STABLE_S` isn't a page. Trim `PAGE_TRIM_S` from each end.
- The page image is the pixel-wise median of up to `MEDIAN_MAX_FRAMES` evenly spaced full-resolution frames from the trimmed run. A page with fewer than `MEDIAN_MIN_FRAMES` is refused.
- Two consecutive pages whose images match (NCC ≥ `PAGE_MERGE_NCC`) are merged.

**Checks on each page:**
- The median brightness must be at least `LIGHT_BG_MIN_LEVEL`, since dark backgrounds are refused for now.
- Stage 3's row finder must find exactly one tab row.
- The row's six line y-positions must match page 1's within `LINE_Y_MATCH_PX`.

**Stitching** consecutive pages n and n+1:
1. **Offset:** search the horizontal shifts at which the right part of page n lies on the left part of page n+1. Each shift must leave an overlap of at least `STRIP_MIN_OVERLAP_FRAC` of the page width. Score each shift by NCC over the tab-row band (top line − `s` to bottom line + `s`), skipping `STRIP_EDGE_SKIP_FRAC` of the width at each page edge, where the tab fades.
2. **Accept or refuse:** the best shift must score at least `STRIP_NCC_MIN`. It must also beat the best shift more than `STRIP_RUNNER_UP_MIN_DIST_PX` away by `STRIP_NCC_MARGIN`. Bar lines in the overlap, found on each page by stage 3's bar-line finder, must line up within `STRIP_MAX_RESIDUAL_PX`. If any check fails, the video is refused with the scores.
3. **Composite:** place the pages on one canvas. Where two pages overlap, each pixel is the darker of the two. Both pages show the same music there, and the faded copy is always the lighter one, so the clear copy is kept and overlapping music appears once.

**Writes:**
- `page_<n>.png`
- `canvas.png`
- `pages.json`: each page's times, frames used and canvas offset, plus each pair's offset, overlap, best and runner-up NCC, and bar-line residual

**Debug overlays:** the change plot with pages marked, each page image, the NCC-by-shift plot for each pair, and the overlaps with bar lines from both pages drawn in two colours.

### 3. Rows: `src/rows.py`
Stage 2 also calls these functions on single pages.
- **Lines:** binarize with Otsu. Open with a horizontal kernel `LINE_MIN_LEN_FRAC` of the image width long, then find line y-positions from the horizontal projection. Each line's y is the centroid of its thickness.
- **Row:** a group of exactly 6 evenly spaced lines, with spacing within ±`LINE_SPACING_TOL_FRAC`. Other line groups are ignored, such as the 5-line notation staff and chord-diagram grids. The row stores its six line y-positions and its spacing `s`. The paged-strip layout needs exactly one row.
- **Extent:** the row's left and right ends are the first and last columns where all six lines are present.
- **Bar lines:** columns where ink covers at least `BARLINE_MIN_COVER_FRAC` of the span from the top line to the bottom line. Columns within `BARLINE_MERGE_FRAC × s` of each other are merged into one bar line at the group's centre, so a double bar line becomes one. A wavy arpeggio line is not straight, so it never covers enough of one column.
- **Origin:** the first bar line if it lies within `s` of the left end of the row, or else the left end itself. Every x-position from here on is measured from the origin.
- **Writes:** `rows.json` (line y-positions, `s`, extent, origin, bar-line x-positions).
- **Debug overlay:** lines, extent, origin and bar lines drawn over the canvas.

### 4. Notes: `src/notes.py`
- **Line-free image:** clear each string line (its y ± half its thickness). Keep columns where the pixels just above and below the line are both ink, because a digit crosses the line there. Clear bar lines the same way.
- **Marks:** connected components of the line-free image. A mark is kept if:
  - its height is between `MARK_MIN_H_FRAC × s` and `MARK_MAX_H_FRAC × s`, and
  - its centre is within `MARK_MAX_DY_FRAC × s` of a string line.

  It is assigned to that string. Marks that are too tall (arpeggio lines) or lie between strings are dropped. Leftovers such as the "TAB" letters and arrowheads are kept here, and stage 5 reads them as "not a number".
- **Joining:** on the same string, marks whose vertical extents overlap by at least `DIGIT_JOIN_OVERLAP_FRAC` and whose horizontal gap is under `DIGIT_JOIN_GAP_FRAC × w` are one note. Here `w` is the median mark width in the video. Each note keeps its digit boxes.
- **Position:** the centre of the note's bounding box, relative to the origin, in canvas pixels. This is the position used in the PDF.
- **Writes:** `notes.json` (ID, string with 1 = high e, x, bounding box, digit boxes) and the line-free image.
- **Debug overlay:** every kept mark boxed and coloured by string, dropped marks in grey, joins drawn as brackets.

### 5. Read: `src/read.py`
**Model reader.**
- Crop each note with `CROP_PAD_FRAC × s` of padding from the canvas (not the line-free image). Scale each crop to `GRID_CELL_H_PX` high.
- Lay out up to `GRID_MAX_CELLS` crops per image in a grid. Each cell's ID is printed in a margin outside the crop, so it can't be mistaken for a digit.
- Send each grid to `MODELS["read"]` (`claude-sonnet-5`) with the prompt `src/prompts/read_v<N>.md`. Request structured output that follows the schema `src/prompts/read_v<N>.schema.json`: one entry per ID, `{"id": int, "fret": 0–24 or null}`, where `null` means "not a number".
- **Validate every response:** the schema, plus the ID set, which must be exactly the grid's IDs with none missing, extra or duplicated. An invalid response is requested once more. If it fails again, every note in that grid is flagged.
- **Settings:** effort `READ_EFFORT["read"]`. No temperature: these models reject it. Repeatable results come from the cache.
- **Refusal:** a refusal (`stop_reason == "refusal"`) is treated like an invalid response. There is no fallback to another model, because that would silently change which model read the notes.
- **Cache:** `cache/<video id>/read/calls/<key>.json`, where the key is a hash of the grid PNG bytes, prompt version, schema version, model ID and effort.
- **Cost:** token counts from each response's `usage`, priced with `PRICES_USD_PER_MTOK`, logged per call in `cost.json` and printed as a total at the end of the run.

**Template reader** (local, no API calls).
- For each digit 0–9, the template is the pixel-wise median of the single-digit crops the model read as that digit. Each crop is binarized and scaled to `TEMPLATE_SIZE_PX`. A template needs at least `TEMPLATE_MIN_SAMPLES` crops.
- **Leave one out:** a crop used in a template is compared against a copy of that template rebuilt without it.
- Crops whose NCC against their own class is below `TEMPLATE_CLASS_MIN_NCC` are removed from the template and flagged. This limits how far a systematic model error can spread.
- Each digit box is read as the best-matching template. The best match must score at least `TEMPLATE_MATCH_MIN_NCC` and beat the second best by `TEMPLATE_MARGIN_NCC`. A multi-digit note is read digit by digit.

**Decision per note.**
- **Verified:** both readers give the same fret, and the number of digits matches the digit boxes.
- **Disagreement, or no template result:** all such notes are re-read in one grid by `MODELS["reread"]` (`claude-opus-5`), with the same schema, validation and cache, at effort `READ_EFFORT["reread"]`. The note is verified if Opus agrees with the template reader. Otherwise it is flagged and shows Opus's reading, or the template's if Opus returned `null`.
- **Model `null` and no template match:** dropped, and counted in the report as "N marks ignored".

**Writes:** `read.json` (each reader's result, the final fret and the status of every note), `cost.json` and the grid images.

**Debug overlays:** the grids and the template sheet.

### 6. Render: `src/render.py`
- **`tab.json`:** the video ID, the tab-area width, one canvas row (origin, extent, `s`, bar lines), and its notes (ID, string, x, fret, status, both readers' results). Every PDF and metric is computed from this file. Its schema is `src/schema/tab.schema.json`.
- **Scale:** one factor for the whole song. The tab-area width in the video (1920 px) maps to the printable width.
- **Rows:** the canvas is cut into PDF rows at bar lines. Each cut is at the last bar line that still fits the printable width. A single measure wider than the printable width is split at a gap between notes and flagged.
- **Drawing** with `reportlab`: six lines, bar lines, and each number centred on its string at its x-position, over a white box that blanks out the line. Flagged numbers are red. The paper size is Letter, or A4 with `--paper a4`.
- **Report (`report.html`):**
  - the headline "X of Y numbers verified", numbers flagged, marks ignored and total cost
  - every automatic decision: page times, stitch offsets and scores, and any refusal
  - for every PDF row with a flag, the video image with its notes boxed, next to its redrawn version
- **Writes:** `out/<video id>.pdf` (or `-o`), `tab.json` and `report.html`.

## Configuration: `src/config.py`

All thresholds are named constants with their unit in the name: `_S` is seconds, `_PX` pixels, `_LEVEL` a 0–255 gray level, `_FRAC` a fraction (of what is given below), `_NCC` a correlation. The values are starting points, and they change only with `eval.py` numbers from before and after.

| Constant | Start | Unit / meaning |
| --- | --- | --- |
| `SAMPLE_FPS` | 12 | samples per second |
| `MOTION_WIDTH_PX` | 480 | width of the change-measure copy |
| `DIFF_PIXEL_LEVEL` | 25 | gray levels |
| `STILL_FRAME_FRAC` | 0.02 | of tab-area pixels |
| `STILL_DRIFT_FRAC` | 0.03 | of tab-area pixels |
| `PAGE_MIN_STABLE_S` | 1.0 | s |
| `PAGE_TRIM_S` | 0.25 | s, from each end |
| `MEDIAN_MAX_FRAMES` / `MEDIAN_MIN_FRAMES` | 31 / 5 | frames |
| `PAGE_MERGE_NCC` | 0.98 | |
| `LIGHT_BG_MIN_LEVEL` | 128 | median gray level |
| `LINE_MIN_LEN_FRAC` | 0.5 | of image width |
| `LINE_SPACING_TOL_FRAC` | 0.15 | of mean spacing |
| `LINE_Y_MATCH_PX` | 1 | px between pages |
| `STRIP_MIN_OVERLAP_FRAC` | 0.08 | of page width |
| `STRIP_EDGE_SKIP_FRAC` | 0.05 | of page width |
| `STRIP_NCC_MIN` / `STRIP_NCC_MARGIN` | 0.8 / 0.05 | |
| `STRIP_RUNNER_UP_MIN_DIST_PX` | 5 | px |
| `STRIP_MAX_RESIDUAL_PX` | 2 | px |
| `BARLINE_MIN_COVER_FRAC` | 0.9 | of top-to-bottom line span |
| `BARLINE_MERGE_FRAC` | 0.5 | of `s` |
| `MARK_MIN_H_FRAC` / `MARK_MAX_H_FRAC` | 0.4 / 1.2 | of `s` |
| `MARK_MAX_DY_FRAC` | 0.4 | of `s` |
| `DIGIT_JOIN_OVERLAP_FRAC` | 0.7 | of the smaller mark's height |
| `DIGIT_JOIN_GAP_FRAC` | 0.35 | of median mark width |
| `CROP_PAD_FRAC` | 0.3 | of `s` |
| `GRID_MAX_CELLS` / `GRID_CELL_H_PX` | 40 / 64 | |
| `TEMPLATE_SIZE_PX` / `TEMPLATE_MIN_SAMPLES` | 32 / 3 | |
| `TEMPLATE_CLASS_MIN_NCC` / `TEMPLATE_MATCH_MIN_NCC` / `TEMPLATE_MARGIN_NCC` | 0.8 / 0.85 / 0.05 | |
| `MATCH_X_TOL_FRAC` | 0.01 | of tab-area width (eval matching) |
| `MODELS` | `{"read": "claude-sonnet-5", "reread": "claude-opus-5"}` | the only place model IDs appear |
| `READ_EFFORT` | `{"read": "medium", "reread": "high"}` | |
| `PRICES_USD_PER_MTOK` | Sonnet 5: 2 in / 10 out; Opus 5: 5 in / 25 out | |

## Evaluation: `eval.py`

- **Ground truth:** `tests/truth/<video id>.json`, in the `tab.json` format but holding only string, x and fret per note.
- **Matching:** a predicted note matches a true note on the same string within `MATCH_X_TOL_FRAC` of the tab-area width (19 px here, under half the closest note spacing). Pairs are matched one to one, nearest first.
- **Metrics per video:**

  | Metric | Target |
  | --- | --- |
  | Note recall (true notes found) | ≥ 99% |
  | Note precision (found notes that are real) | ≥ 99.5% |
  | Fret accuracy on matched notes | ≥ 99% |
  | Unflagged errors (wrong fret, missing or extra, and not red) | ≤ 0.2% of notes |
  | x error, median and maximum, as a fraction of tab-area width | ≤ 0.25% and ≤ 0.5% |
  | Flag rate | ≤ 5% |
  | Flag precision (flags that were real errors) | reported, no target |

  Unflagged errors is the most important number, because it counts the errors a user won't catch. Every video must meet every target on its own.
- **Output:** a table per video, and `tests/results/<date>.json`, which is committed.
- **Review images:** `python eval.py --review <video id>` writes `cache/<video id>/review/part_<n>.png`. Each image is one tab-area width of the canvas, with the predicted numbers drawn under it at their x-positions, flagged ones in red and each with its note ID. If there is no truth file yet, it also writes a draft from `tab.json`.
- **Correcting the truth:** every note is checked against the review image, not only the flagged ones, because a draft makes it easy to accept what is already there. Claude may propose corrections from the review images, but Ethan confirms the file before it is committed.
- **Limits of the x metric:** the true x-positions start as the pipeline's own, so on iteration 1 the x metric catches only positions a person moved by eye. Exact positions are proven by the unit tests on synthetic images. In later iterations, the x metric catches regressions.

## Build steps for iteration 1

Each step ends with its checkpoint met and its unit tests passing.

1. **Setup:** `requirements.txt`, the `.venv`, the `src/` package, `config.py`, `cache.py` (stage directories and keys) and a `main.py` CLI with `--stage`, `--debug`, `-o` and `--paper`. `pytest` runs.
2. **Evaluation:** `src/schema/tab.schema.json` and `eval.py` (matching, metrics, `--review`), tested on hand-written JSON.
   - Checkpoint: the metrics are right on constructed cases, including a missing note, an extra note, a wrong fret and a wrong string.
3. **Ingest.**
   - Checkpoint: 732 samples of 484 × 1920, `meta.json` correct.
4. **Rows on a single page:** line and row finding, extent, bar lines, origin.
   - Checkpoint: on each page image, 6 lines at y ≈ 375–463, `s` ≈ 17.7, the staff and chord diagrams ignored, and the bar-line count equals a count by eye.
5. **Pages.**
   - Checkpoint: exactly 3 pages at ≈ 0–17.3, 17.45–36.9 and 37.0–61 s, and page images with no playhead or red highlights left.
6. **Stitching.**
   - Checkpoint: offsets 1692 ± 1 and 1618 ± 1, bar-line residual ≤ 2 px, a canvas about 5230 px wide, and no doubled or faded notes in the overlaps (checked in the overlay).
7. **Rows on the canvas.**
   - Checkpoint: one row, the origin at the left end of the lines, and every bar line found once, including the final double bar.
8. **Notes.**
   - Checkpoint: in the overlay, no note missed or split in 3 measures checked by eye (including one with two-digit frets and one chord), and arpeggio lines dropped.
9. **Model reader:** prompt and schema v1, grids, validation, the call cache, the cost log, and fixtures recorded with `--record-fixtures`.
   - Checkpoint: a second run makes no API calls, and the cost is printed. The expected cost is under $1 per video.
10. **Template reader and decision,** including the Opus re-read.
    - Checkpoint: the template sheet looks right, and every note has a status.
11. **Render and report.**
    - Checkpoint: the PDF rows start at bar lines, the numbers sit on the right lines, and the report lists the stitch decisions.
12. **Ground truth:** `eval.py --review`, then correct and confirm `tests/truth/mhmDGhkUZt4.json`.
13. **Measure and tune** until the pass bar is met. Any change to a prompt, model or threshold is recorded with its before and after numbers.
14. **Record:** SUPPORTED_VIDEOS.md, the Iterations table, and this plan brought in line with the code.

## Unit tests

Tests use synthetic images and never call the API. Model responses come from recorded files in `tests/fixtures/`, which are re-recorded only on purpose and noted in the change.
- **Pages:** three still pages joined by 0.15 s crossfades, with a thin red line sweeping each page, give 3 pages, and their medians contain no red line. A slow drift is not taken for a page.
- **Stitching:**
  - A long synthetic strip cut into 3 overlapping windows with faded edges gives the exact offsets, and a composite equal to the strip within 1 gray level away from the fades.
  - An overlap that is too small is refused.
  - A strip with a repeating pattern is refused on the runner-up margin.
- **Rows:** a 6-line row is found next to a 5-line staff and chord-diagram grids, with exact y-positions.
- **Bar lines:** they are found, a wavy line is not, a double bar merges into one, and the origin falls on the left end when there is no bar line there.
- **Notes:**
  - Line erasure keeps digits that cross a line.
  - "10" and "12" are joined, but chord notes on neighbouring strings are not.
  - x-positions are exact.
- **Reader:**
  - Responses with a missing, extra or duplicate ID, or a fret outside 0–24, are rejected.
  - The cache key changes with the prompt version, model or effort.
  - The template reader uses leave one out.
- **Render:** a canvas x maps to a PDF x by the one scale factor, and rows are cut at bar lines.
- **Eval:** matching and metrics on constructed cases.

## Project layout

```
main.py  eval.py  requirements.txt
src/  config.py cache.py ingest.py canvas.py rows.py notes.py read.py render.py
      prompts/read_v1.md  prompts/read_v1.schema.json
      schema/tab.schema.json
tests/  test_*.py  videos.md  fixtures/  truth/<id>.json  truth/<id>.args  results/
videos/  cache/  out/  .venv/        (git-ignored)
```

## Dependencies

- `yt-dlp`, `opencv-python`, `numpy`, `anthropic`, `jsonschema`, `reportlab`, `pytest`. ffmpeg isn't needed.
- The API needs `ANTHROPIC_API_KEY`, or an `ant auth login` profile.

## Iterations

| # | Video | What it needed that was new | Result |
| --- | --- | --- | --- |
| 1 | `mhmDGhkUZt4` | Everything above; paged-strip stitching; tab strip above camera video | in progress |

## Ideas for later iterations

These are not designed yet. Each gets designed properly when a video needs it.
- **Paginated pages with several rows each:** one PDF row per video row. If a page repeats the previous page's last measures, match them by measure, and use where the playhead was active to tell an overlap from a genuine repeat.
- **Horizontal scrolling:** one canvas, with each frame's offset fitted to bar lines whose canvas positions are fixed, so drift doesn't build up.
- **Playhead masks:** needed if a playhead pushes change within a page over `STILL_FRAME_FRAC`.
- **Dark backgrounds:** flip the binarization, and take the lighter pixel when compositing.
- **Automatic tab area and layout:** once several videos pass, checked against every video's saved `.args`.
