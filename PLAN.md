# video_to_sheet — Plan

## Goal

Give the tool a URL or file of a guitar tab video. It produces a PDF tab sheet in which:
- every fret number is on the right string, and
- the horizontal spacing of notes and chords is exactly as in the video, scaled by one factor for the whole song.

Nothing else matters. The tool doesn't output rhythm, techniques, notation, tuning, tempo or MusicXML.

Three principles:
- **Image processing decides where, the model decides what.** Image processing measures where each number sits. A model and a local digit matcher each read which number it is. The PDF draws that number at the measured place. The model never outputs positions.
- **Verified means two independent readers agree.** Passing sanity checks only shows a reading is possible, not that it is correct.
- **Flag, don't guess.** Any number that isn't verified is drawn in red and listed in the report. Any automatic decision that could delete or change content, such as removing a page overlap, is listed in the report too.

## Pipeline

```
input → canvases → rows and bar lines → notes → read → PDF + report
  1        2              3              4       5        6
```

Every stage reads and writes files in `cache/<video id>/<stage>/`. Each stage's cache key is a hash of its input key, the config values it uses and its code version (`STAGE_VERSION`). Changing a threshold therefore reruns that stage and every stage after it. Model calls have their own cache (stage 5), so a rerun never repeats a call that already succeeded.

All thresholds live in `src/config.py` as named constants, with their units in the name. The values below are starting points. Tune them only against the test set (see Testing).

### 1. Input — `src/ingest.py`
- Download with `yt-dlp` at the highest available resolution, or accept a local file. Video only, no audio.
- Decode at `SAMPLE_FPS = 12`. Keep full-resolution frames cropped to the tab area. Also keep a copy downscaled to 480 px wide for motion analysis.
- **Tab area:** `--crop x,y,w,h`, or (build step 3) one model call on 5 evenly spaced frames that returns the tab-area rectangle as JSON. Either way, stage 3's line detection then snaps it to the outermost tab lines plus a margin of 2 line spacings.
- **Layout (build step 3):** classified from motion inside the tab area.
  - *Paginated:* long still stretches separated by short changes.
  - *Horizontal scrolling:* a steady sideways shift, measured by phase correlation on a mask with the tab lines removed. The tab lines look the same at every shift, so they can't be used.
  - Anything else is refused, and the refusal gives the reason (for example "vertical motion detected", or "no still stretch of 1 s or longer").
  - Until step 3, the layout is given with `--layout paginated|scroll`.

### 2. Canvases — `src/canvas.py`
A canvas is one clean, still image of tab.

**Change measure (paginated).** For each frame, compute the absolute difference from the previous frame on the downscaled tab area, and threshold it at `DIFF_PIXEL_THRESH = 25` of 255. Before counting, split the changed pixels into two masks:
- **Movers:** thin vertical things such as playheads and cursors. Found by morphological opening with a vertical kernel `MOVER_KERNEL_H_FRAC = 0.5` of the tab-area height tall, and at most `MOVER_MAX_W_FRAC = 0.02` of the tab-area width wide. They are recorded in an **activity map** (for each frame, where movers are) and don't count as change.
- **Content change:** everything else. `change(t)` is the fraction of tab-area pixels in this mask.

**Paginated pages.** A page is a maximal run of frames that meets both of these conditions:
- `change(t) < STILL_FRAME_FRAC = 0.02` for every frame in the run, and
- the content difference between each frame and the run's first frame stays under `STILL_DRIFT_FRAC = 0.03`. This catches slow crossfades, which change very little from one frame to the next.

A run counts as a page if it lasts at least `PAGE_MIN_STABLE_S = 1.0` s. Trim `PAGE_TRIM_S = 0.25` s from each end, which leaves at least 0.5 s, or 6 frames at 12 fps. The canvas is the pixel-wise median of the trimmed frames at full resolution: all of them, or up to `MEDIAN_MAX_FRAMES = 31` evenly spaced. A page with fewer than `MEDIAN_MIN_FRAMES = 5` trimmed frames is refused. The median removes anything that covers a pixel in less than half of the trimmed frames. Two consecutive runs whose medians match (NCC > 0.98) are merged into one page, since a pause in a crossfade can split one page in two.

**Horizontal scrolling.** The whole song becomes one wide canvas.
1. Coarse shift between consecutive frames: phase correlation on the downscaled mask with lines and movers removed.
2. Bar lines are the anchors. Each bar line gets one fixed canvas x-position the first time it is seen, at the frame where it is closest to the middle of the view. In every later frame, the frame's offset is a least-squares fit of the bar lines it can see to their known canvas positions. So an error doesn't carry over from one frame to the next, and drift stays bounded by the bar-line measurement error rather than growing with song length.
3. Each canvas column is the median of every frame that covered it.
4. Abort with a reason if the fit residual exceeds `SCROLL_MAX_RESIDUAL_PX = 2` at full resolution. It is also listed in the report.

### 3. Rows and bar lines — `src/rows.py`
- **Rows:** binarize with Otsu, flipping it on dark backgrounds, which are detected from the median brightness. Find horizontal lines with a horizontal projection after opening with a long horizontal kernel (`LINE_MIN_LEN_FRAC = 0.5` of the canvas width). Group them into rows of exactly 6 evenly spaced lines, allowing a spacing variation of `LINE_SPACING_TOL = 0.15`. Keep each line's exact y-position and the row's line spacing `s`. Lines that don't fit into a row, such as notation staves, are ignored, and so is everything more than `s` above the top line or below the bottom line.
- **Row extent:** the row's left and right ends are where the six lines start and stop.
- **Bar lines:** vertical runs that cover at least 90% of the span from the top line to the bottom line. Runs closer together than `BARLINE_MERGE_FRAC = 0.5 × s` are merged into one bar line, so double and repeat bar lines don't create empty measures. A merged bar line's x-position is the centre of the group.
- **Row origin:** the x-position of the row's first bar line. If the row has no bar line at its left end, the origin is the left end of the six lines. Every x-position in this row is measured from the origin. This removes the indent that a "TAB" label or a first-row clef adds.
- **Page overlap (paginated only):** some videos repeat the last few measures of one page at the start of the next. Compare the last k measures of page n with the first k of page n+1, for k from 4 down to 1. Scale each pair of measure images to the same height, and skip any pair whose widths differ by more than 3%. Score the binarized pair with normalized cross-correlation. A match needs every pair above `OVERLAP_NCC_MIN = 0.9`. Take the largest k that matches.

  An image match alone can't tell an overlap from music that really repeats, so use the activity map from stage 2 as well. A measure is *played* on a page if movers appear inside it during that page.
  - If the k measures are played on only one of the two pages, drop them from the page where they aren't played. Report it as "overlap removed" with the measures involved.
  - If they are played on both pages, keep both, since the music really repeats.
  - If there is no activity signal, for example on a static page with no playhead, keep both, flag the row, and report it as "possible duplicate, kept".

### 4. Notes — `src/notes.py`
- **Line-free image:** in each row, erase the six lines by clearing the rows of pixels at each line's y-position (± half the line thickness). A column is kept where the pixels directly above and below the line are ink, because that means a digit crosses the line there. Erase bar lines the same way.
- **Marks:** connected components in the line-free image. A mark is kept if its height is between `0.4 × s` and `1.2 × s` and its centre is within `0.4 × s` of a string line. Taller marks (stems, text, leftover bar lines) and marks between strings are dropped. Each mark is assigned to its nearest string.
- **Joining multi-digit numbers:** on the same string, marks whose vertical extents overlap by at least 70% and whose horizontal gap is under `DIGIT_JOIN_GAP_FRAC = 0.35 × w` are joined into one note. Here `w` is the median width of all marks in the video. The note keeps the list of its digit boxes, used in stage 5.
- **Position:** each note's x-position is the centre of its bounding box, relative to the row origin, in canvas pixels. This is the position the PDF uses.

### 5. Read — `src/read.py`
Two independent readers classify every note. Only agreement counts as verified.

**Model reader.**
- Crop each note with a padding of `0.3 × s`, and scale all crops to the same height. Arrange up to `GRID_MAX_CELLS = 40` crops per request in a grid. Each cell has its ID printed **outside** the crop, in a margin, so the label can't be mistaken for a digit.
- Model: `claude-sonnet-5`, temperature 0, with structured output that follows `src/prompts/read_v<N>.json`. The response is one entry per ID: `{"id": int, "fret": int 0–24 | null}`, where `null` means "not a number" (letters, symbols, noise). Notes read as `null` are dropped.
- A response is rejected, and requested again once, if it has a missing, extra or duplicate ID, or a fret outside 0–24.
- The cache key is the hash of the grid image + prompt version + model ID + schema version.

**Template reader (local, free).**
- Build per-video digit templates from single-digit notes: for each digit 0–9, the pixel-wise median of the crops the model read as that digit. Each crop is scaled to a fixed size and binarized. A template needs at least `TEMPLATE_MIN_SAMPLES = 3` crops. When classifying a crop that was used to build a template, that template is rebuilt without it (leave one out), so a crop is never matched against itself.
- Crops whose NCC against their own class template is below `TEMPLATE_CLASS_MIN = 0.8` are removed from the template and flagged. This limits how much a systematic model error can poison a template.
- Classify each digit box of every note by the best NCC over the templates. It must score at least `TEMPLATE_MATCH_MIN = 0.85` and beat the second best by `TEMPLATE_MARGIN = 0.05`. A multi-digit note is read by reading its digits in order.

**Decision per note.**
- **Verified:** both readers give the same fret, and the digit count matches the number of digit boxes.
- **Disagreement or no template result:** re-read with `claude-opus-5`, in a grid of just the disputed notes and the same schema. It is verified if Opus agrees with the template reader. Otherwise the note is flagged, and the most likely value is shown: Opus's reading, or the template's if Opus returned `null`.
- **Model `null` and no template match:** dropped, and counted in the report ("N marks ignored").
- Log tokens and cost per call. Print the total per video at the end of the run.

### 6. Output — `src/render.py`
- **Data (`tab.json`):** a list of rows. Each row has its origin in canvas pixels, its extent (left and right ends relative to the origin), its line spacing `s`, its bar-line x-positions and its notes. Each note has its string (1 = high e), x-position, fret, status (`verified`, `flagged`) and the two readers' results. All x-positions are relative to the row origin, in canvas pixels. Every PDF and metric is computed from this file.
- **PDF:** drawn directly with `reportlab`. Six lines per row, bar lines, and each number centred on its string at its x-position, with a background-coloured box behind the number that blanks out the line, as in the video.
  - **One scale factor for the whole song:** the widest video row (from its left end to its right end) fits the printable width. All spacing is therefore proportional to the video.
  - **Row alignment:** every row origin is drawn at the same page x-position: the left margin plus the song's largest distance from a row's left end to its origin. Numbers placed before a row's first bar line (pickups) keep their place.
  - **Paginated:** one PDF row per video row, in order, with pages broken between rows.
  - **Horizontal scrolling:** the canvas is cut into rows at bar lines. Each cut is at the bar line closest to the printable width, without going over it. A single measure wider than the printable width is split at a gap between notes and flagged.
  - Flagged numbers are drawn in red.
- **Report (`report.html`):**
  - Headline: "X of Y numbers verified". Also shown: numbers flagged, marks ignored and total cost.
  - Every automatic decision: overlaps removed, possible duplicates, scroll residuals and refusals.
  - For every row with a flag or a decision: its video image with the notes boxed, next to its redrawn version.

## CLI
```
python main.py <url|file> [-o out.pdf] [--crop x,y,w,h] [--layout paginated|scroll]
               [--paper a4|letter] [--debug]
python eval.py [--videos id,...]      # metrics against ground truth
```
`--debug` writes an overlay image for every decision to `cache/<video id>/debug/`:
- the tab area
- a plot of change over time, with pages marked
- canvases, and the scroll anchor fit
- rows, origins and bar lines
- overlap decisions
- marks, joins and notes
- the model grids
- the template sheet

## Build order
Each step is tested on real videos before the next one starts.
1. **Test set and evaluation first:** collect the videos, write `eval.py`, and define the `tab.json` schema.
2. **Stages 1–3 for paginated videos**, with `--crop` and `--layout`. Check the change plot, pages, rows, origins and bar lines with `--debug`.
3. **Stages 4–6.** Produce the first `tab.json` files, correct them into ground truth (see Testing), then measure.
4. **Automatic tab area, layout classification and horizontal scrolling.**

## Left out
Add one only when real videos show it's needed.
- Rhythm, durations, techniques (h, p, /, b, ~ and so on), notation, tuning, tempo, MusicXML.
- Notes that stay coloured once played, because they break the median.
- Vertical scrolling, several parts, tabs with other than 6 lines.
- Batch mode or a web UI.

## Dependencies
- `yt-dlp`, `ffmpeg`
- `opencv-python`, `numpy`
- `anthropic` (API key required)
- `reportlab`
- `pytest`

## Testing
- **Test set:** 5–10 real videos covering dark backgrounds, several rows per page, notation above the tab, crossfades, a playhead, a static page with no playhead, page overlaps, genuine repeats across pages, two-digit frets, chords and horizontal scrolling. Store them locally, not in git. Their URLs (or local paths) are a plain list in `tests/videos.md`, one per line, optionally followed by a short note. The video ID used for `cache/` and `tests/truth/` is derived from the URL: the YouTube ID, or the file name for a local file.
- **Ground truth:** the pipeline writes a first `tab.json`, which is then corrected by hand into `tests/truth/<video id>.json`. Check every note against the video frame, not only the flagged ones, because a first draft makes it easy to accept what is already there. Ground truth is committed to git.
- **Matching:** a predicted note matches a true note if it is in the same row, on the same string, and within `0.01` of the row width in x.
- **Metrics and pass bar:**

  | Metric | Target |
  | --- | --- |
  | Note recall (true notes found) | ≥ 99% |
  | Note precision (found notes that are real) | ≥ 99.5% |
  | Fret accuracy on matched notes | ≥ 99% |
  | Unflagged errors (wrong fret, wrong string, missing or extra, and not in red) | ≤ 0.2% of notes |
  | x-position error, median and maximum, as a fraction of row width | ≤ 0.25% and ≤ 0.5% |
  | Flag rate | ≤ 5% |
  | Flag precision (flags that were real errors) | reported, no target |

  Unflagged errors is the most important number, because it counts the errors a user won't catch.
- **Every change to a prompt, model or threshold** includes `eval.py` numbers from before and after.
- **Unit tests on synthetic images:**
  - the median removes a moving playhead
  - crossfades and slow drift are not taken for pages
  - bar lines are found and stems ignored
  - double bar lines merge
  - the row origin ignores a "TAB" label
  - overlap removal follows the activity rules, and genuine repeats are kept
  - line erasure keeps digits that cross the line
  - "12" and "10" are joined, while chord notes on neighbouring strings are not
  - x-positions are exact
  - the scroll anchor fit has bounded drift over 100 strips
  - the template reader, including leave one out
- **Fixtures:** recorded model responses in `tests/fixtures/`, so tests never call the API.

## Open questions
- Test video URLs, which are needed for build step 1.
