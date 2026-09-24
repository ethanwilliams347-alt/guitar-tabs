# Supported Videos

Which tab videos video_to_sheet can turn into a tab sheet. This file tracks what [PLAN.md](PLAN.md) describes and what has been tested. Update it whenever the pipeline, its thresholds or the test results change.

**Status:** being built. Stage 1 (ingest) is implemented and has run on `mhmDGhkUZt4`. Stage 2 (page finding and stitching) is implemented and has run on `mhmDGhkUZt4`, stitching its 3 pages into one 5230 px canvas. Stage 3 (lines, row, extent, bar lines, origin) is implemented, runs on each page and on the canvas, and has run on `mhmDGhkUZt4`'s canvas. Stage 4 (notes) is implemented and has run on `mhmDGhkUZt4`. Stage 5 is implemented and has run on `mhmDGhkUZt4`: the model reader (Gemini reads each note's fret from grids of crops, with every call cached; for now Gemini 3.5 Flash, because 3.8 Flash is overloaded), the local template reader, and the decision with a re-read. Stage 6 (PDF, `tab.json` and report) is implemented and has produced a PDF for `mhmDGhkUZt4`. It hasn't been checked against ground truth yet (step 12), so no video is supported yet. The tool is built one video at a time, and it only supports what the videos so far have needed. Iteration 1 (`mhmDGhkUZt4`) is in progress. The plan covers paged-strip videos like it. Everything else is refused for now. Nothing under "Handled correctly" has been verified by a pipeline run yet.
**Last updated:** 2026-09-24

A video will work only if it meets **all** of these conditions.

## Required

1. **Paged-strip layout** (`--layout paged-strip`). The song is one long line of tab shown a screen at a time. Each screen stays still and then changes to the next with a hard cut or a short crossfade. Every screen shows the music at the same scale and height. Each screen starts a little before the previous one ended, overlapping it by at least 8% of the tab-area width, so the screens can be stitched into one line. The tab may fade out towards the screen edges, but the overlaps must still match with an NCC of at least 0.8 over the tab row. On `mhmDGhkUZt4` they match at 0.85–0.86, so a much stronger fade would be refused. The best match must beat every shift more than 5 px away by 0.05, so an overlap made of one repeated pattern is refused. Any bar lines in an overlap must line up on both screens within 2 px.
2. **Each screen still for at least 1 second.** 0.25 s is trimmed from each end, and at least 5 frames must remain for the median.
3. **One tab area at a fixed position, given with `--crop`.** It can't move, zoom or change size. Anything else inside the crop, such as camera video, must stay still while a screen is shown.
4. **Exactly one six-line tab row per screen,** with evenly spaced lines.
5. **A light background.**
6. **Bar lines drawn straight across all six lines, with sharp edges.** A column counts as a bar line when ink covers at least 90% of the tab's height there and the columns on either side are under 50%. A bar line with fret numbers pressed against it along most of its height could be missed.
7. **Playheads and highlights that cover any pixel less than half of a screen's time.**
8. **Readable resolution.** 1080p in practice. Iteration 1's line spacing is 17.60 px.
9. **Downloadable with `yt-dlp` as H.264, or a local file OpenCV can decode.** Age-restricted, members-only and private videos aren't covered.

## Handled correctly (planned)

- Notation staves, chord names and chord diagrams above the tab: ignored (checked on all 3 pages of `mhmDGhkUZt4`).
- Camera video outside the tab area: kept out with `--crop`.
- A playhead line, and notes that change colour only while they play: removed by the median over each screen (checked on `mhmDGhkUZt4`: 0 reddish pixels left on each of the 3 pages).
- Screens that overlap and fade at their edges: stitched into one line, keeping the clear copy of the overlap, so shared music appears once (checked on both overlaps of `mhmDGhkUZt4`).
- Two-digit fret numbers: their digits are joined into one note (checked on `mhmDGhkUZt4`: 42 two-digit notes, including chords made entirely of two-digit frets).
- Chords: each note is found on its own string (checked on chords of up to 5 notes).
- Arpeggio marks crossing the strings: not taken for bar lines (checked on all 3 pages of `mhmDGhkUZt4`), and dropped as notes because they are too tall (all 7 checked).
- A bracket, clef or "TAB" label at the start: positions are measured from the row's origin. On `mhmDGhkUZt4` page 1, the origin is the system start line, not the bracket (checked).
- Double and final bar lines: merged into one bar line (checked on the final bar of `mhmDGhkUZt4`).
- Flagged numbers: drawn in red in the PDF, with the video strip and every reader's result for each one in the report. A flagged mark no reader could read is drawn as a red "?". A measure too wide for one PDF row is split between two notes, marked with a red dashed line, and listed in the report.

## Not output

The PDF has fret numbers, the six lines and bar lines only. Rhythm, techniques (h, p, slides, bends, vibrato), notation, tuning, tempo, chord diagrams and lyrics are left out.

## Not supported yet

Each of these is refused with a reason until a test video needs it.
- Paginated videos with several tab rows per screen, or with screens that don't continue one line.
- Horizontally scrolling tab.
- Dark backgrounds.
- Running without `--crop` and `--layout`.
- Vertical scrolling, tab that pans or zooms, tabs with other than 6 lines, several parts.
- Notes that stay coloured once played, or highlights that cover most of a screen most of the time.

## Tested videos

| Video | Layout | Result | Notes |
| --- | --- | --- | --- |
| `mhmDGhkUZt4` | paged strip | full pipeline run, not yet checked against ground truth (2026-09-24): 732 samples of 484 × 1920 from 1828 frames at 29.97 fps. 3 pages at 0.00–17.25, 17.50–36.83 and 37.08–60.92 s, each a median of 31 frames with no playhead left. On each page, one row at y = 374.5–462.5 with `s` = 17.60, and 6, 6 and 7 bar lines, matching a count by eye. Stitched at offsets 1692 and 1618 px (NCC 0.851 and 0.857, runner-ups 0.662 and 0.715). Bar-line residual: none in the first overlap, 0.5 px in the second. Canvas 5230 × 484, with every overlapping note shown once. On the canvas, one row with its origin on the start line (113.5) and 18 bar lines, each found once, including the final double bar. 163 notes, 42 of them two-digit. In 3 measures checked by eye, no note is missed or split, and all 7 arpeggios are dropped. Model reader (`gemini-3.5-flash`, standing in for the overloaded 3.8 Flash): 5 valid responses, 163 numbers, $0.06 at paid prices, and a rerun makes no calls. All readings match a check by eye. Template reader: templates for all 10 digits. 112 notes verified, 0 disagreements, and 51 flagged (31%, against a 5% target) because their template matches score under 0.85. The re-read agrees with the first read on all 51. The threshold is to be tuned in step 13 against ground truth. PDF: 3 rows on one Letter page, each starting at a bar line, with 51 numbers in red. The report lists every decision. | 3 screens with 228 and 302 px overlaps; tab strip above camera video; chord diagrams and notation above the tab; playhead with note highlights; frets up to 13; arpeggio marks. Run with `--crop 0,0,1920,484 --layout paged-strip`. |
