# Supported Videos

Which tab videos video_to_sheet can turn into a tab sheet. This file tracks what [PLAN.md](PLAN.md) describes and what has been tested. Update it whenever the pipeline, its thresholds or the test results change.

**Status:** planned, not yet implemented. The tool is built one video at a time, and it only supports what the videos so far have needed. Iteration 1 (`mhmDGhkUZt4`) is in progress. The plan covers paged-strip videos like it. Everything else is refused for now. Nothing below has been verified by a pipeline run yet.
**Last updated:** 2026-09-23

A video will work only if it meets **all** of these conditions.

## Required

1. **Paged-strip layout** (`--layout paged-strip`). The song is one long line of tab shown a screen at a time. Each screen stays still and then changes to the next with a hard cut or a short crossfade. Every screen shows the music at the same scale and height. Each screen starts a little before the previous one ended, overlapping it by at least 8% of the tab-area width, so the screens can be stitched into one line. The tab may fade out towards the screen edges.
2. **Each screen still for at least 1 second.** 0.25 s is trimmed from each end, and at least 5 frames must remain for the median.
3. **One tab area at a fixed position, given with `--crop`.** It can't move, zoom or change size. Anything else inside the crop, such as camera video, must stay still while a screen is shown.
4. **Exactly one six-line tab row per screen,** with evenly spaced lines.
5. **A light background.**
6. **Bar lines drawn straight across all six lines.**
7. **Playheads and highlights that cover any pixel less than half of a screen's time.**
8. **Readable resolution.** 1080p in practice. Iteration 1's line spacing is 17.7 px.
9. **Downloadable with `yt-dlp` as H.264, or a local file OpenCV can decode.** Age-restricted, members-only and private videos aren't covered.

## Handled correctly (planned)

- Notation staves, chord names and chord diagrams above the tab: ignored.
- Camera video outside the tab area: kept out with `--crop`.
- A playhead line, and notes that change colour only while they play: removed by the median over each screen.
- Screens that overlap and fade at their edges: stitched into one line, keeping the clear copy of the overlap, so shared music appears once.
- Two-digit fret numbers: their digits are joined into one number.
- Chords.
- Arpeggio marks crossing the strings: dropped.
- A bracket, clef or "TAB" label at the start: positions are measured from the row's origin.
- Double and final bar lines: merged into one bar line.

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
| `mhmDGhkUZt4` | paged strip | examined, pipeline not built yet | 3 screens with 228 and 302 px overlaps; tab strip above camera video; chord diagrams and notation above the tab; playhead with note highlights; frets up to 13; arpeggio marks. Run with `--crop 0,0,1920,484 --layout paged-strip`. |
