# Supported Videos

Which tab videos video_to_sheet can turn into a tab sheet. This file tracks what [PLAN.md](PLAN.md) describes and what has been tested. Update it whenever the pipeline, its thresholds or the test results change.

**Status:** planned, not yet implemented. The plan was revised after review: frames are sampled at 12 fps, change is measured with playheads excluded, overlaps are removed only when playhead activity supports it, two-digit frets are joined, and scrolling is anchored on bar lines. Build order: the test set and evaluation come first, then paginated videos with `--crop`, then horizontal scrolling and automatic tab-area detection (step 4). Nothing below has been verified on real videos yet.
**Last updated:** 2026-09-22

A video will work only if it meets **all** of these conditions.

## Required

1. **Paginated or horizontally scrolling layout.**
   - *Paginated:* the tab sits still, then changes to the next page with a hard cut or a short crossfade. Each page has to stay still for at least 1 second. The 0.25 s trimmed from each end leaves at least 6 frames at 12 fps for the median.
   - *Horizontal scrolling:* the tab slides sideways at a steady rate. This is planned for build step 4. The tab has to contain bar lines, because they are used to line up the strips.
2. **One tab area at a fixed position.** You give its location with `--crop`, or the tool finds it automatically (build step 4). It can't move, zoom or change size.
3. **Six-line tab rows**, evenly spaced. Several rows per page are fine.
4. **Bar lines drawn across the full height of the six lines.**
5. **Moving elements that cover any pixel less than half the time.** This includes playheads, bouncing balls and flashing notes. Thin vertical playheads are also used to tell overlaps from genuine repeats.
6. **Readable resolution.** 1080p in practice.
7. **Downloadable with `yt-dlp`, or a local file.** Age-restricted, members-only and private videos aren't covered.

## Handled correctly

- Dark or light backgrounds.
- Notation, rhythm stems and chord names around the tab: ignored.
- Double and repeat bar lines: merged into one bar line.
- Pages that repeat the previous page's last 1–4 measures: removed when the playhead shows them played on only one page. With no playhead, both copies are kept and flagged.
- Riffs and whole pages that genuinely repeat.
- Static pages with no playhead.
- Two-digit fret numbers: their digits are joined into one number.
- A "TAB" label or first-row indent: positions are measured from each row's first bar line.

## Not output

The PDF has fret numbers, the six lines and bar lines only. Rhythm, techniques (h, p, slides, bends, vibrato), notation, tuning, tempo, chord diagrams and lyrics are left out.

## Not supported

- Vertical scrolling scores.
- Tab that moves on screen: panning, zooming, or picture-in-picture layouts that shift.
- Tabs with other than 6 lines (bass, 7-string).
- Paginated pages still for less than 1 second.
- Page changes with animations longer than a short crossfade, unless each page still has more than a second of stable frames.
- Notes that stay coloured once played.
- Highlights that cover most of the page for most of the time.
- Several parts, unless they form a single top-to-bottom sequence of rows.

## Might work, depending on thresholds

- Heavy compression artifacts.
- A guitarist's camera overlay partly covering the tab.
- Handwritten or unusual fonts.
- Very thin bar lines.

## Tested videos

| Video | Layout | Result | Notes |
| --- | --- | --- | --- |
| _none yet_ | | | |
