"""All thresholds and settings, with the unit in each name.

_S seconds, _PX pixels, _LEVEL a 0-255 gray level, _FRAC a fraction (of what is
said in the comment), _NCC a normalized cross-correlation. Values change only
with eval.py numbers from before and after (see PLAN.md, Configuration).
"""

# Stage 1: ingest
SAMPLE_FPS = 12                  # samples per second
MOTION_WIDTH_PX = 480            # width of the downscaled change-measure copy
YTDLP_FORMAT = "bv*[vcodec^=avc1][height<=1080]"

# Stage 2: canvas (pages)
DIFF_PIXEL_LEVEL = 25            # gray levels
STILL_FRAME_FRAC = 0.02          # of tab-area pixels, change from previous sample
STILL_DRIFT_FRAC = 0.03          # of tab-area pixels, change from a run's first sample
PAGE_MIN_STABLE_S = 1.0
PAGE_TRIM_S = 0.25               # from each end of a run
MEDIAN_MAX_FRAMES = 31
MEDIAN_MIN_FRAMES = 5
PAGE_MERGE_NCC = 0.98
LIGHT_BG_MIN_LEVEL = 128         # median gray level
LINE_Y_MATCH_PX = 1              # between pages

# Stage 2: canvas (stitching)
STRIP_MIN_OVERLAP_FRAC = 0.08    # of page width
STRIP_NCC_MIN = 0.8
STRIP_NCC_MARGIN = 0.05
STRIP_RUNNER_UP_MIN_DIST_PX = 5
STRIP_MAX_RESIDUAL_PX = 2

# Stage 3: rows
LINE_MIN_COVER_FRAC = 0.5        # of image width, ink in one pixel row of a line
LINE_SPACING_TOL_FRAC = 0.15     # of mean spacing
BARLINE_MIN_COVER_FRAC = 0.9     # of top-to-bottom line span
BARLINE_MERGE_FRAC = 0.5         # of s
BARLINE_EDGE_MAX_COVER_FRAC = 0.5  # of top-to-bottom line span, columns beside a bar line
EXTENT_MIN_RUN_FRAC = 1.0        # of s, shortest run of all six lines that can end the row

# Stage 4: notes
BARLINE_CLEAR_PAD_PX = 1         # columns cleared beside a bar line's span, its anti-aliased edge
MARK_MIN_H_FRAC = 0.4            # of s
MARK_MAX_H_FRAC = 1.2            # of s
MARK_MAX_DY_FRAC = 0.4           # of s
DIGIT_JOIN_OVERLAP_FRAC = 0.7    # of the smaller mark's height
DIGIT_JOIN_GAP_FRAC = 0.35       # of median mark width

# Stage 5: read
CROP_PAD_FRAC = 0.3              # of s
GRID_MAX_CELLS = 40
GRID_CELL_H_PX = 64              # height every note crop is scaled to
GRID_COLS = 8                    # cells per grid row
GRID_LABEL_H_PX = 20             # strip above each crop that holds its ID
GRID_GAP_PX = 8                  # white space around each framed crop
GRID_LABEL_FONT_SCALE = 0.5
READ_PROMPT_VERSION = 1          # src/prompts/read_v<N>.md and read_v<N>.schema.json
READ_MAX_TOKENS = 16000          # max_output_tokens per call
READ_MAX_ATTEMPTS = 2            # an invalid response is requested once more
TEMPLATE_SIZE_PX = 32
TEMPLATE_MIN_SAMPLES = 3
TEMPLATE_CLASS_MIN_NCC = 0.8
TEMPLATE_MATCH_MIN_NCC = 0.85
TEMPLATE_MARGIN_NCC = 0.05
# "read" should be gemini-3.8-flash. It is gemini-3.5-flash only while 3.8 Flash is overloaded (503s since at
# least 2026-09-15). Switch back once it answers reliably: see PLAN.md, stage 5, "Temporary model choice".
MODELS = {"read": "gemini-3.5-flash", "reread": "gemini-3.1-pro-preview"}  # the only place model IDs appear
READ_THINKING_LEVEL = {"read": "MEDIUM", "reread": "HIGH"}  # Gemini thinking_level; these models can't turn it off
API_RETRY_ATTEMPTS = 5           # tries per call, the first included, on 408, 429 and 5xx
API_RETRY_INITIAL_DELAY_S = 15   # first wait, doubling to at most 60 s: outlasts a per-minute quota
PRICES_USD_PER_MTOK = {          # Gemini API paid tier; output includes thinking tokens
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},        # to 2026-12-31; 1.50 / 7.50 from 2027-01-01
    "gemini-3.1-pro-preview": {"input": 2.0, "output": 12.0},   # prompts up to 200k tokens
}

# Evaluation
MATCH_X_TOL_FRAC = 0.01          # of tab-area width

# Evaluation pass bar: every video must meet every target on its own
EVAL_MIN_RECALL_FRAC = 0.99
EVAL_MIN_PRECISION_FRAC = 0.995
EVAL_MIN_FRET_ACCURACY_FRAC = 0.99
EVAL_MAX_UNFLAGGED_ERROR_FRAC = 0.002   # of true notes
EVAL_MAX_X_ERR_MEDIAN_FRAC = 0.0025     # of tab-area width
EVAL_MAX_X_ERR_MAX_FRAC = 0.005         # of tab-area width
EVAL_MAX_FLAG_RATE_FRAC = 0.05          # of found notes

# Review images (eval.py --review)
REVIEW_STRING_GAP_PX = 44        # between drawn strings under the canvas
REVIEW_MARGIN_PX = 30            # above and below the drawn strings
REVIEW_FRET_FONT_SCALE = 0.7
REVIEW_ID_FONT_SCALE = 0.35
