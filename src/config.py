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
MARK_MIN_H_FRAC = 0.4            # of s
MARK_MAX_H_FRAC = 1.2            # of s
MARK_MAX_DY_FRAC = 0.4           # of s
DIGIT_JOIN_OVERLAP_FRAC = 0.7    # of the smaller mark's height
DIGIT_JOIN_GAP_FRAC = 0.35       # of median mark width

# Stage 5: read
CROP_PAD_FRAC = 0.3              # of s
GRID_MAX_CELLS = 40
GRID_CELL_H_PX = 64
TEMPLATE_SIZE_PX = 32
TEMPLATE_MIN_SAMPLES = 3
TEMPLATE_CLASS_MIN_NCC = 0.8
TEMPLATE_MATCH_MIN_NCC = 0.85
TEMPLATE_MARGIN_NCC = 0.05
MODELS = {"read": "claude-sonnet-5", "reread": "claude-opus-5"}  # the only place model IDs appear
READ_EFFORT = {"read": "medium", "reread": "high"}
PRICES_USD_PER_MTOK = {
    "claude-sonnet-5": {"input": 2.0, "output": 10.0},
    "claude-opus-5": {"input": 5.0, "output": 25.0},
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
