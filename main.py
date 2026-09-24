"""video_to_sheet CLI: python main.py <URL or file> --crop x,y,w,h --layout paged-strip

Runs ingest -> canvas -> rows -> notes -> read -> render. A stage whose cache key is
unchanged is skipped. --stage <name> reruns just that stage.
"""
import argparse
import importlib
import json
import shutil
import sys

from src import Refused
from src.cache import STAGES, Context, is_current, mark_done, stage_key, video_id

LAYOUTS = ["paged-strip"]
PAPERS = ["letter", "a4"]


def parse_crop(text):
    try:
        parts = [int(v) for v in text.split(",")]
    except ValueError:
        parts = []
    if len(parts) != 4 or parts[0] < 0 or parts[1] < 0 or parts[2] <= 0 or parts[3] <= 0:
        raise argparse.ArgumentTypeError(f"--crop must be x,y,w,h with w,h > 0, got {text!r}")
    return tuple(parts)


def build_parser():
    p = argparse.ArgumentParser(description="Turn a guitar tab video into a PDF tab sheet.")
    p.add_argument("source", help="YouTube URL or local video file")
    p.add_argument("--crop", type=parse_crop, help="tab area in video pixels: x,y,w,h")
    p.add_argument("--layout", help="how the tab is shown; supported: " + ", ".join(LAYOUTS))
    p.add_argument("--stage", choices=STAGES, help="rerun only this stage")
    p.add_argument("--debug", action="store_true", help="write an overlay for every decision")
    p.add_argument("-o", "--output", help="PDF path (default out/<video id>.pdf)")
    p.add_argument("--paper", choices=PAPERS, default="letter")
    p.add_argument("--record-fixtures", action="store_true",
                   help="copy the model calls used to tests/fixtures/read/<video id>/ (re-record only on purpose)")
    return p


def check_inputs(args):
    """Refuse inputs this iteration doesn't handle, with the reason."""
    if args.crop is None:
        raise Refused("--crop is required: the tab area can't be found automatically yet")
    if args.layout is None:
        raise Refused("--layout is required: the layout can't be detected automatically yet")
    if args.layout not in LAYOUTS:
        raise Refused(f"layout {args.layout!r} is not supported; supported: {', '.join(LAYOUTS)}")


def load_stage(name):
    try:
        return importlib.import_module(f"src.{name}")
    except ModuleNotFoundError as e:
        if e.name == f"src.{name}":
            raise SystemExit(f"stage {name!r} is not built yet")
        raise


def run_stage(name, ctx, force=False):
    mod = load_stage(name)
    out = ctx.dir(name)
    key = stage_key(mod.STAGE_VERSION, mod.inputs(ctx), mod.params(ctx))
    debug_dir = out / "debug"
    if not force and is_current(out, key) and (not ctx.debug or debug_dir.is_dir()):
        print(f"[{name}] up to date")
        return
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    if ctx.debug:
        debug_dir.mkdir()
    print(f"[{name}] running")
    mod.run(ctx)
    mark_done(out, key)


def print_cost(ctx):
    """The video's total model cost, from every call its readings used, cached or not."""
    path = ctx.dir("read") / "cost.json"
    if path.exists():
        cost = json.loads(path.read_text())
        print(f"Model cost for {ctx.video_id}: ${cost['total_usd']:.4f} over {len(cost['calls'])} calls "
              f"(${cost['new_usd']:.4f} new when the read stage last ran)")


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        check_inputs(args)
        ctx = Context(source=args.source, video_id=video_id(args.source), crop=args.crop,
                      layout=args.layout, debug=args.debug, output=args.output, paper=args.paper,
                      record_fixtures=args.record_fixtures)
        try:
            for name in [args.stage] if args.stage else STAGES:
                run_stage(name, ctx, force=bool(args.stage))
        finally:
            print_cost(ctx)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
