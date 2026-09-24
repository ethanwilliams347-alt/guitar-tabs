import argparse

import pytest

import main
from src import Refused, config
from src.cache import Context, is_current, mark_done, stage_dir, stage_key, video_id


def test_video_id_from_url_and_file():
    assert video_id("https://www.youtube.com/watch?v=mhmDGhkUZt4&list=PLx&index=8") == "mhmDGhkUZt4"
    assert video_id("https://youtu.be/mhmDGhkUZt4") == "mhmDGhkUZt4"
    assert video_id("videos/mhmDGhkUZt4.mp4") == "mhmDGhkUZt4"
    with pytest.raises(ValueError):
        video_id("https://example.com/no-id")


def test_stage_key_changes_with_inputs_params_and_version(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"one")
    base = stage_key(1, [f], {"X_PX": 3})
    assert stage_key(1, [f], {"X_PX": 3}) == base
    assert stage_key(2, [f], {"X_PX": 3}) != base
    assert stage_key(1, [f], {"X_PX": 4}) != base
    f.write_bytes(b"two")
    assert stage_key(1, [f], {"X_PX": 3}) != base


def test_stage_dir_and_done_marker(tmp_path):
    d = stage_dir("vid", "rows", root=tmp_path)
    assert d == tmp_path / "vid" / "rows" and d.is_dir()
    assert not is_current(d, "k1")
    mark_done(d, "k1")
    assert is_current(d, "k1") and not is_current(d, "k2")
    with pytest.raises(ValueError):
        stage_dir("vid", "nope", root=tmp_path)


def test_context_dir(tmp_path):
    ctx = Context(source="x.mp4", video_id="x", root=tmp_path)
    assert ctx.dir("ingest") == tmp_path / "x" / "ingest"


def test_parse_crop():
    assert main.parse_crop("0,0,1920,484") == (0, 0, 1920, 484)
    for bad in ["0,0,1920", "a,b,c,d", "0,0,0,484", "-1,0,10,10"]:
        with pytest.raises(argparse.ArgumentTypeError):
            main.parse_crop(bad)


def test_cli_refuses_unsupported_inputs():
    parser = main.build_parser()
    for argv in (["v.mp4", "--layout", "paged-strip"],
                 ["v.mp4", "--crop", "0,0,10,10"],
                 ["v.mp4", "--crop", "0,0,10,10", "--layout", "scrolling"]):
        with pytest.raises(Refused):
            main.check_inputs(parser.parse_args(argv))
    main.check_inputs(parser.parse_args(["v.mp4", "--crop", "0,0,10,10", "--layout", "paged-strip"]))
    assert main.main(["v.mp4", "--layout", "scrolling", "--crop", "0,0,10,10"]) == 2


def test_cli_options():
    a = main.build_parser().parse_args(["v.mp4", "--stage", "rows", "--debug", "-o", "x.pdf", "--paper", "a4"])
    assert (a.stage, a.debug, a.output, a.paper) == ("rows", True, "x.pdf", "a4")


def test_model_ids_only_in_config():
    assert set(config.MODELS.values()) <= set(config.PRICES_USD_PER_MTOK)
