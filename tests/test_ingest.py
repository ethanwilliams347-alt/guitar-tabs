import json

import cv2
import numpy as np
import pytest

from src import Refused, config, ingest
from src.cache import Context, is_url


def make_video(path, n_frames, fps, size=(320, 240)):
    """Frame i: left half gray level 8*i (the frame index), right half black."""
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert vw.isOpened()
    for i in range(n_frames):
        f = np.zeros((h, w, 3), np.uint8)
        f[:, : w // 2] = 8 * i
        vw.write(f)
    vw.release()


def test_sample_frames_iteration1_numbers():
    # mhmDGhkUZt4: 1828 frames at 29.97 fps is 60.994 s, so 12 samples/s gives 732.
    times, frames = ingest.sample_frames(30000 / 1001, 1828, 12)
    assert len(times) == 732
    assert frames[0] == 0 and frames[-1] < 1828
    assert frames[12] == 30                       # t = 1 s, frame 29.97 rounds to 30
    assert len(set(frames)) == len(frames)


def test_sample_frames_nearest():
    times, frames = ingest.sample_frames(25, 50, 10)   # 2 s
    assert times == [k / 10 for k in range(20)]
    assert frames == [int(np.floor(t * 25 + 0.5)) for t in times]


def test_check_crop():
    ingest.check_crop((0, 0, 1920, 484), 1920, 1080)
    for bad in [(0, 0, 1921, 484), (0, 700, 1920, 484), (10, 0, 1920, 10)]:
        with pytest.raises(Refused):
            ingest.check_crop(bad, 1920, 1080)


def test_run_on_synthetic_video(tmp_path):
    video = tmp_path / "synth.avi"
    make_video(video, 30, 10)                     # 3 s at 10 fps: below SAMPLE_FPS, so frames repeat
    ctx = Context(source=str(video), video_id="synth", crop=(0, 0, 160, 120), root=tmp_path / "cache",
                  debug=True)
    (ctx.dir("ingest") / "debug").mkdir()
    ingest.run(ctx)
    out = ctx.dir("ingest")
    meta = json.loads((out / "meta.json").read_text())
    motion = np.load(out / "motion.npy")
    n = 3 * config.SAMPLE_FPS
    assert len(meta["sample_times"]) == n and motion.shape[0] == n
    # crop 160x120 scaled to MOTION_WIDTH_PX wide keeps the aspect ratio
    assert motion.shape[1:] == (round(120 * config.MOTION_WIDTH_PX / 160), config.MOTION_WIDTH_PX)
    assert (meta["fps"], meta["frame_count"], meta["crop"]) == (10, 30, [0, 0, 160, 120])
    # each sample holds the frame nearest its time (gray 8*i, within JPEG loss)
    for t, f, img in zip(meta["sample_times"], meta["sample_frames"], motion):
        assert f == int(np.floor(t * 10 + 0.5))
        assert abs(float(np.median(img)) - 8 * f) <= 3
    assert len(set(meta["sample_frames"])) < n
    assert (out / "debug" / "crop.png").exists()


def test_refuses_undecodable_and_missing(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    for src in (junk, tmp_path / "missing.mp4"):
        ctx = Context(source=str(src), video_id=src.stem, crop=(0, 0, 10, 10), root=tmp_path / "cache")
        with pytest.raises(Refused):
            ingest.run(ctx)


def test_refuses_crop_outside_frame(tmp_path):
    video = tmp_path / "v.avi"
    make_video(video, 5, 10)
    ctx = Context(source=str(video), video_id="v", crop=(0, 0, 400, 100), root=tmp_path / "cache")
    with pytest.raises(Refused):
        ingest.run(ctx)


def test_local_file_is_not_url():
    assert not is_url("videos/mhmDGhkUZt4.mp4") and is_url("https://youtu.be/mhmDGhkUZt4")
