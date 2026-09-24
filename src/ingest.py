"""Stage 1: get the video and sample it.

Takes the frame nearest each multiple of 1/SAMPLE_FPS s, crops it to --crop, converts
it to grayscale and stores only a copy downscaled to MOTION_WIDTH_PX wide. Stage 2
rereads the full-resolution frames it needs from the video, by the frame indices in
meta.json.

Writes: meta.json, motion.npy. Debug: crop.png (the crop drawn on the first frame).
"""
import json
from pathlib import Path

import cv2
import numpy as np

from src import Refused, config
from src.cache import VIDEOS_DIR, is_url

STAGE_VERSION = 1


def video_path(ctx):
    """The local video file, downloaded with yt-dlp first if the source is a URL."""
    if not is_url(ctx.source):
        p = Path(ctx.source)
        if not p.exists():
            raise Refused(f"video file {p} does not exist")
        return p
    p = VIDEOS_DIR / f"{ctx.video_id}.mp4"
    if not p.exists():
        download(ctx.source, p)
    return p


def download(url, path):
    import yt_dlp
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = {"format": config.YTDLP_FORMAT, "outtmpl": str(path), "quiet": True, "noprogress": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        raise Refused(f"can't download {url} as H.264 up to 1080p: {e}")


def inputs(ctx):
    return [video_path(ctx)]


def params(ctx):
    return {"crop": ctx.crop, "SAMPLE_FPS": config.SAMPLE_FPS, "MOTION_WIDTH_PX": config.MOTION_WIDTH_PX}


def sample_frames(fps, frame_count, sample_fps):
    """(times, frame indices): the frame nearest each multiple of 1/sample_fps inside the video."""
    duration = frame_count / fps
    n = int(np.ceil(duration * sample_fps))
    times = [k / sample_fps for k in range(n) if k / sample_fps < duration]
    frames = [min(int(np.floor(t * fps + 0.5)), frame_count - 1) for t in times]
    return times, frames


def check_crop(crop, width, height):
    x, y, w, h = crop
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise Refused(f"crop {x},{y},{w},{h} is outside the {width}x{height} frame")


def motion_size(crop):
    _, _, w, h = crop
    return config.MOTION_WIDTH_PX, max(1, int(round(h * config.MOTION_WIDTH_PX / w)))


def to_gray_crop(frame, crop):
    x, y, w, h = crop
    return cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)


def read_frames(path, frame_indices):
    """Yield (index, BGR frame) for the wanted indices, decoding the video once in order."""
    cap = cv2.VideoCapture(str(path))
    wanted = sorted(set(frame_indices))
    i, k = 0, 0
    try:
        while k < len(wanted):
            if not cap.grab():
                raise Refused(f"{path} stopped decoding at frame {i}, before frame {wanted[k]}")
            if i == wanted[k]:
                ok, frame = cap.retrieve()
                if not ok:
                    raise Refused(f"{path}: frame {i} can't be decoded")
                yield i, frame
                k += 1
            i += 1
    finally:
        cap.release()


def run(ctx):
    path = video_path(ctx)
    out = ctx.dir("ingest")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise Refused(f"{path} can't be decoded by OpenCV")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise Refused(f"{path} has no readable frame rate or frame count")
    check_crop(ctx.crop, width, height)

    times, frames = sample_frames(fps, frame_count, config.SAMPLE_FPS)
    size = motion_size(ctx.crop)
    motion = np.empty((len(frames), size[1], size[0]), np.uint8)
    samples_of = {}                  # frame index -> samples that use it (several if SAMPLE_FPS > fps)
    for k, f in enumerate(frames):
        samples_of.setdefault(f, []).append(k)
    for f, frame in read_frames(path, frames):
        small = cv2.resize(to_gray_crop(frame, ctx.crop), size, interpolation=cv2.INTER_AREA)
        for k in samples_of[f]:
            motion[k] = small
        if f == frames[0] and ctx.debug:
            write_crop_overlay(frame, ctx.crop, out / "debug" / "crop.png")

    np.save(out / "motion.npy", motion)
    meta = {"source": ctx.source, "video": str(Path(path).resolve()), "fps": fps, "frame_count": frame_count,
            "width": width, "height": height, "crop": list(ctx.crop), "sample_fps": config.SAMPLE_FPS,
            "sample_times": times, "sample_frames": frames}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[ingest] {len(frames)} samples of {ctx.crop[3]}x{ctx.crop[2]} "
          f"(stored at {size[1]}x{size[0]}) from {frame_count} frames at {fps:.3f} fps")


def write_crop_overlay(frame, crop, path):
    x, y, w, h = crop
    img = frame.copy()
    img[:y] //= 3
    img[y + h:] //= 3
    img[:, :x] //= 3
    img[:, x + w:] //= 3
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 200, 0), 3)
    cv2.imwrite(str(path), img)
