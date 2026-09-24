"""Stage directories and stage cache keys.

Every stage reads and writes files in cache/<video id>/<stage>/. A stage's key is a
hash of its input files, the config values it uses and its STAGE_VERSION. If the key
stored in the stage directory matches, the stage is skipped. Because a stage's inputs
are the previous stage's outputs, a stage that reruns with a different result changes
the key of every stage after it.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_ROOT = ROOT / "cache"
VIDEOS_DIR = ROOT / "videos"
KEY_FILE = "key.txt"

STAGES = ["ingest", "canvas", "rows", "notes", "read", "render"]

_YOUTUBE_ID = re.compile(r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})")


def is_url(source):
    return source.startswith(("http://", "https://"))


def video_id(source):
    """The YouTube ID for a URL, or the file name without its extension for a file."""
    if is_url(source):
        m = _YOUTUBE_ID.search(source)
        if not m:
            raise ValueError(f"no YouTube video ID in {source!r}")
        return m.group(1)
    return Path(source).stem


def video_dir(vid, root=None):
    return Path(root or CACHE_ROOT) / vid


def stage_dir(vid, stage, root=None):
    """cache/<video id>/<stage>/, created if missing."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    d = video_dir(vid, root) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash_file(path, h):
    h.update(str(Path(path).name).encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)


def stage_key(stage_version, input_files=(), params=None):
    """Hash of the input files' contents, the params (config values, CLI inputs) and the version."""
    h = hashlib.sha256()
    h.update(f"v{stage_version}".encode())
    for path in sorted(str(p) for p in input_files):
        _hash_file(path, h)
    h.update(json.dumps(params or {}, sort_keys=True, default=str).encode())
    return h.hexdigest()


def is_current(directory, key):
    f = Path(directory) / KEY_FILE
    return f.exists() and f.read_text().strip() == key


def mark_done(directory, key):
    """Write the key last, so a stage that crashes midway is rerun."""
    (Path(directory) / KEY_FILE).write_text(key)


@dataclass
class Context:
    """What every stage gets: the video, its CLI inputs and where the cache is."""
    source: str
    video_id: str
    crop: tuple = None           # (x, y, w, h) in video pixels
    layout: str = None
    debug: bool = False
    output: str = None
    paper: str = "letter"
    record_fixtures: bool = False  # copy the model calls used to tests/fixtures/read/<video id>/
    root: Path = None            # cache root; None means cache/

    def dir(self, stage):
        return stage_dir(self.video_id, stage, self.root)
