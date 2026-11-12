import contextlib
import io
import re
import sqlite3
import struct
import traceback
from compression import zstd
from datetime import datetime
from pathlib import Path
from typing import Any

import av

from motion_correlation import d435_motion, mirrorless_motion
from mp4_video import Mp4Video, validate_mp4
from recording_reader import (
    CompressedImageMessage,
    D435Validator,
    MirrorlessRos2Loader,
    MirrorlessRos2Validator,
    MirrorlessRosValidator,
    RealSenseLoader,
    Recording,
)

RECORDING_SUFFIXES = {".bag", ".db3", ".mp4"}

type Loaded = Recording | list[CompressedImageMessage] | Mp4Video

# the errors of a damaged or unreadable recording: pyrealsense2 raises RuntimeError, PyAV raises FFmpegError, a .db3 file can raise
# sqlite3, struct, or zstd errors, and the readers raise ValueError and IndexError
READ_ERRORS = (OSError, RuntimeError, ValueError, IndexError, av.FFmpegError, sqlite3.Error, struct.error, zstd.ZstdError)

# severity: a wrong camera setting or a stream that is not there is critical; the checks that lose frames are medium for 1 or 2
# lost frames in the file (all the gaps of the check added) and critical for more, but a frame_number_gaps gap at a missing_frame
# frameset takes the severity of the missing_frame (frame_number_gap_severity); every other check is medium
CRITICAL_CHECKS = {"missing_stream", "fps", "size", "format", "serial_number", "unexpected_stream"}
GAP_CHECKS = {"frame_number_gaps", "barcode_gaps", "timestamp_gaps"}
MAX_MEDIUM_LOST_FRAMES = 2
# "4 frame numbers missing" (frame_number_gaps, barcode_gaps) or "about 867 frames missing" (timestamp_gaps)
LOST_FRAMES = re.compile(r"(\d+) frame numbers missing|about (\d+) frames missing")
# one range per gap after "missing: " in a frame_number_gaps message: "[90430, 90434]" or "77"
GAP_RANGE = re.compile(r"\[(\d+), (\d+)\]|(\d+)")


# every .mp4 file is the mirrorless camera; a .bag or .db3 file has -r (RealSense D435) or -m (mirrorless) in its name
def camera_or_none(path: Path) -> str | None:
    if path.suffix == ".mp4":
        return "mirrorless"
    is_d435, is_mirrorless = "-r" in path.name, "-m" in path.name
    if is_d435 == is_mirrorless:
        return None
    return "d435" if is_d435 else "mirrorless"


def camera_of(path: Path) -> str:
    camera = camera_or_none(path)
    if camera is None:
        raise ValueError(f"{path.name}: the file name must contain exactly one of -r (RealSense D435) and -m (mirrorless)")
    return camera


# 1, 2, 10 in number order, not 1, 10, 2
def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def find_recordings(folder: Path) -> list[Path]:
    paths = [path for path in folder.rglob("*") if path.suffix in RECORDING_SUFFIXES and path.is_file()]
    return sorted(paths, key=lambda path: natural_key(path.relative_to(folder).as_posix()))


# the -r and -m recordings of one take have the same key: 1-r.bag and 1-m.bag (or 1-m.mp4) in one folder, or
# pipeline-r_20261012_151250.db3 and pipeline-m_20261012_151250_color/pipeline-m_20261012_151250_color.db3
def take_key(path: Path) -> tuple[Path, str]:
    folder = path.parent.parent if path.parent.name == path.stem else path.parent
    return folder, path.stem.replace("-r", "-x").replace("-m", "-x").removesuffix("_color")


def find_partner(path: Path, recordings: list[Path]) -> Path | None:
    camera = camera_or_none(path)
    partners = [other for other in recordings if camera_or_none(other) not in {None, camera} and take_key(other) == take_key(path)]
    return partners[0] if partners else None


def lost_frames(issue: dict[str, Any]) -> int | None:
    if issue["check"] == "missing_frame":
        return len(issue["frames"])
    if issue["check"] not in GAP_CHECKS:
        return None
    match = LOST_FRAMES.search(issue["message"])
    if match is None:
        raise ValueError(f"{issue['check']}: no count of lost frames in the message: {issue['message']}")
    return int(match[1] or match[2])


# the number of lost frame numbers of each gap, in the order of issue["frames"] (the frameset after each gap)
def gap_sizes(issue: dict[str, Any]) -> list[int]:
    ranges = GAP_RANGE.findall(issue["message"].split("missing: ", 1)[1])
    return [int(end) - int(start) + 1 if start else 1 for start, end, _ in ranges]


def count_severity(count: int) -> str:
    return "critical" if count > MAX_MEDIUM_LOST_FRAMES else "medium"


# a gap that ends at a missing_frame frameset (or right after one) is the same event as the missing_frame, so it takes the
# severity of the missing_frame; the other gaps of the check are added and follow the count rule
def frame_number_gap_severity(issue: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    missing_frames = [other for other in issues if other["check"] == "missing_frame"]
    severities, other_lost = set(), 0
    for gap_end, size in zip(issue["frames"], gap_sizes(issue), strict=True):
        event = next((other for other in missing_frames if gap_end in other["frames"] or gap_end - 1 in other["frames"]), None)
        if event is None:
            other_lost += size
            continue
        severities.add(count_severity(len(event["frames"])))
    if other_lost:
        severities.add(count_severity(other_lost))
    return "critical" if "critical" in severities else "medium"


# issues: all the issues of the run, because a frame_number_gaps issue depends on the missing_frame issues
def issue_severity(issue: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    if issue["check"] in CRITICAL_CHECKS:
        return "critical"
    if issue["check"] == "frame_number_gaps":
        return frame_number_gap_severity(issue, issues)
    count = lost_frames(issue)
    return "medium" if count is None else count_severity(count)


# a file that cannot be read is critical
def run_severity(run: dict[str, Any]) -> str:
    if run["error"]:
        return "critical"
    severities = {issue_severity(issue, run["issues"]) for issue in run["issues"]}
    if "critical" in severities:
        return "critical"
    return "medium" if severities else "OK"


def load_recording(path: Path) -> Loaded:
    if path.suffix == ".mp4":
        return Mp4Video(path)
    if camera_of(path) == "mirrorless" and path.suffix == ".db3":
        return MirrorlessRos2Loader(path).load()
    return RealSenseLoader(path, forward_fill=False).load()


# the validators print the measured fps; the printed lines are kept with the issues
def validate_recording(path: Path, loaded: Loaded) -> dict[str, Any]:
    if isinstance(loaded, Mp4Video):
        fps_lines, issues = validate_mp4(loaded)
        return {"fps": fps_lines, "issues": [issue.model_dump(mode="json") for issue in issues], "error": None}
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        if isinstance(loaded, list):
            issues = MirrorlessRos2Validator(loaded).validate(raise_error=False)
        elif camera_of(path) == "d435":
            issues = D435Validator(loaded).validate(raise_error=False)
        else:
            issues = MirrorlessRosValidator(loaded).validate(raise_error=False)
    return {"fps": printed.getvalue().splitlines(), "issues": [issue.model_dump(mode="json") for issue in issues], "error": None}


# the motion signal for the cross-correlation; an .mp4 file needs a decode of every frame, so it is made only on request
def motion_of(path: Path, loaded: Loaded) -> dict:
    if camera_of(path) == "d435":
        return d435_motion(loaded)
    return mirrorless_motion(loaded)


def validation_run(path: Path, loaded: Loaded) -> dict[str, Any]:
    run = {"validated_at": datetime.now().isoformat(timespec="seconds")}
    try:
        return run | validate_recording(path, loaded)
    except READ_ERRORS:
        return run | {"fps": [], "issues": [], "error": traceback.format_exc()}


# one file of Validate all, from one load: the run for the history, and the motion signal for the cross-correlation (None for an
# .mp4 file or an error); the loaded frames are freed when the function returns
def validate_file(path: Path) -> tuple[dict[str, Any], dict | None]:
    try:
        loaded = load_recording(path)
    except READ_ERRORS:
        run = {"validated_at": datetime.now().isoformat(timespec="seconds")}
        return run | {"fps": [], "issues": [], "error": traceback.format_exc()}, None
    run = validation_run(path, loaded)
    if isinstance(loaded, Mp4Video):
        return run, None
    try:
        return run, motion_of(path, loaded)
    except READ_ERRORS:
        return run, None
