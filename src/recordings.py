import contextlib
import csv
import io
import json
import re
import sqlite3
import statistics
import struct
import traceback
from collections.abc import Callable, Sequence
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
    Db3,
    MirrorlessRos2Loader,
    MirrorlessRos2Validator,
    MirrorlessRosValidator,
    RealSenseLoader,
    Recording,
    Ros1BagIndex,
    ValidationIssue,
    measure,
)

RECORDING_SUFFIXES = {".bag", ".db3", ".mp4"}

type Loaded = Recording | list[CompressedImageMessage] | Mp4Video

# the errors of a damaged or unreadable recording: pyrealsense2 raises RuntimeError, PyAV raises FFmpegError, a .db3 file can raise
# sqlite3, struct, or zstd errors, and the readers raise ValueError and IndexError
READ_ERRORS = (OSError, RuntimeError, ValueError, IndexError, av.FFmpegError, sqlite3.Error, struct.error, zstd.ZstdError)

# Hardware Instability: a known group of issues, the D435 notifications at a stream start; the depth data after each one was at the
# level of a take with no notification (recording_survey.md, the D435 notification issue)
HARDWARE_INSTABILITY = {"Left MIPI error", "Right MIPI error", "Depth stream start failure"}
# Half image rate: a known group of issues, the checks of a file whose color copies come in pairs (each image in 2 frames); in the
# stored runs, the files with this pattern had 93.3% to 100% steps of 2 frames between 2 copies, and the other files 0% to 33.6%
HALF_IMAGE_RATE_CHECKS = {"duplicate_image", "image_fps", "report_no_frame_drops"}
MIN_PAIR_STEPS = 0.9
# the text of each known group for the app
GROUP_ABOUT = {
    "Hardware Instability": (
        f"The D435 notifications {', '.join(sorted(HARDWARE_INSTABILITY))} at a stream start. The depth data after each one was at the "
        "level of a take with no notification. This level wins over notification."
    ),
    "Half image rate": (
        f"The issues {', '.join(sorted(HALF_IMAGE_RATE_CHECKS))} of a file whose color copies come in pairs: at least "
        f"{MIN_PAIR_STEPS:.0%} of the steps between 2 copies are 2 frames. So about half of the frames are different images. "
        "Possible causes, not confirmed: a slow CPU, or the capture card downsampling the input (for example, a 60 Hz monitor on its "
        "HDMI OUT port). This level wins over the levels of these checks."
    ),
}
# the severity levels, the worst first
LEVELS = ["critical", "medium", "low"]
# the count rule of the checks that lose frames: medium for 1 or 2 lost frames in the file (all the gaps of the check added) and
# critical for more; a frame_number_gaps gap at a missing_frame frameset takes the severity of the missing_frame
# (frame_number_gap_severity); for duplicate_image the rule is a share: medium for copies in up to 5% of the frames, critical for more
BY_LOST_FRAMES = "by lost frames"
# the default severity of each known group and each check: a wrong camera setting or a stream that is not there is critical; the
# Hardware Instability group is low and the Half image rate group is critical, and a group level wins over the level of its checks;
# every other check is medium; the app settings can change each value (set_severity_levels)
DEFAULT_SEVERITY_LEVELS = {
    "Hardware Instability": "low",
    "Half image rate": "critical",
    **dict.fromkeys(["missing_stream", "fps", "size", "format", "serial_number", "unexpected_stream"], "critical"),
    **dict.fromkeys(["missing_frame", "duplicate_image", "frame_number_gaps", "barcode_gaps", "timestamp_gaps"], BY_LOST_FRAMES),
    **dict.fromkeys(
        [
            "color_depth_offset", "constant_value", "decreasing_barcode", "decreasing_frame_number", "depth_to_color",
            "dropped_capture_frames", "duplicate_barcode", "duplicate_frame_number", "frames_not_loaded", "image_fps", "measured_fps",
            "metadata_keys", "missing_barcode", "missing_partner", "notification", "pair_duration", "report", "report_no_frame_drops",
            "timestamp_domain", "timestamp_order",
        ],
        "medium",
    ),
}
severity_levels = dict(DEFAULT_SEVERITY_LEVELS)
GAP_CHECKS = {"frame_number_gaps", "barcode_gaps", "timestamp_gaps"}
MAX_MEDIUM_LOST_FRAMES = 2
MAX_MEDIUM_COPY_SHARE = 0.05
# "the same image as the previous frame in 30 of 1000 frames" (duplicate_image)
COPY_SHARE = re.compile(r"in (\d+) of (\d+) frames")
# "4 frame numbers missing" (frame_number_gaps, barcode_gaps) or "about 867 frames missing" (timestamp_gaps)
LOST_FRAMES = re.compile(r"(\d+) frame numbers missing|about (\d+) frames missing")
# one range per gap after "missing: " in a frame_number_gaps message: "[90430, 90434]" or "77"
GAP_RANGE = re.compile(r"\[(\d+), (\d+)\]|(\d+)")


# r and m are camera markers at the start of the name or after a "-", and only when no letter or digit follows: 1-m.bag,
# pipeline-r_20261012_151250.db3, r_20261113_134220.db3, -m_20261113_143019.mp4, and 1-m-retake.bag have a marker, and the -r of
# 1-m-retake.bag is not one
CAMERA_MARKER = re.compile(r"(?:^|-)([rm])(?![A-Za-z0-9])")
CAMERAS = {"r": "d435", "m": "mirrorless"}
CAMERA_LABELS = {"d435": "D435", "mirrorless": "mirrorless"}
# the durations of the -r and the -m recording of one take can differ by this time
MAX_PAIR_DURATION_DIFFERENCE_MS = 1000
# the image topics of a .db3 file: RealSense (/device_0/sensor_0/Depth_0/image/data) and poc1 UvcRos2Bag
DB3_IMAGE_TOPIC = re.compile(r"/image/data$|/image_raw/compressed$")


# a file has the marker r (RealSense D435) or m (mirrorless) in its name; an .mp4 file with no marker is the mirrorless camera
def camera_or_none(path: Path) -> str | None:
    markers = set(CAMERA_MARKER.findall(path.stem))
    if len(markers) == 1:
        return CAMERAS[markers.pop()]
    return "mirrorless" if path.suffix == ".mp4" and not markers else None


def camera_of(path: Path) -> str:
    camera = camera_or_none(path)
    if camera is None:
        raise ValueError(f"{path.name}: the file name must contain exactly one of the markers r (RealSense D435) and m (mirrorless)")
    return camera


# 1, 2, 10 in number order, not 1, 10, 2
def natural_key(text: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def find_recordings(folder: Path) -> list[Path]:
    paths = [path for path in folder.rglob("*") if path.suffix in RECORDING_SUFFIXES and path.is_file()]
    return sorted(paths, key=lambda path: natural_key(path.relative_to(folder).as_posix()))


# the recordings of one take have the same key: 1-r.bag and 1-m.bag (or 1-m.mp4) in one folder, or r_20261113_134220.db3,
# r_20261113_134220.mp4, m_20261113_134220.mp4, and m_20261113_134220_color/m_20261113_134220_color.db3 (a mirrorless .db3 file
# is in a subfolder with its own name)
def take_key(path: Path) -> tuple[Path, str]:
    folder = path.parent.parent if path.parent.name == path.stem else path.parent
    return folder, CAMERA_MARKER.sub("-x", path.stem).removesuffix("_color")


# the name of the take of a file for the list of the app: the file name with no camera marker, no _color, and no suffix:
# 20261113_144012 for m_20261113_144012_color.db3, pipeline_20261012_151250 for pipeline-r_20261012_151250.db3, 1 for 1-m.bag
def take_label(path: Path) -> str:
    return CAMERA_MARKER.sub("", path.stem.removesuffix("_color")).strip("-_") or path.stem


# the name of a file under its take in the list of the app: the camera and the file type; a file with no camera marker keeps its name
def file_label(path: Path) -> str:
    camera = camera_or_none(path)
    return f"{CAMERA_LABELS[camera]} {path.suffix}" if camera is not None else path.name


# the takes of the recordings, in the order of their first file: (label, files of the take); the label of a take in a subfolder
# starts with the subfolder, because 2 subfolders can have a take with the same name (1-m.bag)
def group_takes(folder: Path, recordings: list[Path]) -> list[tuple[str, list[Path]]]:
    takes: dict[tuple[Path, str], list[Path]] = {}
    for path in recordings:
        takes.setdefault(take_key(path), []).append(path)
    groups = []
    for (take_folder, _), paths in takes.items():
        relative = take_folder.relative_to(folder).as_posix()
        label = take_label(paths[0])
        groups.append((label if relative == "." else f"{relative}/{label}", paths))
    return groups


# the recordings of one camera in the take of path; the files of the type of path are first: an .mp4 file for an .mp4 file, and a
# .bag or .db3 file for a .bag or .db3 file
def take_files(path: Path, recordings: list[Path], camera: str) -> list[Path]:
    files = [other for other in recordings if camera_or_none(other) == camera and take_key(other) == take_key(path)]
    return sorted(files, key=lambda other: (other.suffix == ".mp4") != (path.suffix == ".mp4"))


# the recording of the other camera in the take of path, of the same file type when the take has one
def find_partner(path: Path, recordings: list[Path]) -> Path | None:
    camera = camera_or_none(path)
    if camera is None:
        return None
    partners = take_files(path, recordings, "mirrorless" if camera == "d435" else "d435")
    return partners[0] if partners else None


# the 2 recordings for the cross-correlation of the take of path: the D435 .bag or .db3 file (a D435 .mp4 file has no depth and no
# absolute time), then the mirrorless recording when the take has one; None when the take has no D435 .bag or .db3 file
def correlation_pair(path: Path, recordings: list[Path]) -> list[Path] | None:
    camera = camera_or_none(path)
    if camera is None:
        return None
    d435_files = [other for other in take_files(path, recordings, "d435") if other.suffix != ".mp4"]
    if not d435_files:
        return None
    d435 = path if path in d435_files else d435_files[0]
    mirrorless = path if camera == "mirrorless" else find_partner(path, recordings)
    return [d435] if mirrorless is None else [d435, mirrorless]


# the recordings in the folder of the take and in its direct subfolders (a -m .db3 file is in its own subfolder)
def take_recordings(path: Path) -> list[Path]:
    folder = take_key(path)[0]
    return [other for pattern in ("*", "*/*") for other in folder.glob(pattern) if other.suffix in RECORDING_SUFFIXES and other.is_file()]


# the time from the first image to the last image, from the index of the file with no read of the images
def recording_duration_ms(path: Path) -> float:
    if path.suffix == ".mp4":
        return float(Mp4Video(path).times_ms[-1])
    if path.suffix == ".bag":
        return Ros1BagIndex(path).duration_ms
    db3 = Db3(path)
    spans = [(first, last) for topic, (_, first, last) in db3.topic_stats().items() if DB3_IMAGE_TOPIC.search(topic)]
    db3.connection.close()
    if not spans:
        raise ValueError(f"{path.name}: no image topic")
    return (max(last for _, last in spans) - min(first for first, _ in spans)) / 1e6


# the checks of the 2 recordings of one take: the other recording is there, and the 2 durations agree
def pair_issues(path: Path, loaded: Loaded | None = None) -> list[ValidationIssue]:
    camera = camera_of(path)
    partner = find_partner(path, take_recordings(path))
    if partner is None:
        wanted = "-m (mirrorless)" if camera == "d435" else "-r (RealSense D435)"
        return [ValidationIssue(check="missing_partner", message=f"no {wanted} recording of this take next to {path.name}")]
    duration_ms = float(loaded.times_ms[-1]) if isinstance(loaded, Mp4Video) else recording_duration_ms(path)
    try:
        partner_duration_ms = recording_duration_ms(partner)
    except READ_ERRORS as error:
        return [ValidationIssue(check="pair_duration", message=f"pair duration: {partner.name} cannot be read: {error}")]
    difference_ms = duration_ms - partner_duration_ms
    if abs(difference_ms) <= MAX_PAIR_DURATION_DIFFERENCE_MS:
        return []
    message = (
        f"pair duration: {path.name} {duration_ms / 1000:.3f} s, {partner.name} {partner_duration_ms / 1000:.3f} s, "
        f"difference {difference_ms:+.0f} ms, expected within {MAX_PAIR_DURATION_DIFFERENCE_MS} ms"
    )
    return [ValidationIssue(check="pair_duration", message=message)]


# poc1 writes meta/<take>.report.json and meta/<take>_sysmon.csv in the folder of the take; <take> is the file name with no suffix and
# no _color: m_20261113_144012 for m_20261113_144012_color/m_20261113_144012_color.db3; a RealSense Viewer file has no meta file
def meta_path(path: Path, suffix: str) -> Path:
    return take_key(path)[0] / "meta" / f"{path.stem.removesuffix('_color')}{suffix}"


# the requested_fps of the poc1 report of the take, or None when there is no report or no value
def requested_fps(path: Path) -> int | None:
    try:
        value = json.loads(meta_path(path, ".report.json").read_text()).get("requested_fps")
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, int) and value > 0 else None


# the checks of the poc1 report against the file: copies is the number of duplicate_image frames, None for an .mp4 file (its
# check has no copy count); dropped_capture_frames is in the reports of the recorder that counts the ffmpeg drop warnings
def report_issues(path: Path, copies: int | None) -> list[ValidationIssue]:
    report_path = meta_path(path, ".report.json")
    if not report_path.is_file():
        return []
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [ValidationIssue(check="report", message=f"report: {report_path.name} cannot be read: {error}")]
    issues = []
    if report.get("no_frame_drops") is True and copies:
        message = f"report: no_frame_drops is true in {report_path.name}, but {copies} frames are copies of the previous image"
        issues.append(ValidationIssue(check="report_no_frame_drops", message=message))
    dropped = report.get("dropped_capture_frames")
    if dropped:
        message = f"report: ffmpeg dropped {dropped} capture frames (dropped_capture_frames in {report_path.name})"
        issues.append(ValidationIssue(check="dropped_capture_frames", message=message))
    return issues


# the CPU median and the total RAM of the PC during the take, from the poc1 sysmon file; the first row is not in the CPU median,
# because psutil.cpu_percent(interval=None) gives the time since an earlier call (or 0.0 at the first call), not 1 second
def sysmon_measures(path: Path) -> list[dict[str, str]]:
    sysmon_path = meta_path(path, "_sysmon.csv")
    if not sysmon_path.is_file():
        return []
    try:
        rows = list(csv.DictReader(sysmon_path.open(newline="")))
        cpu = [float(row["cpu_percent"]) for row in rows[1:]]
        ram_gb = [float(row["ram_used_mb"]) / float(row["ram_percent"]) * 100 / 1024 for row in rows if float(row["ram_percent"]) > 0]
    except (OSError, KeyError, ValueError) as error:
        return [measure("PC", "sysmon", f"{sysmon_path.name} cannot be read: {error}")]
    if not cpu or not ram_gb:
        return [measure("PC", "sysmon", f"not available, {len(rows)} rows in {sysmon_path.name}")]
    return [
        measure("PC", "CPU median", f"{statistics.median(cpu):.1f}% in {len(cpu)} samples"),
        measure("PC", "total RAM", f"{statistics.median(ram_gb):.1f} GB"),
    ]


# the sysmon measures as one text line, as the validators print their measures
def sysmon_lines(path: Path) -> list[str]:
    measures = sysmon_measures(path)
    if not measures:
        return []
    return ["sysmon: " + ", ".join(f"{row['name']} {row['value']}" for row in measures)]


def lost_frames(issue: dict[str, Any]) -> int | None:
    if issue["check"] in ("missing_frame", "duplicate_image"):
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


def copy_severity(issue: dict[str, Any]) -> str:
    match = COPY_SHARE.search(issue["message"])
    if match is None:
        raise ValueError(f"duplicate_image: no count of frames in the message: {issue['message']}")
    return "critical" if int(match[1]) / int(match[2]) > MAX_MEDIUM_COPY_SHARE else "medium"


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
        severities.add(issue_severity(event, issues))
    if other_lost:
        severities.add(count_severity(other_lost))
    return min(severities, key=LEVELS.index)


# True when the color copies of the run come in pairs: at least MIN_PAIR_STEPS of the steps between 2 copies are 2 frames
def copies_in_pairs(issues: Sequence[dict[str, Any]]) -> bool:
    copies = next((issue["frames"] for issue in issues if issue["check"] == "duplicate_image" and issue["stream"] == "color"), [])
    steps = [b - a for a, b in zip(copies, copies[1:])]
    return bool(steps) and sum(1 for step in steps if step == 2) / len(steps) >= MIN_PAIR_STEPS


# the known group of an issue, or None; issues: all the issues of the run, because the Half image rate group depends on the
# duplicate_image issue; the message of a notification is
# "notification: Left MIPI error (hardware_error, error, Stereo Module), 1 time. What to do: ..."
def issue_group(issue: dict[str, Any], issues: Sequence[dict[str, Any]] = ()) -> str | None:
    description = issue["message"].removeprefix("notification: ").split(" (", 1)[0]
    if issue["check"] == "notification" and description in HARDWARE_INSTABILITY:
        return "Hardware Instability"
    if issue["check"] in HALF_IMAGE_RATE_CHECKS and copies_in_pairs(issues):
        return "Half image rate"
    return None


# the severity levels of the app settings, on top of the defaults; a key or a level that the app does not know is not used
def set_severity_levels(levels: dict[str, str]) -> None:
    known = {
        key: level for key, level in levels.items()
        if key in DEFAULT_SEVERITY_LEVELS and (level in LEVELS or level == BY_LOST_FRAMES == DEFAULT_SEVERITY_LEVELS[key])
    }
    severity_levels.clear()
    severity_levels.update(DEFAULT_SEVERITY_LEVELS | known)


# issues: all the issues of the run, because a frame_number_gaps issue depends on the missing_frame issues; the level of a known group
# wins over the level of the check; a check that the app does not know (in an older run) is medium
def issue_severity(issue: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    group = issue_group(issue, issues)
    level = severity_levels[group] if group else severity_levels.get(issue["check"], "medium")
    if level != BY_LOST_FRAMES:
        return level
    if issue["check"] == "frame_number_gaps":
        return frame_number_gap_severity(issue, issues)
    if issue["check"] == "duplicate_image":
        return copy_severity(issue)
    return count_severity(lost_frames(issue))


# a file that cannot be read is critical
def run_severity(run: dict[str, Any]) -> str:
    if run["error"]:
        return "critical"
    severities = {issue_severity(issue, run["issues"]) for issue in run["issues"]}
    if "critical" in severities:
        return "critical"
    if "medium" in severities:
        return "medium"
    return "low" if severities else "OK"


# should_stop: True stops the read of a RealSense file with LoadCancelled; the 2 other loads are short and do not ask
def load_recording(path: Path, should_stop: Callable[[], bool] | None = None) -> Loaded:
    if path.suffix == ".mp4":
        return Mp4Video(path)
    if camera_of(path) == "mirrorless" and path.suffix == ".db3":
        return MirrorlessRos2Loader(path).load()
    return RealSenseLoader(path, forward_fill=False).load(should_stop)


# the validators print the measured fps and return the measures; the printed lines ("fps") and the measures are kept with the issues
def validate_recording(path: Path, loaded: Loaded) -> dict[str, Any]:
    # a mirrorless take is checked at the fps that poc1 requested (120 or 60); with no report, the validators use 120
    fps = requested_fps(path) if camera_of(path) == "mirrorless" else None
    if isinstance(loaded, Mp4Video):
        fps_lines, issues, measures = validate_mp4(loaded, camera_of(path), fps)
        issues += pair_issues(path, loaded) + report_issues(path, copies=None)
    else:
        if isinstance(loaded, list):
            validator = MirrorlessRos2Validator(loaded, fps)
        elif camera_of(path) == "d435":
            validator = D435Validator(loaded)
        else:
            validator = MirrorlessRosValidator(loaded)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            issues = validator.validate(raise_error=False)
        fps_lines, measures = printed.getvalue().splitlines(), validator.measures
        copies = sum(len(issue.frames) for issue in issues if issue.check == "duplicate_image" and issue.stream == "color")
        issues += pair_issues(path) + report_issues(path, copies)
    return {
        "fps": fps_lines + sysmon_lines(path),
        "measures": measures + sysmon_measures(path),
        "issues": [issue.model_dump(mode="json") for issue in issues],
        "error": None,
    }


# the motion signal for the cross-correlation; an .mp4 file needs a decode of every frame, so it is made only on request
# should_stop: True stops the work with LoadCancelled
def motion_of(path: Path, loaded: Loaded, should_stop: Callable[[], bool] | None = None) -> dict:
    if camera_of(path) == "d435":
        return d435_motion(loaded, should_stop)
    return mirrorless_motion(loaded, should_stop)


def validation_run(path: Path, loaded: Loaded) -> dict[str, Any]:
    run = {"validated_at": datetime.now().isoformat(timespec="seconds")}
    try:
        return run | validate_recording(path, loaded)
    except READ_ERRORS:
        return run | {"fps": [], "measures": [], "issues": [], "error": traceback.format_exc()}


# one file of a validation, from one load: the run for the history, and the motion signal for the cross-correlation (None for an
# .mp4 file or an error); the loaded frames are freed when the function returns
def validate_file(path: Path) -> tuple[dict[str, Any], dict | None]:
    try:
        loaded = load_recording(path)
    except READ_ERRORS:
        run = {"validated_at": datetime.now().isoformat(timespec="seconds")}
        return run | {"fps": [], "measures": [], "issues": [], "error": traceback.format_exc()}, None
    run = validation_run(path, loaded)
    if isinstance(loaded, Mp4Video):
        return run, None
    try:
        return run, motion_of(path, loaded)
    except READ_ERRORS:
        return run, None
