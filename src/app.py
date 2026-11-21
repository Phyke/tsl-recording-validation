import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import (
    QElapsedTimer,
    QLocale,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPalette,
    QPixmap,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QTabWidget,
    QToolTip,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from motion_correlation import correlation_from_motion
from mp4_video import Mp4Video
from recording_reader import LoadCancelled, decode_jpeg, decode_jpeg_half
from recordings import (
    BY_LOST_FRAMES,
    DEFAULT_SEVERITY_LEVELS,
    GROUP_ABOUT,
    HALF_IMAGE_RATE_CHECKS,
    LEVELS,
    READ_ERRORS,
    Loaded,
    camera_or_none,
    correlation_pair,
    file_label,
    find_recordings,
    group_takes,
    issue_group,
    issue_severity,
    load_recording,
    motion_of,
    run_severity,
    set_severity_levels,
    severity_levels,
    validate_file,
    validation_run,
)

# the default font size; the Settings dialog changes it and saves it in SETTINGS_PATH
FONT_POINT_SIZE = 16
FONT_POINT_SIZE_RANGE = (8, 40)
THUMBNAIL_FONT_POINT_SIZE = 10
THUMBNAIL_MIN_HEIGHT = 100
MEASURE_COLUMNS = [
    ("Measure", "The name of the measure, under its camera stream (color or depth), or under PC for the computer of a poc1 take. "
                "Put the pointer on a measure to see how it is calculated and its full value. Click a stream row to open or close it."),
    ("Value", "The value of the measure. The measures have no limit, so they give no issue; the checks give the issues. A value in "
              "the color of a severity is the value of an issue, with the expected value."),
]
# the tooltip of each measure in the Measures table
MEASURE_ABOUT = {
    "file fps": "Frames per second in the file: (frames - 1) / duration.",
    "camera fps": "Frames per second that the camera numbered: (last frame number - first frame number) / duration.",
    "barcode fps": "Frames per second that poc1 numbered with the barcodes on the frames: (last barcode - first barcode) / duration. "
                   "poc1 gives a copy of the previous image its own barcode.",
    "kept": "The frames in the file as a percentage of the frames that the camera (or poc1) numbered. "
            "Less than 100% means that the file does not have all the numbered frames.",
    "frames": "The number of frames in the file.",
    "duration": "The time from the first frame to the last frame.",
    "image fps": "Different images per second: (different images - 1) / duration. A copy of the previous image is not a different image.",
    "different images": "The number of frames with an image that is different from the previous frame.",
    "longest run": "The largest number of frames in a row with the same image.",
    "run lengths": "The number of runs of each length of frames with the same image. "
                   "Runs of 3 to 5 frames are the pattern of a capture that gives 120 frames per second with about 30 new images.",
    "fill rate": "The part of the depth pixels that has a depth value: the mean of all the frames.",
    "lowest fill rate": "The fill rate of the frame with the lowest fill rate.",
    "noise": "How far the depth values of the largest flat surface are from a plane: the median RMS distance in pixels of disparity, "
             "in millimeters, at the distance of the surface.",
    "flat surface": "The part of the image that the flat surface of the noise measure covers.",
    "CPU median": "The median CPU load of the PC during the poc1 take, from meta/<take>_sysmon.csv. The first row is not in the median.",
    "total RAM": "The total RAM of the PC, from meta/<take>_sysmon.csv.",
}
# one "length: count" pair of the run lengths measure
RUN_LENGTH_PAIR = re.compile(r"(\d+): (\d+)")
# the expected value at the end of an issue message: "color.image_fps: 30.27 (...), expected 120 within 1%"
EXPECTED_TEXT = re.compile(r"expected (.+)$")
ISSUE_COLUMNS = [
    ("Check", "The name of the validation check that found the issue. The issues are under their known group. The issues of a "
              "known group have a known cause. Click a group row to open or close it."),
    ("Stream", "The camera stream that has the issue."),
    ("Frames", "The frame of the issue, or the number of frames with the issue. Select the row to go to these frames."),
    ("Message", "What the check found. Put the pointer on a row to see the full text, or select the row to see it in the box below."),
]
HISTORY_PATH = Path(__file__).resolve().parent.parent / "validation_history.json"
SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"
STATUS_COLORS = {"OK": "#1a7f37", "low": "#0969da", "medium": "#d97706", "critical": "#cf222e"}
# the item data of the recording list: the path of a file item (None for a take item), and the label of a take item
PATH_ROLE = Qt.ItemDataRole.UserRole
TAKE_LABEL_ROLE = Qt.ItemDataRole.UserRole + 1
# the item data of the issue list: the issue of an issue item (None for a group item)
ISSUE_ROLE = Qt.ItemDataRole.UserRole
# the group item of the issues that are in no known group
NO_GROUP = "No group"
# a plain text for each known group and each check: the Settings dialog shows it under the name
PLAIN_TEXTS = {
    "Hardware Instability": "The RealSense camera gave a short hardware message when its stream started.",
    "Half image rate": "The mirrorless camera repeated each image in 2 frames, so the video has about half of the expected images.",
    "error": "The file cannot be read.",
    "missing_stream": "A camera stream is not in the recording.",
    "fps": "The frame rate setting of the recording is not the expected value.",
    "size": "The image size is not the expected value.",
    "format": "The image format is not the expected format.",
    "serial_number": "The RealSense camera is not one of the known studio cameras.",
    "unexpected_stream": "The recording has a stream that it must not have.",
    "missing_frame": "Some frames of one camera stream are missing.",
    "duplicate_image": "Some frames repeat the previous image.",
    "frame_number_gaps": "The RealSense camera skipped some frame numbers, so some frames are lost.",
    "barcode_gaps": "The mirrorless camera skipped some frame numbers, so some frames are lost.",
    "timestamp_gaps": "The video has long time gaps between frames, so some frames are lost.",
    "color_depth_offset": "In some frames, the color image and the depth image were not taken at the same time.",
    "constant_value": "A camera value changed during the recording. This value must stay the same.",
    "decreasing_barcode": "The frame numbers of the mirrorless camera go back.",
    "decreasing_frame_number": "The frame numbers of the RealSense camera go back.",
    "depth_to_color": "The recording does not have the data that aligns the depth image with the color image.",
    "dropped_capture_frames": "The recording software reported lost frames from the capture card.",
    "duplicate_barcode": "The mirrorless camera gave the same frame number 2 times.",
    "duplicate_frame_number": "The RealSense camera gave the same frame number 2 times.",
    "frames_not_loaded": "Some frames in the file cannot be used.",
    "image_fps": "The video has fewer different images per second than expected.",
    "measured_fps": "The measured frame rate is not the expected value.",
    "metadata_keys": "Some frames have different frame information than the first frame.",
    "missing_barcode": "Some frames of the mirrorless camera have no readable frame number.",
    "missing_partner": "The recording of the other camera for this take is not in the folder.",
    "notification": "The RealSense camera gave a hardware message.",
    "pair_duration": "The 2 camera recordings of this take have different lengths.",
    "report": "The report of the recording software cannot be read.",
    "report_no_frame_drops": "The report of the recording software says that no frames were lost, but some images repeat.",
    "timestamp_domain": "The RealSense camera times come from the clock of the PC, not from the clock of the camera.",
    "timestamp_order": "Some frame times are not in order.",
}
# the dashes of the peak line of each cross-correlation curve
CURVE_DASHES = [(1, 3), (6, 4)]
# the plot colors for a light and a dark app palette; the surface and the text come from the palette (chart_colors); the curve
# colors are the first 2 colors of the dataviz palette, with the steps of each mode, checked against the surface of each mode
CHART_COLORS = {
    "light": {"curves": ["#2a78d6", "#eb6834"], "grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781"},
    "dark": {"curves": ["#3987e5", "#d95926"], "grid": "#2c2c2a", "axis": "#383835", "muted": "#898781"},
}
PLAYBACK_SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
# the app looks for new and removed recordings in the open folder at this interval
RESCAN_MS = 3000
# the jet colormap for depth: dark blue for near, dark red for far
JET = (np.clip(1.5 - np.abs(4 * np.linspace(0, 1, 256)[:, None] - np.array([3, 2, 1])), 0, 1) * 255).astype(np.uint8)


# "frame 16 " at the start of a message for one frame; the Frames column shows the frame
FRAME_PREFIX = re.compile(r"^frame \d+ ")
# the text box below the issue table shows this number of lines; a longer message scrolls
ISSUE_DETAIL_LINES = 3
# the note in the Result box for a run of an older version of the app, which has only the printed lines and no measures
OLD_RUN_TEXT = "An older version of the app made this result, so the Measures tab has no measures. Click Validate selected to get them."

# auto_load: load the video when a recording is selected; when False, a click on the player loads it
# last_folder: the folder that was open last; the app opens it at the next start when no folder is given
# severity_levels: the severity of each known group and each check that is not the default (recordings.DEFAULT_SEVERITY_LEVELS)
# hidden_issues: the known groups and the checks whose issues the issue list does not show; they still count in the status and the colors
# drawing: the processor that scales the video frames, CPU (ImageView) or GPU (GpuImageView); a change takes effect at the next start
DEFAULT_SETTINGS = {
    "font_point_size": FONT_POINT_SIZE,
    "auto_load": True,
    "last_folder": None,
    "severity_levels": {},
    "hidden_issues": [],
    "drawing": "CPU",
}
DRAWING_MODES = ["CPU", "GPU"]
DRAWING_ABOUT = (
    "CPU: the processor scales the video frames. GPU: the graphics card scales the frames with OpenGL. In both modes the CPU decodes "
    "the frames. Use CPU on a PC with no OpenGL driver, because there the GPU mode uses a slow software OpenGL. "
    "The change takes effect at the next start of the app."
)


def load_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return dict(DEFAULT_SETTINGS)
    return DEFAULT_SETTINGS | json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def save_settings(settings: dict[str, Any]) -> None:
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")


# the folder at the start of the app: a folder on the command line, else the last folder when it is still there
def start_folder(arguments: list[str], settings: dict[str, Any]) -> Path | None:
    if len(arguments) > 1:
        return Path(arguments[1])
    last_folder = settings["last_folder"]
    return Path(last_folder) if last_folder and Path(last_folder).is_dir() else None


# the font of every widget that has no font of its own, and of the large matplotlib plots
def set_app_font_size(size: int) -> None:
    app = QApplication.instance()
    font = app.font()
    font.setPointSize(size)
    app.setFont(font)
    mpl.rcParams["font.size"] = size


# the number of issues of each severity, the worst first: "1 critical, 2 medium"
def severity_counts_text(severities: list[str]) -> str:
    return ", ".join(f"{severities.count(level)} {level}" for level in LEVELS if level in severities)


def run_summary(run: dict[str, Any]) -> str:
    if run["error"]:
        return "critical: validation error"
    if not run["issues"]:
        return "OK"
    return severity_counts_text([issue_severity(issue, run["issues"]) for issue in run["issues"]])


# the text and the color of a run in the recording list and the Result box: the severity and the number of issues
def run_label(run: dict[str, Any]) -> tuple[str, str]:
    return run_summary(run), STATUS_COLORS[run_severity(run)]


# the Frames column: "frame 77" for one frame and "30 frames" for more, so that a frame index does not look like a count
def frames_text(frames: list[int]) -> str:
    if not frames:
        return ""
    return f"frame {frames[0]}" if len(frames) == 1 else f"{len(frames)} frames"


# "{1: 5, 3: 123}" of the run lengths measure as "5 runs of 1 frame, 123 runs of 3 frames"
def run_lengths_text(value: str) -> str:
    pairs = RUN_LENGTH_PAIR.findall(value)
    if not pairs:
        return value
    return ", ".join(
        f"{count} {'run' if count == '1' else 'runs'} of {length} {'frame' if length == '1' else 'frames'}" for length, count in pairs
    )


# the issues of a measure: a check names the measure at the start of its message, "color.image_fps: 30.27 ..." for the image fps
# of the color stream, and "color.camera_fps: ..." for the camera fps
def measure_issues(values: dict[str, str], issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix = f"{values['stream']}.{values['name'].replace(' ', '_')}: "
    return [issue for issue in issues if issue["message"].startswith(prefix)]


# the value of a measure with the expected value of its first issue: "30.27 (expected 120 within 1%)"
def value_with_expected(value: str, issues: list[dict[str, Any]]) -> str:
    expected = EXPECTED_TEXT.search(issues[0]["message"].splitlines()[0]) if issues else None
    return f"{value} (expected {expected.group(1)})" if expected else value


# the text of an issue for the Message column: the last line of the message with no frame, stream, or check name at its start,
# because the other columns show them; the tooltip of the row and the text box show the full message
def short_message(issue: dict[str, Any]) -> str:
    text = FRAME_PREFIX.sub("", issue["message"].splitlines()[-1])
    if issue["stream"]:
        text = text.removeprefix(f"{issue['stream']}.").removeprefix(f"{issue['stream']}: ")
    label, separator, rest = text.partition(": ")
    # "pair duration: ..." is the label of the check pair_duration
    return rest if separator and label.replace(" ", "_") == issue["check"] else text


# the settings key of an issue (severity_levels, hidden_issues): its known group, else its check; the group wins over the check
def issue_key(issue: dict[str, Any], issues: list[dict[str, Any]]) -> str:
    return issue_group(issue, issues) or issue["check"]


# the last validation run of each recording; a file of an older version of the app has a list of runs for each recording, and its
# last run is kept
class LastRuns:
    def __init__(self, path: Path) -> None:
        self.path = path
        stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.run_by_recording: dict[str, dict[str, Any]] = {
            key: value[-1] if isinstance(value, list) else value for key, value in stored.items() if value
        }

    @staticmethod
    def key(recording: Path) -> str:
        return recording.absolute().as_posix()

    def get(self, recording: Path) -> dict[str, Any] | None:
        return self.run_by_recording.get(self.key(recording))

    # the file is replaced in one step, so that a failure during the write keeps the previous runs
    def set(self, recording: Path, run: dict[str, Any]) -> None:
        self.run_by_recording[self.key(recording)] = run
        part_path = self.path.with_name(self.path.name + ".part")
        part_path.write_text(json.dumps(self.run_by_recording, indent=2), encoding="utf-8")
        part_path.replace(self.path)


# the frames of one loaded recording for the player
class Frames:
    def __init__(self, path: Path, loaded: Loaded) -> None:
        self.path = path
        self.loaded = loaded
        self.has_color, self.has_depth = True, False
        if isinstance(loaded, Mp4Video):
            times_ms = loaded.times_ms.tolist()
        elif isinstance(loaded, list):
            times_ms = [message.stamp_s * 1000 for message in loaded]
        else:
            times_ms = [frame.color.timestamp_ms if frame.color is not None else frame.depth.timestamp_ms for frame in loaded.frames]
            self.has_color = any(frame.color is not None for frame in loaded.frames)
            self.has_depth = any(frame.depth is not None for frame in loaded.frames)
        # a D435 .bag or .db3 recording has a depth view even with no depth stream, so that the view can show the missing_stream text
        self.expects_depth = self.has_depth or (camera_or_none(path) == "d435" and not isinstance(loaded, Mp4Video))
        self.count = len(times_ms)
        if self.count == 0:
            raise ValueError(f"{path.name}: no frames")
        self.times_ms = np.array(times_ms) - times_ms[0]
        intervals = np.diff(self.times_ms)
        intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
        # playback shows one frame per median frame period, so that a missing frame or a time gap does not stop the playback
        self.period_ms = float(np.median(intervals)) if intervals.size else 1000 / 30
        self.depth_colors = self.depth_color_table() if self.has_depth else None

    # one color for each raw depth value (65536 values), from the 1st to 99th percentile of the depth > 0 in about 50 frames,
    # so that a depth keeps its color in every frame and a frame needs only one table lookup
    def depth_color_table(self) -> np.ndarray:
        depths = [frame.depth for frame in self.loaded.frames if frame.depth is not None]
        samples = [depth.frame[::4, ::4] for depth in depths[:: max(1, len(depths) // 50)]]
        raw = np.concatenate([sample[sample > 0] for sample in samples])
        low, high = np.percentile(raw, [1, 99]) if raw.size else (0.0, 1.0)
        index = np.clip((np.arange(65536) - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
        colors = JET[index]
        colors[0] = 0
        return colors

    # playing: a JPEG frame decodes at half size, because a full decode takes longer than the frame period of a 120 fps take; a
    # paused frame has the full size
    def color(self, i: int, playing: bool = False) -> np.ndarray | None:
        if isinstance(self.loaded, Mp4Video):
            return self.loaded.frame(i)
        if isinstance(self.loaded, list):
            return decode_jpeg_half(self.loaded[i].data) if playing else decode_jpeg(self.loaded[i].data)
        color = self.loaded.frames[i].color
        if color is None:
            return None
        return color.frame[..., ::-1] if color.format.endswith("bgr8") else color.frame

    def depth(self, i: int) -> np.ndarray | None:
        if self.depth_colors is None:
            return None
        depth = self.loaded.frames[i].depth
        return self.depth_colors[depth.frame] if depth is not None else None

    # the frame index starts at 0, as in the frame lists of the validation issues; the frame_number of a stream is the number that
    # the camera gave the frame
    def info(self, i: int) -> str:
        text = f"frame {i} (0 to {self.count - 1}), {self.times_ms[i] / 1000:.3f} s"
        if isinstance(self.loaded, (list, Mp4Video)):
            return text
        frame = self.loaded.frames[i]
        streams = [("color", frame.color), ("depth", frame.depth)] if self.has_depth else [("color", frame.color)]
        for name, stream in streams:
            text += f", {name} frame_number {stream.frame_number}" if stream is not None else f", {name} MISSING"
        return text


class WorkerSignals(QObject):
    loaded = Signal(str, object)
    load_failed = Signal(str, str)
    validated = Signal(str, object, object)
    correlated = Signal(object, object)
    progress = Signal(str)
    task_failed = Signal(str)
    validation_finished = Signal()
    correlation_cancelled = Signal()


def reuse_or_load(path: Path, player_frames: Frames | None) -> Loaded:
    if player_frames is not None and player_frames.path == path:
        return player_frames.loaded
    return load_recording(path)


# the load of the player; the motion signal for the cross-correlation comes only at a Cross-correlation click (CorrelateTask), so that
# it does not slow the playback after a load
# stopped: the recording is no longer selected; a task in the queue then does nothing, and a task that runs stops its load and sends
# no signal
class LoadTask(QRunnable):
    def __init__(self, signals: WorkerSignals, path: Path) -> None:
        super().__init__()
        self.signals, self.path = signals, path
        self.stopped = False
        self.finished = False

    def run(self) -> None:
        try:
            self.load()
        finally:
            self.finished = True

    def load(self) -> None:
        if self.stopped:
            return
        try:
            loaded = load_recording(self.path, lambda: self.stopped)
        except LoadCancelled:
            return
        except READ_ERRORS:
            self.signals.load_failed.emit(str(self.path), traceback.format_exc())
            return
        self.signals.loaded.emit(str(self.path), loaded)


# one recording at a time, so that only one recording is in memory; the recording in the player is not loaded again
class ValidateTask(QRunnable):
    def __init__(self, signals: WorkerSignals, paths: list[Path], player_frames: Frames | None) -> None:
        super().__init__()
        self.signals, self.paths, self.player_frames = signals, paths, player_frames
        self.stopped = False

    def run(self) -> None:
        try:
            for n, path in enumerate(self.paths, start=1):
                if self.stopped:
                    return
                self.signals.progress.emit(f"validating {n} of {len(self.paths)}: {path.name}")
                if self.player_frames is not None and self.player_frames.path == path:
                    self.signals.validated.emit(str(path), validation_run(path, self.player_frames.loaded), None)
                    continue
                run, motion = validate_file(path)
                self.signals.validated.emit(str(path), run, motion)
        finally:
            self.signals.validation_finished.emit()


# loads only the recordings with no motion signal yet
class CorrelateTask(QRunnable):
    def __init__(self, signals: WorkerSignals, paths: list[Path], motions: dict[Path, dict], player_frames: Frames | None) -> None:
        super().__init__()
        self.signals, self.paths, self.motions, self.player_frames = signals, paths, dict(motions), player_frames
        self.stopped = False

    def run(self) -> None:
        try:
            for path in self.paths:
                if self.stopped:
                    self.signals.correlation_cancelled.emit()
                    return
                if path not in self.motions:
                    self.signals.progress.emit(f"cross-correlation: reading {path.name}")
                    self.motions[path] = motion_of(path, reuse_or_load(path, self.player_frames))
            result = correlation_from_motion(self.motions[self.paths[0]], self.motions[self.paths[1]] if len(self.paths) > 1 else None)
        except READ_ERRORS:
            self.signals.task_failed.emit(traceback.format_exc())
            return
        self.signals.correlated.emit(result | {"paths": self.paths}, self.motions)


# a missing frame keeps the last frame on the screen with a dark cover and the text, so that the image does not flash
class ImageView(QLabel):
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)
        self.image: QImage | None = None
        self.missing_text = ""
        self.fast = False

    def show_array(self, rgb: np.ndarray | None, missing_text: str) -> None:
        if rgb is None:
            self.missing_text = missing_text
        else:
            self.missing_text = ""
            rgb = np.ascontiguousarray(rgb)
            height, width, _ = rgb.shape
            self.image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.render_image()

    def clear_image(self, text: str) -> None:
        self.image, self.missing_text = None, ""
        self.setPixmap(QPixmap())
        self.setText(text)

    def render_image(self) -> None:
        if self.image is None:
            self.setText(self.missing_text)
            return
        mode = Qt.TransformationMode.FastTransformation if self.fast else Qt.TransformationMode.SmoothTransformation
        pixmap = QPixmap.fromImage(self.image.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, mode))
        if self.missing_text:
            painter = QPainter(pixmap)
            painter.fillRect(pixmap.rect(), QColor(0, 0, 0, 150))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor("#ff6b6b"))
            painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"{self.missing_text}\n(the previous image is shown)")
            painter.end()
        self.setPixmap(pixmap)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.render_image()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit()


# the ImageView of the GPU drawing mode (Settings): it draws with OpenGL; during the playback the GPU scales the frame, and a paused
# frame has the smooth scale of the CPU, as in ImageView
class GpuImageView(QOpenGLWidget):
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(1, 1)
        self.image: QImage | None = None
        self.text = ""
        self.missing_text = ""
        self.fast = False

    def show_array(self, rgb: np.ndarray | None, missing_text: str) -> None:
        if rgb is None:
            self.missing_text = missing_text
        else:
            self.missing_text = ""
            rgb = np.ascontiguousarray(rgb)
            height, width, _ = rgb.shape
            self.image = QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()
        self.update()

    def clear_image(self, text: str) -> None:
        self.image, self.missing_text, self.text = None, "", text
        self.update()

    def paintGL(self) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QPalette.ColorRole.Window))
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        if self.image is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.missing_text or self.text)
            painter.end()
            return
        target = QRect(QPoint(0, 0), self.image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio))
        target.moveCenter(self.rect().center())
        if self.fast:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(target, self.image)
        else:
            smooth = self.image.scaled(target.size(), Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawImage(target, smooth)
        if self.missing_text:
            painter.fillRect(target, QColor(0, 0, 0, 150))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor("#ff6b6b"))
            painter.drawText(target, Qt.AlignmentFlag.AlignCenter, f"{self.missing_text}\n(the previous image is shown)")
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.clicked.emit()


# the plot colors of the app palette: a dark window gives the dark colors
def chart_colors() -> dict[str, Any]:
    palette = QApplication.palette()
    surface = palette.color(QPalette.ColorRole.Window)
    mode = "dark" if surface.lightness() < 128 else "light"
    return CHART_COLORS[mode] | {"surface": surface.name(), "ink": palette.color(QPalette.ColorRole.WindowText).name()}


# the matplotlib settings for a plot in the colors of the app; a figure gets its color when it is made, so set_chart_surface sets
# the color of a figure that is already there
def chart_style(colors: dict[str, Any]) -> dict[str, Any]:
    return {
        "figure.facecolor": colors["surface"], "axes.facecolor": colors["surface"], "axes.edgecolor": colors["axis"],
        "axes.labelcolor": colors["ink"], "axes.titlecolor": colors["ink"], "text.color": colors["ink"], "xtick.color": colors["muted"],
        "ytick.color": colors["muted"], "xtick.labelcolor": colors["ink"], "ytick.labelcolor": colors["ink"], "grid.color": colors["grid"],
        "legend.edgecolor": colors["axis"],
    }


def set_chart_surface(figure: Figure, colors: dict[str, Any]) -> None:
    figure.set_facecolor(colors["surface"])


def draw_correlation(axes: Axes, result: dict, legend_size: float | None = None, with_legend: bool = True) -> None:
    colors = chart_colors()
    shifts = result["shifts"]
    for curve, color, dashes in zip(result["curves"], colors["curves"], CURVE_DASHES, strict=False):
        if curve["peak"] is None:
            axes.plot(shifts, curve["values"], color=color, linewidth=2, label=f"{curve['label']}: no value")
            continue
        peak_shift, peak_value = curve["peak"]
        label = f"{curve['label']}: peak at {peak_shift:+d} frames, r {peak_value:.3f}"
        axes.plot(shifts, curve["values"], color=color, linewidth=2, label=label)
        axes.axvline(peak_shift, color=color, linewidth=2, linestyle=(0, dashes))
        # a label with the values at the peak; on the right half of the plot, the label is on the left side of the line
        right_half = peak_shift > (shifts[0] + shifts[-1]) / 2
        axes.plot(peak_shift, peak_value, "o", color=color, markersize=6)
        # with no legend, the label also names the curve
        shift_text = f"shift {peak_shift:+d} frames, r {peak_value:.3f}"
        text = shift_text if with_legend else f"{curve['label']}: {shift_text}"
        # the text has the text color and the box edge has the curve color, so that the text is easy to read on the surface
        axes.annotate(
            text, xy=(peak_shift, peak_value), xytext=(-6 if right_half else 6, 6),
            textcoords="offset points", ha="right" if right_half else "left", va="bottom", color=colors["ink"], fontsize=legend_size,
            fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "facecolor": colors["surface"], "edgecolor": color},
        )
    # room above the highest peak for its label
    axes.set_ymargin(0.3)
    axes.set_xlim(shifts[0], shifts[-1])
    axes.set_xticks(np.arange(shifts[0], shifts[-1] + 1, 10))
    axes.grid(color=colors["grid"])
    if with_legend:
        axes.legend(loc="best", fontsize=legend_size)


def correlation_title(result: dict) -> str:
    names = [path.name for path in result["paths"]]
    title = " and ".join(names) if len(names) > 1 else f"{names[0]} (no mirrorless recording)"
    return f"{title}\n{result['note']}" if result["note"] else title


class PlotWindow(QWidget):
    def __init__(self, result: dict) -> None:
        super().__init__()
        self.setWindowTitle(f"Cross-correlation: {correlation_title(result).splitlines()[0]}")
        colors = chart_colors()
        with mpl.rc_context(chart_style(colors)):
            figure = Figure(layout="constrained")
            canvas = FigureCanvasQTAgg(figure)
            layout = QVBoxLayout(self)
            layout.addWidget(NavigationToolbar2QT(canvas, self))
            layout.addWidget(canvas)
            axes = figure.add_subplot()
            draw_correlation(axes, result)
            axes.set_xlabel(f"shift k in D435 frames of {result['period_ms']:.1f} ms: second signal at t + k")
            axes.set_ylabel("correlation of the motion signals")
            axes.set_title(correlation_title(result))


# the small plot in the right panel; a click opens the large plot; the Cross-correlation button of the left panel runs the
# cross-correlation
class CorrelationThumbnail(FigureCanvasQTAgg):
    def __init__(self, open_large: Any) -> None:
        super().__init__(Figure(layout="constrained"))
        self.open_large = open_large
        self.result: dict | None = None
        self.drawn = False
        self.setMinimumHeight(THUMBNAIL_MIN_HEIGHT)
        self.show_result(None)

    def show_result(self, result: dict | None) -> None:
        # matplotlib draws slowly, so the same plot is not drawn again
        if self.drawn and result is self.result:
            return
        self.result, self.drawn = result, True
        self.setToolTip("Click Cross-correlation in the left panel to run it." if result is None else "Click to enlarge.")
        self.setCursor(Qt.CursorShape.ArrowCursor if result is None else Qt.CursorShape.PointingHandCursor)
        self.figure.clear()
        colors = chart_colors()
        set_chart_surface(self.figure, colors)
        with mpl.rc_context(chart_style(colors) | {"font.size": THUMBNAIL_FONT_POINT_SIZE}):
            axes = self.figure.add_subplot()
            if result is None:
                axes.set_axis_off()
                axes.text(0.5, 0.5, "No cross-correlation yet.\nClick Cross-correlation in the left panel.", ha="center", va="center")
            else:
                # the small plot has no legend and no title, so that the curves get the height; the peak labels name the curves
                draw_correlation(axes, result, THUMBNAIL_FONT_POINT_SIZE, with_legend=False)
        self.draw_idle()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.result is not None:
            self.open_large(self.result)


# the bar chart of the run lengths measure in the Measures tab: the number of runs of each length of frames with the same image,
# one plot for each stream with the measure; the chart is hidden when the run has no run lengths
class RunLengthChart(FigureCanvasQTAgg):
    def __init__(self) -> None:
        super().__init__(Figure(layout="constrained"))
        self.run: dict[str, Any] | None = None
        self.drawn = False
        self.bars: list[tuple[Any, str, int, int]] = []
        self.setMinimumHeight(THUMBNAIL_MIN_HEIGHT)
        self.mpl_connect("motion_notify_event", self.show_bar_tip)
        self.hide()

    @staticmethod
    def run_lengths(run: dict[str, Any] | None) -> list[tuple[str, list[tuple[int, int]]]]:
        measures = run.get("measures", []) if run else []
        streams = [(values["stream"], RUN_LENGTH_PAIR.findall(values["value"])) for values in measures if values["name"] == "run lengths"]
        return [(stream, [(int(length), int(count)) for length, count in pairs]) for stream, pairs in streams if pairs]

    def show_run(self, run: dict[str, Any] | None) -> None:
        # matplotlib draws slowly, so the chart of the same run is not drawn again
        if self.drawn and run is self.run:
            return
        self.run, self.drawn = run, True
        streams = self.run_lengths(run)
        self.setVisible(bool(streams))
        self.figure.clear()
        self.bars = []
        if not streams:
            return
        colors = chart_colors()
        set_chart_surface(self.figure, colors)
        with mpl.rc_context(chart_style(colors) | {"font.size": THUMBNAIL_FONT_POINT_SIZE}):
            for n, (stream, pairs) in enumerate(streams, start=1):
                axes = self.figure.add_subplot(1, len(streams), n)
                lengths, counts = zip(*pairs, strict=True)
                drawn = axes.bar(lengths, counts, width=0.6, color=colors["curves"][0])
                self.bars += [(bar, stream, length, count) for bar, length, count in zip(drawn, lengths, counts, strict=True)]
                axes.set_title(f"{stream}: runs of frames with the same image")
                axes.set_xlabel("run length (frames)")
                axes.set_ylabel("runs")
                axes.xaxis.set_major_locator(MaxNLocator(integer=True))
                axes.grid(axis="y", color=colors["grid"])
                axes.set_axisbelow(True)
                axes.spines[["top", "right"]].set_visible(False)
        self.draw_idle()

    # the pointer on a bar shows its values
    def show_bar_tip(self, event: Any) -> None:
        for bar, stream, length, count in self.bars:
            if bar.contains(event)[0]:
                runs, frames = "run" if count == 1 else "runs", "frame" if length == 1 else "frames"
                QToolTip.showText(QCursor.pos(), f"{stream}: {count} {runs} of {length} {frames}")
                return
        QToolTip.hideText()


# a range of 0 to 0 gives a busy bar that moves until it is hidden
def busy_bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 0)
    bar.setTextVisible(False)
    bar.hide()
    return bar


# the strip under the player slider: a mark at each issue frame of the shown issues, in the color of the issue severity, at the
# position of the frame on the slider; the marks of the selected issue are tall, the other marks are short; a click on a mark goes
# to its frame
class IssueStrip(QWidget):
    frame_clicked = Signal(int)
    # a click goes to the nearest mark within this distance in pixels
    CLICK_DISTANCE = 6
    HEIGHT = 14

    def __init__(self, slider: QSlider) -> None:
        super().__init__()
        self.slider = slider
        self.count = 0
        self.issues: list[tuple[dict[str, Any], str]] = []
        self.selected: dict[str, Any] | None = None
        self.setFixedHeight(self.HEIGHT)
        self.setToolTip(
            "The frames of the shown issues, in the color of their severity. The tall marks are the frames of the selected issue. "
            "Click a mark to go to its frame."
        )

    def set_count(self, count: int) -> None:
        self.count = count
        self.update()

    def set_issues(self, issues: list[tuple[dict[str, Any], str]]) -> None:
        self.issues, self.selected = issues, None
        self.update()

    def set_selected(self, issue: dict[str, Any] | None) -> None:
        self.selected = issue
        self.update()

    # the x of frame 0 and the distance between 2 frames: frame 0 and the last frame are at the center of the slider handle at its
    # 2 ends, so that a mark is under the handle at its frame
    def frame_scale(self) -> tuple[float, float]:
        option = QStyleOptionSlider()
        self.slider.initStyleOption(option)
        handle = self.slider.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self.slider)
        return handle.width() / 2, (self.width() - handle.width()) / max(self.count - 1, 1)

    # the marks from the lowest severity to the worst, and the selected issue last, so that the worst and the selected marks are on
    # top; the item data of the issue list can give a copy of an issue, so the selected issue is found by its value
    def marks(self) -> list[tuple[int, str, bool]]:
        if self.count == 0:
            return []
        order = sorted(self.issues, key=lambda entry: (entry[0] == self.selected, -LEVELS.index(entry[1])))
        return [(frame, severity, issue == self.selected) for issue, severity in order for frame in issue["frames"] if frame < self.count]

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.ColorRole.Mid))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        start, step = self.frame_scale()
        for frame, severity, selected in self.marks():
            painter.setPen(QColor(STATUS_COLORS[severity]))
            x = round(start + step * frame)
            painter.drawLine(x, 0 if selected else self.height() // 2, x, self.height() - 1)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        frames = [frame for frame, _, _ in self.marks()]
        if not frames:
            return
        start, step = self.frame_scale()
        x = event.position().x()
        frame = min(frames, key=lambda frame: abs(start + step * frame - x))
        if abs(start + step * frame - x) <= self.CLICK_DISTANCE:
            self.frame_clicked.emit(frame)


# a new font size applies at once; Cancel puts back the size from before the dialog
# the checks whose levels a known group wins over (recordings.issue_group): the Settings dialog names them on the row of the group
GROUP_CHECKS = {"Hardware Instability": ["notification"], "Half image rate": sorted(HALF_IMAGE_RATE_CHECKS)}
# the source of the values of each check, the section of the check in the Settings dialog: Metadata (what the camera or the
# recorder wrote into the file), Measured (what the app calculates from the frame numbers, the times, or the images), or Other files
# (the folder and the poc1 report)
CHECK_SOURCES = {
    **dict.fromkeys(
        [
            "missing_stream", "unexpected_stream", "fps", "size", "format", "serial_number", "depth_to_color", "timestamp_domain",
            "constant_value", "metadata_keys", "notification",
        ],
        "Metadata",
    ),
    **dict.fromkeys(
        [
            "measured_fps", "image_fps", "missing_frame", "frame_number_gaps", "barcode_gaps", "timestamp_gaps", "duplicate_image",
            "duplicate_frame_number", "duplicate_barcode", "decreasing_frame_number", "decreasing_barcode", "missing_barcode",
            "timestamp_order", "color_depth_offset", "frames_not_loaded", "pair_duration",
        ],
        "Measured",
    ),
    **dict.fromkeys(["missing_partner", "report", "dropped_capture_frames", "report_no_frame_drops"], "Other files"),
}


class SettingsDialog(QDialog):
    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.setWindowTitle("Settings")
        self.size_box = QSpinBox()
        self.size_box.setRange(*FONT_POINT_SIZE_RANGE)
        self.size_box.setSuffix(" pt")
        self.size_box.setValue(QApplication.font().pointSize())
        self.size_box.valueChanged.connect(window.apply_font_size)
        self.auto_load_box = QCheckBox("Load the video when a recording is selected")
        self.auto_load_box.setToolTip("When this is off, click the player to load the video.")
        self.auto_load_box.setChecked(window.settings["auto_load"])
        self.drawing_box = QComboBox()
        self.drawing_box.addItems(DRAWING_MODES)
        self.drawing_box.setCurrentText(window.settings["drawing"])
        self.drawing_box.setToolTip(DRAWING_ABOUT)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        reset_button = buttons.addButton("Reset to defaults", QDialogButtonBox.ButtonRole.ResetRole)
        reset_button.setToolTip("Set each setting to its default: the font size, the video load, the drawing, each level, and each Show box. "
                                "OK saves the change.")
        reset_button.clicked.connect(self.reset_settings)
        layout = QFormLayout(self)
        layout.addRow("Font size", self.size_box)
        layout.addRow(self.auto_load_box)
        layout.addRow("Draw frames with", self.drawing_box)
        layout.addRow(self.checks_box(set(window.settings["hidden_issues"])))
        layout.addRow(buttons)

    # one entry for each known group and each check, under a heading (the known groups, then the checks by the source of their values),
    # in columns that use the width of the screen, in a box that scrolls: the name with its plain text under it, the severity list,
    # the Default button, and the Show box; the stored results get the new levels with no new validation
    def checks_box(self, hidden: set[str]) -> QGroupBox:
        self.level_boxes: dict[str, QComboBox] = {}
        self.show_boxes: dict[str, QCheckBox] = {}
        self.name_labels: dict[str, QLabel] = {}
        self.default_buttons: dict[str, QPushButton] = {}
        rows = QWidget()
        grid = QGridLayout(rows)
        sections: dict[str, list[tuple[QWidget, QComboBox, QPushButton, QCheckBox]]] = {
            section: [] for section in ["Known groups", "Metadata", "Measured", "Other files"]
        }
        for key, default in DEFAULT_SEVERITY_LEVELS.items():
            section = "Known groups" if key in GROUP_ABOUT else CHECK_SOURCES[key]
            box = QComboBox()
            box.addItems(LEVELS + ([BY_LOST_FRAMES] if default == BY_LOST_FRAMES else []))
            box.setCurrentText(severity_levels[key])
            tooltip = f"Default: {default}."
            if key == "duplicate_image":
                tooltip += f" {BY_LOST_FRAMES}: medium for copies in up to 5% of the frames, critical for more."
            elif default == BY_LOST_FRAMES:
                tooltip += f" {BY_LOST_FRAMES}: medium for 1 or 2 lost frames in the file, critical for more."
            if key in GROUP_ABOUT:
                tooltip += f" A known group: {GROUP_ABOUT[key]}"
            label = QLabel(key)
            about_text = PLAIN_TEXTS.get(key, "")
            if key in GROUP_CHECKS:
                about_text += f" Wins over: {', '.join(GROUP_CHECKS[key])}."
            about = QLabel(about_text)
            about.setWordWrap(True)
            about.setStyleSheet("color: gray")
            # about 42 characters on one line, so that most plain texts take 1 or 2 lines and 2 columns fit on a 1920 px screen
            about.setMinimumWidth(42 * about.fontMetrics().averageCharWidth())
            cell = QWidget()
            cell.setToolTip(tooltip)
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(0)
            cell_layout.addWidget(label)
            cell_layout.addWidget(about)
            # the name and the plain text stay at the top when the entry next to it in the other column is taller
            cell_layout.addStretch(1)
            box.setToolTip(tooltip)
            show_box = QCheckBox("Show")
            show_box.setChecked(key not in hidden)
            show_box.setToolTip("Show these issues in the issue list. A hidden issue still counts in the status and the colors.")
            # shows only when the level is not the default; its space stays when it is hidden, so that the rows do not move; it is a child
            # of the rows from the start, so that it does not open as a window of its own
            default_button = QPushButton("Default", rows)
            default_button.setToolTip(f"Set this level back to its default: {default}.")
            policy = default_button.sizePolicy()
            policy.setRetainSizeWhenHidden(True)
            default_button.setSizePolicy(policy)
            default_button.clicked.connect(lambda _checked=False, key=key: self.level_boxes[key].setCurrentText(DEFAULT_SEVERITY_LEVELS[key]))
            sections.setdefault(section, []).append((cell, box, default_button, show_box))
            self.level_boxes[key] = box
            self.show_boxes[key] = show_box
            self.name_labels[key] = label
            self.default_buttons[key] = default_button
            self.mark_change(key)
            box.currentTextChanged.connect(lambda _text, key=key: self.mark_change(key))
        # each column has 5 grid columns: the name, the severity list, the Default button, the Show box, and a gap; as many columns as
        # fit in 90% of the screen width, at most 3
        entries = [entry for entries in sections.values() for entry in entries]
        gap = 3 * grid.horizontalSpacing()
        column_width = sum(max(widget.sizeHint().width() for widget in parts) for parts in zip(*entries)) + 4 * grid.horizontalSpacing() + gap
        columns = max(1, min(3, int(0.9 * self.screen().availableGeometry().width()) // column_width))
        for column in range(columns):
            grid.setColumnStretch(5 * column, 1)
            if column < columns - 1:
                grid.setColumnMinimumWidth(5 * column + 4, gap)
        for section, entries in sections.items():
            if not entries:
                continue
            # a line above each section after the first
            if grid.rowCount() > 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFrameShadow(QFrame.Shadow.Sunken)
                grid.addWidget(line, grid.rowCount(), 0, 1, 5 * columns - 1)
            title = QLabel(section)
            title.setStyleSheet("font-weight: bold")
            first_row = grid.rowCount()
            grid.addWidget(title, first_row, 0, 1, 5 * columns - 1)
            # the entries of a section go down the first column, then down the next column
            per_column = -(-len(entries) // columns)
            for i, (cell, box, default_button, show_box) in enumerate(entries):
                row, column = first_row + 1 + i % per_column, 5 * (i // per_column)
                grid.addWidget(cell, row, column)
                grid.addWidget(box, row, column + 1, Qt.AlignmentFlag.AlignTop)
                grid.addWidget(default_button, row, column + 2, Qt.AlignmentFlag.AlignTop)
                grid.addWidget(show_box, row, column + 3, Qt.AlignmentFlag.AlignTop)
        area = QScrollArea()
        area.setWidget(rows)
        area.setWidgetResizable(True)
        # the list gets the width of its rows and the scroll bar, so it has no horizontal scroll bar, and up to 70% of the screen height
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setMinimumWidth(rows.sizeHint().width() + area.verticalScrollBar().sizeHint().width() + 2 * area.frameWidth())
        area.setMinimumHeight(min(rows.sizeHint().height(), round(0.7 * self.screen().availableGeometry().height())))
        group = QGroupBox("Checks")
        QVBoxLayout(group).addWidget(area)
        return group

    # a level that is not the default: the name in bold, with the default after it, and the Default button
    def mark_change(self, key: str) -> None:
        default = DEFAULT_SEVERITY_LEVELS[key]
        changed = self.level_boxes[key].currentText() != default
        label = self.name_labels[key]
        label.setText(f"{key} (default: {default})" if changed else key)
        label.setStyleSheet("font-weight: bold" if changed else "")
        self.default_buttons[key].setVisible(changed)

    # each setting back to its default: the font size (the window shows it at once), the video load, the drawing, each level, and each
    # Show box; OK saves the change, Cancel keeps the settings from before
    def reset_settings(self) -> None:
        self.size_box.setValue(DEFAULT_SETTINGS["font_point_size"])
        self.auto_load_box.setChecked(DEFAULT_SETTINGS["auto_load"])
        self.drawing_box.setCurrentText(DEFAULT_SETTINGS["drawing"])
        for key, box in self.level_boxes.items():
            box.setCurrentText(DEFAULT_SEVERITY_LEVELS[key])
        for show_box in self.show_boxes.values():
            show_box.setChecked(True)

    # the levels that are not the default
    def changed_levels(self) -> dict[str, str]:
        return {key: box.currentText() for key, box in self.level_boxes.items() if box.currentText() != DEFAULT_SEVERITY_LEVELS[key]}

    def hidden_issues(self) -> list[str]:
        return [key for key, box in self.show_boxes.items() if not box.isChecked()]


# the issue list and the measure list: one item for each group, with its rows under it; a click on a group opens or closes it
def group_tree(columns: list[tuple[str, str]]) -> QTreeWidget:
    widget = QTreeWidget()
    widget.setColumnCount(len(columns))
    widget.setHeaderLabels([name for name, _ in columns])
    for column, (_, tooltip) in enumerate(columns):
        widget.headerItem().setToolTip(column, tooltip)
    widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    widget.setWordWrap(False)
    widget.setTextElideMode(Qt.TextElideMode.ElideRight)
    widget.setUniformRowHeights(True)
    widget.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    widget.header().setStretchLastSection(True)
    # a click on the arrow of a group does not send itemClicked; a double click is 2 clicks, so it does not toggle the group again
    widget.setExpandsOnDoubleClick(False)
    return widget


class MainWindow(QMainWindow):
    def __init__(self, folder: Path | None) -> None:
        super().__init__()
        self.setWindowTitle("TSL recording validation")
        self.last_runs = LastRuns(HISTORY_PATH)
        self.pool = QThreadPool()
        # one task at a time: one recording in memory, and never 2 RealSense playbacks at the same time
        self.pool.setMaxThreadCount(1)
        self.signals = WorkerSignals()
        self.signals.loaded.connect(self.on_loaded)
        self.signals.load_failed.connect(self.on_load_failed)
        self.signals.validated.connect(self.on_validated)
        self.signals.correlated.connect(self.on_correlated)
        self.signals.progress.connect(self.statusBar().showMessage)
        self.signals.task_failed.connect(self.on_task_failed)
        self.signals.validation_finished.connect(self.on_validation_finished)
        self.signals.correlation_cancelled.connect(self.on_correlation_cancelled)
        self.folder: Path | None = None
        self.recordings: list[Path] = []
        self.recording_items: dict[Path, QTreeWidgetItem] = {}
        self.requested: Path | None = None
        self.frames: Frames | None = None
        self.index = 0
        self.playing = False
        self.speed = 1.0
        self.play_start_index = 0
        self.play_clock = QElapsedTimer()
        self.issue_frames: list[int] = []
        self.issue_position = 0
        self.motions: dict[Path, dict] = {}
        self.correlations: dict[tuple[Path, ...], dict] = {}
        self.load_task: LoadTask | None = None
        # the stopped load tasks that the pool can still have; a task stays here until it ends, so that Python does not delete it
        # before that
        self.stopped_load_tasks: list[LoadTask] = []
        self.validate_task: ValidateTask | None = None
        self.correlate_task: CorrelateTask | None = None
        self.plot_windows: list[PlotWindow] = []
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.play_step)
        self.rescan_timer = QTimer(self)
        self.rescan_timer.setInterval(RESCAN_MS)
        self.rescan_timer.timeout.connect(self.rescan_folder)
        self.settings = load_settings()
        set_severity_levels(self.settings["severity_levels"])
        self.build_ui()
        self.apply_font_size(self.settings["font_point_size"])
        if folder is not None:
            self.open_folder(folder)

    def build_ui(self) -> None:
        splitter = QSplitter()
        for widget in [self.build_left(), self.build_middle(), self.build_right()]:
            splitter.addWidget(widget)
        # the widths of the 3 panels in the ratio 2 : 5 : 4
        splitter.setSizes([2000, 5000, 4000])
        self.setCentralWidget(splitter)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.toggle_play)

    def build_left(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        self.folder_label = QLabel("no folder")
        self.folder_label.setWordWrap(True)
        open_button = QPushButton("Open folder")
        open_button.clicked.connect(self.choose_folder)
        # one item for each take, with its files under it: a click on a take opens or closes it, a click on a file selects the file
        self.recording_list = QTreeWidget()
        self.recording_list.setHeaderHidden(True)
        # a long row is cut at its start, so that the take name and the status stay in view; the tooltip of the row has the full text
        self.recording_list.setTextElideMode(Qt.TextElideMode.ElideLeft)
        self.recording_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recording_list.currentItemChanged.connect(self.select_item)
        # a click on the arrow of a take does not send itemClicked; a double click is 2 clicks, so it does not toggle the take again
        self.recording_list.setExpandsOnDoubleClick(False)
        self.recording_list.itemClicked.connect(self.toggle_take)
        self.validate_selected_button = QPushButton("Validate selected")
        self.validate_selected_button.clicked.connect(self.validate_selected)
        self.correlate_button = QPushButton("Cross-correlation")
        self.correlate_button.clicked.connect(self.correlate_selected)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.open_settings)
        folder_row = QHBoxLayout()
        folder_row.addWidget(open_button, 1)
        folder_row.addWidget(settings_button)
        # the busy bar moves and Cancel shows while a validation or a cross-correlation runs
        self.task_bar = busy_bar()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setToolTip("Stop after the recording that is open now. The reader cannot stop inside one file.")
        self.cancel_button.clicked.connect(self.cancel_task)
        self.cancel_button.hide()
        task_row = QHBoxLayout()
        task_row.addWidget(self.task_bar, 1)
        task_row.addWidget(self.cancel_button)
        layout.addWidget(self.folder_label)
        layout.addLayout(folder_row)
        for widget in [self.recording_list, self.validate_selected_button, self.correlate_button]:
            layout.addWidget(widget)
        layout.addLayout(task_row)
        return left

    def build_middle(self) -> QWidget:
        middle = QWidget()
        layout = QVBoxLayout(middle)
        views = QHBoxLayout()
        view_class = GpuImageView if self.settings["drawing"] == "GPU" else ImageView
        self.color_view, self.depth_view = view_class(), view_class()
        self.color_view.clicked.connect(self.load_clicked)
        self.depth_view.clicked.connect(self.load_clicked)
        self.depth_column = QWidget()
        for title, view, column in [("color", self.color_view, QWidget()), ("depth", self.depth_view, self.depth_column)]:
            column_layout = QVBoxLayout(column)
            column_layout.addWidget(QLabel(title), 0)
            column_layout.addWidget(view, 1)
            views.addWidget(column, 1)
        layout.addLayout(views, 1)
        # moves while the player loads a recording
        self.load_bar = busy_bar()
        layout.addWidget(self.load_bar)
        self.info_label = QLabel("select a recording")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        controls = QHBoxLayout()
        self.previous_button, self.play_button, self.next_button = QPushButton("◀"), QPushButton("Play"), QPushButton("▶")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.play_button.clicked.connect(self.toggle_play)
        self.next_button.clicked.connect(lambda: self.step(1))
        self.speed_combo = QComboBox()
        for speed in PLAYBACK_SPEEDS:
            self.speed_combo.addItem(f"{speed:g}x", speed)
        self.speed_combo.setCurrentIndex(PLAYBACK_SPEEDS.index(1.0))
        self.speed_combo.currentIndexChanged.connect(self.set_speed)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.on_slider)
        # the issue frames under the slider, with the same width as the slider
        self.issue_strip = IssueStrip(self.slider)
        self.issue_strip.frame_clicked.connect(self.go_to_strip_frame)
        slider_column = QVBoxLayout()
        slider_column.setSpacing(0)
        slider_column.addWidget(self.slider)
        slider_column.addWidget(self.issue_strip)
        for widget in [self.previous_button, self.play_button, self.next_button, self.speed_combo]:
            controls.addWidget(widget)
        controls.addLayout(slider_column, 1)
        layout.addLayout(controls)
        return middle

    def build_right(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)

        status_box = QGroupBox("Result")
        status_column = QVBoxLayout(status_box)
        status_layout = QHBoxLayout()
        self.status_label = QLabel("not validated")
        # bold; apply_font_size sets the size
        status_font = QFont(self.font())
        status_font.setBold(True)
        self.status_label.setFont(status_font)
        # the time of the last validation of the file: a validation before a change of the checks can give a different result
        self.validated_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.validated_label)
        # shows for a run of an older version of the app, which has no measures
        self.old_run_label = QLabel(OLD_RUN_TEXT)
        self.old_run_label.setWordWrap(True)
        self.old_run_label.setStyleSheet("color: gray")
        self.old_run_label.hide()
        status_column.addLayout(status_layout)
        status_column.addWidget(self.old_run_label)

        measure_tab = QWidget()
        measure_layout = QVBoxLayout(measure_tab)
        self.measure_tree = group_tree(MEASURE_COLUMNS)
        self.measure_tree.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.measure_tree.itemClicked.connect(self.toggle_group)
        self.run_length_chart = RunLengthChart()
        measure_layout.addWidget(self.measure_tree, 3)
        measure_layout.addWidget(self.run_length_chart, 2)

        issue_tab = QWidget()
        issue_layout = QVBoxLayout(issue_tab)
        self.issue_tree = group_tree(ISSUE_COLUMNS)
        self.issue_tree.itemSelectionChanged.connect(self.select_issue)
        self.issue_tree.itemClicked.connect(self.toggle_group)
        # in place of the issue list, the text box, and the issue frame buttons when the run has no issue to show
        self.issue_empty_label = QLabel("")
        self.issue_empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.issue_empty_label.setWordWrap(True)
        self.issue_empty_label.hide()
        self.issue_detail = QPlainTextEdit()
        self.issue_detail.setReadOnly(True)
        issue_controls = QHBoxLayout()
        self.issue_previous_button, self.issue_next_button = QPushButton("◀ issue frame"), QPushButton("issue frame ▶")
        self.issue_previous_button.clicked.connect(lambda: self.step_issue(-1))
        self.issue_next_button.clicked.connect(lambda: self.step_issue(1))
        self.issue_frame_label = QLabel("")
        self.issue_controls = QWidget()
        issue_controls.setContentsMargins(0, 0, 0, 0)
        self.issue_controls.setLayout(issue_controls)
        issue_controls.addWidget(self.issue_previous_button)
        issue_controls.addWidget(self.issue_frame_label, 1, Qt.AlignmentFlag.AlignCenter)
        issue_controls.addWidget(self.issue_next_button)
        issue_layout.addWidget(self.issue_tree, 1)
        issue_layout.addWidget(self.issue_empty_label, 1)
        issue_layout.addWidget(self.issue_detail)
        issue_layout.addWidget(self.issue_controls)

        self.thumbnail = CorrelationThumbnail(self.open_large_plot)

        # each tab gets the full height below the Result row
        self.tabs = QTabWidget()
        self.tabs.addTab(issue_tab, "Issues")
        self.tabs.setTabToolTip(0, "Critical issues first. Select a row to go to its frames.")
        self.tabs.addTab(measure_tab, "Measures")
        self.tabs.addTab(self.thumbnail, "Cross-correlation")
        layout.addWidget(status_box)
        layout.addWidget(self.tabs, 1)
        return right

    # the app font, and the parts with a size from the font: the status text, the issue text box, and the table heights
    def apply_font_size(self, size: int) -> None:
        set_app_font_size(size)
        status_font = QFont(self.status_label.font())
        status_font.setPointSize(round(size * 1.3))
        self.status_label.setFont(status_font)
        # the tables and the issue text box get the new font in the font change events, so their sizes are measured after these events
        QTimer.singleShot(0, self.refresh_table_sizes)

    # the issue list and the measure list get their row height from the font
    def refresh_table_sizes(self) -> None:
        # ISSUE_DETAIL_LINES full lines; a longer message (an error trace) scrolls
        margins = 2 * (self.issue_detail.document().documentMargin() + self.issue_detail.frameWidth())
        self.issue_detail.setMaximumHeight(round(self.issue_detail.fontMetrics().lineSpacing() * ISSUE_DETAIL_LINES + margins))
        self.show_result(self.requested)

    def open_settings(self) -> None:
        size_before = QApplication.font().pointSize()
        dialog = SettingsDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.apply_font_size(size_before)
            return
        self.settings |= {
            "font_point_size": dialog.size_box.value(),
            "auto_load": dialog.auto_load_box.isChecked(),
            "severity_levels": dialog.changed_levels(),
            "hidden_issues": dialog.hidden_issues(),
        }
        drawing_before = self.settings["drawing"]
        self.settings["drawing"] = dialog.drawing_box.currentText()
        save_settings(self.settings)
        if self.settings["drawing"] != drawing_before:
            self.statusBar().showMessage(f"the frames are drawn with the {self.settings['drawing']} at the next start of the app")
        # the colors of the recording list and the result panel come from the levels
        set_severity_levels(self.settings["severity_levels"])
        for path in self.recording_items:
            self.refresh_item(path)
        self.show_result(self.requested)

    def choose_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Open a recording folder", str(self.folder or ""))
        if chosen:
            self.open_folder(Path(chosen))

    def open_folder(self, folder: Path) -> None:
        self.folder = folder
        self.settings["last_folder"] = str(folder.absolute())
        save_settings(self.settings)
        # a zero-width space after each separator lets a long path wrap
        self.folder_label.setText(str(folder).replace("\\", "\\\u200b").replace("/", "/\u200b"))
        takes = self.fill_recording_list(find_recordings(folder), keep_view=False)
        self.statusBar().showMessage(f"{len(self.recordings)} recordings in {takes} takes in {folder}")
        self.rescan_timer.start()
        if self.recordings:
            self.select_path(self.recordings[0])

    # puts the takes of recordings in the list and returns the number of takes; with keep_view, the takes that were open stay open,
    # the selected file stays selected with no new load, and the list keeps its scroll position
    def fill_recording_list(self, recordings: list[Path], keep_view: bool) -> int:
        top_items = [self.recording_list.topLevelItem(i) for i in range(self.recording_list.topLevelItemCount())]
        expanded = {item.data(0, TAKE_LABEL_ROLE) for item in top_items if item.isExpanded()} if keep_view else set()
        current = self.recording_list.currentItem()
        current_path = current.data(0, PATH_ROLE) if keep_view and current is not None else None
        scroll = self.recording_list.verticalScrollBar().value()
        self.recordings = recordings
        # clear() sends item changes with items of the previous list, which are not in the new self.recordings, so the list sends no
        # signal while it changes
        self.recording_list.blockSignals(True)
        self.recording_list.clear()
        self.recording_items = {}
        takes = group_takes(self.folder, recordings)
        for label, paths in takes:
            take_item = QTreeWidgetItem(self.recording_list)
            take_item.setData(0, PATH_ROLE, None)
            take_item.setData(0, TAKE_LABEL_ROLE, label)
            for path in paths:
                item = QTreeWidgetItem(take_item)
                item.setData(0, PATH_ROLE, str(path))
                self.recording_items[path] = item
                self.refresh_item(path)
            take_item.setExpanded(label in expanded)
        if current_path is not None and Path(current_path) in self.recording_items:
            self.recording_list.setCurrentItem(self.recording_items[Path(current_path)])
        self.recording_list.blockSignals(False)
        if keep_view:
            self.recording_list.verticalScrollBar().setValue(scroll)
        return len(takes)

    # a new or removed recording in the open folder changes the list; the open recording and its player do not change
    def rescan_folder(self) -> None:
        if self.folder is None or not self.folder.is_dir():
            return
        recordings = find_recordings(self.folder)
        if recordings != self.recordings:
            takes = self.fill_recording_list(recordings, keep_view=True)
            self.statusBar().showMessage(f"the list changed: {len(recordings)} recordings in {takes} takes in {self.folder}")

    # opens the take of the file and selects the file, which loads it
    def select_path(self, path: Path) -> None:
        item = self.recording_items[path]
        item.parent().setExpanded(True)
        self.recording_list.setCurrentItem(item)

    def refresh_item(self, path: Path) -> None:
        item = self.recording_items.get(path)
        if item is None:
            return
        run = self.last_runs.get(path)
        text, color = run_label(run) if run else ("", "")
        status = f"  [{text}]" if run else ""
        item.setText(0, file_label(path) + status)
        item.setToolTip(0, path.relative_to(self.folder).as_posix() + status)
        item.setForeground(0, QColor(color) if run else self.palette().text().color())
        self.refresh_take(item.parent())

    # a take shows the worst severity of the last runs of its files; a take with no validated file shows only its name
    def refresh_take(self, take_item: QTreeWidgetItem) -> None:
        paths = [Path(take_item.child(i).data(0, PATH_ROLE)) for i in range(take_item.childCount())]
        severities = {run_severity(run) for run in (self.last_runs.get(path) for path in paths) if run}
        worst = next((severity for severity in ["critical", "medium", "low", "OK"] if severity in severities), None)
        label = take_item.data(0, TAKE_LABEL_ROLE)
        take_item.setToolTip(0, "\n".join(path.relative_to(self.folder).as_posix() for path in paths))
        if worst is None:
            take_item.setText(0, label)
            take_item.setForeground(0, self.palette().text().color())
            return
        take_item.setText(0, f"{label}  [{worst}]")
        take_item.setForeground(0, QColor(STATUS_COLORS[worst]))

    def toggle_take(self, item: QTreeWidgetItem) -> None:
        if item.data(0, PATH_ROLE) is None:
            item.setExpanded(not item.isExpanded())

    # a take item is not a recording, so its selection changes nothing
    def select_item(self, current: QTreeWidgetItem | None) -> None:
        if current is not None and current.data(0, PATH_ROLE) is not None:
            self.select_recording(Path(current.data(0, PATH_ROLE)))

    def select_recording(self, path: Path) -> None:
        self.pause()
        self.stop_load()
        # the previous recording is freed before the next one loads, so that only one recording is in memory
        self.requested, self.frames = path, None
        self.issue_strip.set_count(0)
        self.depth_column.setVisible(camera_or_none(path) != "mirrorless" and path.suffix != ".mp4")
        if self.settings["auto_load"]:
            self.start_load(path)
        else:
            self.load_bar.hide()
            self.color_view.clear_image("click to load the video")
            self.depth_view.clear_image("click to load the video")
            self.info_label.setText(f"click the player to load {path.name}")
        self.show_result(path)
        self.thumbnail.show_result(self.correlations.get(tuple(correlation_pair(path, self.recordings) or [])))

    # the load of the previous recording is not wanted now: a new click does not wait for it
    def stop_load(self) -> None:
        if self.load_task is None:
            return
        self.load_task.stopped = True
        self.stopped_load_tasks = [task for task in self.stopped_load_tasks if not task.finished] + [self.load_task]
        self.load_task = None

    def start_load(self, path: Path) -> None:
        self.stop_load()
        self.color_view.clear_image("loading")
        self.depth_view.clear_image("loading")
        self.info_label.setText(f"loading {path.name}")
        self.load_bar.show()
        self.load_task = LoadTask(self.signals, path)
        self.load_task.setAutoDelete(False)
        self.pool.start(self.load_task)

    # with the automatic load off (Settings), a click on the player loads the selected recording
    def load_clicked(self) -> None:
        if self.requested is None or self.frames is not None or self.load_bar.isVisible():
            return
        self.start_load(self.requested)

    def on_loaded(self, path_text: str, loaded: Loaded) -> None:
        path = Path(path_text)
        if path != self.requested:
            return
        self.load_bar.hide()
        try:
            self.frames = Frames(path, loaded)
        except ValueError as error:
            self.info_label.setText(str(error))
            return
        self.depth_column.setVisible(self.frames.expects_depth)
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.frames.count - 1)
        self.slider.blockSignals(False)
        self.issue_strip.set_count(self.frames.count)
        self.show_frame(0)
        self.statusBar().showMessage(f"{path.name}: {self.frames.count} frames")

    def on_load_failed(self, path_text: str, trace: str) -> None:
        if Path(path_text) == self.requested:
            self.load_bar.hide()
            self.color_view.clear_image("")
            self.depth_view.clear_image("")
            self.info_label.setText(f"load failed: {trace.strip().splitlines()[-1]}")
        self.statusBar().showMessage(f"load failed: {Path(path_text).name}")

    def show_frame(self, i: int) -> None:
        if self.frames is None:
            return
        self.index = min(max(i, 0), self.frames.count - 1)
        try:
            color = self.frames.color(self.index, self.playing)
        except ValueError as error:
            self.info_label.setText(str(error))
            return
        # a stream with no frame in the whole recording is a missing_stream, not a missing_frame
        color_text = "missing_frame: no color image" if self.frames.has_color else "missing_stream: no color stream in this recording"
        self.color_view.show_array(color, color_text)
        if self.frames.has_depth:
            self.depth_view.show_array(self.frames.depth(self.index), "missing_frame: no depth image")
        elif self.frames.expects_depth:
            self.depth_view.show_array(None, "missing_stream: no depth stream in this recording")
        self.info_label.setText(self.frames.info(self.index))
        self.slider.blockSignals(True)
        self.slider.setValue(self.index)
        self.slider.blockSignals(False)

    def on_slider(self, value: int) -> None:
        self.pause()
        self.show_frame(value)

    def step(self, delta: int) -> None:
        self.pause()
        self.show_frame(self.index + delta)

    def set_fast_scaling(self, fast: bool) -> None:
        self.color_view.fast = self.depth_view.fast = fast

    def set_speed(self, combo_index: int) -> None:
        self.speed = self.speed_combo.itemData(combo_index)
        if self.playing:
            self.play_start_index = self.index
            self.play_clock.restart()
            self.schedule_tick()

    def toggle_play(self) -> None:
        if self.playing:
            self.pause()
            return
        if self.frames is None:
            return
        if self.index >= self.frames.count - 1:
            self.show_frame(0)
        self.playing = True
        self.set_fast_scaling(True)
        self.play_button.setText("Pause")
        self.play_start_index = self.index
        self.play_clock.start()
        self.schedule_tick()

    # each frame has a due time on the wall clock; a frame that is late is skipped, so that the playback keeps its speed
    def schedule_tick(self) -> None:
        due_ms = (self.index + 1 - self.play_start_index) * self.frames.period_ms / self.speed
        self.timer.start(max(0, round(due_ms - self.play_clock.elapsed())))

    def play_step(self) -> None:
        if not self.playing or self.frames is None:
            return
        target = self.play_start_index + int(self.play_clock.elapsed() * self.speed / self.frames.period_ms)
        self.show_frame(min(max(target, self.index + 1), self.frames.count - 1))
        if self.index >= self.frames.count - 1:
            self.pause()
            return
        self.schedule_tick()

    def pause(self) -> None:
        was_playing = self.playing
        self.playing = False
        self.timer.stop()
        self.play_button.setText("Play")
        self.set_fast_scaling(False)
        if was_playing and self.frames is not None:
            self.show_frame(self.index)

    def set_task_buttons_enabled(self, enabled: bool) -> None:
        for button in [self.validate_selected_button, self.correlate_button]:
            button.setEnabled(enabled)
        self.task_bar.setVisible(not enabled)
        self.cancel_button.setVisible(not enabled)
        self.cancel_button.setEnabled(True)

    # a task checks the stop flag before each recording, so the recording that is open now finishes first
    def cancel_task(self) -> None:
        for task in [self.validate_task, self.correlate_task]:
            if task is not None:
                task.stopped = True
        self.cancel_button.setEnabled(False)
        self.statusBar().showMessage("cancelling: the recording that is open now finishes first")

    def on_correlation_cancelled(self) -> None:
        self.set_task_buttons_enabled(True)
        self.statusBar().showMessage("cross-correlation cancelled")

    def validate(self, paths: list[Path]) -> None:
        if not paths:
            return
        self.set_task_buttons_enabled(False)
        self.validate_task = ValidateTask(self.signals, paths, self.frames)
        self.validate_task.setAutoDelete(False)
        self.pool.start(self.validate_task)

    def validate_selected(self) -> None:
        if self.requested is not None:
            self.validate([self.requested])

    def on_validated(self, path_text: str, run: dict[str, Any], motion: dict | None) -> None:
        path = Path(path_text)
        self.last_runs.set(path, run)
        if motion is not None:
            self.motions[path] = motion
        self.refresh_item(path)
        if path == self.requested:
            self.show_result(path)

    def on_validation_finished(self) -> None:
        self.set_task_buttons_enabled(True)
        self.statusBar().showMessage("validation cancelled" if self.validate_task.stopped else "validation done")

    # the last run of the file: the status, the measures, and the issues with the critical issues first
    def show_result(self, path: Path | None) -> None:
        self.measure_tree.clear()
        self.issue_tree.clear()
        self.issue_detail.clear()
        self.set_issue_frames([])
        run = self.last_runs.get(path) if path is not None else None
        self.run_length_chart.show_run(run)
        self.old_run_label.setVisible(run is not None and "measures" not in run)
        if run is None:
            self.status_label.setText("not validated")
            self.status_label.setStyleSheet("")
            self.validated_label.setText("click Validate selected" if path is not None else "")
            self.tabs.setTabText(0, "Issues")
            self.issue_strip.set_issues([])
            self.show_issue_text("Not validated." if path is not None else "No recording is selected.")
            return
        text, color = run_label(run)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}")
        self.validated_label.setText(f"validated {run['validated_at'].replace('T', ' ')}")
        self.fill_measure_tree(run)
        if run["error"]:
            rows = [{"check": "error", "stream": "", "frames": [], "message": run["error"].strip()}]
        else:
            # sorted() keeps the check order inside each severity
            rows = sorted(run["issues"], key=lambda issue: issue_severity(issue, run["issues"]) != "critical")
        # each issue with its severity and its known group, in the order of the rows; the Settings dialog hides the issues of some
        # groups and checks; a read error always shows
        hidden = set(self.settings["hidden_issues"])
        members, hidden_members = [], []
        for issue in rows:
            member = (issue, "critical" if run["error"] else issue_severity(issue, run["issues"]), issue_group(issue, run["issues"]))
            shown = run["error"] or issue_key(issue, run["issues"]) not in hidden
            (members if shown else hidden_members).append(member)
        self.fill_issue_groups(members)
        self.issue_strip.set_issues([(issue, severity) for issue, severity, _ in members])
        shown_count, hidden_count = len(members), len(hidden_members)
        self.tabs.setTabText(0, f"Issues ({shown_count}, {hidden_count} hidden)" if hidden_count else f"Issues ({shown_count})")
        if hidden_count and not shown_count:
            self.show_issue_text(f"No issue to show: the Settings dialog hides the {hidden_count} issues of this file.")
        else:
            self.show_issue_text("" if shown_count else "No issues.")
        # the first issue of the first open row
        top = next((item for item in map(self.issue_tree.topLevelItem, range(self.issue_tree.topLevelItemCount())) if item.isExpanded()), None)
        if top is not None and top.childCount():
            self.issue_tree.setCurrentItem(top.child(0))

    # the issue list: one row for each known group (and No group), with each issue under it
    def fill_issue_groups(self, members: list[tuple[dict[str, Any], str, str | None]]) -> None:
        groups: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for issue, severity, group in members:
            groups.setdefault(group or NO_GROUP, []).append((issue, severity))
        # the group with the worst issue first; sorted() keeps the order of the rows for groups of the same severity
        for group, members in sorted(groups.items(), key=lambda entry: min(LEVELS.index(severity) for _, severity in entry[1])):
            worst = LEVELS[min(LEVELS.index(severity) for _, severity in members)]
            group_item = QTreeWidgetItem(self.issue_tree)
            group_item.setText(0, f"{group}  [{severity_counts_text([severity for _, severity in members])}]")
            group_item.setFirstColumnSpanned(True)
            group_item.setForeground(0, QColor(STATUS_COLORS[worst]))
            group_item.setToolTip(0, GROUP_ABOUT.get(group, "The issues that are in no known group."))
            group_item.setData(0, ISSUE_ROLE, None)
            for issue, severity in members:
                label = severity if group == NO_GROUP else f"{severity}, {group}"
                item = QTreeWidgetItem(group_item, [issue["check"], issue["stream"] or "", frames_text(issue["frames"]), short_message(issue)])
                item.setData(0, ISSUE_ROLE, issue)
                # the row has the color of its severity, so that the issues that make the file critical are clear
                for column in range(len(ISSUE_COLUMNS)):
                    item.setForeground(column, QColor(STATUS_COLORS[severity]))
                    item.setToolTip(column, f"{label}: {issue['message']}")
            group_item.setExpanded(True)

    # text: the text in place of the issue list, the text box, and the issue frame buttons; an empty text shows the issue list
    def show_issue_text(self, text: str) -> None:
        self.issue_empty_label.setText(text)
        self.issue_empty_label.setVisible(bool(text))
        for widget in [self.issue_tree, self.issue_detail, self.issue_controls]:
            widget.setVisible(not text)

    # the measures under their stream, in the order of the streams in the run; a measure of an issue has the color of the issue
    # severity and the expected value; the tooltip has the explanation, the full value, and the issue message
    # a run of an older version of the app has no measures, only the printed lines; each line fills one row
    def fill_measure_tree(self, run: dict[str, Any]) -> None:
        if "measures" not in run:
            for line in run["fps"]:
                item = QTreeWidgetItem(self.measure_tree, [line])
                item.setFirstColumnSpanned(True)
                item.setToolTip(0, line)
            return
        stream_items: dict[str, QTreeWidgetItem] = {}
        for values in run["measures"]:
            if values["stream"] not in stream_items:
                stream_item = QTreeWidgetItem(self.measure_tree, [values["stream"]])
                stream_item.setFirstColumnSpanned(True)
                stream_item.setExpanded(True)
                stream_items[values["stream"]] = stream_item
            issues = measure_issues(values, run["issues"])
            value = run_lengths_text(values["value"]) if values["name"] == "run lengths" else values["value"]
            item = QTreeWidgetItem(stream_items[values["stream"]], [values["name"], value_with_expected(value, issues)])
            tooltip = "\n\n".join(text for text in [MEASURE_ABOUT.get(values["name"], ""), f"Value: {value}"] if text)
            severities = [issue_severity(issue, run["issues"]) for issue in issues]
            for issue, severity in zip(issues, severities, strict=True):
                tooltip += f"\n\n{severity} issue: {issue['message']}"
            for column in range(len(MEASURE_COLUMNS)):
                item.setToolTip(column, tooltip)
                if severities:
                    item.setForeground(column, QColor(STATUS_COLORS[min(severities, key=LEVELS.index)]))

    # a group item is not an issue, so it clears the text box and the issue frames
    def select_issue(self) -> None:
        items = self.issue_tree.selectedItems()
        if not items:
            return
        issue = items[0].data(0, ISSUE_ROLE)
        self.issue_strip.set_selected(issue)
        if issue is None:
            self.issue_detail.clear()
            self.set_issue_frames([])
            return
        self.issue_detail.setPlainText(issue["message"])
        self.set_issue_frames(issue["frames"])
        if self.issue_frames:
            self.go_to_issue_frame()

    def toggle_group(self, item: QTreeWidgetItem) -> None:
        if item.parent() is None:
            item.setExpanded(not item.isExpanded())

    # a click on a mark of the issue strip goes to its frame; a frame of the selected issue also sets the issue frame buttons
    def go_to_strip_frame(self, frame: int) -> None:
        if frame in self.issue_frames:
            self.issue_position = self.issue_frames.index(frame)
            self.go_to_issue_frame()
            return
        self.pause()
        self.show_frame(frame)

    def set_issue_frames(self, frames: list[int]) -> None:
        self.issue_frames, self.issue_position = frames, 0
        self.issue_previous_button.setEnabled(len(frames) > 1)
        self.issue_next_button.setEnabled(len(frames) > 1)
        self.issue_frame_label.setText(f"frame {frames[0]} (issue frame 1 of {len(frames)})" if frames else "")

    def go_to_issue_frame(self) -> None:
        frame = self.issue_frames[self.issue_position]
        self.issue_frame_label.setText(f"frame {frame} (issue frame {self.issue_position + 1} of {len(self.issue_frames)})")
        if self.frames is not None and self.frames.path == self.requested:
            self.pause()
            self.show_frame(frame)

    def step_issue(self, delta: int) -> None:
        if not self.issue_frames:
            return
        self.issue_position = (self.issue_position + delta) % len(self.issue_frames)
        self.go_to_issue_frame()

    # with the motion signals in memory (from a validation or an earlier Cross-correlation) the curves need no file read
    def correlate_selected(self) -> None:
        # a click on the small plot while a task runs does not start a second task
        if self.requested is None or not self.correlate_button.isEnabled():
            return
        paths = correlation_pair(self.requested, self.recordings)
        if paths is None:
            self.statusBar().showMessage(f"cross-correlation: no -r recording for {self.requested.name}")
            return
        if all(path in self.motions for path in paths):
            result = correlation_from_motion(self.motions[paths[0]], self.motions[paths[1]] if len(paths) > 1 else None)
            self.on_correlated(result | {"paths": paths}, {})
            return
        self.set_task_buttons_enabled(False)
        self.correlate_task = CorrelateTask(self.signals, paths, self.motions, self.frames)
        self.correlate_task.setAutoDelete(False)
        self.pool.start(self.correlate_task)

    def on_correlated(self, result: dict, motions: dict[Path, dict]) -> None:
        self.set_task_buttons_enabled(True)
        self.motions |= motions
        # a D435 .db3 file is in 2 results (with the mirrorless .db3 file and with the mirrorless .mp4 file), so the key is the pair
        self.correlations[tuple(result["paths"])] = result
        if self.requested is not None and correlation_pair(self.requested, self.recordings) == result["paths"]:
            self.thumbnail.show_result(result)
        self.statusBar().showMessage("cross-correlation done: click the small plot to enlarge it")

    def open_large_plot(self, result: dict) -> None:
        window = PlotWindow(result)
        self.plot_windows.append(window)
        window.showMaximized()

    def on_task_failed(self, trace: str) -> None:
        self.set_task_buttons_enabled(True)
        self.statusBar().showMessage(f"failed: {trace.strip().splitlines()[-1]}")

    def closeEvent(self, event: Any) -> None:
        self.pause()
        for task in [self.load_task, self.validate_task, self.correlate_task]:
            if task is not None:
                task.stopped = True
        self.pool.clear()
        super().closeEvent(event)


# the number boxes of Qt follow the number format of Windows; a Thai format gives Thai digits, so the app uses the English format
def use_english_numbers() -> None:
    QLocale.setDefault(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))


def main() -> None:
    app = QApplication(sys.argv)
    use_english_numbers()
    settings = load_settings()
    set_app_font_size(settings["font_point_size"])
    window = MainWindow(start_folder(sys.argv, settings))
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
