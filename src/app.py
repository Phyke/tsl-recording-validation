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
from PySide6.QtCore import (
    QElapsedTimer,
    QObject,
    QRunnable,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPixmap,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
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
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from motion_correlation import correlation_from_motion
from mp4_video import Mp4Video
from recording_reader import LoadCancelled, decode_jpeg
from recordings import (
    BY_LOST_FRAMES,
    DEFAULT_SEVERITY_LEVELS,
    GROUP_ABOUT,
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
    ("Stream", "The camera stream (color or depth), or PC for the computer of a poc1 take."),
    ("Measure", "The name of the measure. Put the pointer on a measure to see how it is calculated."),
    ("Value", "The value of the measure. The measures have no limit, so they give no issue; the checks give the issues."),
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
# the simple view (the Details box is off) is for the people who record: one verdict for each severity, and a plain text for each
# known group and each check
VERDICTS = {"critical": "Problem", "medium": "Check", "low": "Good", "OK": "Good"}
VERDICT_COLORS = {"Problem": "#cf222e", "Check": "#d97706", "Good": "#1a7f37"}
# the simple view puts the low issues in this closed row
OTHER_NOTES = "Other notes"
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
CURVE_STYLES = [("#2a78d6", (1, 3)), ("#eb6834", (6, 4))]
PLAYBACK_SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
# the app looks for new and removed recordings in the open folder at this interval
RESCAN_MS = 3000
# the jet colormap for depth: dark blue for near, dark red for far
JET = (np.clip(1.5 - np.abs(4 * np.linspace(0, 1, 256)[:, None] - np.array([3, 2, 1])), 0, 1) * 255).astype(np.uint8)


# "frame 16 " at the start of a message for one frame; the Frames column shows the frame
FRAME_PREFIX = re.compile(r"^frame \d+ ")
# the text box below the issue table shows this number of lines; a longer message scrolls
ISSUE_DETAIL_LINES = 3

# auto_load: load the video when a recording is selected; when False, a click on the player loads it
# last_folder: the folder that was open last; the app opens it at the next start when no folder is given
# severity_levels: the severity of each known group and each check that is not the default (recordings.DEFAULT_SEVERITY_LEVELS)
# hidden_issues: the known groups and the checks whose issues the issue list does not show; they still count in the status and the colors
# details: the full view (the Details box); off: the simple view
DEFAULT_SETTINGS = {
    "font_point_size": FONT_POINT_SIZE,
    "auto_load": True,
    "last_folder": None,
    "severity_levels": {},
    "hidden_issues": [],
    "details": False,
}


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


def run_summary(run: dict[str, Any]) -> str:
    if run["error"]:
        return "critical: validation error"
    if not run["issues"]:
        return "OK"
    count = len(run["issues"])
    return f"{run_severity(run)}: {count} {'issue' if count == 1 else 'issues'}"


# the text and the color of a run in the recording list and the Result box: the verdict in the simple view, the severity and the number
# of issues in the full view
def run_label(run: dict[str, Any], details: bool) -> tuple[str, str]:
    severity = run_severity(run)
    if details:
        return run_summary(run), STATUS_COLORS[severity]
    return VERDICTS[severity], VERDICT_COLORS[VERDICTS[severity]]


# the Frames column: "frame 77" for one frame and "30 frames" for more, so that a frame index does not look like a count
def frames_text(frames: list[int]) -> str:
    if not frames:
        return ""
    return f"frame {frames[0]}" if len(frames) == 1 else f"{len(frames)} frames"


# the row of an issue in the simple view: one row for each known group, and one row for each check and stream in no group
def verdict_key(issue: dict[str, Any], severity: str, group: str | None) -> tuple:
    return (severity, group) if group else (severity, issue["check"], issue["stream"])


# "{1: 5, 3: 123}" of the run lengths measure as "5 runs of 1 frame, 123 runs of 3 frames"
def run_lengths_text(value: str) -> str:
    pairs = RUN_LENGTH_PAIR.findall(value)
    if not pairs:
        return value
    return ", ".join(
        f"{count} {'run' if count == '1' else 'runs'} of {length} {'frame' if length == '1' else 'frames'}" for length, count in pairs
    )


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

    def color(self, i: int) -> np.ndarray | None:
        if isinstance(self.loaded, Mp4Video):
            return self.loaded.frame(i)
        if isinstance(self.loaded, list):
            return decode_jpeg(self.loaded[i].data)
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
    motion_ready = Signal(str, object)
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


# the player gets the frames first; the motion signal for the cross-correlation comes after (not for .mp4: it needs a full decode)
# stopped: the recording is no longer selected; a task in the queue then does nothing, and a task that runs stops its load or its
# motion signal and sends no signal
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
        if self.stopped or isinstance(loaded, Mp4Video):
            return
        try:
            self.signals.motion_ready.emit(str(self.path), motion_of(self.path, loaded, lambda: self.stopped))
        except LoadCancelled:
            return
        except READ_ERRORS:
            self.signals.progress.emit(f"motion signal failed: {traceback.format_exc().strip().splitlines()[-1]}")


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


def draw_correlation(axes: Axes, result: dict, legend_size: float | None = None, with_legend: bool = True) -> None:
    shifts = result["shifts"]
    for curve, (color, dashes) in zip(result["curves"], CURVE_STYLES, strict=False):
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
        axes.annotate(
            text, xy=(peak_shift, peak_value), xytext=(-6 if right_half else 6, 6),
            textcoords="offset points", ha="right" if right_half else "left", va="bottom", color=color, fontsize=legend_size,
            fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": color},
        )
    # room above the highest peak for its label
    axes.set_ymargin(0.3)
    axes.set_xlim(shifts[0], shifts[-1])
    axes.set_xticks(np.arange(shifts[0], shifts[-1] + 1, 10))
    axes.grid(color="#e6e5e0")
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


# the small plot in the right panel; a click opens the large plot, or runs the cross-correlation when no plot is there yet
class CorrelationThumbnail(FigureCanvasQTAgg):
    def __init__(self, open_large: Any, run_correlation: Any) -> None:
        super().__init__(Figure(layout="constrained"))
        self.open_large, self.run_correlation = open_large, run_correlation
        self.result: dict | None = None
        self.drawn = False
        self.setMinimumHeight(THUMBNAIL_MIN_HEIGHT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.show_result(None)

    def show_result(self, result: dict | None) -> None:
        # matplotlib draws slowly, so the same plot is not drawn again
        if self.drawn and result is self.result:
            return
        self.result, self.drawn = result, True
        self.setToolTip("click to run the cross-correlation" if result is None else "click to enlarge")
        self.figure.clear()
        with mpl.rc_context({"font.size": THUMBNAIL_FONT_POINT_SIZE}):
            axes = self.figure.add_subplot()
            if result is None:
                axes.set_axis_off()
                axes.text(0.5, 0.5, "no cross-correlation yet\nclick here to run it", ha="center", va="center")
            else:
                # the small plot has no legend and no title, so that the curves get the height; the peak labels name the curves
                draw_correlation(axes, result, THUMBNAIL_FONT_POINT_SIZE, with_legend=False)
        self.draw_idle()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.result is None:
            self.run_correlation()
            return
        self.open_large(self.result)


# a range of 0 to 0 gives a busy bar that moves until it is hidden
def busy_bar() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 0)
    bar.setTextVisible(False)
    bar.hide()
    return bar


# a new font size applies at once; Cancel puts back the size from before the dialog
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
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QFormLayout(self)
        layout.addRow("Font size", self.size_box)
        layout.addRow(self.auto_load_box)
        layout.addRow(self.checks_box(set(window.settings["hidden_issues"])))
        layout.addRow(buttons)

    # one row for each known group and each check, in a box that scrolls: the severity list and the Show box; the stored results get the
    # new levels with no new validation
    def checks_box(self, hidden: set[str]) -> QGroupBox:
        self.level_boxes: dict[str, QComboBox] = {}
        self.show_boxes: dict[str, QCheckBox] = {}
        rows = QWidget()
        rows_layout = QFormLayout(rows)
        for key, default in DEFAULT_SEVERITY_LEVELS.items():
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
            label.setToolTip(tooltip)
            box.setToolTip(tooltip)
            show_box = QCheckBox("Show")
            show_box.setChecked(key not in hidden)
            show_box.setToolTip("Show these issues in the issue list. A hidden issue still counts in the status and the colors.")
            row = QHBoxLayout()
            row.addWidget(box, 1)
            row.addWidget(show_box)
            rows_layout.addRow(label, row)
            self.level_boxes[key] = box
            self.show_boxes[key] = show_box
        area = QScrollArea()
        area.setWidget(rows)
        area.setWidgetResizable(True)
        area.setMinimumHeight(10 * box.sizeHint().height())
        group = QGroupBox("Checks")
        QVBoxLayout(group).addWidget(area)
        return group

    # the levels that are not the default
    def changed_levels(self) -> dict[str, str]:
        return {key: box.currentText() for key, box in self.level_boxes.items() if box.currentText() != DEFAULT_SEVERITY_LEVELS[key]}

    def hidden_issues(self) -> list[str]:
        return [key for key, box in self.show_boxes.items() if not box.isChecked()]


def table(columns: list[tuple[str, str]]) -> QTableWidget:
    widget = QTableWidget(0, len(columns))
    widget.setHorizontalHeaderLabels([name for name, _ in columns])
    for column, (_, tooltip) in enumerate(columns):
        widget.horizontalHeaderItem(column).setToolTip(tooltip)
    widget.verticalHeader().setVisible(False)
    widget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    widget.setWordWrap(False)
    widget.setTextElideMode(Qt.TextElideMode.ElideRight)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    widget.horizontalHeader().setStretchLastSection(True)
    return widget


# the issue list: one item for each group, with its issues under it; a click on a group opens or closes it
def issue_tree(columns: list[tuple[str, str]]) -> QTreeWidget:
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
        self.signals.motion_ready.connect(self.on_motion_ready)
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
        self.apply_view()
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
        self.validate_all_button = QPushButton("Validate all")
        self.validate_all_button.clicked.connect(self.validate_all)
        self.correlate_button = QPushButton("Cross-correlation")
        self.correlate_button.clicked.connect(self.correlate_selected)
        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self.open_settings)
        # on: the full view for the admin; off: the simple view for the people who record
        self.details_box = QCheckBox("Details")
        self.details_box.setToolTip("Show the full result: each issue with its check and severity, the Measures tab, and the "
                                    "Cross-correlation tab.")
        self.details_box.setChecked(self.settings["details"])
        self.details_box.toggled.connect(self.set_details)
        folder_row = QHBoxLayout()
        folder_row.addWidget(open_button, 1)
        folder_row.addWidget(settings_button)
        folder_row.addWidget(self.details_box)
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
        for widget in [self.recording_list, self.validate_selected_button, self.validate_all_button, self.correlate_button]:
            layout.addWidget(widget)
        layout.addLayout(task_row)
        return left

    def build_middle(self) -> QWidget:
        middle = QWidget()
        layout = QVBoxLayout(middle)
        views = QHBoxLayout()
        self.color_view, self.depth_view = ImageView(), ImageView()
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
        for widget in [self.previous_button, self.play_button, self.next_button, self.speed_combo]:
            controls.addWidget(widget)
        controls.addWidget(self.slider, 1)
        layout.addLayout(controls)
        return middle

    def build_right(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)

        status_box = QGroupBox("Result")
        status_layout = QHBoxLayout(status_box)
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

        self.measure_table = table(MEASURE_COLUMNS)
        self.measure_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        issue_tab = QWidget()
        issue_layout = QVBoxLayout(issue_tab)
        self.issue_tree = issue_tree(ISSUE_COLUMNS)
        self.issue_tree.itemSelectionChanged.connect(self.select_issue)
        self.issue_tree.itemClicked.connect(self.toggle_issue_group)
        self.row_padding = self.measure_table.verticalHeader().defaultSectionSize() - self.measure_table.verticalHeader().fontMetrics().height()
        self.issue_detail = QPlainTextEdit()
        self.issue_detail.setReadOnly(True)
        issue_controls = QHBoxLayout()
        self.issue_previous_button, self.issue_next_button = QPushButton("◀ issue frame"), QPushButton("issue frame ▶")
        self.issue_previous_button.clicked.connect(lambda: self.step_issue(-1))
        self.issue_next_button.clicked.connect(lambda: self.step_issue(1))
        self.issue_frame_label = QLabel("")
        issue_controls.addWidget(self.issue_previous_button)
        issue_controls.addWidget(self.issue_frame_label, 1, Qt.AlignmentFlag.AlignCenter)
        issue_controls.addWidget(self.issue_next_button)
        issue_layout.addWidget(self.issue_tree, 1)
        issue_layout.addWidget(self.issue_detail)
        issue_layout.addLayout(issue_controls)

        self.thumbnail = CorrelationThumbnail(self.open_large_plot, self.correlate_selected)

        # each tab gets the full height below the Result row
        self.tabs = QTabWidget()
        self.tabs.addTab(issue_tab, "Issues")
        self.tabs.setTabToolTip(0, "Critical issues first. Select a row to go to its frames.")
        self.tabs.addTab(self.measure_table, "Measures")
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

    # the row height follows the font: the font height and the padding that Qt gave the rows at the start
    def refresh_table_sizes(self) -> None:
        # the issue list gets its row height from the font
        header = self.measure_table.verticalHeader()
        header.setDefaultSectionSize(header.fontMetrics().height() + self.row_padding)
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
        save_settings(self.settings)
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
        text, color = run_label(run, self.settings["details"]) if run else ("", "")
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
        # the simple view shows the verdict of the worst severity
        text, color = (worst, STATUS_COLORS[worst]) if self.settings["details"] else (VERDICTS[worst], VERDICT_COLORS[VERDICTS[worst]])
        take_item.setText(0, f"{label}  [{text}]")
        take_item.setForeground(0, QColor(color))

    def set_details(self, checked: bool) -> None:
        self.settings["details"] = checked
        save_settings(self.settings)
        self.apply_view()

    # the parts of the full view: the Cross-correlation button, the Measures and Cross-correlation tabs, and the Stream and Message
    # columns of the issue list; the recording list and the result get the labels of the view
    def apply_view(self) -> None:
        details = self.settings["details"]
        self.correlate_button.setVisible(details)
        self.tabs.setTabVisible(self.tabs.indexOf(self.measure_table), details)
        self.tabs.setTabVisible(self.tabs.indexOf(self.thumbnail), details)
        header = self.issue_tree.headerItem()
        header.setText(0, ISSUE_COLUMNS[0][0] if details else "Issue")
        header.setToolTip(0, ISSUE_COLUMNS[0][1] if details else "What the validation found. Select a row to see more and to go to its frames.")
        for column in (1, 3):
            self.issue_tree.setColumnHidden(column, not details)
        for path in self.recording_items:
            self.refresh_item(path)
        self.show_result(self.requested)

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
        self.show_frame(0)
        self.statusBar().showMessage(f"{path.name}: {self.frames.count} frames")

    def on_motion_ready(self, path_text: str, motion: dict) -> None:
        self.motions[Path(path_text)] = motion

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
            color = self.frames.color(self.index)
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
        for button in [self.validate_selected_button, self.validate_all_button, self.correlate_button]:
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

    def validate_all(self) -> None:
        self.validate(list(self.recordings))

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
        self.measure_table.setRowCount(0)
        self.issue_tree.clear()
        self.issue_detail.clear()
        self.set_issue_frames([])
        run = self.last_runs.get(path) if path is not None else None
        if run is None:
            self.status_label.setText("not validated")
            self.status_label.setStyleSheet("")
            self.validated_label.setText("click Validate selected" if path is not None else "")
            self.tabs.setTabText(0, "Issues")
            return
        text, color = run_label(run, self.settings["details"])
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}")
        self.validated_label.setText(f"validated {run['validated_at'].replace('T', ' ')}")
        self.fill_measure_table(run)
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
        # the tab counts the issue rows of the view: in the simple view, one row can hold many issues
        if self.settings["details"]:
            self.fill_issue_groups(members)
            shown_count, hidden_count = len(members), len(hidden_members)
        else:
            self.fill_issue_verdicts(members)
            shown_count = len({verdict_key(*member) for member in members})
            hidden_count = len({verdict_key(*member) for member in hidden_members})
        self.tabs.setTabText(0, f"Issues ({shown_count}, {hidden_count} hidden)" if hidden_count else f"Issues ({shown_count})")
        # the first issue of the first open row
        top = next((item for item in map(self.issue_tree.topLevelItem, range(self.issue_tree.topLevelItemCount())) if item.isExpanded()), None)
        if top is not None and top.childCount():
            self.issue_tree.setCurrentItem(top.child(0))

    # the full view: one row for each known group (and No group), with each issue under it
    def fill_issue_groups(self, members: list[tuple[dict[str, Any], str, str | None]]) -> None:
        groups: dict[str, list[tuple[dict[str, Any], str]]] = {}
        for issue, severity, group in members:
            groups.setdefault(group or NO_GROUP, []).append((issue, severity))
        # the group with the worst issue first; sorted() keeps the order of the rows for groups of the same severity
        for group, members in sorted(groups.items(), key=lambda entry: min(LEVELS.index(severity) for _, severity in entry[1])):
            worst = LEVELS[min(LEVELS.index(severity) for _, severity in members)]
            count = len(members)
            group_item = QTreeWidgetItem(self.issue_tree)
            group_item.setText(0, f"{group}  [{worst}: {count} {'issue' if count == 1 else 'issues'}]")
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

    # the simple view: a row for each verdict (Problem, Check, and the closed Other notes row for the low issues), with one plain item
    # for each known group and for each check and stream in no group (their issues together)
    def fill_issue_verdicts(self, members: list[tuple[dict[str, Any], str, str | None]]) -> None:
        entries: dict[str, list[dict[str, Any]]] = {"critical": [], "medium": [], "low": []}
        by_key: dict[tuple, dict[str, Any]] = {}
        for issue, severity, group in members:
            key = verdict_key(issue, severity, group)
            if key in by_key:
                entry = by_key[key]
                entry["frames"] = sorted(set(entry["frames"]) | set(issue["frames"]))
                entry["message"] += "\n" + issue["message"]
                continue
            stream = f" ({issue['stream']})" if issue["stream"] and group is None else ""
            entry = {
                "check": group or issue["check"],
                "stream": issue["stream"],
                "frames": list(issue["frames"]),
                "message": issue["message"],
                "plain": PLAIN_TEXTS.get(group or issue["check"], (group or issue["check"]).replace("_", " ")) + stream,
            }
            by_key[key] = entry
            entries[severity].append(entry)
        for severity, title in [("critical", "Problem"), ("medium", "Check"), ("low", OTHER_NOTES)]:
            if not entries[severity]:
                continue
            row_item = QTreeWidgetItem(self.issue_tree)
            row_item.setText(0, f"{title} ({len(entries[severity])})")
            row_item.setFirstColumnSpanned(True)
            if title in VERDICT_COLORS:
                row_item.setForeground(0, QColor(VERDICT_COLORS[title]))
            else:
                row_item.setToolTip(0, "Minor notes. They do not make the take a problem.")
            row_item.setData(0, ISSUE_ROLE, None)
            for entry in entries[severity]:
                item = QTreeWidgetItem(row_item, [entry["plain"], "", frames_text(entry["frames"]), ""])
                item.setData(0, ISSUE_ROLE, entry)
                item.setToolTip(0, entry["plain"])
            # the Problem and Check rows are open; the Other notes row is closed
            row_item.setExpanded(title != OTHER_NOTES)

    # a run of an older version of the app has no measures, only the printed lines; each line fills one row
    def fill_measure_table(self, run: dict[str, Any]) -> None:
        if "measures" not in run:
            for line in run["fps"]:
                row = self.measure_table.rowCount()
                self.measure_table.insertRow(row)
                self.measure_table.setItem(row, 0, QTableWidgetItem(line))
                self.measure_table.setSpan(row, 0, 1, self.measure_table.columnCount())
            return
        for values in run["measures"]:
            row = self.measure_table.rowCount()
            self.measure_table.insertRow(row)
            value = run_lengths_text(values["value"]) if values["name"] == "run lengths" else values["value"]
            for column, text in enumerate([values["stream"], values["name"], value]):
                item = QTableWidgetItem(text)
                item.setToolTip(MEASURE_ABOUT.get(values["name"], ""))
                self.measure_table.setItem(row, column, item)

    # a group item is not an issue, so it clears the text box and the issue frames
    def select_issue(self) -> None:
        items = self.issue_tree.selectedItems()
        if not items:
            return
        issue = items[0].data(0, ISSUE_ROLE)
        if issue is None:
            self.issue_detail.clear()
            self.set_issue_frames([])
            return
        # an item of the simple view has a plain text before the message of the check
        self.issue_detail.setPlainText(f"{issue['plain']}\n\n{issue['message']}" if "plain" in issue else issue["message"])
        self.set_issue_frames(issue["frames"])
        if self.issue_frames:
            self.go_to_issue_frame()

    def toggle_issue_group(self, item: QTreeWidgetItem) -> None:
        if item.parent() is None:
            item.setExpanded(not item.isExpanded())

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

    # with the motion signals in memory (from a load or a validation) the curves need no file read
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


def main() -> None:
    app = QApplication(sys.argv)
    settings = load_settings()
    set_app_font_size(settings["font_point_size"])
    window = MainWindow(start_folder(sys.argv, settings))
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
