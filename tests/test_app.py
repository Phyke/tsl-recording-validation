import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

import app


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


def make_folder(folder: Path, names: list[str]) -> Path:
    folder.mkdir()
    for name in names:
        (folder / name).touch()
    return folder


def test_run_summary_counts_one_issue_and_more() -> None:
    issue = {"check": "missing_partner", "stream": None, "frames": [], "message": "no -m (mirrorless) recording of this take next to 40-r.bag"}
    assert app.run_summary({"error": None, "issues": []}) == "OK"
    assert app.run_summary({"error": None, "issues": [issue]}) == "medium: 1 issue"
    assert app.run_summary({"error": None, "issues": [issue, issue]}) == "medium: 2 issues"


@pytest.mark.parametrize(
    ("check", "stream", "message", "short"),
    [
        ("notification", None, "notification: Right MIPI error (hardware_error, error, Stereo Module), 1 time", "Right MIPI error (hardware_error, error, Stereo Module), 1 time"),
        ("timestamp_domain", "depth", "depth.timestamp_domain: system_time in 116 of 116 frames, expected global_time", "system_time in 116 of 116 frames, expected global_time"),
        ("measured_fps", "color", "color.camera_fps: 109.70, expected 120 within 1%", "camera_fps: 109.70, expected 120 within 1%"),
        ("pair_duration", None, "pair duration: 3-m.bag 10.670 s, 3-r.bag 12.687 s, difference -2018 ms, expected within 1000 ms", "3-m.bag 10.670 s, 3-r.bag 12.687 s, difference -2018 ms, expected within 1000 ms"),
        ("constant_value", "color", "frame 5 color.intrinsics.fx: 1.0 != 2.0", "intrinsics.fx: 1.0 != 2.0"),
        ("color_depth_offset", None, "frame 3 color.timestamp_ms - depth.timestamp_ms: +20.00 ms, expected within 16.67 ms", "color.timestamp_ms - depth.timestamp_ms: +20.00 ms, expected within 16.67 ms"),
        ("frames_not_loaded", "depth", "depth: 3 of the 9 frames in the file are not in the loaded framesets, expected at most 1", "3 of the 9 frames in the file are not in the loaded framesets, expected at most 1"),
        ("missing_stream", "color", "no color stream in this recording", "no color stream in this recording"),
        ("error", "", "Traceback (most recent call last):\nValueError: 1-r.bag: the .bag file has no index", "ValueError: 1-r.bag: the .bag file has no index"),
    ],
)
def test_short_message_has_no_part_that_a_column_shows(check: str, stream: str | None, message: str, short: str) -> None:
    assert app.short_message({"check": check, "stream": stream, "frames": [], "message": message}) == short


def test_open_a_second_folder_with_fewer_recordings(application: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # with 108 recordings and row 106, QListWidget.clear() sends a row change for row 54
    large = make_folder(tmp_path / "large", [f"{n}-r.bag" for n in range(1, 109)])
    small = make_folder(tmp_path / "small", [f"{n}-r.bag" for n in range(1, 11)])
    # Qt gives an exception in a slot to sys.excepthook and continues
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(large)
    window.recording_list.setCurrentRow(106)
    window.open_folder(small)
    assert errors == []
    assert window.recording_list.count() == 10
    assert window.recording_list.currentRow() == 0
    assert window.requested == window.recordings[0]
    window.close()
