import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.colors import to_hex
from PySide6.QtCore import QLocale, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QSpinBox

import app


@pytest.fixture(scope="module")
def application() -> QApplication:
    return QApplication.instance() or QApplication([])


# the app writes the last folder and the last runs; the tests use files in tmp_path, not the files of the repository
@pytest.fixture(autouse=True)
def app_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(app, "HISTORY_PATH", tmp_path / "validation_history.json")


def make_folder(folder: Path, names: list[str]) -> Path:
    folder.mkdir()
    for name in names:
        (folder / name).touch()
    return folder


def test_run_summary_counts_the_issues_of_each_severity() -> None:
    issue = {"check": "missing_partner", "stream": None, "frames": [], "message": "no -m (mirrorless) recording of this take next to 40-r.bag"}
    assert app.run_summary({"error": None, "issues": []}) == "OK"
    assert app.run_summary({"error": None, "issues": [issue]}) == "1 medium"
    assert app.run_summary({"error": None, "issues": [issue, issue]}) == "2 medium"
    message = "notification: Left MIPI error (hardware_error, error, Stereo Module), 1 time"
    low = {"check": "notification", "stream": None, "frames": [], "message": message}
    assert app.run_summary({"error": None, "issues": [low]}) == "1 low"
    # the worst severity comes first, and each severity has its own count
    critical = {"check": "fps", "stream": "color", "frames": [], "message": "color.fps: 60, expected 30"}
    assert app.run_summary({"error": None, "issues": [low, issue, critical, issue]}) == "1 critical, 2 medium, 1 low"
    assert app.run_summary({"error": "Traceback", "issues": []}) == "critical: validation error"


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
    # clear() of the list can send an item change for an item of the previous folder
    large = make_folder(tmp_path / "large", [f"{n}-r.bag" for n in range(1, 109)])
    small = make_folder(tmp_path / "small", [f"{n}-r.bag" for n in range(1, 11)])
    # Qt gives an exception in a slot to sys.excepthook and continues
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *error: errors.append(error))
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(large)
    window.select_path(window.recordings[106])
    window.open_folder(small)
    assert errors == []
    assert window.recording_list.topLevelItemCount() == 10
    assert window.recording_list.currentItem() is window.recording_items[window.recordings[0]]
    assert window.requested == window.recordings[0]
    window.close()


# a poc1 take with 4 files (the mirrorless .db3 file in its own subfolder) and a take in a subfolder
def test_the_list_has_one_item_for_each_take_with_its_files(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["m_20261113_144012.mp4", "r_20261113_144012.db3", "r_20261113_144012.mp4"])
    (folder / "m_20261113_144012_color").mkdir()
    (folder / "m_20261113_144012_color" / "m_20261113_144012_color.db3").touch()
    (folder / "station_a").mkdir()
    for name in ["1-m.bag", "1-r.bag"]:
        (folder / "station_a" / name).touch()
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    takes = [window.recording_list.topLevelItem(i) for i in range(window.recording_list.topLevelItemCount())]
    assert [take.text(0) for take in takes] == ["20261113_144012", "station_a/1"]
    files = sorted(takes[0].child(i).text(0) for i in range(takes[0].childCount()))
    assert files == ["D435 .db3", "D435 .mp4", "mirrorless .db3", "mirrorless .mp4"]
    # the first file is selected and its take is open; a click on a take opens or closes it and selects no file
    assert takes[0].isExpanded()
    assert not takes[1].isExpanded()
    selected = window.requested
    window.toggle_take(takes[1])
    window.recording_list.setCurrentItem(takes[1])
    assert takes[1].isExpanded()
    assert window.requested == selected
    window.recording_list.setCurrentItem(takes[1].child(0))
    assert window.requested == folder / "station_a" / "1-m.bag"
    window.close()


# a file that is added to or removed from the open folder changes the list at the next scan; the open take and the selected file stay,
# and the selected file does not load again
def test_the_list_finds_new_and_removed_files(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-m.bag", "1-r.bag", "2-r.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    assert window.rescan_timer.isActive()
    window.select_path(folder / "1-r.bag")
    selections = []
    window.select_recording = selections.append
    (folder / "3-r.bag").touch()
    (folder / "2-r.bag").unlink()
    window.rescan_folder()
    assert window.recordings == [folder / "1-m.bag", folder / "1-r.bag", folder / "3-r.bag"]
    takes = [window.recording_list.topLevelItem(i) for i in range(window.recording_list.topLevelItemCount())]
    assert [take.text(0) for take in takes] == ["1", "3"]
    assert takes[0].isExpanded()
    assert not takes[1].isExpanded()
    assert window.recording_list.currentItem() is window.recording_items[folder / "1-r.bag"]
    assert selections == []
    window.close()


def test_the_last_folder_opens_at_the_next_start(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-r.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    window.close()
    settings = app.load_settings()
    assert settings["last_folder"] == str(folder.absolute())
    assert app.start_folder(["app.py"], settings) == folder.absolute()
    # a folder on the command line comes first; a last folder that is no longer there gives no folder
    assert app.start_folder(["app.py", "other"], settings) == Path("other")
    assert app.start_folder(["app.py"], settings | {"last_folder": str(tmp_path / "removed")}) is None


def test_an_older_file_of_runs_keeps_the_last_run(tmp_path: Path) -> None:
    path = tmp_path / "validation_history.json"
    first, last = {"validated_at": "1", "issues": [], "error": None}, {"validated_at": "2", "issues": [], "error": None}
    path.write_text(json.dumps({app.LastRuns.key(tmp_path / "1-r.bag"): [first, last]}))
    runs = app.LastRuns(path)
    assert runs.get(tmp_path / "1-r.bag") == last
    runs.set(tmp_path / "1-r.bag", first)
    assert json.loads(path.read_text()) == {app.LastRuns.key(tmp_path / "1-r.bag"): first}


def issue(check: str, frames: list[int]) -> dict:
    return {"check": check, "stream": "color", "frames": frames, "message": f"color.{check}: text"}


def test_the_issue_list_puts_the_issues_under_their_known_group(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-r.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    message = "notification: Left MIPI error (hardware_error, error, Stereo Module), 1 time"
    notification = {"check": "notification", "stream": None, "frames": [], "message": message}
    run = {"validated_at": "2026-11-18T10:00:00", "fps": [], "measures": [], "error": None}
    run["issues"] = [notification, issue("measured_fps", [])]
    window.on_validated(str(window.recordings[0]), run, None)
    assert window.status_label.text() == "1 medium, 1 low"
    # the group with the worst issue first
    groups = [window.issue_tree.topLevelItem(row) for row in range(window.issue_tree.topLevelItemCount())]
    assert [group.text(0) for group in groups] == ["No group  [1 medium]", "Hardware Instability  [1 low]"]
    assert [group.child(0).text(0) for group in groups] == ["measured_fps", "notification"]
    assert groups[1].child(0).toolTip(0).startswith("low, Hardware Instability: notification: Left MIPI error")
    window.close()


def test_the_right_panel_shows_the_last_run(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-m.bag", "2-m.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    # the critical duplicate_image (copies in 3 of 10 frames) comes before the medium issue
    measures = [{"stream": "color", "name": "image fps", "value": "29.62"}, {"stream": "PC", "name": "total RAM", "value": "15.9 GB"}]
    run = {"validated_at": "2026-11-16T10:00:00", "fps": [], "measures": measures, "error": None}
    message = "color.frame: the same image as the previous frame in 3 of 10 frames: [1, 2, 3]"
    copies = {**issue("duplicate_image", [1, 2, 3]), "message": message}
    run["issues"] = [issue("timestamp_order", [4]), copies]
    window.on_validated(str(window.recordings[0]), run, None)
    assert window.status_label.text() == "1 critical, 1 medium"
    assert window.validated_label.text() == "validated 2026-11-16 10:00:00"
    assert window.tabs.tabText(0) == "Issues (2)"
    rows = [window.issue_tree.topLevelItem(row) for row in range(window.issue_tree.topLevelItemCount())]
    assert [row.text(0) for row in rows] == ["No group  [1 critical, 1 medium]"]
    # the Frames column gives the number of frames, or the frame index of a single frame
    assert [(rows[0].child(i).text(0), rows[0].child(i).text(2)) for i in range(rows[0].childCount())] == [
        ("duplicate_image", "3 frames"),
        ("timestamp_order", "frame 4"),
    ]
    assert window.issue_frame_label.text() == "frame 1 (issue frame 1 of 3)"
    # the measures are under their stream
    streams = [window.measure_tree.topLevelItem(row) for row in range(window.measure_tree.topLevelItemCount())]
    assert [(stream.text(0), stream.child(0).text(0)) for stream in streams] == [("color", "image fps"), ("PC", "total RAM")]
    assert window.old_run_label.isHidden()
    assert window.issue_empty_label.isHidden()
    # a run of an older version of the app has only the printed lines, and the Result box has a note
    old_run = {"validated_at": "2026-11-15T10:00:00", "fps": ["color: file fps 33.62, 376 frames in 11.154 s"], "issues": [], "error": None}
    window.on_validated(str(window.recordings[1]), old_run, None)
    window.select_path(window.recordings[1])
    assert window.measure_tree.topLevelItemCount() == 1
    assert window.measure_tree.topLevelItem(0).text(0) == "color: file fps 33.62, 376 frames in 11.154 s"
    assert not window.old_run_label.isHidden()
    assert window.tabs.tabText(0) == "Issues (0)"
    # a run with no issue shows a text in place of the empty issue list
    assert window.issue_empty_label.text() == "No issues."
    assert not window.issue_empty_label.isHidden()
    assert window.issue_tree.isHidden()
    assert window.issue_controls.isHidden()
    window.close()


def test_a_measure_of_an_issue_has_the_color_and_the_expected_value(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-m.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    message = "color.image_fps: 30.27 (527 different images in 2090 frames), expected 120 within 1%"
    image_fps = {"check": "image_fps", "stream": "color", "frames": [], "message": message}
    measures = [
        {"stream": "color", "name": "image fps", "value": "30.27"},
        {"stream": "color", "name": "run lengths", "value": "{1: 5, 3: 123, 4: 279}"},
    ]
    run = {"validated_at": "2026-11-16T10:00:00", "fps": [], "measures": measures, "issues": [image_fps], "error": None}
    window.on_validated(str(window.recordings[0]), run, None)
    color = window.measure_tree.topLevelItem(0)
    fps_item, runs_item = color.child(0), color.child(1)
    severity = app.issue_severity(image_fps, [image_fps])
    assert fps_item.text(1) == "30.27 (expected 120 within 1%)"
    assert fps_item.foreground(1).color().name() == app.STATUS_COLORS[severity]
    assert fps_item.toolTip(1).endswith(f"{severity} issue: {message}")
    # a measure of no issue keeps its value, and its tooltip has the full value under the explanation
    assert runs_item.text(1) == "5 runs of 1 frame, 123 runs of 3 frames, 279 runs of 4 frames"
    assert runs_item.toolTip(1) == f"{app.MEASURE_ABOUT['run lengths']}\n\nValue: {runs_item.text(1)}"
    # the run lengths also show as a bar chart
    assert not window.run_length_chart.isHidden()
    assert [(stream, length, count) for _, stream, length, count in window.run_length_chart.bars] == [
        ("color", 1, 5), ("color", 3, 123), ("color", 4, 279),
    ]
    window.close()


def test_the_issue_strip_marks_the_issue_frames(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-m.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    copies = {**issue("duplicate_image", [1, 2, 3]), "message": "color.frame: the same image as the previous frame in 3 of 10 frames"}
    run = {"validated_at": "2026-11-16T10:00:00", "fps": [], "measures": [], "issues": [issue("timestamp_order", [4]), copies], "error": None}
    window.on_validated(str(window.recordings[0]), run, None)
    strip = window.issue_strip
    # no marks before the player has the frames
    assert strip.marks() == []
    strip.set_count(10)
    # the selected issue (the critical duplicate_image) is last, so that its tall marks are on top
    assert strip.marks() == [(4, "medium", False), (1, "critical", True), (2, "critical", True), (3, "critical", True)]
    strip.resize(400, strip.HEIGHT)
    start, step = strip.frame_scale()
    clicked = []
    strip.frame_clicked.connect(clicked.append)
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(round(start + step * 4), 5))
    # a click far from a mark goes nowhere
    QTest.mouseClick(strip, Qt.MouseButton.LeftButton, pos=QPoint(round(start + step * 8), 5))
    assert clicked == [4]
    window.close()


def test_the_empty_cross_correlation_plot_does_not_start_a_task(application: QApplication) -> None:
    opened = []
    thumbnail = app.CorrelationThumbnail(opened.append)
    QTest.mouseClick(thumbnail, Qt.MouseButton.LeftButton)
    assert opened == []
    assert to_hex(thumbnail.figure.get_facecolor()) == app.chart_colors()["surface"]


def test_each_check_and_group_has_a_plain_text() -> None:
    assert set(app.DEFAULT_SEVERITY_LEVELS) <= set(app.PLAIN_TEXTS)


# each check has one source; a known group has none
def test_each_check_has_a_source() -> None:
    checks = set(app.DEFAULT_SEVERITY_LEVELS) - set(app.GROUP_ABOUT)
    assert set(app.CHECK_SOURCES) == checks


def test_the_number_boxes_use_english_digits(application: QApplication) -> None:
    previous = QLocale()
    try:
        app.use_english_numbers()
        box = QSpinBox()
        box.setSuffix(" pt")
        box.setValue(16)
        assert box.locale().language() == QLocale.Language.English
        assert box.text() == "16 pt"
    finally:
        QLocale.setDefault(previous)


def test_the_settings_dialog_restores_the_defaults(application: QApplication) -> None:
    window = app.MainWindow(None)
    dialog = app.SettingsDialog(window)
    texts = [label.text() for label in dialog.findChildren(QLabel)]
    # the headings: the known groups, then the source of the values of the checks
    headings = ["Known groups", "Metadata", "Measured", "Other files"]
    assert [text for text in texts if text in headings] == headings
    # a group row names the checks that its level wins over
    assert f"{app.PLAIN_TEXTS['Hardware Instability']} Wins over: notification." in texts
    assert app.PLAIN_TEXTS["fps"] in texts
    assert dialog.name_labels["measured_fps"].text() == "measured_fps"
    # the Default button of a row shows only for a changed level, and it sets only that row back
    assert dialog.default_buttons["measured_fps"].isHidden()
    dialog.level_boxes["measured_fps"].setCurrentText("critical")
    dialog.level_boxes["fps"].setCurrentText("medium")
    assert not dialog.default_buttons["measured_fps"].isHidden()
    assert dialog.name_labels["measured_fps"].text() == "measured_fps (default: medium)"
    dialog.default_buttons["fps"].click()
    assert dialog.changed_levels() == {"measured_fps": "critical"}
    assert dialog.default_buttons["fps"].isHidden()
    # Reset to defaults sets every setting back
    dialog.show_boxes["notification"].setChecked(False)
    dialog.auto_load_box.setChecked(not app.DEFAULT_SETTINGS["auto_load"])
    dialog.drawing_box.setCurrentText("GPU")
    dialog.size_box.setValue(app.DEFAULT_SETTINGS["font_point_size"] + 2)
    assert dialog.hidden_issues() == ["notification"]
    dialog.reset_settings()
    assert dialog.name_labels["measured_fps"].text() == "measured_fps"
    assert dialog.changed_levels() == {}
    assert dialog.hidden_issues() == []
    assert dialog.size_box.value() == app.DEFAULT_SETTINGS["font_point_size"]
    assert dialog.auto_load_box.isChecked() == app.DEFAULT_SETTINGS["auto_load"]
    assert dialog.drawing_box.currentText() == app.DEFAULT_SETTINGS["drawing"]
    dialog.close()
    window.close()


def test_stopped_load_task_sends_no_signal(application: QApplication, tmp_path: Path) -> None:
    signals = app.WorkerSignals()
    received = []
    for signal in [signals.loaded, signals.load_failed, signals.progress]:
        signal.connect(lambda *values: received.append(values))
    # a file that is not there: a task that is not stopped sends load_failed
    task = app.LoadTask(signals, tmp_path / "1-r.bag")
    task.run()
    assert len(received) == 1
    assert task.finished
    received.clear()
    stopped_task = app.LoadTask(signals, tmp_path / "1-r.bag")
    stopped_task.stopped = True
    stopped_task.run()
    assert received == []
    assert stopped_task.finished


def test_a_new_selection_stops_the_previous_load(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-r.bag", "2-r.bag", "3-r.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = True
    window.open_folder(folder)
    first = window.load_task
    assert first is not None
    assert not first.stopped
    window.select_path(window.recordings[1])
    second = window.load_task
    assert first.stopped
    assert second is not first
    assert not second.stopped
    assert not window.load_bar.isHidden()
    # with the automatic load off, the selection stops the load and starts no new one
    window.settings["auto_load"] = False
    window.select_path(window.recordings[2])
    assert second.stopped
    assert window.load_task is None
    assert window.load_bar.isHidden()
    window.pool.waitForDone()
    application.processEvents()
    window.close()
