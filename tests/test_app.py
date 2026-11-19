import json
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


def test_run_summary_counts_one_issue_and_more() -> None:
    issue = {"check": "missing_partner", "stream": None, "frames": [], "message": "no -m (mirrorless) recording of this take next to 40-r.bag"}
    assert app.run_summary({"error": None, "issues": []}) == "OK"
    assert app.run_summary({"error": None, "issues": [issue]}) == "medium: 1 issue"
    assert app.run_summary({"error": None, "issues": [issue, issue]}) == "medium: 2 issues"
    message = "notification: Left MIPI error (hardware_error, error, Stereo Module), 1 time"
    low = {"check": "notification", "stream": None, "frames": [], "message": message}
    assert app.run_summary({"error": None, "issues": [low]}) == "low: 1 issue"


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


def test_the_full_view_puts_the_issues_under_their_known_group(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-r.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.details_box.setChecked(True)
    window.open_folder(folder)
    message = "notification: Left MIPI error (hardware_error, error, Stereo Module), 1 time"
    notification = {"check": "notification", "stream": None, "frames": [], "message": message}
    run = {"validated_at": "2026-11-18T10:00:00", "fps": [], "measures": [], "error": None}
    run["issues"] = [notification, issue("measured_fps", [])]
    window.on_validated(str(window.recordings[0]), run, None)
    assert window.status_label.text() == "medium: 2 issues"
    # the group with the worst issue first
    groups = [window.issue_tree.topLevelItem(row) for row in range(window.issue_tree.topLevelItemCount())]
    assert [group.text(0) for group in groups] == ["No group  [medium: 1 issue]", "Hardware Instability  [low: 1 issue]"]
    assert [group.child(0).text(0) for group in groups] == ["measured_fps", "notification"]
    assert groups[1].child(0).toolTip(0).startswith("low, Hardware Instability: notification: Left MIPI error")
    window.close()


def test_the_right_panel_shows_the_last_run(application: QApplication, tmp_path: Path) -> None:
    folder = make_folder(tmp_path / "takes", ["1-m.bag", "2-m.bag"])
    window = app.MainWindow(None)
    window.settings["auto_load"] = False
    window.open_folder(folder)
    # the simple view is the default; the critical duplicate_image (copies in 3 of 10 frames) comes before the medium issue
    measures = [{"stream": "color", "name": "image fps", "value": "29.62"}, {"stream": "PC", "name": "total RAM", "value": "15.9 GB"}]
    run = {"validated_at": "2026-11-16T10:00:00", "fps": [], "measures": measures, "error": None}
    message = "color.frame: the same image as the previous frame in 3 of 10 frames: [1, 2, 3]"
    copies = {**issue("duplicate_image", [1, 2, 3]), "message": message}
    run["issues"] = [issue("timestamp_order", [4]), copies]
    window.on_validated(str(window.recordings[0]), run, None)
    assert window.status_label.text() == "Problem"
    assert window.validated_label.text() == "validated 2026-11-16 10:00:00"
    assert window.tabs.tabText(0) == "Issues (2)"
    rows = [window.issue_tree.topLevelItem(row) for row in range(window.issue_tree.topLevelItemCount())]
    assert [row.text(0) for row in rows] == ["Problem (1)", "Check (1)"]
    # the Frames column gives the number of frames, or the frame index of a single frame
    assert [(row.child(0).text(0), row.child(0).text(2)) for row in rows] == [
        (f"{app.PLAIN_TEXTS['duplicate_image']} (color)", "3 frames"),
        (f"{app.PLAIN_TEXTS['timestamp_order']} (color)", "frame 4"),
    ]
    assert window.issue_frame_label.text() == "frame 1 (issue frame 1 of 3)"
    assert [window.measure_table.item(row, 1).text() for row in range(2)] == ["image fps", "total RAM"]
    # a run of an older version of the app has only the printed lines
    old_run = {"validated_at": "2026-11-15T10:00:00", "fps": ["color: file fps 33.62, 376 frames in 11.154 s"], "issues": [], "error": None}
    window.on_validated(str(window.recordings[1]), old_run, None)
    window.select_path(window.recordings[1])
    assert window.measure_table.rowCount() == 1
    assert window.measure_table.item(0, 0).text() == "color: file fps 33.62, 376 frames in 11.154 s"
    assert window.tabs.tabText(0) == "Issues (0)"
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
