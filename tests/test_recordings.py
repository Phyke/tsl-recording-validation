import subprocess
import sys
from pathlib import Path

import av
import numpy as np
import pytest
from conftest import compressed_image_cdr, mirrorless_messages, write_db3, write_ros1_bag

from recording_reader import Db3, MirrorlessRos2Loader, Ros1BagIndex
from recordings import (
    camera_or_none,
    correlation_pair,
    file_label,
    find_partner,
    group_takes,
    issue_group,
    issue_severity,
    load_recording,
    pair_issues,
    recording_duration_ms,
    report_issues,
    run_severity,
    sysmon_lines,
    take_key,
    take_label,
    validate_recording,
)

DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"
COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
VALIDATE = Path(__file__).resolve().parent.parent / "src" / "validate.py"


@pytest.mark.parametrize(
    ("name", "camera"),
    [
        ("1-r.bag", "d435"),
        ("1-m.bag", "mirrorless"),
        ("pipeline-r_20261012_151250.db3", "d435"),
        ("pipeline-m_20261012_151250_color.db3", "mirrorless"),
        ("1-m-retake.bag", "mirrorless"),
        ("take-rgb-m.bag", "mirrorless"),
        ("3-right-m.bag", "mirrorless"),
        ("demo-reel-r.bag", "d435"),
        ("1-m.mp4", "mirrorless"),
        ("r_20261113_134220.db3", "d435"),
        ("r_20261113_134220.mp4", "d435"),
        ("m_20261113_134220.mp4", "mirrorless"),
        ("m_20261113_134220_color.db3", "mirrorless"),
        ("-r_20261113_143019.db3", "d435"),
        ("-m_20261113_143019.mp4", "mirrorless"),
        ("video.mp4", "mirrorless"),
        ("1-r-m.bag", None),
        ("1-r-m.mp4", None),
        ("x.bag", None),
        ("retake_1.bag", None),
        ("meta.db3", None),
    ],
)
def test_camera_comes_from_the_marker_in_the_file_name(name: str, camera: str | None) -> None:
    assert camera_or_none(Path(name)) == camera


# a take of the pipeline: 2 files of each camera, and the mirrorless .db3 file is in its own subfolder
def test_a_take_with_4_files_pairs_the_files_of_the_same_type() -> None:
    for prefix in ["", "-"]:
        d435_db3, d435_mp4 = Path(f"x/{prefix}r_20261113_134220.db3"), Path(f"x/{prefix}r_20261113_134220.mp4")
        mirrorless_mp4 = Path(f"x/{prefix}m_20261113_134220.mp4")
        mirrorless_db3 = Path(f"x/{prefix}m_20261113_134220_color/{prefix}m_20261113_134220_color.db3")
        other_take = [Path("x/r_20261113_144012.db3"), Path("x/m_20261113_144012.mp4")]
        recordings = [mirrorless_mp4, mirrorless_db3, d435_mp4, d435_db3, *other_take]
        assert len({take_key(path) for path in recordings}) == 2
        assert find_partner(d435_db3, recordings) == mirrorless_db3
        assert find_partner(d435_mp4, recordings) == mirrorless_mp4
        assert find_partner(mirrorless_db3, recordings) == d435_db3
        assert find_partner(mirrorless_mp4, recordings) == d435_mp4
        # the cross-correlation has the D435 .db3 file as its first recording
        assert correlation_pair(d435_db3, recordings) == [d435_db3, mirrorless_db3]
        assert correlation_pair(mirrorless_db3, recordings) == [d435_db3, mirrorless_db3]
        assert correlation_pair(d435_mp4, recordings) == [d435_db3, mirrorless_mp4]
        assert correlation_pair(mirrorless_mp4, recordings) == [d435_db3, mirrorless_mp4]


def test_a_file_pairs_with_the_other_file_type_when_the_take_has_no_file_of_its_type() -> None:
    d435_db3, d435_mp4, mirrorless_mp4 = Path("x/r_1.db3"), Path("x/r_1.mp4"), Path("x/m_1.mp4")
    assert find_partner(d435_db3, [d435_db3, mirrorless_mp4]) == mirrorless_mp4
    assert find_partner(mirrorless_mp4, [d435_db3, mirrorless_mp4]) == d435_db3
    assert correlation_pair(d435_db3, [d435_db3]) == [d435_db3]
    assert correlation_pair(mirrorless_mp4, [d435_mp4, mirrorless_mp4]) is None
    assert find_partner(Path("x/1.bag"), [d435_db3, mirrorless_mp4]) is None


def write_mp4(path: Path, width: int, height: int, fps: int, count: int) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for n in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((height, width, 3), 10 * n, dtype=np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())


def test_an_mp4_file_is_checked_with_the_settings_of_its_camera(tmp_path: Path) -> None:
    d435, mirrorless = tmp_path / "r_20261113_134220.mp4", tmp_path / "m_20261113_134220.mp4"
    write_mp4(d435, 640, 480, 30, 12)
    write_mp4(mirrorless, 640, 480, 30, 12)
    result = validate_recording(d435, load_recording(d435))
    assert result["issues"] == []
    assert [(row["name"], row["value"]) for row in result["measures"]] == [("file fps", "30.00"), ("frames", "12"), ("duration", "0.367 s")]
    # the frame period of 30 fps is longer than 1.5 frame periods of 120 fps, so the mirrorless check also finds gaps
    checks = [issue["check"] for issue in validate_recording(mirrorless, load_recording(mirrorless))["issues"]]
    assert checks == ["size", "fps", "timestamp_gaps"]


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("m_20261113_144012_color/m_20261113_144012_color.db3", "20261113_144012"),
        ("r_20261113_144012.mp4", "20261113_144012"),
        ("-m_20261113_143019.mp4", "20261113_143019"),
        ("pipeline-r_20261012_151250.db3", "pipeline_20261012_151250"),
        ("1-m.bag", "1"),
        ("1-m-retake.bag", "1-retake"),
        ("demo-xray-r.bag", "demo-xray"),
        ("20261012_151612.db3", "20261012_151612"),
    ],
)
def test_take_label_has_no_camera_marker(name: str, label: str) -> None:
    assert take_label(Path(name)) == label


def test_takes_are_grouped_in_the_order_of_their_first_file() -> None:
    folder = Path("x")
    recordings = [
        folder / "a" / "1-m.bag",
        folder / "a" / "1-r.bag",
        folder / "m_1_color" / "m_1_color.db3",
        folder / "r_1.db3",
        folder / "b" / "1-r.bag",
    ]
    groups = group_takes(folder, recordings)
    assert [(label, [path.name for path in paths]) for label, paths in groups] == [
        ("a/1", ["1-m.bag", "1-r.bag"]),
        ("1", ["m_1_color.db3", "r_1.db3"]),
        ("b/1", ["1-r.bag"]),
    ]
    assert [file_label(path) for path in recordings] == ["mirrorless .bag", "D435 .bag", "mirrorless .db3", "D435 .db3", "D435 .bag"]


def test_the_2_recordings_of_a_take_have_the_same_key() -> None:
    assert take_key(Path("x/demo-reel-r.bag")) == take_key(Path("x/demo-reel-m.bag")) == (Path("x"), "demo-reel-x")
    d435 = Path("x/pipeline-r_20261012_151250.db3")
    mirrorless = Path("x/pipeline-m_20261012_151250_color/pipeline-m_20261012_151250_color.db3")
    assert take_key(d435) == take_key(mirrorless)
    assert find_partner(d435, [d435, mirrorless]) == mirrorless


def test_ros1_bag_index_gives_the_counts_and_the_duration(tmp_path: Path) -> None:
    path = tmp_path / "1-r.bag"
    chunks = [(5000, 5400, {DEPTH_TOPIC: 12, COLOR_TOPIC: 11}), (5400, 7250, {DEPTH_TOPIC: 30, COLOR_TOPIC: 31, "/file_version": 1})]
    write_ros1_bag(path, [DEPTH_TOPIC, COLOR_TOPIC, "/file_version"], chunks)
    index = Ros1BagIndex(path)
    assert index.message_counts == {DEPTH_TOPIC: 42, COLOR_TOPIC: 42, "/file_version": 1}
    assert index.duration_ms == 2250


def test_db3_topic_stats_and_duration(tmp_path: Path) -> None:
    path = tmp_path / "1-r.db3"
    write_db3(path, [("/device_0/info", 0, b""), (DEPTH_TOPIC, 2_000_000_000, b""), (DEPTH_TOPIC, 2_500_000_000, b""), (COLOR_TOPIC, 2_600_000_000, b"")])
    assert Db3(path).topic_stats()[DEPTH_TOPIC] == (2, 2_000_000_000, 2_500_000_000)
    assert recording_duration_ms(path) == 600


def test_pair_checks(tmp_path: Path) -> None:
    d435, mirrorless = tmp_path / "1-r.bag", tmp_path / "1-m.bag"
    write_ros1_bag(d435, [DEPTH_TOPIC], [(1000, 9000, {DEPTH_TOPIC: 240})])
    assert [issue.check for issue in pair_issues(d435)] == ["missing_partner"]
    write_ros1_bag(mirrorless, [COLOR_TOPIC], [(2000, 9500, {COLOR_TOPIC: 900})])
    assert pair_issues(d435) == []
    assert pair_issues(mirrorless) == []
    write_ros1_bag(mirrorless, [COLOR_TOPIC], [(2000, 8500, {COLOR_TOPIC: 780})])
    issues = pair_issues(d435)
    assert [issue.check for issue in issues] == ["pair_duration"]
    assert issues[0].message == "pair duration: 1-r.bag 8.000 s, 1-m.bag 6.500 s, difference +1500 ms, expected within 1000 ms"
    assert [issue.check for issue in pair_issues(mirrorless)] == ["pair_duration"]


def test_copies_in_up_to_5_percent_of_the_frames_are_medium() -> None:
    def duplicate_image(copies: int, frames: int) -> dict:
        indices = list(range(1, 3 * copies, 3))
        message = f"color.frame: the same image as the previous frame in {copies} of {frames} frames: {indices}"
        return {"check": "duplicate_image", "stream": "color", "frames": indices, "message": message}

    assert issue_severity(duplicate_image(3, 1000), []) == "medium"
    assert issue_severity(duplicate_image(50, 1000), []) == "medium"
    assert issue_severity(duplicate_image(51, 1000), []) == "critical"


def test_the_d435_start_notifications_are_low_hardware_instability() -> None:
    def notification(description: str) -> dict:
        message = f"notification: {description} (hardware_error, error, Stereo Module), 1 time. What to do: Start the stream again."
        return {"check": "notification", "stream": None, "frames": [], "message": message}

    for description in ["Left MIPI error", "Right MIPI error", "Depth stream start failure"]:
        assert issue_group(notification(description)) == "Hardware Instability"
        assert issue_severity(notification(description), []) == "low"
    assert issue_group(notification("RT MIPI error")) is None
    assert issue_severity(notification("RT MIPI error"), []) == "medium"
    low, medium = notification("Left MIPI error"), notification("RT MIPI error")
    assert run_severity({"error": None, "issues": [low]}) == "low"
    assert run_severity({"error": None, "issues": [low, medium]}) == "medium"


# a poc1 take: the meta folder is next to the m_1_color folder of the mirrorless .db3 file
def test_poc1_report_against_the_copies(tmp_path: Path) -> None:
    mirrorless = tmp_path / "m_1_color" / "m_1_color.db3"
    assert report_issues(mirrorless, copies=3) == []
    (tmp_path / "meta").mkdir()
    report = tmp_path / "meta" / "m_1.report.json"
    report.write_text('{"no_frame_drops": true}')
    assert report_issues(mirrorless, copies=0) == []
    issues = report_issues(mirrorless, copies=3)
    assert [issue.check for issue in issues] == ["report_no_frame_drops"]
    assert issues[0].message == "report: no_frame_drops is true in m_1.report.json, but 3 frames are copies of the previous image"
    report.write_text('{"no_frame_drops": false, "dropped_capture_frames": 4}')
    issues = report_issues(tmp_path / "m_1.mp4", copies=None)
    assert [(issue.check, issue.message) for issue in issues] == [
        ("dropped_capture_frames", "report: ffmpeg dropped 4 capture frames (dropped_capture_frames in m_1.report.json)"),
    ]


def test_sysmon_line_has_the_cpu_median_and_the_total_ram(tmp_path: Path) -> None:
    d435 = tmp_path / "r_1.db3"
    assert sysmon_lines(d435) == []
    (tmp_path / "meta").mkdir()
    rows = [
        "timestamp,cpu_percent,ram_percent,ram_used_mb,disk_write_mb_since_start",
        "1,5.0,50.0,8192.0,0",
        "2,50.0,50.0,8192.0,0",
        "3,70.0,50.0,8192.0,0",
    ]
    (tmp_path / "meta" / "r_1_sysmon.csv").write_text("\n".join(rows) + "\n")
    # the first row is not in the CPU median
    assert sysmon_lines(d435) == ["sysmon: CPU median 60.0% in 2 samples, total RAM 16.0 GB"]


def test_validate_exit_code(tmp_path: Path) -> None:
    folder = tmp_path / "take-m_color"
    folder.mkdir()
    mirrorless = folder / "take-m_color.db3"
    messages = mirrorless_messages([10, 20, 30])
    rows = [(MirrorlessRos2Loader.TOPIC, round(message.stamp_s * 1e9), compressed_image_cdr(message)) for message in messages]
    write_db3(mirrorless, rows)
    assert [message.stamp_s for message in MirrorlessRos2Loader(mirrorless).load()] == pytest.approx([message.stamp_s for message in messages])
    result = subprocess.run([sys.executable, str(VALIDATE), str(mirrorless)], capture_output=True, text=True, check=False)
    assert ": 1 issue\n" in result.stdout
    assert "no -r (RealSense D435) recording of this take" in result.stdout
    assert result.returncode == 1
    write_db3(tmp_path / "take-r.db3", [(DEPTH_TOPIC, 1_000_000_000_000, b""), (DEPTH_TOPIC, 1_000_033_000_000, b"")])
    result = subprocess.run([sys.executable, str(VALIDATE), str(mirrorless)], capture_output=True, text=True, check=False)
    assert ": 0 issues\n" in result.stdout
    assert result.returncode == 0
