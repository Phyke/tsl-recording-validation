import subprocess
import sys
from pathlib import Path

import pytest
from conftest import compressed_image_cdr, mirrorless_messages, write_db3, write_ros1_bag

from recording_reader import Db3, MirrorlessRos2Loader, Ros1BagIndex
from recordings import camera_or_none, find_partner, pair_issues, recording_duration_ms, take_key

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
        ("1-r-m.bag", None),
        ("x.bag", None),
    ],
)
def test_camera_comes_from_the_marker_in_the_file_name(name: str, camera: str | None) -> None:
    assert camera_or_none(Path(name)) == camera


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
