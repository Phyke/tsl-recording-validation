import numpy as np
import pytest
from conftest import barcode_image, mirrorless_messages

from recording_reader import MirrorlessRos2Validator, decode_jpeg, read_barcode


def test_barcode_is_read_from_a_jpeg_frame() -> None:
    for number in [0, 1, 26, 123456, 2**31 + 5]:
        assert read_barcode(barcode_image(number, shade=10)) == number
    assert read_barcode(decode_jpeg(mirrorless_messages([10], first_number=1061)[0].data)) == 1061


def test_black_frame_and_white_frame_have_no_barcode() -> None:
    assert read_barcode(np.zeros((1080, 1920, 3), np.uint8)) is None
    assert read_barcode(np.full((1080, 1920, 3), 255, np.uint8)) is None


def test_mirrorless_db3_frames_with_no_fault_have_no_issue() -> None:
    assert MirrorlessRos2Validator(mirrorless_messages([10, 20, 30, 40, 50])).validate(raise_error=False) == []


def test_same_camera_image_with_a_different_barcode_is_an_issue() -> None:
    issues = MirrorlessRos2Validator(mirrorless_messages([10, 20, 20, 30, 30])).validate(raise_error=False)
    assert [(issue.check, issue.frames) for issue in issues] == [("duplicate_image", [2, 4]), ("image_fps", [])]
    # the barcodes go up by 1, so the barcode fps is 120, but only 3 of the 5 frames have a new image
    assert issues[1].message == "color.image_fps: 60.00 (3 different images in 5 frames), expected 120 within 1%"


def test_the_runs_of_equal_images_are_printed(capsys: pytest.CaptureFixture[str]) -> None:
    validator = MirrorlessRos2Validator(mirrorless_messages([10, 20, 20, 30, 30, 30]))
    validator.validate(raise_error=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "color: image fps 48.00, 3 different images in 6 frames, longest run 3 frames, run lengths {1: 1, 2: 1, 3: 1}"
    # the same values are the measures of the Measures table
    assert {row["name"]: row["value"] for row in validator.measures} == {
        "file fps": "120.00",
        "barcode fps": "120.00",
        "kept": "100.0%",
        "frames": "6",
        "duration": "0.042 s",
        "image fps": "48.00",
        "different images": "3 of 6 frames",
        "longest run": "3 frames",
        "run lengths": "{1: 1, 2: 1, 3: 1}",
    }


def test_barcode_fps_out_of_tolerance_is_an_issue() -> None:
    messages = mirrorless_messages([10, 20, 30, 40, 50])
    for i, message in enumerate(messages):
        message.stamp_s = 1000 + i / 100
    issues = MirrorlessRos2Validator(messages).validate(raise_error=False)
    assert [(issue.check, issue.message) for issue in issues] == [("measured_fps", "color.barcode_fps: 100.00, expected 120 within 1%")]
