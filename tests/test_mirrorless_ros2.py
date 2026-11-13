import numpy as np
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
    assert [(issue.check, issue.frames) for issue in issues] == [("duplicate_image", [2, 4])]


def test_barcode_fps_out_of_tolerance_is_an_issue() -> None:
    messages = mirrorless_messages([10, 20, 30, 40, 50])
    for i, message in enumerate(messages):
        message.stamp_s = 1000 + i / 100
    issues = MirrorlessRos2Validator(messages).validate(raise_error=False)
    assert [(issue.check, issue.message) for issue in issues] == [("measured_fps", "color.barcode_fps: 100.00, expected 120 within 1%")]
