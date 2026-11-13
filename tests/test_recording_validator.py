from conftest import d435_recording, mirrorless_recording

from recording_reader import D435Validator, MirrorlessRosValidator, Notification, Recording


def checks(issues: list) -> list[str]:
    return sorted(issue.check for issue in issues)


def test_d435_recording_with_no_fault_has_no_issue() -> None:
    assert D435Validator(d435_recording()).validate(raise_error=False) == []


def test_mirrorless_recording_with_no_fault_has_no_issue() -> None:
    assert MirrorlessRosValidator(mirrorless_recording()).validate(raise_error=False) == []


def test_empty_recording_is_a_missing_stream() -> None:
    d435 = D435Validator(Recording(serial_number="317622070655", frames=[])).validate(raise_error=False)
    assert checks(d435) == ["depth_to_color", "missing_stream", "missing_stream"]
    assert checks(MirrorlessRosValidator(Recording(frames=[])).validate(raise_error=False)) == ["missing_stream"]


def test_notification_is_an_issue_with_the_advice() -> None:
    recording = d435_recording()
    notification = Notification(
        sensor="Stereo Module", timestamp_ms=0, category="hardware_error", severity="error", description="Right MIPI error",
    )
    recording.notifications = [notification, notification]
    issues = D435Validator(recording).validate(raise_error=False)
    assert checks(issues) == ["notification"]
    assert "Right MIPI error" in issues[0].message
    assert "2 times" in issues[0].message
    assert "What to do: Stop and start the stream again." in issues[0].message


def test_unknown_serial_number_is_an_issue() -> None:
    recording = d435_recording()
    recording.serial_number = "000000000000"
    assert checks(D435Validator(recording).validate(raise_error=False)) == ["serial_number"]
    assert D435Validator(recording).validate(expected_serial_number="000000000000", raise_error=False) == []


def test_system_time_is_an_issue_for_a_d435() -> None:
    issues = D435Validator(d435_recording(domain="system_time")).validate(raise_error=False)
    assert [(issue.check, issue.stream, len(issue.frames)) for issue in issues] == [
        ("timestamp_domain", "depth", 6),
        ("timestamp_domain", "color", 6),
    ]


def test_camera_fps_out_of_tolerance_is_an_issue() -> None:
    assert D435Validator(d435_recording(period_ms=1000 / 30.2)).validate(raise_error=False) == []
    issues = D435Validator(d435_recording(period_ms=1000 / 29)).validate(raise_error=False)
    assert [(issue.check, issue.stream) for issue in issues] == [("measured_fps", "depth"), ("measured_fps", "color")]
    assert issues[0].message == "depth.camera_fps: 29.00, expected 30 within 1%"


def test_same_image_as_the_previous_frame_is_an_issue() -> None:
    recording = d435_recording()
    recording.frames[3].color.frame = recording.frames[2].color.frame.copy()
    recording.frames[5].color.frame = recording.frames[4].color.frame.copy()
    issues = D435Validator(recording).validate(raise_error=False)
    assert [(issue.check, issue.stream, issue.frames) for issue in issues] == [("duplicate_image", "color", [3, 5])]


def test_frames_of_the_file_that_are_not_loaded() -> None:
    recording = d435_recording()
    recording.file_frame_counts = {"depth": 7, "color": 6}
    assert D435Validator(recording).validate(raise_error=False) == []
    recording.file_frame_counts = {"depth": 9, "color": 6}
    issues = D435Validator(recording).validate(raise_error=False)
    assert [(issue.check, issue.stream) for issue in issues] == [("frames_not_loaded", "depth")]
    assert issues[0].message == "depth: 3 of the 9 frames in the file are not in the loaded framesets, expected at most 1"
