from ..pydantic_models import ValidationIssue
from ..validator import RecordingValidationError, RecordingValidator
from .notifications import D400_FW_ERROR_REPORT, D400_WHAT_TO_DO

# the serial numbers of the D435 cameras of the project; a recording from a different camera gets a serial_number issue
KNOWN_SERIAL_NUMBERS = {"317622070655", "239722070026"}
# global_time: the capture time from the camera clock, converted to the clock of the computer; system_time shows that the
# computer got no hardware timestamps from the camera, so the time is the arrival time of the frame
TIMESTAMP_DOMAIN = "global_time"
NOTIFICATION_TYPES = {text: notification_type for notification_type, text in D400_FW_ERROR_REPORT.items()}


# -r files: RealSense D435, color and depth, 640x480 at 30 fps
class D435Validator(RecordingValidator):
    def check_known_serial_number(self) -> list[ValidationIssue]:
        if self.recording.serial_number in KNOWN_SERIAL_NUMBERS:
            return []
        message = f"serial_number: {self.recording.serial_number}, expected one of {sorted(KNOWN_SERIAL_NUMBERS)}"
        return [ValidationIssue(check="serial_number", message=message)]

    def check_depth_to_color(self) -> list[ValidationIssue]:
        if self.recording.depth_to_color is not None:
            return []
        return [ValidationIssue(check="depth_to_color", message="depth_to_color: no depth-to-color extrinsics in this recording")]

    def notification_advice(self, description: str) -> str:
        return D400_WHAT_TO_DO.get(NOTIFICATION_TYPES.get(description), "")

    # a pair further apart than half a frame period has a closer partner frame in the other stream
    def check_color_depth_pairs(self, fps: int) -> list[ValidationIssue]:
        tolerance_ms = 1000 / fps / 2
        issues = []
        for i, frameset in enumerate(self.recording.frames):
            if frameset.color is None or frameset.depth is None:
                continue
            offset_ms = frameset.color.timestamp_ms - frameset.depth.timestamp_ms
            if abs(offset_ms) > tolerance_ms:
                message = f"frame {i} color.timestamp_ms - depth.timestamp_ms: {offset_ms:+.2f} ms, expected within {tolerance_ms:.2f} ms"
                issues.append(ValidationIssue(check="color_depth_offset", frames=[i], message=message))
        return issues

    # with no expected_serial_number, the serial number must be one of KNOWN_SERIAL_NUMBERS
    def validate(self, expected_serial_number: str | None = None, raise_error: bool = True) -> list[ValidationIssue]:
        if expected_serial_number is not None:
            issues = self.check_serial_number(expected_serial_number)
        else:
            issues = self.check_known_serial_number()
        issues += self.check_notifications()
        issues += self.check_depth_to_color()
        issues += self.check_every_frameset_has_stream("color")
        issues += self.check_every_frameset_has_stream("depth")
        issues += self.check_frames_loaded("depth")
        issues += self.check_frames_loaded("color")
        issues += self.check_stream("depth", fps=30, width=640, height=480, timestamp_domain=TIMESTAMP_DOMAIN)
        issues += self.check_stream("color", fps=30, width=640, height=480, timestamp_domain=TIMESTAMP_DOMAIN)
        issues += self.check_color_depth_pairs(fps=30)
        self.print_measured_fps("depth")
        self.print_measured_fps("color")
        if issues and raise_error:
            raise RecordingValidationError(issues)
        return issues
