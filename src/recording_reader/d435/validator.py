from ..pydantic_models import ValidationIssue
from ..validator import RecordingValidationError, RecordingValidator


# -r files: RealSense D435, color and depth, 640x480 at 30 fps
class D435Validator(RecordingValidator):
    def check_depth_to_color(self) -> list[ValidationIssue]:
        if self.recording.depth_to_color is not None:
            return []
        return [ValidationIssue(check="depth_to_color", message="depth_to_color: no depth-to-color extrinsics in this recording")]

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

    def validate(self, expected_serial_number: str | None = None, raise_error: bool = True) -> list[ValidationIssue]:
        issues = self.check_serial_number(expected_serial_number) if expected_serial_number is not None else []
        issues += self.check_depth_to_color()
        issues += self.check_every_frameset_has_stream("color")
        issues += self.check_every_frameset_has_stream("depth")
        issues += self.check_stream("depth", fps=30, width=640, height=480)
        issues += self.check_stream("color", fps=30, width=640, height=480)
        issues += self.check_color_depth_pairs(fps=30)
        self.print_measured_fps("depth")
        self.print_measured_fps("color")
        if issues and raise_error:
            raise RecordingValidationError(issues)
        return issues
