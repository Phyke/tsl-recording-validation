import numpy as np

from ..pydantic_models import ValidationIssue
from ..validator import RecordingValidationError, RecordingValidator, measure
from .depth_quality import fill_rates, flat_surface, surface_noise
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

    # fill rate: the part of the depth pixels that has a depth value, the mean of all the frames and the lowest frame
    # noise: how far the depth values of a flat surface are from a plane (depth_quality.surface_noise), the median of all the frames
    # the 2 values have no limit, so they give no issue
    def print_depth_quality(self) -> list[dict[str, str]]:
        frames = self.stream_frames("depth")
        if not frames:
            print("depth: fill rate not available, no depth stream")
            return [measure("depth", "fill rate", "not available, no depth stream")]
        depths = [frame.frame for _, frame in frames]
        fill = fill_rates(depths)
        lowest = int(np.argmin(fill))
        print(f"depth: fill rate {fill.mean():.4f}, lowest {fill[lowest]:.4f} in frame {frames[lowest][0]}")
        measures = [
            measure("depth", "fill rate", f"{fill.mean():.4f}"),
            measure("depth", "lowest fill rate", f"{fill[lowest]:.4f} in frame {frames[lowest][0]}"),
        ]
        baseline_mm = self.recording.sensor_options.get("Stereo Module", {}).get("stereo_baseline")
        if baseline_mm is None:
            reason = "no stereo_baseline option in this recording"
        elif len({depth.shape for depth in depths}) > 1:
            reason = "the depth frames have different sizes"
        else:
            first = frames[0][1]
            # disparity in pixels = fx * baseline / depth
            focal_baseline = first.intrinsics.fx * baseline_mm
            disparity_scale = focal_baseline / (first.depth_scale * 1000)
            surface = flat_surface(depths, disparity_scale)
            reason = "no flat surface in this recording" if surface is None else None
        if reason is None:
            u, v, part = surface
            noise, raw_distance = surface_noise(depths, u, v, disparity_scale)
            reason = "no depth values on the flat surface" if np.isnan(noise).all() else None
        if reason is not None:
            print(f"depth: noise not available, {reason}")
            return [*measures, measure("depth", "noise", f"not available, {reason}")]
        noise_px = float(np.nanmedian(noise))
        distance_mm = raw_distance * first.depth_scale * 1000
        noise_mm = noise_px * distance_mm**2 / focal_baseline
        print(f"depth: noise {noise_px:.4f} px = {noise_mm:.1f} mm at {distance_mm:.0f} mm, on a flat surface that covers {part:.1%} of the image")
        return [
            *measures,
            measure("depth", "noise", f"{noise_px:.4f} px = {noise_mm:.1f} mm at {distance_mm:.0f} mm"),
            measure("depth", "flat surface", f"{part:.1%} of the image"),
        ]

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
        self.measures = self.print_measured_fps("depth") + self.print_measured_fps("color") + self.print_depth_quality()
        if issues and raise_error:
            raise RecordingValidationError(issues)
        return issues
