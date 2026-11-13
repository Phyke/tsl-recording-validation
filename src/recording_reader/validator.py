import itertools
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .pydantic_models import DepthFrame, Recording, StreamFrame, ValidationIssue

# the camera fps (frame numbers / duration) can differ from the expected fps by this fraction
MEASURED_FPS_TOLERANCE = 0.01
# rs.pipeline does not return a frame at the start or the end of the file that has no frame of the other stream at its time;
# 1 frame of a stream is usual, more frames are an issue
MAX_UNPAIRED_FRAMES = 1


class RecordingValidationError(Exception):
    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        super().__init__("\n".join([f"{len(issues)} problems", *(f"    {issue.message}" for issue in issues)]))


class RecordingValidator:
    def __init__(self, recording: Recording) -> None:
        self.recording = recording

    def stream_frames(self, stream_name: str) -> list[tuple[int, StreamFrame]]:
        return [(i, getattr(frame, stream_name)) for i, frame in enumerate(self.recording.frames) if getattr(frame, stream_name) is not None]

    @staticmethod
    def constant_values(stream_frame: StreamFrame) -> dict[str, Any]:
        values = stream_frame.model_dump(include={"timestamp_domain", "format", "fps"})
        values |= {f"intrinsics.{key}": value for key, value in stream_frame.intrinsics.model_dump().items()}

        values["shape"] = stream_frame.frame.shape
        if isinstance(stream_frame, DepthFrame):
            values["depth_scale"] = stream_frame.depth_scale
        return values

    def check_serial_number(self, expected: str) -> list[ValidationIssue]:
        if self.recording.serial_number == expected:
            return []
        return [ValidationIssue(check="serial_number", message=f"serial_number: {self.recording.serial_number}, expected {expected}")]

    def check_every_frameset_has_stream(self, stream_name: str) -> list[ValidationIssue]:
        missing = [i for i, frameset in enumerate(self.recording.frames) if getattr(frameset, stream_name) is None]
        # a recording with no framesets has no stream
        if len(missing) == len(self.recording.frames):
            return [ValidationIssue(check="missing_stream", stream=stream_name, message=f"no {stream_name} stream in this recording")]
        if not missing:
            return []
        return [
            ValidationIssue(
                check="missing_frame",
                stream=stream_name,
                frames=missing,
                message=f"no {stream_name} frame in {len(missing)} of {len(self.recording.frames)} framesets: {missing}",
            ),
        ]

    def check_no_frameset_has_stream(self, stream_name: str) -> list[ValidationIssue]:
        present = [i for i, frameset in enumerate(self.recording.frames) if getattr(frameset, stream_name) is not None]
        if not present:
            return []
        return [
            ValidationIssue(
                check="unexpected_stream",
                stream=stream_name,
                frames=present,
                message=f"{stream_name} frame in {len(present)} of {len(self.recording.frames)} framesets, expected no {stream_name} stream",
            ),
        ]

    @staticmethod
    def check_expected_settings(
        stream_name: str, frames: list[tuple[int, StreamFrame]], fps: int, width: int, height: int
    ) -> list[ValidationIssue]:
        issues = []
        fps_frames, size_frames = defaultdict(list), defaultdict(list)
        for i, stream_frame in frames:
            fps_frames[stream_frame.fps].append(i)
            size_frames[(stream_frame.frame.shape[1], stream_frame.frame.shape[0])].append(i)
        for value, indices in fps_frames.items():
            if value != fps:
                message = f"{stream_name}.fps: {value} in {len(indices)} of {len(frames)} frames, expected {fps}"
                issues.append(ValidationIssue(check="fps", stream=stream_name, frames=indices, message=message))
        for (frame_width, frame_height), indices in size_frames.items():
            if (frame_width, frame_height) != (width, height):
                message = (
                    f"{stream_name}.size: {frame_width}x{frame_height} in {len(indices)} of {len(frames)} frames, expected {width}x{height}"
                )
                issues.append(ValidationIssue(check="size", stream=stream_name, frames=indices, message=message))
        return issues

    @staticmethod
    def check_timestamp_domain(stream_name: str, frames: list[tuple[int, StreamFrame]], expected: str) -> list[ValidationIssue]:
        issues = []
        domain_frames = defaultdict(list)
        for i, stream_frame in frames:
            domain_frames[stream_frame.timestamp_domain].append(i)
        for value, indices in domain_frames.items():
            if value != expected:
                message = f"{stream_name}.timestamp_domain: {value} in {len(indices)} of {len(frames)} frames, expected {expected}"
                issues.append(ValidationIssue(check="timestamp_domain", stream=stream_name, frames=indices, message=message))
        return issues

    # a frame with the same pixels as the previous frame: the camera or the capture card gave one image 2 times
    @staticmethod
    def check_duplicate_images(stream_name: str, frames: list[tuple[int, StreamFrame]]) -> list[ValidationIssue]:
        duplicates = []
        for (_, previous), (i, current) in itertools.pairwise(frames):
            if previous.frame.shape != current.frame.shape:
                continue
            # the middle row is compared first, because almost all frames are different there
            middle = current.frame.shape[0] // 2
            if np.array_equal(previous.frame[middle], current.frame[middle]) and np.array_equal(previous.frame, current.frame):
                duplicates.append(i)
        if not duplicates:
            return []
        message = f"{stream_name}.frame: the same image as the previous frame in {len(duplicates)} of {len(frames)} frames: {duplicates}"
        return [ValidationIssue(check="duplicate_image", stream=stream_name, frames=duplicates, message=message)]

    @staticmethod
    def check_constant_values(stream_name: str, frames: list[tuple[int, StreamFrame]]) -> list[ValidationIssue]:
        issues = []
        reference = RecordingValidator.constant_values(frames[0][1])
        for i, stream_frame in frames[1:]:
            for key, value in RecordingValidator.constant_values(stream_frame).items():
                if value != reference[key]:
                    message = f"frame {i} {stream_name}.{key}: {value} != {reference[key]}"
                    issues.append(ValidationIssue(check="constant_value", stream=stream_name, frames=[i], message=message))
        return issues

    @staticmethod
    def check_metadata_keys(stream_name: str, frames: list[tuple[int, StreamFrame]]) -> list[ValidationIssue]:
        issues = []
        reference = set(frames[0][1].metadata)
        for i, stream_frame in frames[1:]:
            keys = set(stream_frame.metadata)
            if keys != reference:
                message = f"frame {i} {stream_name}.metadata: missing {sorted(reference - keys)}, extra {sorted(keys - reference)}"
                issues.append(ValidationIssue(check="metadata_keys", stream=stream_name, frames=[i], message=message))
        return issues

    @staticmethod
    def check_frame_numbers(stream_name: str, frames: list[tuple[int, StreamFrame]]) -> list[ValidationIssue]:
        issues = []
        gaps = []
        # the frameset index of the first frame after each gap
        gap_frames = []
        for (_, previous), (i, current) in itertools.pairwise(frames):
            step = current.frame_number - previous.frame_number
            if step == 0:
                message = f"frame {i} {stream_name}.frame_number: {current.frame_number} is a duplicate of the previous frame"
                issues.append(ValidationIssue(check="duplicate_frame_number", stream=stream_name, frames=[i], message=message))
            elif step < 0:
                message = f"frame {i} {stream_name}.frame_number: {current.frame_number} < previous {previous.frame_number}"
                issues.append(ValidationIssue(check="decreasing_frame_number", stream=stream_name, frames=[i], message=message))
            elif step > 1:
                gaps.append((previous.frame_number + 1, current.frame_number - 1))
                gap_frames.append(i)
        if gaps:
            n_missing = sum(end - start + 1 for start, end in gaps)
            ranges = ", ".join(str(start) if start == end else f"[{start}, {end}]" for start, end in gaps)
            message = f"{stream_name}.frame_number: {len(gaps)} gaps, {n_missing} frame numbers missing: {ranges}"
            issues.append(ValidationIssue(check="frame_number_gaps", stream=stream_name, frames=gap_frames, message=message))
        return issues

    @staticmethod
    def check_timestamps(stream_name: str, frames: list[tuple[int, StreamFrame]]) -> list[ValidationIssue]:
        issues = []
        for (_, previous), (i, current) in itertools.pairwise(frames):
            if current.timestamp_ms <= previous.timestamp_ms:
                message = f"frame {i} {stream_name}.timestamp_ms: {current.timestamp_ms} <= previous {previous.timestamp_ms}"
                issues.append(ValidationIssue(check="timestamp_order", stream=stream_name, frames=[i], message=message))
        return issues

    # the camera fps is the same value as in print_measured_fps: (last frame number - first frame number) / duration
    @staticmethod
    def check_measured_fps(stream_name: str, frames: list[tuple[int, StreamFrame]], fps: int) -> list[ValidationIssue]:
        first, last = frames[0][1], frames[-1][1]
        duration_s = (last.timestamp_ms - first.timestamp_ms) / 1000
        numbered = last.frame_number - first.frame_number
        if duration_s <= 0 or numbered <= 0:
            return []
        camera_fps = numbered / duration_s
        if abs(camera_fps - fps) <= fps * MEASURED_FPS_TOLERANCE:
            return []
        message = f"{stream_name}.camera_fps: {camera_fps:.2f}, expected {fps} within {MEASURED_FPS_TOLERANCE:.0%}"
        return [ValidationIssue(check="measured_fps", stream=stream_name, message=message)]

    def check_stream(self, stream_name: str, fps: int, width: int, height: int, timestamp_domain: str) -> list[ValidationIssue]:
        frames = self.stream_frames(stream_name)
        if not frames:
            return []
        return (
            self.check_expected_settings(stream_name, frames, fps, width, height)
            + self.check_timestamp_domain(stream_name, frames, timestamp_domain)
            + self.check_constant_values(stream_name, frames)
            + self.check_metadata_keys(stream_name, frames)
            + self.check_frame_numbers(stream_name, frames)
            + self.check_timestamps(stream_name, frames)
            + self.check_measured_fps(stream_name, frames, fps)
            + self.check_duplicate_images(stream_name, frames)
        )

    def check_frames_loaded(self, stream_name: str) -> list[ValidationIssue]:
        file_count = self.recording.file_frame_counts.get(stream_name)
        if file_count is None:
            return []
        not_loaded = file_count - sum(1 for frameset in self.recording.frames if getattr(frameset, stream_name) is not None)
        if not_loaded <= MAX_UNPAIRED_FRAMES:
            return []
        message = (
            f"{stream_name}: {not_loaded} of the {file_count} frames in the file are not in the loaded framesets, "
            f"expected at most {MAX_UNPAIRED_FRAMES}"
        )
        return [ValidationIssue(check="frames_not_loaded", stream=stream_name, message=message)]

    # the text that tells the user what to do about a notification; a camera validator gives the text for its camera
    def notification_advice(self, description: str) -> str:
        return ""

    # one issue for each different notification, with the number of times that it is in the recording
    def check_notifications(self) -> list[ValidationIssue]:
        counts = Counter((n.sensor, n.category, n.severity, n.description) for n in self.recording.notifications)
        issues = []
        for (sensor, category, severity, description), count in counts.items():
            message = f"notification: {description} ({category}, {severity}, {sensor}), {count} time{'s' if count > 1 else ''}"
            advice = self.notification_advice(description)
            if advice:
                message += f". What to do: {advice}"
            issues.append(ValidationIssue(check="notification", message=message))
        return issues

    # file fps: frames in the file, (frames - 1) / duration
    # camera fps: frames that the camera numbered, (last frame number - first frame number) / duration
    # kept: frames in the file as a percentage of the frames that the camera numbered
    def print_measured_fps(self, stream_name: str) -> None:
        frames = self.stream_frames(stream_name)
        if not frames:
            print(f"{stream_name}: measured fps not available, no {stream_name} stream")
            return
        if len(frames) < 2:
            print(f"{stream_name}: measured fps not available, only 1 frame")
            return
        first, last = frames[0][1], frames[-1][1]
        duration_s = (last.timestamp_ms - first.timestamp_ms) / 1000
        if duration_s <= 0:
            print(f"{stream_name}: measured fps not available, duration {duration_s} s")
            return
        file_fps = (len(frames) - 1) / duration_s
        numbered = last.frame_number - first.frame_number
        if numbered <= 0:
            print(f"{stream_name}: file fps {file_fps:.2f}, camera fps not available, frame numbers do not increase")
            return
        camera_fps = numbered / duration_s
        kept = (len(frames) - 1) / numbered * 100
        print(
            f"{stream_name}: file fps {file_fps:.2f}, camera fps {camera_fps:.2f}, kept {kept:.1f}%, {len(frames)} frames in {duration_s:.3f} s"
        )
