import io
import itertools
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image

from ..pydantic_models import CompressedImageMessage, ValidationIssue
from ..validator import RecordingValidationError, RecordingValidator
from .barcode import read_barcode


# -m ROS (.bag) files: mirrorless camera, color only, 1920x1080 at 120 fps
class MirrorlessRosValidator(RecordingValidator):
    def validate(self, raise_error: bool = True) -> list[ValidationIssue]:
        issues = self.check_every_frameset_has_stream("color")
        issues += self.check_no_frameset_has_stream("depth")
        issues += self.check_stream("color", fps=120, width=1920, height=1080)
        self.print_measured_fps("color")
        if issues and raise_error:
            raise RecordingValidationError(issues)
        return issues


# -m ROS2 (.db3) files from poc1 UvcRos2Bag: JPEG frames of the mirrorless camera, 1920x1080 at 120 fps,
# with the frame number as a barcode along the top edge of each frame
class MirrorlessRos2Validator:
    def __init__(self, frames: list[CompressedImageMessage]) -> None:
        if not frames:
            raise ValueError("no frames")
        self.frames = frames

    def frame_values(self) -> list[dict[str, Any]]:
        values = []
        for message in self.frames:
            image = Image.open(io.BytesIO(message.data))
            values.append({"format": image.format, "size": image.size, "barcode": read_barcode(np.asarray(image.convert("RGB")))})
        return values

    @staticmethod
    def check_format(values: list[dict[str, Any]]) -> list[ValidationIssue]:
        frames = [i for i, value in enumerate(values) if value["format"] != "JPEG"]
        if not frames:
            return []
        formats = sorted({str(values[i]["format"]) for i in frames})
        message = f"color.format: {formats} in {len(frames)} of {len(values)} frames, expected JPEG"
        return [ValidationIssue(check="format", stream="color", frames=frames, message=message)]

    @staticmethod
    def check_size(values: list[dict[str, Any]], width: int, height: int) -> list[ValidationIssue]:
        issues = []
        size_frames = defaultdict(list)
        for i, value in enumerate(values):
            size_frames[value["size"]].append(i)
        for (frame_width, frame_height), indices in size_frames.items():
            if (frame_width, frame_height) != (width, height):
                message = f"color.size: {frame_width}x{frame_height} in {len(indices)} of {len(values)} frames, expected {width}x{height}"
                issues.append(ValidationIssue(check="size", stream="color", frames=indices, message=message))
        return issues

    @staticmethod
    def check_barcodes(values: list[dict[str, Any]]) -> list[ValidationIssue]:
        issues = []
        missing = [i for i, value in enumerate(values) if value["barcode"] is None]
        if missing:
            message = f"color.barcode: no barcode in {len(missing)} of {len(values)} frames: {missing}"
            issues.append(ValidationIssue(check="missing_barcode", stream="color", frames=missing, message=message))
        numbered = [(i, value["barcode"]) for i, value in enumerate(values) if value["barcode"] is not None]
        gaps = []
        # the frame index of the first frame after each gap
        gap_frames = []
        for (_, previous), (i, current) in itertools.pairwise(numbered):
            step = current - previous
            if step == 0:
                message = f"frame {i} color.barcode: {current} is a duplicate of the previous frame"
                issues.append(ValidationIssue(check="duplicate_barcode", stream="color", frames=[i], message=message))
            elif step < 0:
                message = f"frame {i} color.barcode: {current} < previous {previous}"
                issues.append(ValidationIssue(check="decreasing_barcode", stream="color", frames=[i], message=message))
            elif step > 1:
                gaps.append((previous + 1, current - 1))
                gap_frames.append(i)
        if gaps:
            n_missing = sum(end - start + 1 for start, end in gaps)
            ranges = ", ".join(str(start) if start == end else f"[{start}, {end}]" for start, end in gaps)
            message = f"color.barcode: {len(gaps)} gaps, {n_missing} frame numbers missing: {ranges}"
            issues.append(ValidationIssue(check="barcode_gaps", stream="color", frames=gap_frames, message=message))
        return issues

    def check_timestamps(self) -> list[ValidationIssue]:
        issues = []
        for i, (previous, current) in enumerate(itertools.pairwise(self.frames), start=1):
            if current.stamp_s <= previous.stamp_s:
                message = f"frame {i} color.stamp_s: {current.stamp_s} <= previous {previous.stamp_s}"
                issues.append(ValidationIssue(check="timestamp_order", stream="color", frames=[i], message=message))
        return issues

    # file fps: frames in the file, (frames - 1) / duration
    # barcode fps: frames that the pipeline numbered, (last barcode - first barcode) / duration
    # kept: frames in the file as a percentage of the frames that the pipeline numbered
    def print_measured_fps(self, values: list[dict[str, Any]]) -> None:
        duration_s = self.frames[-1].stamp_s - self.frames[0].stamp_s
        if duration_s <= 0:
            print(f"color: measured fps not available, {len(self.frames)} frames in {duration_s} s")
            return
        file_fps = (len(self.frames) - 1) / duration_s
        barcodes = [value["barcode"] for value in values if value["barcode"] is not None]
        numbered = barcodes[-1] - barcodes[0] if len(barcodes) >= 2 else 0
        if numbered <= 0:
            print(f"color: file fps {file_fps:.2f}, barcode fps not available, barcodes do not increase")
            return
        barcode_fps = numbered / duration_s
        kept = (len(barcodes) - 1) / numbered * 100
        print(
            f"color: file fps {file_fps:.2f}, barcode fps {barcode_fps:.2f}, kept {kept:.1f}%, {len(self.frames)} frames in {duration_s:.3f} s"
        )

    def validate(self, raise_error: bool = True) -> list[ValidationIssue]:
        values = self.frame_values()
        issues = self.check_format(values)
        issues += self.check_size(values, width=1920, height=1080)
        issues += self.check_barcodes(values)
        issues += self.check_timestamps()
        self.print_measured_fps(values)
        if issues and raise_error:
            raise RecordingValidationError(issues)
        return issues
