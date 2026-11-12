from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np

from recording_reader import ValidationIssue

# an .mp4 file is checked as the mirrorless camera, as in MirrorlessRosValidator
EXPECTED_WIDTH, EXPECTED_HEIGHT, EXPECTED_FPS = 1920, 1080, 120
# a frame up to this many frames after the next one is reached by a decode forward, which is faster than a seek
MAX_DECODE_AHEAD = 240


# an .mp4 file: the frame times come from the packets with no decode; frame() seeks only for a frame before the decoder position
# or far after it
class Mp4Video:
    def __init__(self, path: Path) -> None:
        self.path = path
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            pts = sorted(packet.pts for packet in container.demux(stream) if packet.pts is not None)
            self.width, self.height = stream.codec_context.width, stream.codec_context.height
            self.declared_fps = float(stream.average_rate) if stream.average_rate is not None else np.nan
        if not pts:
            raise ValueError(f"{path.name}: no video frames")
        self.pts = np.array(pts, dtype=np.int64)
        # milliseconds from the start of the file: an .mp4 file has no absolute time for each frame
        self.times_ms = (self.pts - self.pts[0]) * time_base * 1000
        self.container: av.container.InputContainer | None = None
        self.decoded: Iterator[av.VideoFrame] | None = None
        self.next_index = -1

    def __len__(self) -> int:
        return len(self.pts)

    def frame(self, i: int) -> np.ndarray:
        if self.container is None:
            self.container = av.open(str(self.path))
        stream = self.container.streams.video[0]
        if self.decoded is None or i < self.next_index or i > self.next_index + MAX_DECODE_AHEAD:
            self.container.seek(int(self.pts[i]), stream=stream, backward=True, any_frame=False)
            self.decoded = self.container.decode(stream)
        for frame in self.decoded:
            if frame.pts is not None and frame.pts >= self.pts[i]:
                self.next_index = i + 1
                return frame.to_ndarray(format="rgb24")
        raise ValueError(f"{self.path.name}: frame {i} could not be decoded")

    # every frame in order with its own container, so that a worker thread does not share the decoder of the player
    def iter_frames(self) -> Iterator[tuple[float, np.ndarray]]:
        with av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            for frame in container.decode(stream):
                if frame.pts is not None:
                    yield (frame.pts - self.pts[0]) * time_base * 1000, frame.to_ndarray(format="rgb24")


# the mirrorless checks of an .mp4 file: size, declared fps, frames with the same time, and gaps longer than 1.5 frame periods;
# returns the measured fps lines and the issues
def validate_mp4(video: Mp4Video) -> tuple[list[str], list[ValidationIssue]]:
    issues = []
    if (video.width, video.height) != (EXPECTED_WIDTH, EXPECTED_HEIGHT):
        message = f"color.size: {video.width}x{video.height}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
        issues.append(ValidationIssue(check="size", stream="color", message=message))
    if not np.isfinite(video.declared_fps) or round(video.declared_fps) != EXPECTED_FPS:
        message = f"color.fps: {video.declared_fps:.2f} in the file header, expected {EXPECTED_FPS}"
        issues.append(ValidationIssue(check="fps", stream="color", message=message))
    intervals = np.diff(video.times_ms)
    same_time = (np.flatnonzero(intervals <= 0) + 1).tolist()
    if same_time:
        message = f"color.timestamp: {len(same_time)} frames with the same time as the previous frame"
        issues.append(ValidationIssue(check="timestamp_order", stream="color", frames=same_time, message=message))
    period_ms = 1000 / EXPECTED_FPS
    gaps = np.flatnonzero(intervals > 1.5 * period_ms)
    if gaps.size:
        missing = int(np.round(intervals[gaps] / period_ms).sum()) - gaps.size
        message = f"color.timestamp: {gaps.size} gaps longer than 1.5 frame periods, about {missing} frames missing"
        issues.append(ValidationIssue(check="timestamp_gaps", stream="color", frames=(gaps + 1).tolist(), message=message))
    duration_s = (video.times_ms[-1] - video.times_ms[0]) / 1000
    if duration_s <= 0:
        return [f"color: measured fps not available, {len(video)} frames in {duration_s} s"], issues
    return [f"color: file fps {(len(video) - 1) / duration_s:.2f}, {len(video)} frames in {duration_s:.3f} s"], issues
