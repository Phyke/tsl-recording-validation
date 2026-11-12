import itertools

import numpy as np

from mp4_video import Mp4Video
from recording_reader import CompressedImageMessage, Recording, decode_jpeg

SHIFTS = np.arange(-30, 31)
# the frames are cut to about this width before the difference, so that 640 and 1920 pixel wide frames give a similar scale
SMALL_WIDTH = 160


def small_gray(rgb: np.ndarray) -> np.ndarray:
    step = max(1, rgb.shape[1] // SMALL_WIDTH)
    return rgb[::step, ::step].astype(np.float32).mean(axis=2)


# the motion signal: the mean absolute change between 2 frames in a row, per second, at the middle time of the 2 frames
def motion_between(previous: np.ndarray, current: np.ndarray, previous_ms: float, current_ms: float) -> tuple[float, float]:
    dt_s = (current_ms - previous_ms) / 1000
    if dt_s <= 0:
        return (previous_ms + current_ms) / 2, np.nan
    return (previous_ms + current_ms) / 2, float(np.abs(current - previous).mean()) / dt_s


# the D435 motion of color and of depth, one value per frameset from the second on, NaN where a frame of the pair is missing;
# the depth change counts only the pixels with depth > 0 in both frames, in meters
def d435_motion(recording: Recording) -> dict:
    frames = recording.frames
    grays = [small_gray(frame.color.frame) if frame.color is not None else None for frame in frames]
    time_ms, color, depth = (np.full(len(frames) - 1, np.nan) for _ in range(3))
    for t, (previous, current) in enumerate(itertools.pairwise(frames)):
        if grays[t] is not None and grays[t + 1] is not None:
            time_ms[t], color[t] = motion_between(grays[t], grays[t + 1], previous.color.timestamp_ms, current.color.timestamp_ms)
        if previous.depth is not None and current.depth is not None:
            a = previous.depth.frame[::4, ::4] * previous.depth.depth_scale
            b = current.depth.frame[::4, ::4] * current.depth.depth_scale
            valid = (a > 0) & (b > 0)
            if valid.any():
                _, depth[t] = motion_between(a[valid], b[valid], previous.depth.timestamp_ms, current.depth.timestamp_ms)
    color_times = [frame.color.timestamp_ms for frame in frames if frame.color is not None]
    return {"time_ms": time_ms, "color": color, "depth": depth, "first_color_ms": min(color_times) if color_times else np.nan}


# the mirrorless color motion between the frames in the file, in file order; an .mp4 file has times from its own start
def mirrorless_motion(loaded: Recording | list[CompressedImageMessage] | Mp4Video) -> dict:
    if isinstance(loaded, Mp4Video):
        times, grays = [], []
        for time_ms, rgb in loaded.iter_frames():
            times.append(time_ms)
            grays.append(small_gray(rgb))
    elif isinstance(loaded, list):
        times = [message.stamp_s * 1000 for message in loaded]
        grays = [small_gray(decode_jpeg(message.data)) for message in loaded]
    else:
        colors = [frame.color for frame in loaded.frames if frame.color is not None]
        times = [color.timestamp_ms for color in colors]
        grays = [small_gray(color.frame) for color in colors]
    pairs = [motion_between(a, b, ta, tb) for (a, ta), (b, tb) in itertools.pairwise(zip(grays, times, strict=True))]
    return {
        "time_ms": np.array([time for time, _ in pairs]),
        "color": np.array([value for _, value in pairs]),
        "relative_time": isinstance(loaded, Mp4Video),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return np.nan
    a, b = a[valid] - a[valid].mean(), b[valid] - b[valid].mean()
    scale = np.sqrt((a**2).sum() * (b**2).sum())
    return float((a * b).sum() / scale) if scale > 0 else np.nan


# correlation of first[t] and second[t + k] for each shift k, on the frameset index
def index_curve(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    n = len(first)
    values = []
    for k in SHIFTS:
        low, high = max(0, -k), n - max(0, k)
        values.append(pearson(first[low:high], second[low + k:high + k]) if high - low >= 3 else np.nan)
    return np.array(values)


# correlation of the D435 color motion at t and the mirrorless color motion at t + k D435 frame periods, on the timestamps of
# the 2 cameras (D435 global_time and mirrorless system_time are both milliseconds since 1970)
def time_curve(d435_time_ms: np.ndarray, d435_color: np.ndarray, mirrorless_time_ms: np.ndarray, mirrorless_color: np.ndarray,
               period_ms: float) -> np.ndarray:
    valid = np.isfinite(mirrorless_time_ms) & np.isfinite(mirrorless_color)
    order = np.argsort(mirrorless_time_ms[valid])
    times, motion = mirrorless_time_ms[valid][order], mirrorless_color[valid][order]
    if len(times) < 3:
        return np.full(len(SHIFTS), np.nan)
    values = []
    for k in SHIFTS:
        query = d435_time_ms + k * period_ms
        inside = np.isfinite(query) & (query >= times[0]) & (query <= times[-1])
        values.append(pearson(d435_color, np.where(inside, np.interp(np.nan_to_num(query), times, motion), np.nan)))
    return np.array(values)


def peak(values: np.ndarray) -> tuple[int, float] | None:
    if not np.isfinite(values).any():
        return None
    k = int(np.nanargmax(values))
    return int(SHIFTS[k]), float(values[k])


# the curves from the motion signals of one D435 recording and of its mirrorless recording when there is one; an .mp4 file has no
# absolute time, so its first frame is put at the time of the first D435 color frame: shift 0 means that the 2 files started together
def correlation_from_motion(d435: dict, mirrorless: dict | None) -> dict:
    period_ms = float(np.nanmedian(np.diff(d435["time_ms"])))
    curves = [{"label": "color-depth (D435)", "values": index_curve(d435["color"], d435["depth"])}]
    note = ""
    if mirrorless is not None:
        mirrorless_time_ms = mirrorless["time_ms"]
        if mirrorless["relative_time"]:
            mirrorless_time_ms = mirrorless_time_ms + d435["first_color_ms"]
            note = "mp4: no absolute time, shift 0 = the 2 files started together"
        values = time_curve(d435["time_ms"], d435["color"], mirrorless_time_ms, mirrorless["color"], period_ms)
        curves.append({"label": "color-color (D435 vs mirrorless)", "values": values})
    for curve in curves:
        curve["peak"] = peak(curve["values"])
    return {"shifts": SHIFTS, "period_ms": period_ms, "curves": curves, "note": note}


def correlation_curves(d435: Recording, mirrorless: Recording | list[CompressedImageMessage] | Mp4Video | None) -> dict:
    return correlation_from_motion(d435_motion(d435), mirrorless_motion(mirrorless) if mirrorless is not None else None)
