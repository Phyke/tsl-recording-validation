import re

import numpy as np
import pytest

from conftest import d435_recording
from recording_reader import D435Validator, Recording
from recording_reader.d435.depth_quality import fill_rates, flat_surface, surface_noise

# conftest.intrinsics has fx 600; with a baseline of 50 mm and a depth unit of 1 mm, disparity in pixels = 30000 / depth value
BASELINE_MM = 50.0
DISPARITY_SCALE = 600 * BASELINE_MM
NOISE_PX = 0.1


# depth frames of a flat wall at about 2000 mm (15 pixels of disparity) with Gaussian noise in disparity, a nearer block in front
# of the wall (a person), and 10 columns with no depth value
def wall_frames(count: int, noise_px: float = NOISE_PX) -> list[np.ndarray]:
    random = np.random.default_rng(0)
    v, u = np.mgrid[0:480, 0:640]
    disparity = 15 + 0.002 * (u - 320) + 0.001 * (v - 240)
    frames = []
    for _ in range(count):
        depth = np.round(DISPARITY_SCALE / (disparity + random.normal(0, noise_px, disparity.shape))).astype(np.uint16)
        depth[160:480, 260:380] = 1200
        depth[:, :10] = 0
        frames.append(depth)
    return frames


def wall_recording(count: int) -> Recording:
    recording = d435_recording(count)
    for frameset, depth in zip(recording.frames, wall_frames(count), strict=True):
        frameset.depth.frame = depth
    recording.sensor_options = {"Stereo Module": {"stereo_baseline": BASELINE_MM}}
    return recording


def test_fill_rate_is_the_part_of_the_pixels_with_a_depth_value() -> None:
    frames = wall_frames(2)
    frames[1][:240] = 0
    assert fill_rates(frames).tolist() == [630 / 640, 630 / 640 / 2]


def test_flat_surface_is_the_wall_and_not_the_block() -> None:
    u, v, part = flat_surface(wall_frames(9), DISPARITY_SCALE)
    assert not np.any((u >= 260) & (u < 380) & (v >= 160))
    assert u.min() >= 10
    assert len(u) <= 1024
    # the wall is 0.86 of the image; the edge margin removes a band at its border
    assert 0.7 < part < 0.86


def test_surface_noise_is_the_noise_of_the_wall() -> None:
    frames = wall_frames(9)
    u, v, _ = flat_surface(frames, DISPARITY_SCALE)
    noise, raw_distance = surface_noise(frames, u, v, DISPARITY_SCALE)
    # the 3 sigma limit of the fit removes the ends of the Gaussian noise, so the RMS is a little below the sigma
    assert np.median(noise) == pytest.approx(NOISE_PX, rel=0.05)
    assert raw_distance == pytest.approx(2000, rel=0.02)
    more_noise, _ = surface_noise(wall_frames(9, 2 * NOISE_PX), u, v, DISPARITY_SCALE)
    assert np.median(more_noise) == pytest.approx(2 * NOISE_PX, rel=0.05)


def test_surface_noise_ignores_an_arm_in_front_of_the_wall() -> None:
    frames = wall_frames(9)
    u, v, _ = flat_surface(frames, DISPARITY_SCALE)
    expected, _ = surface_noise(frames, u, v, DISPARITY_SCALE)
    # the arm is in one frame only, so the median image of the surface search does not have it
    frames[4] = frames[4].copy()
    frames[4][100:140, 400:600] = 1200
    noise, _ = surface_noise(frames, u, v, DISPARITY_SCALE)
    assert noise[4] == pytest.approx(expected[4], rel=0.05)


def test_surface_noise_is_nan_for_a_frame_with_no_depth_on_the_surface() -> None:
    frames = wall_frames(9)
    u, v, _ = flat_surface(frames, DISPARITY_SCALE)
    frames[4] = np.zeros_like(frames[4])
    noise, _ = surface_noise(frames, u, v, DISPARITY_SCALE)
    assert np.isnan(noise[4])
    assert np.isfinite(np.delete(noise, 4)).all()


def test_random_depth_has_no_flat_surface() -> None:
    frames = [frameset.depth.frame for frameset in d435_recording(6).frames]
    assert flat_surface(frames, DISPARITY_SCALE) is None


def test_validator_prints_the_fill_rate_and_the_noise_and_gives_no_issue(capsys: pytest.CaptureFixture[str]) -> None:
    issues = D435Validator(wall_recording(9)).validate(raise_error=False)
    lines = capsys.readouterr().out.splitlines()
    assert [issue.check for issue in issues] == []
    assert "depth: fill rate 0.9844, lowest 0.9844 in frame 0" in lines
    noise_lines = [line for line in lines if line.startswith("depth: noise")]
    assert len(noise_lines) == 1
    match = re.fullmatch(r"depth: noise (0\.\d{4}) px = (\d+\.\d) mm at (\d+) mm, on a flat surface that covers (\d+\.\d)% of the image", noise_lines[0])
    assert match is not None
    assert float(match[1]) == pytest.approx(NOISE_PX, rel=0.05)
    # 1 pixel of disparity at 2000 mm = 2000 * 2000 / 30000 = 133.3 mm
    assert float(match[2]) == pytest.approx(float(match[1]) * float(match[3]) ** 2 / DISPARITY_SCALE, abs=0.1)
    assert float(match[3]) == pytest.approx(2000, rel=0.02)


def test_validator_prints_why_the_noise_is_not_available(capsys: pytest.CaptureFixture[str]) -> None:
    recording = d435_recording(6)
    D435Validator(recording).validate(raise_error=False)
    assert "depth: noise not available, no stereo_baseline option in this recording" in capsys.readouterr().out.splitlines()
    recording.sensor_options = {"Stereo Module": {"stereo_baseline": BASELINE_MM}}
    D435Validator(recording).validate(raise_error=False)
    assert "depth: noise not available, no flat surface in this recording" in capsys.readouterr().out.splitlines()
    for frameset in recording.frames:
        frameset.depth = None
    D435Validator(recording).validate(raise_error=False)
    assert "depth: fill rate not available, no depth stream" in capsys.readouterr().out.splitlines()
