import itertools

import numpy as np
import pytest
from conftest import d435_recording, mirrorless_messages

from motion_correlation import d435_motion, mirrorless_motion, motion_between, small_gray
from recording_reader import LoadCancelled, decode_jpeg


def test_mirrorless_motion_on_threads_is_equal_to_the_motion_of_one_frame_after_the_other() -> None:
    messages = mirrorless_messages([10, 20, 20, 30, 50, 90])
    grays = [small_gray(decode_jpeg(message.data)) for message in messages]
    times = [message.stamp_s * 1000 for message in messages]
    pairs = [motion_between(a, b, ta, tb) for (a, ta), (b, tb) in itertools.pairwise(zip(grays, times, strict=True))]
    motion = mirrorless_motion(messages)
    assert np.array_equal(motion["time_ms"], np.array([time for time, _ in pairs]))
    assert np.array_equal(motion["color"], np.array([value for _, value in pairs]))
    assert (motion["color"] > 0).all()


def test_mirrorless_motion_stops_with_load_cancelled() -> None:
    with pytest.raises(LoadCancelled):
        mirrorless_motion(mirrorless_messages([10, 20, 30]), should_stop=lambda: True)


def test_d435_motion_stops_with_load_cancelled() -> None:
    recording = d435_recording()
    assert len(d435_motion(recording, should_stop=lambda: False)["color"]) == len(recording.frames) - 1
    with pytest.raises(LoadCancelled):
        d435_motion(recording, should_stop=lambda: True)
