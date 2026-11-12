from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs

from .pydantic_models import (
    ColorFrame,
    DepthFrame,
    Extrinsics,
    Frame,
    Intrinsics,
    Notification,
    Recording,
)
from .ros2 import Db3, cdr_string


def print_device_info(device: rs.device) -> None:
    for key, info in rs.camera_info.__members__.items():
        if device.supports(info):
            print(f"{key} = {device.get_info(info)}")


class RealSenseLoader:
    # forward_fill: rs.pipeline puts the last frame of a stream again into a frameset that has no new frame of that stream
    # (librealsense src/pipeline/aggregator.cpp), False sets that repeated frame to None
    def __init__(self, path: str | Path, forward_fill: bool = True) -> None:
        self.path = Path(path)
        self.forward_fill = forward_fill

    @staticmethod
    def read_common_frame_metadata(frame: rs.frame) -> dict[str, Any]:
        stream_profile = frame.get_profile().as_video_stream_profile()
        intrinsics = stream_profile.get_intrinsics()
        return {
            "timestamp_ms": frame.get_timestamp(),
            "timestamp_domain": frame.get_frame_timestamp_domain().name,
            "frame_number": frame.get_frame_number(),
            "format": stream_profile.format().name,
            "fps": stream_profile.fps(),
            "intrinsics": Intrinsics(
                width=intrinsics.width,
                height=intrinsics.height,
                ppx=intrinsics.ppx,
                ppy=intrinsics.ppy,
                fx=intrinsics.fx,
                fy=intrinsics.fy,
                model=intrinsics.model.name,
                coeffs=intrinsics.coeffs,
            ),
            "metadata": {
                key: frame.get_frame_metadata(value)
                for key, value in rs.frame_metadata_value.__members__.items()
                if frame.supports_frame_metadata(value)
            },
            "frame": np.asanyarray(frame.get_data()).copy(),
        }

    @staticmethod
    def read_depth_to_color(profile: rs.pipeline_profile) -> Extrinsics | None:
        streams = {stream.stream_type(): stream for stream in profile.get_streams()}
        if rs.stream.depth not in streams or rs.stream.color not in streams:
            return None
        extrinsics = streams[rs.stream.depth].get_extrinsics_to(streams[rs.stream.color])
        return Extrinsics(rotation=extrinsics.rotation, translation=extrinsics.translation)

    # each sensor gets its own callback, so that the callback knows the sensor name
    @staticmethod
    def notification_callback(sensor_name: str, notifications: list[Notification]) -> Callable[[rs.notification], None]:
        def callback(notification: rs.notification) -> None:
            notifications.append(
                Notification(
                    sensor=sensor_name,
                    timestamp_ms=notification.get_timestamp(),
                    category=notification.get_category().name,
                    severity=notification.get_severity().name,
                    description=notification.get_description(),
                ),
            )

        return callback

    # the topics are /device_0/sensor_<index>/option/<Name>/value
    def read_db3_options(self) -> dict[int, dict[str, float]]:
        db3 = Db3(self.path)
        options = {}
        for topic in db3.topic_names():
            parts = topic.split("/")
            if len(parts) != 6 or parts[3] != "option" or parts[5] != "value":
                continue
            values = {float(cdr_string(blob)) for blob in db3.messages(topic)}
            if len(values) != 1:
                raise ValueError(f"{self.path.name}: {topic} has {len(values)} different values")
            options.setdefault(int(parts[2].removeprefix("sensor_")), {})[parts[4].lower()] = values.pop()
        db3.connection.close()
        return options

    # fallback for a .db3 file: the raw topics add each option that the SDK did not return
    def read_sensor_options(self, device: rs.device) -> dict[str, dict[str, float]]:
        db3_options = self.read_db3_options() if self.path.suffix == ".db3" else {}
        sensor_options = {}
        for index, sensor in enumerate(device.query_sensors()):
            options = {option.name: sensor.get_option(option) for option in sensor.get_supported_options()}
            for name, value in db3_options.get(index, {}).items():
                options.setdefault(name, value)
            sensor_options[sensor.get_info(rs.camera_info.name)] = options
        return sensor_options

    def start(self) -> tuple[rs.pipeline, rs.pipeline_profile]:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device_from_file(str(self.path), repeat_playback=False)
        profile = pipeline.start(config)
        profile.get_device().as_playback().set_real_time(False)
        return pipeline, profile

    def read_frames(self, pipeline: rs.pipeline) -> Iterator[Frame]:
        # the finally block also stops the pipeline when the caller leaves the loop early
        previous_frame_numbers = {"depth": None, "color": None}
        try:
            while True:
                # Wait for the next set of frames from the pipeline
                received, frameset = pipeline.try_wait_for_frames(1000)

                # Return if no frames were received in 1 second.
                if not received:
                    return

                # Separate the depth and color frames from the frameset
                frames = {"depth": frameset.get_depth_frame(), "color": frameset.get_color_frame()}

                # Run 2 checks on each frame
                for stream_name, frame in frames.items():
                    # If there is a frame at all
                    # This case is when the entire stream of color or depth is missing.
                    # Set to None and jump to the next frame
                    if not frame:
                        frames[stream_name] = None
                        continue

                    # Check if the current frame is a duplicate of the previous one
                    # By default, librealsense forward-fill missing frames while still maintaining frame numbers as an indicator.
                    # If forward_fill is disabled, with the help of tracking previous frame numbers,
                    # we can detect and set missing frames to None.
                    # This allows us to do accurate missing frame detection.
                    # Also allows us to use better filling method like interpolation.
                    frame_number = frame.get_frame_number()
                    if not self.forward_fill and frame_number == previous_frame_numbers[stream_name]:
                        frames[stream_name] = None
                    previous_frame_numbers[stream_name] = frame_number

                depth_frame, color_frame = frames["depth"], frames["color"]

                # If either depth or color frame is missing,
                # we still yield the available frame while setting the missing one to None
                # But no pair is missing both simultaneously.
                depth = None
                if depth_frame:
                    depth = DepthFrame(**self.read_common_frame_metadata(depth_frame), depth_scale=depth_frame.get_units())

                color = None
                if color_frame:
                    color = ColorFrame(**self.read_common_frame_metadata(color_frame))

                yield Frame(depth=depth, color=color)
        finally:
            pipeline.stop()

    def iter_frames(self) -> Iterator[Frame]:
        pipeline, _ = self.start()
        yield from self.read_frames(pipeline)

    def load(self) -> Recording:
        pipeline, profile = self.start()
        device = profile.get_device()
        playback = device.as_playback()
        # pause so that the notifications play after the callbacks are registered
        # a notification in the time between pipeline.start() and pause() is lost
        playback.pause()
        notifications = []
        for sensor in device.query_sensors():
            sensor_name = sensor.get_info(rs.camera_info.name)
            sensor.set_notifications_callback(self.notification_callback(sensor_name, notifications))
        serial_number = device.get_info(rs.camera_info.serial_number) if device.supports(rs.camera_info.serial_number) else None
        depth_to_color = self.read_depth_to_color(profile)
        playback.resume()
        frames = list(self.read_frames(pipeline))
        # the SDK loads the options in the background after pipeline.start(), so they are read after the frames
        sensor_options = self.read_sensor_options(device)
        return Recording(
            serial_number=serial_number,
            depth_to_color=depth_to_color,
            sensor_options=sensor_options,
            notifications=notifications,
            frames=frames,
        )

    def load_numpy(self, stream_name: str) -> np.ndarray:
        frames = [getattr(frameset, stream_name).frame for frameset in self.iter_frames() if getattr(frameset, stream_name) is not None]
        if not frames:
            raise ValueError(f"{self.path.name}: no {stream_name} frames")
        return np.stack(frames)
