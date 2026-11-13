from collections.abc import Iterator
from datetime import timedelta
from typing import Any, Self

import numpy as np
import pyrealsense2 as rs
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field


class RealSenseDeviceInfo(BaseModel):
    name: str | None = None
    serial_number: str | None = None
    firmware_version: str | None = None
    recommended_firmware_version: str | None = None
    physical_port: str | None = None
    debug_op_code: str | None = None
    advanced_mode: str | None = None
    product_id: str | None = None
    camera_locked: str | None = None
    usb_type_descriptor: str | None = None
    product_line: str | None = None
    asic_serial_number: str | None = None
    firmware_update_id: str | None = None
    ip_address: str | None = None
    dfu_device_path: str | None = None
    connection_type: str | None = None
    smcu_fw_version: str | None = None
    imu_type: str | None = None
    mipi_driver_version: str | None = None

    @classmethod
    def from_realsense_device_object(cls, device: rs.device) -> Self:
        values = {key: device.get_info(info) for key, info in rs.camera_info.__members__.items() if device.supports(info)}
        return cls.model_validate(values)

    def pretty_print(self) -> None:
        fields = self.model_dump(exclude_none=True)
        width = max((len(key) for key in fields), default=0)
        for key, value in fields.items():
            print(f"{key:<{width}} : {value}")


class RealSensePlaybackInfo(BaseModel):
    file_name: str
    duration: timedelta
    position_ns: int
    real_time: bool
    status: str

    @classmethod
    def from_playback(cls, playback: rs.playback) -> Self:
        return cls(
            file_name=playback.file_name(),
            duration=playback.get_duration(),
            position_ns=playback.get_position(),
            real_time=playback.is_real_time(),
            status=playback.current_status().name,
        )

    def pretty_print(self) -> None:
        fields = self.model_dump()
        width = max((len(key) for key in fields), default=0)
        for key, value in fields.items():
            print(f"{key:<{width}} : {value}")


class Intrinsics(BaseModel):
    width: int = Field(description="Image width")
    height: int = Field(description="Image height")
    ppx: float = Field(description="Principal point x-coordinate")
    ppy: float = Field(description="Principal point y-coordinate")
    fx: float = Field(description="Focal length x")
    fy: float = Field(description="Focal length y")
    model: str = Field(description="Camera model")
    coeffs: list[float] = Field(description="Distortion coefficients")

    def pretty_print(self) -> None:
        camera_matrix = np.array([[self.fx, 0, self.ppx], [0, self.fy, self.ppy], [0, 0, 1]])
        print(f"K ({self.width}x{self.height}, {self.model}):")
        print(np.array2string(camera_matrix, precision=4, floatmode="fixed", suppress_small=True))
        print(f"coeffs: {np.array2string(np.array(self.coeffs), precision=6, floatmode='fixed', suppress_small=True)}")


class Extrinsics(BaseModel):
    rotation: list[float] = Field(description="3x3 rotation matrix in column-major order")
    translation: list[float] = Field(description="Translation in meters")

    def pretty_print(self) -> None:
        transform = np.eye(4)
        # order="F" fills the matrix one column at a time, because rotation is column-major
        transform[:3, :3] = np.reshape(self.rotation, (3, 3), order="F")
        transform[:3, 3] = self.translation
        print("[R | t], t in meters:")
        print(np.array2string(transform, precision=6, floatmode="fixed", suppress_small=True))


class StreamFrame(BaseModel):
    timestamp_ms: float
    timestamp_domain: str
    frame_number: int
    format: str
    fps: int
    intrinsics: Intrinsics
    metadata: dict[str, int]


class DepthFrame(StreamFrame):
    frame: NDArray[Shape["*, *"], np.uint16]
    depth_scale: float


class ColorFrame(StreamFrame):
    frame: NDArray[Shape["*, *, 3"], np.uint8]


class Frame(BaseModel):
    depth: DepthFrame | None = None
    color: ColorFrame | None = None


class Notification(BaseModel):
    sensor: str
    timestamp_ms: float
    category: str
    severity: str
    description: str


class Recording(BaseModel):
    serial_number: str | None = None
    depth_to_color: Extrinsics | None = None
    sensor_options: dict[str, dict[str, float]] = Field(default_factory=dict, description="Option values by sensor name, then option name")
    notifications: list[Notification] = Field(default_factory=list)
    file_frame_counts: dict[str, int] = Field(
        default_factory=dict, description="Image messages in the file by stream name (depth, color), from the index of the file",
    )
    frames: list[Frame]

    def to_numpy(self, stream_name: str) -> np.ndarray:
        frames = [getattr(frameset, stream_name).frame for frameset in self.frames if getattr(frameset, stream_name) is not None]
        if not frames:
            raise ValueError(f"no {stream_name} frames in this recording")
        return np.stack(frames)


class CompressedImageMessage(BaseModel):
    stamp_s: float
    format: str
    data: bytes

    # print the length of data, not every byte
    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        for name, value in super().__repr_args__():
            yield name, f"<{len(value)} bytes>" if name == "data" else value


class ValidationIssue(BaseModel):
    check: str = Field(description="Name of the check that found the problem")
    stream: str | None = Field(default=None, description="depth or color, None for a problem of the whole recording")
    frames: list[int] = Field(default_factory=list, description="Frameset indices of the problem, empty for a problem of the whole stream")
    message: str
