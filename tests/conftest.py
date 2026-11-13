import io
import sqlite3
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from recording_reader import (
    BARCODE_BIT_WIDTH,
    BARCODE_BITS,
    BARCODE_HEIGHT,
    ColorFrame,
    CompressedImageMessage,
    DepthFrame,
    Extrinsics,
    Frame,
    Intrinsics,
    Recording,
)

D435_SERIAL_NUMBER = "317622070655"


def intrinsics(width: int, height: int) -> Intrinsics:
    return Intrinsics(width=width, height=height, ppx=width / 2, ppy=height / 2, fx=600, fy=600, model="none", coeffs=[0.0] * 5)


def stream_values(number: int, timestamp_ms: float, fps: int, width: int, height: int, domain: str) -> dict:
    return {
        "timestamp_ms": timestamp_ms,
        "timestamp_domain": domain,
        "frame_number": number,
        "fps": fps,
        "intrinsics": intrinsics(width, height),
        "metadata": {"frame_counter": number},
    }


# a D435 recording with no fault: each frame has different pixels, the frame numbers have no gap, and the fps is 30
def d435_recording(count: int = 6, domain: str = "global_time", period_ms: float = 1000 / 30) -> Recording:
    random = np.random.default_rng(0)
    frames = []
    for i in range(count):
        values = stream_values(100 + i, 1_000_000 + i * period_ms, 30, 640, 480, domain)
        depth = DepthFrame(**values, format="z16", frame=random.integers(1, 5000, (480, 640), dtype=np.uint16), depth_scale=0.001)
        color = ColorFrame(**values, format="rgb8", frame=random.integers(0, 255, (480, 640, 3), dtype=np.uint8))
        frames.append(Frame(depth=depth, color=color))
    return Recording(
        serial_number=D435_SERIAL_NUMBER,
        depth_to_color=Extrinsics(rotation=[1, 0, 0, 0, 1, 0, 0, 0, 1], translation=[0.015, 0, 0]),
        frames=frames,
    )


# a mirrorless .bag recording with no fault: color only, 1920x1080 at 120 fps
def mirrorless_recording(count: int = 4) -> Recording:
    random = np.random.default_rng(0)
    frames = []
    for i in range(count):
        values = stream_values(100 + i, 1_000_000 + i * 1000 / 120, 120, 1920, 1080, "system_time")
        frames.append(Frame(color=ColorFrame(**values, format="rgb8", frame=random.integers(0, 255, (1080, 1920, 3), dtype=np.uint8))))
    return Recording(frames=frames)


# a 1920x1080 camera image with the frame number as a barcode along the top edge, as poc1 embed_seq_barcode() makes it
def barcode_image(number: int, shade: int) -> np.ndarray:
    image = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    image[:, :, 0] = np.linspace(40, 200, 1920, dtype=np.uint8)
    image[200:400, 200:400] = shade
    for bit in range(BARCODE_BITS):
        value = 255 if number >> (BARCODE_BITS - 1 - bit) & 1 else 0
        image[:BARCODE_HEIGHT, bit * BARCODE_BIT_WIDTH : (bit + 1) * BARCODE_BIT_WIDTH] = value
    return image


def jpeg_bytes(image: np.ndarray) -> bytes:
    data = io.BytesIO()
    Image.fromarray(image).save(data, format="JPEG", quality=95)
    return data.getvalue()


# the frames of a mirrorless .db3 recording with no fault; shades gives the camera image of each frame
def mirrorless_messages(shades: list[int], first_number: int = 5) -> list[CompressedImageMessage]:
    return [
        CompressedImageMessage(stamp_s=1000 + i / 120, format="jpeg", data=jpeg_bytes(barcode_image(first_number + i, shade)))
        for i, shade in enumerate(shades)
    ]


def cdr_sequence(data: bytes, position: int) -> bytes:
    padding = b"\x00" * (-position % 4)
    return padding + struct.pack("<I", len(data)) + data


# a sensor_msgs/msg/CompressedImage message in CDR, as CdrReader reads it
def compressed_image_cdr(message: CompressedImageMessage) -> bytes:
    seconds = int(message.stamp_s)
    body = struct.pack("<iI", seconds, round((message.stamp_s - seconds) * 1e9))
    for value in [b"camera\x00", message.format.encode() + b"\x00", message.data]:
        body += cdr_sequence(value, len(body))
    return b"\x00\x01\x00\x00" + body


# a rosbag2 .db3 file with the given messages: (topic, timestamp in nanoseconds, data)
def write_db3(path: Path, messages: list[tuple[str, int, bytes]]) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE topics (id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT)")
    connection.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB)")
    topic_ids = {}
    for topic, timestamp, data in messages:
        if topic not in topic_ids:
            topic_ids[topic] = len(topic_ids) + 1
            connection.execute("INSERT INTO topics VALUES (?, ?, 'unknown', 'cdr')", (topic_ids[topic], topic))
        connection.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)", (topic_ids[topic], timestamp, data))
    connection.commit()
    connection.close()


def ros1_record(fields: dict[str, bytes], data: bytes) -> bytes:
    header = b"".join(struct.pack("<I", len(name) + 1 + len(value)) + name.encode() + b"=" + value for name, value in fields.items())
    return struct.pack("<I", len(header)) + header + struct.pack("<I", len(data)) + data


# a ROS1 .bag file with only the bag header and the index: one chunk info record for each chunk
# chunks: (start time in ms, end time in ms, message count of each topic)
def write_ros1_bag(path: Path, topics: list[str], chunks: list[tuple[int, int, dict[str, int]]]) -> None:
    def time_bytes(time_ms: int) -> bytes:
        return struct.pack("<II", time_ms // 1000, time_ms % 1000 * 1_000_000)

    index = b""
    for connection, topic in enumerate(topics):
        index += ros1_record({"op": bytes([7]), "conn": struct.pack("<I", connection), "topic": topic.encode()}, b"")
    for start_ms, end_ms, counts in chunks:
        data = b"".join(struct.pack("<II", topics.index(topic), count) for topic, count in counts.items())
        fields = {
            "op": bytes([6]),
            "ver": struct.pack("<I", 1),
            "chunk_pos": struct.pack("<Q", 0),
            "start_time": time_bytes(start_ms),
            "end_time": time_bytes(end_ms),
            "count": struct.pack("<I", len(counts)),
        }
        index += ros1_record(fields, data)
    magic = b"#ROSBAG V2.0\n"
    header_fields = {"op": bytes([3]), "index_pos": b"\x00" * 8, "conn_count": struct.pack("<I", len(topics)), "chunk_count": struct.pack("<I", len(chunks))}
    index_position = len(magic) + len(ros1_record(header_fields, b" " * 64))
    header_fields["index_pos"] = struct.pack("<Q", index_position)
    path.write_bytes(magic + ros1_record(header_fields, b" " * 64) + index)
