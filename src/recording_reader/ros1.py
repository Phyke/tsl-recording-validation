import struct
from pathlib import Path
from typing import BinaryIO

MAGIC = b"#ROSBAG V2.0\n"
OP_BAG_HEADER = 3
OP_CHUNK_INFO = 6
OP_CONNECTION = 7


def read_record(file: BinaryIO) -> tuple[dict[str, bytes], bytes]:
    (header_length,) = struct.unpack("<I", file.read(4))
    header = file.read(header_length)
    fields, position = {}, 0
    while position < header_length:
        (field_length,) = struct.unpack_from("<I", header, position)
        name, _, value = header[position + 4 : position + 4 + field_length].partition(b"=")
        fields[name.decode()] = value
        position += 4 + field_length
    (data_length,) = struct.unpack("<I", file.read(4))
    return fields, file.read(data_length)


def record_time_ms(value: bytes) -> float:
    seconds, nanoseconds = struct.unpack("<II", value)
    return seconds * 1000 + nanoseconds / 1e6


# the index at the end of a ROS1 .bag file: the message count of each topic, and the time from the first message to the last
# message, with no read of the messages
class Ros1BagIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with self.path.open("rb") as file:
            if file.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"{self.path.name}: not a ROS1 .bag file")
            fields, _ = read_record(file)
            if fields["op"][0] != OP_BAG_HEADER:
                raise ValueError(f"{self.path.name}: no bag header record")
            (index_position,) = struct.unpack("<Q", fields["index_pos"])
            (connection_count,) = struct.unpack("<I", fields["conn_count"])
            (chunk_count,) = struct.unpack("<I", fields["chunk_count"])
            # a recording that did not close correctly has no index
            if index_position == 0:
                raise ValueError(f"{self.path.name}: the .bag file has no index")
            file.seek(index_position)
            topics = {}
            for _ in range(connection_count):
                fields, _ = read_record(file)
                if fields["op"][0] != OP_CONNECTION:
                    raise ValueError(f"{self.path.name}: no connection record in the index")
                topics[struct.unpack("<I", fields["conn"])[0]] = fields["topic"].decode()
            self.message_counts = dict.fromkeys(topics.values(), 0)
            start_times, end_times = [], []
            for _ in range(chunk_count):
                fields, data = read_record(file)
                if fields["op"][0] != OP_CHUNK_INFO:
                    raise ValueError(f"{self.path.name}: no chunk info record in the index")
                start_times.append(record_time_ms(fields["start_time"]))
                end_times.append(record_time_ms(fields["end_time"]))
                for connection, count in struct.iter_unpack("<II", data):
                    self.message_counts[topics[connection]] += count
        self.duration_ms = max(end_times) - min(start_times) if start_times else 0.0
