import sqlite3
import struct
from compression import zstd
from pathlib import Path

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def decompress(blob: bytes) -> bytes:
    return zstd.decompress(blob) if blob[:4] == ZSTD_MAGIC else blob


class CdrReader:
    def __init__(self, blob: bytes) -> None:
        self.data = decompress(blob)
        # skip the 4-byte encapsulation header, alignment counts from its end
        self.position = 4

    def align(self, size: int) -> None:
        self.position = 4 + (self.position - 4 + size - 1) // size * size

    def unpack(self, fmt: str) -> int:
        size = struct.calcsize(fmt)
        self.align(size)
        (value,) = struct.unpack_from(f"<{fmt}", self.data, self.position)
        self.position += size
        return value

    def sequence(self) -> bytes:
        length = self.unpack("I")
        value = self.data[self.position : self.position + length]
        self.position += length
        return value

    def string(self) -> str:
        return self.sequence().rstrip(b"\x00").decode("utf-8", "replace")


def cdr_string(blob: bytes) -> str:
    return CdrReader(blob).string()


def key_values(text: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in text.split(";") if "=" in item)


class Db3:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)

    def topic_names(self) -> list[str]:
        return [name for (name,) in self.connection.execute("SELECT name FROM topics ORDER BY name")]

    def messages(self, topic: str) -> list[bytes]:
        rows = self.connection.execute(
            "SELECT m.data FROM messages m JOIN topics t ON t.id = m.topic_id WHERE t.name = ? ORDER BY m.timestamp",
            (topic,),
        )
        return [data for (data,) in rows]

    def message(self, topic: str, index: int) -> bytes:
        row = self.connection.execute(
            "SELECT m.data FROM messages m JOIN topics t ON t.id = m.topic_id WHERE t.name = ? ORDER BY m.timestamp LIMIT 1 OFFSET ?",
            (topic, index),
        ).fetchone()
        if row is None:
            raise IndexError(f"{self.path.name}: no message {index} in {topic}")
        return row[0]
