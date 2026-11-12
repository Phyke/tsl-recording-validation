import io

import numpy as np
from PIL import Image

from ..pydantic_models import CompressedImageMessage
from ..ros2 import CdrReader, Db3


def parse_compressed_image(blob: bytes) -> CompressedImageMessage:
    reader = CdrReader(blob)
    stamp_s = reader.unpack("i") + reader.unpack("I") * 1e-9
    reader.string()  # frame_id
    image_format = reader.string()
    return CompressedImageMessage(stamp_s=stamp_s, format=image_format, data=reader.sequence())


def decode_jpeg(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


# a .db3 folder from poc1 UvcRos2Bag: JPEG frames of the mirrorless camera through the Elgato
class MirrorlessRos2Loader(Db3):
    TOPIC = "/camera/color/image_raw/compressed"

    def load(self) -> list[CompressedImageMessage]:
        return [parse_compressed_image(blob) for blob in self.messages(self.TOPIC)]
