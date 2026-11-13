import numpy as np

# poc1 embed_seq_barcode(): a 32-bit frame number along the top edge, each bit a 10x20 block, white for 1
BARCODE_BITS = 32


BARCODE_BIT_WIDTH = 10


BARCODE_HEIGHT = 20


# JPEG puts the pixels in blocks of up to 16x16, so the barcode changes the pixels of the rows above this one; the rows from this
# one down have only the camera image
BARCODE_JPEG_ROWS = 32


def read_barcode(image: np.ndarray) -> int | None:
    row = image[BARCODE_HEIGHT // 2].astype(float)
    if row.ndim == 2:
        row = row.mean(axis=1)
    samples = row[BARCODE_BIT_WIDTH // 2 :: BARCODE_BIT_WIDTH][:BARCODE_BITS]
    # a sample between 60 and 195 is neither a black block nor a white block
    if len(samples) < BARCODE_BITS or np.any((samples > 60) & (samples < 195)):
        return None
    # 32 white blocks are not a frame number that a recording can get to: the frame is white there
    if samples.min() > 127:
        return None
    # 32 black blocks are the frame number 0 only when the row has the camera image after the barcode; a black frame is black
    # there also
    rest = row[BARCODE_BITS * BARCODE_BIT_WIDTH :]
    if samples.max() <= 127 and rest.size and rest.max() <= 60:
        return None
    return int("".join("1" if value > 127 else "0" for value in samples), 2)
