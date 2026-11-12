import numpy as np

# poc1 embed_seq_barcode(): a 32-bit frame number along the top edge, each bit a 10x20 block, white for 1
BARCODE_BITS = 32


BARCODE_BIT_WIDTH = 10


BARCODE_HEIGHT = 20


def read_barcode(image: np.ndarray) -> int | None:
    samples = image[BARCODE_HEIGHT // 2, BARCODE_BIT_WIDTH // 2 :: BARCODE_BIT_WIDTH][:BARCODE_BITS].astype(float)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    # a sample between 60 and 195 is neither a black block nor a white block
    if len(samples) < BARCODE_BITS or np.any((samples > 60) & (samples < 195)):
        return None
    return int("".join("1" if value > 127 else "0" for value in samples), 2)
