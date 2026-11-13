# TSL recording validation

This repository validates the recordings of a TSL take. A take has 2 recordings:

- the `-r` recording from the RealSense D435 (color and depth, 640x480 at 30 fps), as a `.bag` or `.db3` file
- the `-m` recording from the mirrorless camera (color, 1920x1080 at 120 fps), as a `.bag`, `.db3`, or `.mp4` file

The marker `-r` or `-m` in the file name gives the camera, for example `1-r.bag`, `1-m.bag`, `pipeline-r_20261012_151250.db3`, and
`pipeline-m_20261012_151250_color.db3`. A marker has no letter or digit after it.

## Commands

Run each command in the repository folder.

Validate one `.bag` or `.db3` file. The exit code is 1 when the file has issues.

```bash
uv run src/validate.py <path to the recording>
```

Open the app. The folder is optional.

```bash
uv run src/app.py <path to a folder with recordings>
```

Run the tests.

```bash
uv run pytest
```

## Checks

The severity rule is in `src/recordings.py`.

| Check | The check finds | Files |
| --- | --- | --- |
| `serial_number` | a D435 serial number that is not in `KNOWN_SERIAL_NUMBERS` | `-r` |
| `notification` | a hardware notification in the file, with the action to take | `-r`, `-m` `.bag` |
| `depth_to_color` | no depth-to-color extrinsics | `-r` |
| `missing_stream` | a stream with no frame, or a recording with no frame | `-r`, `-m` `.bag` |
| `missing_frame` | a frameset with no frame of a stream | `-r`, `-m` `.bag` |
| `unexpected_stream` | a depth frame in a mirrorless recording | `-m` `.bag` |
| `frames_not_loaded` | more than 1 frame of the file that the reader did not put in a frameset | `-r`, `-m` `.bag` |
| `fps`, `size`, `format` | a setting that is not the expected setting | all |
| `timestamp_domain` | a time domain that is not `global_time` (D435) or `system_time` (mirrorless `.bag`) | `-r`, `-m` `.bag` |
| `constant_value`, `metadata_keys` | a value or a metadata key that changes in the file | `-r`, `-m` `.bag` |
| `frame_number_gaps`, `duplicate_frame_number`, `decreasing_frame_number` | frame numbers that do not go up by 1 | `-r`, `-m` `.bag` |
| `missing_barcode`, `barcode_gaps`, `duplicate_barcode`, `decreasing_barcode` | barcode numbers that do not go up by 1 | `-m` `.db3` |
| `timestamp_order` | a time that is not later than the previous time | all |
| `timestamp_gaps` | a time gap longer than 1.5 frame periods | `.mp4` |
| `measured_fps` | a camera fps (frame numbers or barcodes / duration) more than 1% from the expected fps | `-r`, `-m` `.bag`, `-m` `.db3` |
| `duplicate_image` | a frame with the same image as the previous frame | `-r`, `-m` `.bag`, `-m` `.db3` |
| `color_depth_offset` | a color frame and a depth frame of one frameset more than half a frame period apart | `-r` |
| `missing_partner` | no recording of the other camera for the take | all |
| `pair_duration` | durations of the 2 recordings of the take that differ by more than 1000 ms | all |
