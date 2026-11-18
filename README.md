# TSL recording validation

This repository validates the recordings of a TSL take. A take has recordings from 2 cameras:

- the `r` recordings from the RealSense D435: color and depth as a `.bag` or `.db3` file, and color as an `.mp4` file, 640x480 at
  30 fps
- the `m` recordings from the mirrorless camera: color as a `.bag`, `.db3`, or `.mp4` file, 1920x1080 at 120 fps

The marker `r` or `m` in the file name gives the camera. A marker is at the start of the name or after a `-`, and it has no letter
or digit after it. Examples are `1-r.bag`, `1-m.bag`, `pipeline-r_20261012_151250.db3`, `r_20261113_134220.db3`,
`r_20261113_134220.mp4`, `m_20261113_134220.mp4`, and `-m_20261113_143019.mp4`. An `.mp4` file with no marker is a mirrorless
recording.

The recordings of one take are in one folder, and they have the same name after the marker. A mirrorless `.db3` file can be in a
subfolder with its own name, for example `m_20261113_134220_color/m_20261113_134220_color.db3`.

The partner of a recording is the recording of the other camera in the same take. When the other camera has 2 files, the partner
is the file of the same type: an `.mp4` file for an `.mp4` file, and a `.bag` or `.db3` file for a `.bag` or `.db3` file. The
cross-correlation always uses the `.bag` or `.db3` file of the D435.

## Commands

Run each command in the repository folder.

Validate one `.bag` or `.db3` file. The exit code is 1 when the file has issues.

```bash
uv run src/validate.py <path to the recording>
```

Open the app. The folder is optional. With no folder, the app opens the folder that was open last (`last_folder` in `settings.json`).
The app keeps the last validation result of each file in `validation_history.json`.

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
| `duplicate_image` | a frame with the same image as the previous frame; more than 2 copies are critical | `-r`, `-m` `.bag`, `-m` `.db3` |
| `image_fps` | copies that make the different images per second, (different images - 1) / duration, more than 1% below 120 | `-m` `.bag`, `-m` `.db3` |
| `report_no_frame_drops` | `no_frame_drops` true in the poc1 `meta/<take>.report.json`, but the file has copies | poc1 `.db3` |
| `dropped_capture_frames` | `dropped_capture_frames` above 0 in the poc1 `meta/<take>.report.json` | poc1 `.db3`, `.mp4` |
| `color_depth_offset` | a color frame and a depth frame of one frameset more than half a frame period apart | `-r` |
| `missing_partner` | no recording of the other camera for the take | all |
| `pair_duration` | durations of the 2 recordings of the take that differ by more than 1000 ms | all |

## Depth measures

The validation of a `-r` recording prints 2 measures of the depth stream. They have no limit, so they give no issue.

- `fill rate`: the part of the depth pixels that has a depth value. The line shows the mean of all the frames and the lowest frame.
- `noise`: how far the depth values of a flat surface are from a plane. The validation finds the largest flat surface of the
  scene, fits a plane to its disparity in each frame, and prints the median RMS distance in pixels of disparity and in millimeters.
  The code is in `src/recording_reader/d435/depth_quality.py`.

## Image runs and PC load

These lines have no limit, so they give no issue.

- `image fps` (`-m` `.bag`, `-m` `.db3`): the different images per second, the number of different images, the longest run of equal
  images, and the number of runs of each length. Runs of 3 to 5 frames are the pattern of a capture that gives 120 frames per second
  with about 30 new images.
- `sysmon` (poc1 files): the CPU median and the total RAM of the PC during the take, from `meta/<take>_sysmon.csv`. The first row is
  not in the CPU median.
