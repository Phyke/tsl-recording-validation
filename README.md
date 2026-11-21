# TSL recording validation

This app validates the recordings of TSL takes and shows the issues of each recording.

## Usage

1. Open the app. Run this command in the repository folder:

   ```bash
   uv run src/app.py
   ```

2. Click **Open folder** and select the folder with the recordings. At the next start, the app opens this folder again.
3. Click a take in the list to open it, and click a file of the take. The player loads the video.
4. Click **Validate selected**. The Result box and the Issues tab show the result.
5. Select an issue in the Issues tab. The player goes to the first frame of the issue.
6. To compare the D435 and the mirrorless recordings of the take, click **Cross-correlation**.

## Recordings

A take has recordings from 2 cameras:

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

## The app window

The app window has 3 panels: the recording list on the left, the player in the middle, and the result on the right.

### Recording list

- The list has one row for each take, with the files of the take under it. Click a take to open or close it. Click a file to select
  it.
- A file row shows the camera and the file type, for example `D435 .db3`, and the result of its last validation, for example
  `[1 critical, 2 medium]`. A take row shows the worst severity of its files. The color of a row is the color of its severity.
- The app looks for new and removed files in the open folder every 3 seconds.
- **Validate selected** validates the selected file.
- **Cross-correlation** compares the motion in the color and the depth stream of the D435 `.bag` or `.db3` file of the take, and in
  its mirrorless file. The Cross-correlation tab shows the result.
- While a task runs, a busy bar and a **Cancel** button show. Cancel stops the task after the file that is open now.

### Player

- The player shows the color stream and, for a D435 `.bag` or `.db3` file, the depth stream. A missing frame shows the previous
  image with a dark cover and a text.
- The line under the views gives the frame index (from 0) and the time. For a `.bag` file or a D435 `.db3` file, it also gives the
  frame number of each stream.
- The buttons are ◀ (previous frame), Play, ▶ (next frame), and the speed list. The keys are Left, Right, and Space.
- The strip under the slider has a mark at each issue frame of the shown issues, in the color of the issue severity. The marks of
  the selected issue are tall. Click a mark to go to its frame.
- When **Load the video when a recording is selected** is off in the Settings dialog, click the player to load the video.

### Result

- The Result box shows the number of issues of each severity, for example `1 critical, 2 medium`, and the time of the validation.
- A result of an older version of the app has no measures. For such a result, the Result box tells you to validate the file again.
- The **Issues** tab lists the issues under their known group, the worst group first. Issues that are in no known group are under
  `No group`. Select an issue to see its full message in the text box and to go to its first frame. The **issue frame** buttons go
  to the previous and the next frame of the issue.
- The **Measures** tab lists the measures under their stream (`color`, `depth`, or `PC`). A measure of an issue has the color of the
  issue severity and the expected value, for example `30.27 (expected 120 within 1%)`. Put the pointer on a measure to see how it is
  calculated and its full value. A file with the `run lengths` measure also shows a bar chart of the run lengths.
- The **Cross-correlation** tab shows a small plot of the last cross-correlation of the take. Click the plot to open a large plot.

### Settings

The **Settings** button opens the Settings dialog. **OK** saves the settings, and **Cancel** discards the changes.

- **Font size**: the size of the text in the app. The change shows at once.
- **Load the video when a recording is selected**: when this is off, the app does not load a video until you click the player.
- **Draw frames with**: `CPU` or `GPU`. In the GPU mode, the graphics card scales the frames with OpenGL. Use CPU on a PC with no
  OpenGL driver. The change takes effect at the next start of the app.
- **Checks**: one row for each known group and each check, under the headings `Known groups`, `Metadata`, `Measured`, and
  `Other files`. Each row has a plain text, the severity list, and the **Show** box.
  - After OK, a changed severity applies to the stored results, with no new validation.
  - The **Default** button of a row shows when its severity is not the default. It sets the default severity.
  - When **Show** is off, the Issues tab does not show the issues of that group or check. They still count in the result.
- **Reset to defaults** sets each setting to its default.

### Files of the app

The app writes 2 files in the repository folder. Git does not track them.

- `settings.json`: the settings and the folder that was open last
- `validation_history.json`: the last validation result of each file

## Severity

Each issue has one of 3 severities: critical, medium, or low. A file has the severity of its worst issue. A file with no issue is
OK. A file that cannot be read is critical. The rules are in `src/recordings.py`.

The default severities are:

- critical: `missing_stream`, `fps`, `size`, `format`, `serial_number`, `unexpected_stream`, and the Half image rate group
- `by lost frames`: `missing_frame`, `frame_number_gaps`, `barcode_gaps`, and `timestamp_gaps`
  - The severity is medium for 1 or 2 lost frames in the file, and critical for more.
  - A `frame_number_gaps` gap at a `missing_frame` frameset has the severity of the `missing_frame` issue.
- `by lost frames` for `duplicate_image`: medium for copies in up to 5% of the frames, and critical for more
- low: the Hardware Instability group
- medium: each other check

A known group is a set of issues with a known cause. The severity of a group wins over the severity of its checks.

- **Hardware Instability**: the D435 notifications `Left MIPI error`, `Right MIPI error`, and `Depth stream start failure` at a
  stream start. The depth data after each one was at the level of a take with no notification.
- **Half image rate**: the `duplicate_image`, `image_fps`, and `report_no_frame_drops` issues of a file whose color copies come in
  pairs. In such a file, at least 90% of the steps between 2 copies are 2 frames, so about half of the frames are different images.

## Checks

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
| `image_fps` | copies that make the different images per second, (different images - 1) / duration, more than 1% below the expected fps | `-m` `.bag`, `-m` `.db3` |
| `report` | a poc1 `meta/<take>.report.json` that cannot be read | poc1 files |
| `report_no_frame_drops` | `no_frame_drops` true in the poc1 `meta/<take>.report.json`, but the file has copies | poc1 `.db3` |
| `dropped_capture_frames` | `dropped_capture_frames` above 0 in the poc1 `meta/<take>.report.json` | poc1 `.db3`, `.mp4` |
| `color_depth_offset` | a color frame and a depth frame of one frameset more than half a frame period apart | `-r` |
| `missing_partner` | no recording of the other camera for the take | all |
| `pair_duration` | durations of the 2 recordings of the take that differ by more than 1000 ms | all |

A poc1 mirrorless `.db3` file is checked at the `requested_fps` of its poc1 report, or at 120 fps when it has no report.

## Measures

The measures have no limit, so they give no issue. The Measures tab of the app shows them.

- Frame rate (each stream): `file fps`, `frames`, and `duration` for all files. A `-r` or `-m` `.bag` file also has `camera fps`
  and `kept`. A `-m` `.db3` file also has `barcode fps` and `kept`.
- Image runs (`-m` `.bag`, `-m` `.db3`): `image fps`, `different images`, `longest run`, and `run lengths`. Runs of 3 to 5 frames
  are the pattern of a capture that gives 120 frames per second with about 30 new images.
- Depth (`-r` `.bag`, `-r` `.db3`):
  - `fill rate` and `lowest fill rate`: the part of the depth pixels that has a depth value, as the mean of all the frames and for
    the frame with the lowest value.
  - `noise` and `flat surface`: how far the depth values of a flat surface are from a plane. The validation finds the largest flat
    surface of the scene, fits a plane to its disparity in each frame, and gives the median RMS distance in pixels of disparity and
    in millimeters. The code is in `src/recording_reader/d435/depth_quality.py`.
- PC (poc1 files): `CPU median` and `total RAM` of the PC during the take, from `meta/<take>_sysmon.csv`. The first row is not in
  the CPU median.

## Other commands

These commands are for development. Run each command in the repository folder.

Run the tests.

```bash
uv run pytest
```

Validate one `.bag` or `.db3` file with no app. The exit code is 1 when the file has issues. This command does not run the
`report`, `report_no_frame_drops`, and `dropped_capture_frames` checks, and it does not print the PC measures.

```bash
uv run src/validate.py <path to the recording>
```

Open the app with a given folder in place of the folder that was open last.

```bash
uv run src/app.py <path to a folder with recordings>
```
