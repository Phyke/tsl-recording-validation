import argparse
import sys
from pathlib import Path

from recording_reader import (
    D435Validator,
    MirrorlessRos2Loader,
    MirrorlessRos2Validator,
    MirrorlessRosValidator,
    RealSenseLoader,
)
from recordings import camera_of, pair_issues

parser = argparse.ArgumentParser(description="Validate one recording file")
parser.add_argument("path", type=Path, help="a -r (RealSense D435) or -m (mirrorless) recording, .bag or .db3")
path = parser.parse_args().path
if not path.is_file():
    raise FileNotFoundError(f"{path}: no such file")
if path.suffix not in {".bag", ".db3"}:
    raise ValueError(f"{path}: the extension must be .bag or .db3")
# the camera comes from the marker in the file name, for example 1-r.bag, 1-m.bag, pipeline-r_20261012_151250.db3, and
# pipeline-m_20261012_151250_color.db3
camera = camera_of(path)

if camera == "d435":
    issues = D435Validator(RealSenseLoader(path, forward_fill=False).load()).validate(raise_error=False)
elif path.suffix == ".bag":
    issues = MirrorlessRosValidator(RealSenseLoader(path, forward_fill=False).load()).validate(raise_error=False)
else:
    issues = MirrorlessRos2Validator(MirrorlessRos2Loader(path).load()).validate(raise_error=False)
issues += pair_issues(path)
print(f"{path}: {len(issues)} {'issue' if len(issues) == 1 else 'issues'}")
for issue in issues:
    print(f"    {issue.message}")
# the exit code is 1 when the recording has issues, so that a script can test the result
sys.exit(1 if issues else 0)
