"""Profile selection and explicitly supported torque release for the GUI modes."""
import json
import subprocess
from .client import ROOT, IO_PYTHON
from .mapping import Mapping


def control_profile(requested, allow_visual=False):
    path=requested
    if path==ROOT/'config/follower.json' and not path.exists():
        path=ROOT/'config/follower_visual_checked.json';allow_visual=True
    mapping=Mapping(json.loads(path.read_text()),allow_visual_confirmation=allow_visual)
    return path,allow_visual,mapping


def release_supported(port, supported, writes_allowed=True):
    if not writes_allowed:raise ValueError('Motor writes are disabled in this test')
    if not supported:raise ValueError('Support the arm and check Arm supported before releasing torque')
    result=subprocess.run([str(IO_PYTHON),str(ROOT/'scripts/hardware.py'),'release','--port',port],
        input='SUPPORTED\n',text=True,capture_output=True,timeout=5)
    if result.returncode:raise RuntimeError(result.stderr or result.stdout or 'Torque release failed')
    return result.stdout
