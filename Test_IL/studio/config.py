import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEACH = ROOT.parent / 'Test_teach'
SOURCE = ROOT.parent / 'Sim_real'
ISAAC = Path('/path/to/isaac/env_isaaclab/bin/python')
IL_PYTHON = Path('/path/to/venv/bin/python')
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
CAMERAS = ('gripper', 'table')


def load():
    value=json.loads((ROOT / 'config/settings.json').read_text())
    from .robot import profile
    cfg=profile();value.update(follower=cfg['follower_port'],leader=cfg.get('leader_port',''))
    return value


def camera_devices():
    # Identical UVC modules often ship with the same serial number. USB topology
    # links distinguish them even when one by-id alias shadows the other camera.
    seen = set()
    depth_nodes={p.resolve() for p in Path('/dev/v4l/by-id').glob('*') if 'realsense' in p.name.lower()}
    result = []
    for folder in ('/dev/v4l/by-id','/dev/v4l/by-path'):
        for path in sorted(Path(folder).glob('*-video-index0')):
            if 'realsense' in path.name.lower():continue
            real = path.resolve()
            if real not in seen and real not in depth_nodes:
                seen.add(real)
                result.append(str(path))
    from .cameras import realsense_devices
    return result+realsense_devices()


def fingerprint():
    from .robot import reference_config
    cfg=json.loads((ROOT/'config/scene.json').read_text())
    ref=reference_config()
    h=hashlib.sha256(json.dumps(dict(robot_type='so101',gripper_model=ref['gripper_model'],follower=ref['follower']),sort_keys=True).encode())
    for path in (ROOT/'config/scene.json',Path(cfg['model_source']),ROOT/'assets/so101/provenance.json'):
        h.update(path.read_bytes())
    return h.hexdigest()
