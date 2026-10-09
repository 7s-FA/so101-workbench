"""Nominal fixed-finger TCP from the assembled URDF/STL; no device access."""
from copy import deepcopy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

MODEL = Path(__file__).resolve().parent / 'assets' / 'so101'


def read_stl(path):
    data = Path(path).read_bytes()
    count = int.from_bytes(data[80:84], 'little')
    if len(data) != 84 + 50 * count:
        raise ValueError('모델 TCP 계산에는 원본 binary STL이 필요합니다.')
    record = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    return np.frombuffer(data, dtype=record, offset=84, count=count)['vertices'].astype(float)


def origin_mm(origin):
    matrix = np.eye(4)
    if origin is not None:
        matrix[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in origin.get('rpy', '0 0 0').split()]).as_matrix()
        matrix[:3, 3] = np.array([float(v) for v in origin.get('xyz', '0 0 0').split()]) * 1000
    return matrix


def fixed_finger_tip(model=MODEL):
    """Centre of the distal mesh edge, along gripper_frame_link's +Z axis.

    This pinned tapered finger ends in an edge, not a flat end face. Require
    exactly two adjacent tip vertices so a changed model cannot silently choose
    an arbitrary bounding-box centre or a point on the moving jaw.
    """
    model = Path(model)
    urdf = model / 'so101_modified.urdf'
    root = ET.parse(urdf).getroot()
    visual = root.find("link[@name='gripper_link']/visual[@name='xlerobot_fixed_finger']")
    frame_joint = root.find("joint[@name='gripper_frame_joint']")
    if visual is None or frame_joint is None or frame_joint.find('parent').get('link') != 'gripper_link':
        raise ValueError('고정 집게와 TCP 기준 프레임의 조립 관계를 확인하세요.')
    mesh = visual.find('geometry/mesh')
    mesh_path = model / mesh.get('filename')
    scale_mm = np.array([float(v) for v in mesh.get('scale', '1 1 1').split()]) * 1000
    triangles_mm = read_stl(mesh_path) * scale_mm
    transform = np.linalg.inv(origin_mm(frame_joint.find('origin'))) @ origin_mm(visual.find('origin'))
    assembled = triangles_mm @ transform[:3, :3].T + transform[:3, 3]
    vertices = np.unique(assembled.reshape(-1, 3), axis=0)
    tip = vertices[vertices[:, 2] >= vertices[:, 2].max() - .01]
    if len(tip) != 2 or not 5 < np.linalg.norm(tip[1] - tip[0]) < 50:
        raise ValueError('고정 집게의 끝단 형상이 변경됐습니다. 모델 기준점을 다시 확인하세요.')
    adjacent = [np.linalg.norm(assembled - p, axis=2).min(axis=1) < 1e-6 for p in tip]
    if not (adjacent[0] & adjacent[1]).any():
        raise ValueError('고정 집게의 끝단 꼭짓점이 같은 모서리에 있지 않습니다.')
    return {
        'kind': 'model_fixed_finger_tip',
        'label': '왼쪽 고정 집게 끝단 중심 · 모델 기준',
        'frame': 'gripper_frame_link',
        'visual': 'xlerobot_fixed_finger',
        'definition': 'Midpoint of the two adjacent distal-edge vertices; +Z extension, 0.01 mm selection tolerance.',
        'xyz_mm': tip.mean(axis=0).tolist(),
        'edge_endpoints_mm': tip.tolist(),
        'edge_width_mm': float(np.linalg.norm(tip[1] - tip[0])),
        'source_sha256': {
            'so101_modified.urdf': hashlib.sha256(urdf.read_bytes()).hexdigest(),
            str(mesh_path.relative_to(model)): hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        },
        'physically_verified': False,
    }


def model_tip_tool(current):
    """Set only the reference point; preserve axes and joint corrections."""
    tool = deepcopy(current)
    reference = fixed_finger_tip()
    tool['xyz_mm'] = reference['xyz_mm']
    tool['reference'] = reference
    tool['verified'] = False
    return tool


def reference_matches(tool, reference):
    return (isinstance(reference, dict) and reference.get('kind') == 'model_fixed_finger_tip'
            and np.allclose(tool['xyz_mm'], reference['xyz_mm'], rtol=0, atol=1e-8))


def reference_label(tool):
    reference = tool.get('reference')
    return reference['label'] if reference_matches(tool, reference) else '사용자 지정 오프셋'


def execution_summary(tool):
    xyz=' / '.join(f'{v:.4f}' for v in tool['xyz_mm'])
    rpy=' / '.join(f'{v:g}' for v in tool['rpy_deg'])
    return f'사용 기준: {reference_label(tool)}\nTCP XYZ: {xyz} mm · 회전: {rpy}°\n저장된 값을 실행에 바로 사용합니다. 설치 변경 시에만 아래 값을 수정하세요.'
