"""CAD-based overhead camera initialization, explicitly not a measured extrinsic."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from vision_teaching import digest, number, require_intrinsics

SOURCE = Path(__file__).resolve().parent/'calibration/mount_geometry/nominal_pose.json'


def model_extrinsics(station, optical_offset_mm=0., image_roll_deg=0.):
    require_intrinsics(station, [station['camera']['width'], station['camera']['height']])
    offset = number(optical_offset_mm, 0, 100)
    roll = number(image_roll_deg, -180, 270)
    geometry = json.loads(SOURCE.read_text())
    z = np.array(geometry['nominal_optical_axis_in_robot_axes'], float)
    z /= np.linalg.norm(z)
    # Current unrotated image: right points to robot right (-Y), down towards
    # the robot side of the table. This axis choice is an installation estimate.
    x = np.cross(z, [0., 0., 1.]); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x, y, z]) @ Rotation.from_euler('z', roll, degrees=True).as_matrix()
    transform[:3, 3] = np.array(geometry['camera_board_center_in_base_link']) + offset*z
    return {'source': 'cad_estimate', 'verified_for_robot_motion': False,
            'base_from_camera': transform.tolist(),
            'intrinsics_signature': digest(station['intrinsics']),
            'optical_offset_mm': offset, 'image_roll_deg': roll,
            'geometry_source': str(SOURCE), 'geometry_sha256': digest(geometry),
            'lens_protrusion_mm_reference_only': 20.,
            'note': 'User specifies optical centre at PCB plane centre (0 mm). Lens protrusion of 20 mm is not added. CAD mounting pose and image axes remain estimates; confirm with physical points.'}


def calibration_label(station):
    ext = station.get('extrinsics')
    if not ext:
        return '실측 필요'
    if ext.get('source') == 'cad_estimate':
        return f"모델 추정 · 광학 중심 {ext['optical_offset_mm']:g}mm · 실측 미확인"
    return '체커보드 보정 저장됨'
