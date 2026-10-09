"""Convert saved SO-101 simulation geometry into camera-relative teaching candidates.

This module never opens hardware and never changes station calibration. Each
candidate picks one part, lifts it, returns it to the same seat and withdraws.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from vision_teaching import (number, vector, validate_target, calibration_signature,
                             reference_key, make_steps, compile_steps)
from jig_detection import PROFILE_DEFAULTS

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'Test_teach/config/so101_scene.json'
PARTS = {'cap': '상단', 'ball': '구', 'housing': '하단'}


def read_source(path=SOURCE):
    from episode_preview import read_episode
    config = read_episode(path)
    if config.get('robot') != 'so101' or config.get('part_order') != list(PARTS):
        raise ValueError('SO-101 부품 A의 상단·구·하단 시뮬 설정이 필요합니다.')
    return config


def convert_target(config, kind, *, carrier_bottom_z_mm=None, carrier_yaw_deg=None):
    """X/Y/yaw relative to carrier; Z absolute in the robot base, all in mm."""
    if kind not in PARTS:
        raise ValueError('상단·구·하단 중 하나를 선택하세요.')
    p = config['parts'][kind]
    center = vector(config['carrier']['center_base_mm'], 3, -600, 600)
    bottom = center[2] if carrier_bottom_z_mm is None else number(carrier_bottom_z_mm, -500, 1000)
    carrier_yaw = number(config['carrier']['yaw_deg'] if carrier_yaw_deg is None else carrier_yaw_deg, -360, 360)
    jig_yaw = number(p['jig_yaw_deg'], -360, 360)
    part_yaw = number(p['part_yaw_deg'], -360, 360)
    jig = np.array(vector(p['jig_center_carrier_mm'], 3))
    seat = np.array(vector(p['part_offset_jig_mm'], 3))
    grasp = np.array(vector(p['grasp_tcp_part_mm'], 3))
    mirrored = grasp[0] > 0
    grasp[0] = -abs(grasp[0])  # Fixed finger contacts the part-local left (-X) side.
    rz = lambda yaw: Rotation.from_euler('z', yaw, degrees=True)
    local = jig + rz(jig_yaw).apply(seat + rz(part_yaw).apply(grasp))
    size = vector(config['carrier']['size_mm'], 3, .01, 1500)
    # This carrier's continuous outer rim is at 4.8 mm. The 6 mm tabs are not
    # the plane traced by the camera detector. Do not substitute overall height.
    if not np.allclose(size, [220, 148, 6], atol=.01):
        raise ValueError('등록된 220×148×6 mm 하위 지그 모델만 가져올 수 있습니다.')
    yaw = (jig_yaw + part_yaw + number(p['grasp_yaw_deg'], -360, 360) + (180 if mirrored else 0) + 180) % 360 - 180
    entry = rz(jig_yaw+part_yaw).apply([-number(p.get('approach_clearance_mm', 0), 0, 10), 0, 0])
    return validate_target({**PROFILE_DEFAULTS, 'name': 'A '+PARTS[kind],
        'product': '제품 A', 'mode': 'rectangle', 'operation': 'pick',
        'return_to_start': True, 'grasp_side':'part_left', 'upper_strategy':'height_first',
        
        'jig_name': '시뮬 A 하위 지그 220×148', 'long_mm': size[0], 'short_mm': size[1],
        'surface_z_mm': bottom+4.8, 'expected_yaw_deg': carrier_yaw,
        'x_mm': float(local[0]), 'y_mm': float(local[1]), 'z_mm': bottom+float(local[2]),
        'yaw_deg': yaw, 'pitch_deg': p['grasp_pitch_deg'],
        'hover_mm': p['hover_mm'], 'retreat_mm': p['lift_mm'],
        'approach_x_mm': float(entry[0]), 'approach_y_mm': float(entry[1]),
        'lift_tilt_deg': p.get('lift_tilt_deg', 0), 'lift_clearance_mm': p.get('lift_clearance_mm', 40),
        'retreat_wait_s': 3.,
        'open': p['open_percent'], 'closed': p['close_percent'],
        'speed_mm_s': 10., 'wait_s': .4, 'geometry_verified': False})


def candidate(config, kind, station, *, source_path=SOURCE):
    from teach_left_arm import UNITS, stamp, validate_program
    target = convert_target(config, kind)
    tool = deepcopy(station['tool'])
    seed_values = vector(config['home_deg'], 6, -180, 180)
    from vision_teaching import JOINTS
    seed = {**{name+'.pos': seed_values[i] for i, name in enumerate(JOINTS)},
            'gripper.pos': number(seed_values[5], 0, 100)}
    ident = 'sim_A_'+kind
    center = config['carrier']['center_base_mm']
    observation = {reference_key(target): {'x_mm': center[0], 'y_mm': center[1],
                                         'yaw_deg': target['expected_yaw_deg']}}
    reference = {'kind': 'simulation_candidate', 'hardware_verified': False,
        'source': str(source_path),
        'configuration_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'part': kind, 'jig_center_carrier_mm': deepcopy(config['parts'][kind]['jig_center_carrier_mm']),
        'part_offset_jig_mm': deepcopy(config['parts'][kind]['part_offset_jig_mm']),
        'source_tool': deepcopy(config['tcp']), 'contact_reference_only': deepcopy(config['contact']),
        'mass_reference_only': config.get('mass_source'),
        'part_settings': deepcopy(config['parts'][kind]),
        'actuators_reference_only': deepcopy(config.get('actuators', {})),
        'source_home_deg_reference_only': deepcopy(config['home_deg']),
        'source_safe_deg_reference_only': deepcopy(config.get('safe_deg')),
        'motion_transfer': 'Carrier-relative left-side fixed-finger contact; height-first upper poses aim at grasp orientation; XY converges during descent; high hold 3 s; reverse path to original seat, release, withdraw. Physical torque is not set. Return placement has not been tested in contact simulation.',
        'nominal_carrier_base_mm': deepcopy(center),
        'note': 'Z and heading are simulation installation candidates. Verify actual rim height, docking direction, camera extrinsics and TCP. Mass/friction do not change motor commands.'}
    plan = {'targets': {ident: target}, 'calibration_signature': calibration_signature(station),
            'preview_tool': tool, 'preview_kind': 'simulation_candidate',
            'execution_trial':{'position_mm':4.,'angle_deg':6.},
            'simulation_reference': reference}
    from safe_episode import model_safe_pose, with_safe_endpoints
    safe=model_safe_pose(config,tool)
    plan['safe_pose_reference']={'source':'Test_teach safe_deg','model_candidate':True,'pose':deepcopy(safe)}
    steps = compile_steps(with_safe_endpoints(make_steps(ident, target, seed),safe), plan, station, observation, seed)
    doc = {'schema': 2, 'kind': 'teaching', 'identity': deepcopy(station['identity']),
           'units': deepcopy(UNITS), 'title': 'A '+PARTS[kind]+' · 집고 제자리에 놓기',
           'updated_at': stamp(), 'steps': steps, 'vision_plan': plan}
    validate_program(doc, doc['identity'])
    return doc


def save_candidates(directory, station, path=SOURCE):
    from teach_left_arm import save_named_teaching
    config = read_source(path)
    # Compute all candidates first so invalid geometry cannot leave a partial set.
    docs = {kind: candidate(config, kind, station, source_path=path) for kind in PARTS}
    return {kind: save_named_teaching(directory, doc) for kind, doc in docs.items()}
