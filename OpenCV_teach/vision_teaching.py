"""Geometry and persistent contracts for fixed overhead-camera teaching.

All public distances are millimetres and angles degrees. No motor or camera is
opened on import. A rectangular jig frame has its origin at the rectangle centre, +X along
the directed long edge, +Y to its left, and +Z upward. Station calibration and episode targets are separate records.
"""
from copy import deepcopy
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
PHASES = ('approach', 'descend', 'grip', 'retreat')
PHASE_LABELS = ('위에서 대기', '잡기·놓기 높이로 내리기', '집게 열기·닫기', '위로 빼기')
RETURN_PHASES = ('return', 'release', 'withdraw')
RETURN_LABELS = ('같은 자리에 내려놓기', '집게 열어 놓기', '빈 집게를 위로 빼기')
DICTIONARIES = ('DICT_4X4_50', 'DICT_5X5_100', 'DICT_6X6_250')
ROOT = Path(__file__).resolve().parent


def number(value, low=-10000, high=10000):
    if isinstance(value, bool):
        raise ValueError('숫자 값을 입력하세요.')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('숫자 값을 입력하세요.') from None
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'값은 {low:g}~{high:g} 범위여야 합니다.')
    return value


def vector(value, size, low=-10000, high=10000):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != size:
        raise ValueError(f'{size}개 값을 입력하세요.')
    return [number(v, low, high) for v in value]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def default_station(identity):
    return {'schema': 2, 'identity': deepcopy(identity), 'camera': {'source': '0', 'width': 1280, 'height': 720},
            'board': {'cols': 13, 'rows': 9, 'square_mm': 20.0}, 'intrinsics': None, 'extrinsics': None,
            'tool': {'xyz_mm': [0., 0., 0.], 'rpy_deg': [0., 0., 0.], 'joint_offsets_deg': [0.] * 5,
                     'verified': False}, 'jigs': {}}


def calibration_signature(station):
    return digest({k: station[k] for k in ('identity', 'camera', 'intrinsics', 'extrinsics', 'tool')})


def validate_target(target):
    t = deepcopy(target)
    for key in ('name', 'product'):
        if not isinstance(t.get(key), str) or not 1 <= len(t[key].strip()) <= 60:
            raise ValueError('목표 이름과 제품 이름을 1~60자로 입력하세요.')
    if t.get('mode') not in ('rectangle', 'marker', 'fixed') or t.get('operation') not in ('pick', 'place'):
        raise ValueError('목표 기준 또는 집기/놓기 설정이 잘못됐습니다.')
    if t['mode'] == 'rectangle':
        from jig_detection import validate_profile
        t.update(validate_profile(t))
    elif t['mode'] == 'marker':
        # Load old files for explicit conversion; they are never executed as rectangles.
        if t.get('dictionary') not in DICTIONARIES:
            raise ValueError('기존 마커 사전 정보가 잘못됐습니다.')
        n = number(t.get('marker_id'), 0, {'DICT_4X4_50':49, 'DICT_5X5_100':99, 'DICT_6X6_250':249}[t['dictionary']])
        if n != int(n):
            raise ValueError('마커 ID는 정수여야 합니다.')
        t['marker_id'] = int(n)
        t['marker_size_mm'] = number(t.get('marker_size_mm'), 5, 500)
        t['marker_z_mm'] = number(t.get('marker_z_mm'), -500, 1000)
    for k, lo, hi in (('x_mm', -1000, 1000), ('y_mm', -1000, 1000), ('z_mm', -500, 1000),
                       ('yaw_deg', -360, 360), ('hover_mm', 5, 300), ('open', 0, 100),
                       ('closed', 0, 100), ('speed_mm_s', 1, 100), ('wait_s', 0, 30)):
        t[k] = number(t.get(k), lo, hi)
    if t['open'] <= t['closed']:
        raise ValueError('열림 값은 닫힘 값보다 커야 합니다. 0~100 정규화 단위입니다.')
    t['upper_strategy']=t.get('upper_strategy','fixed_xy')
    if t['upper_strategy'] not in ('fixed_xy','height_first'):raise ValueError('상단 이동 방식이 올바르지 않습니다.')
    t['pitch_deg'] = number(t.get('pitch_deg', 180), 0, 360)
    t['roll_deg'] = number(t.get('roll_deg', 180), -360, 360)
    t['retreat_mm'] = number(t.get('retreat_mm', t['hover_mm']), 5, 300)
    for key in ('approach_x_mm', 'approach_y_mm'):
        t[key] = number(t.get(key, 0), -10, 10)
    t['lift_tilt_deg'] = number(t.get('lift_tilt_deg', 0), 0, 30)
    t['lift_clearance_mm'] = number(t.get('lift_clearance_mm', 40), 5, 150)
    t['retreat_wait_s'] = number(t.get('retreat_wait_s', t['wait_s']), 0, 30)
    t['return_to_start'] = t.get('return_to_start', False)
    if type(t['return_to_start']) is not bool or (t['return_to_start'] and t['operation'] != 'pick'):
        raise ValueError('제자리 왕복은 집기 동작에서만 사용할 수 있습니다.')
    return t


def reference_key(target):
    if target['mode'] == 'marker':
        raise ValueError('기존 마커 목표입니다. 직사각형 기준으로 전환하고 칸별 위치를 다시 입력하세요.')
    return 'rectangle:' + target.get('jig_id',target['jig_name'])


def board_points(board):
    cols, rows = int(number(board['cols'], 3, 20)), int(number(board['rows'], 3, 20))
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * number(board['square_mm'], 1, 100)
    return points


def chessboard(frame, board):
    import cv2
    shape = (int(board['cols']), int(board['rows']))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, shape, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise ValueError('체커보드 전체 내부 코너가 보이지 않습니다. 크기 설정·초점·조명을 확인하세요.')
    return corners


def calibrate_intrinsics(samples, image_size, board, camera):
    import cv2
    if len(samples) < 10:
        raise ValueError('서로 다른 위치·기울기의 선명한 사진을 10장 이상 수집하세요. 권장 20~30장.')
    obj = board_points(board)
    rms, k, d, rvecs, tvecs = cv2.calibrateCamera([obj] * len(samples), samples, tuple(image_size), None, None)
    errors = []
    for corners, r, t in zip(samples, rvecs, tvecs):
        projection, _ = cv2.projectPoints(obj, r, t, k, d)
        errors.append(float(np.sqrt(np.mean(np.sum((corners - projection) ** 2, axis=2)))))
    if not np.isfinite(k).all() or not np.isfinite(d).all() or rms > 2:
        raise ValueError(f'보정 오차 {rms:.3f}px: 흐린 사진·비슷한 자세·체커 크기를 확인하세요.')
    normals=np.array([cv2.Rodrigues(rv)[0][:,2] for rv in rvecs])
    spread=float(np.degrees(np.arccos(np.clip(normals@normals.T,-1,1))).max())
    return {'normal_spread_deg':spread, 'usable_for_metric':spread>=10.,
            'quality_note': '' if spread>=10. else '보드 기울기 변화가 부족합니다. 앞뒤·좌우로 기울인 사진을 추가하세요.',
            'K': k.tolist(), 'D': d.ravel().tolist(), 'size': list(image_size), 'rms_px': float(rms),
            'errors_px': errors, 'camera': deepcopy(camera), 'board': deepcopy(board)}


def require_intrinsics(station, size):
    intrinsic = station.get('intrinsics')
    if not intrinsic or intrinsic['camera'] != station['camera'] or list(size) != intrinsic['size']:
        raise ValueError('카메라·해상도가 보정값과 다릅니다. 같은 설정의 내부 보정을 먼저 적용하세요.')
    if intrinsic.get('usable_for_metric') is False:
        raise ValueError('내부 보정 후보: 보드를 기울인 사진을 추가해야 로봇 좌표를 계산할 수 있습니다.')
    return np.asarray(intrinsic['K'], float), np.asarray(intrinsic['D'], float)


def calibrate_extrinsics(corners, size, station, origin_mm, yaw_deg, row_sign=-1):
    import cv2
    k, d = require_intrinsics(station, size)
    obj = board_points(station['board']).astype(float)
    if row_sign not in (-1, 1):
        raise ValueError('체커 행 방향은 +Y 또는 -Y로 지정하세요.')
    obj[:, 1] *= row_sign
    r = Rotation.from_euler('z', number(yaw_deg, -360, 360), degrees=True).as_matrix()
    obj = obj @ r.T + vector(origin_mm, 3)
    ok, rv, tv = cv2.solvePnP(obj, corners, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('로봇 좌표 보정에 실패했습니다.')
    proj, _ = cv2.projectPoints(obj, rv, tv, k, d)
    err = float(np.sqrt(np.mean(np.sum((corners - proj) ** 2, axis=2))))
    if err > 2 or not np.isfinite(err):
        raise ValueError(f'좌표 보정 오차가 큽니다: {err:.3f}px')
    camera_from_base = np.eye(4)
    camera_from_base[:3, :3] = cv2.Rodrigues(rv)[0]
    camera_from_base[:3, 3] = tv.ravel()
    if np.min((obj @ camera_from_base[:3, :3].T + tv.ravel())[:, 2]) <= 0:
        raise ValueError('기준판의 앞뒤 또는 좌표를 확인하세요.')
    base_from_camera = np.linalg.inv(camera_from_base)
    if base_from_camera[2, 3] <= origin_mm[2]:
        raise ValueError('계산된 카메라가 기준판 아래에 있습니다. 0번 코너·열 방향·행의 ±Y 방향을 다시 확인하세요.')
    return {'base_from_camera': base_from_camera.tolist(), 'error_px': err, 'row_sign': row_sign,
            'intrinsics_signature': digest(station['intrinsics']), 'origin_mm': list(origin_mm),
            'yaw_deg': yaw_deg}


def pixels_on_plane(pixels, z_mm, station, size):
    import cv2
    k, d = require_intrinsics(station, size)
    ext = station.get('extrinsics')
    if not ext or ext['intrinsics_signature'] != digest(station['intrinsics']):
        raise ValueError('로봇 좌표 보정이 필요합니다.')
    tf = np.asarray(ext['base_from_camera'], float)
    points = cv2.undistortPoints(np.asarray(pixels, float).reshape(-1, 1, 2), k, d).reshape(-1, 2)
    rays = np.column_stack((points, np.ones(len(points)))) @ tf[:3, :3].T
    if np.any(np.abs(rays[:, 2]) < 1e-6):
        raise ValueError('작업 평면과 시선이 평행합니다.')
    scale = (z_mm - tf[2, 3]) / rays[:, 2]
    if np.any(scale <= 0):
        raise ValueError('지그 윗면 높이 또는 카메라 좌표가 잘못됐습니다.')
    return tf[:3, 3] + rays * scale[:, None]


def angle_delta(a, b):
    return (a - b + 180) % 360 - 180


class StableJigs:
    """Rolling 3 s consensus, with >=2 precise inliers and a quality-weighted centre.

    Isolated outliers cannot drag the centre or count as precise evidence.
    Sustained displacement, missing/low frames and gaps start a new window.
    """
    duration = 3.
    max_gap = .6
    minimum_precise = 2
    minimum_inlier_fraction = .7

    def __init__(self, *, position_tolerance=1., angle_period=360., max_gap=.6):
        self.history = {}
        self.failures = {}
        self.max_gap = float(max_gap)
        self.position_tolerance = position_tolerance
        self.angle_period = angle_period

    def clear(self):
        self.history.clear()
        self.failures.clear()

    def missing(self, key, timestamp=None, frame_id=None, *, hard=False):
        if timestamp is None or hard:
            self.history.pop(key, None);self.failures.pop(key, None);return
        hist=self.history.get(key)
        if not hist:return
        if timestamp-hist[-1]['at']>self.max_gap:
            self.history.pop(key,None);self.failures.pop(key,None);return
        failures=self.failures.setdefault(key,deque(maxlen=1024))
        if not failures or failures[-1][1]!=frame_id:failures.append((timestamp,frame_id))
        while failures and timestamp-failures[0][0]>self.duration:failures.popleft()

    def _delta(self, a, b):
        return (a-b+self.angle_period/2)%self.angle_period-self.angle_period/2

    def _statistics(self, hist):
        xy = np.array([[v['x_mm'], v['y_mm']] for v in hist])
        raw = np.array([v['yaw_deg'] for v in hist])
        # Choose a circular medoid before unwrapping: a first-frame angular
        # outlier must not split a valid cluster across the wrap boundary.
        distance = np.abs(self._delta(raw[:,None],raw[None,:]))
        anchor = raw[np.argmin(distance.sum(axis=1))]
        angles = anchor+self._delta(raw,anchor)
        return xy, angles

    def observe(self, key, value, timestamp, frame_id):
        if value.get('quality', 'precise') not in ('precise', 'medium'):
            self.missing(key,timestamp,frame_id)
            return
        hist = self.history.setdefault(key, deque(maxlen=1024))
        if hist and frame_id == hist[-1]['frame_id']:
            return
        if hist and (timestamp <= hist[-1]['at'] or
                     value.get('profile_signature') != hist[-1].get('profile_signature') or
                     timestamp-hist[-1]['at'] > self.max_gap):
            hist.clear();self.failures.pop(key,None)
        if len(hist)>=3:
            xy, angles = self._statistics(list(hist)[:-1])
            centre, angle = np.median(xy,axis=0), np.median(angles)
            def displaced(v):
                return (np.linalg.norm(np.array([v['x_mm'],v['y_mm']])-centre)>2*self.position_tolerance or
                        abs(self._delta(v['yaw_deg'],angle))>2)
            nearby=(math.hypot(value['x_mm']-hist[-1]['x_mm'],value['y_mm']-hist[-1]['y_mm'])<=2*self.position_tolerance
                    and abs(self._delta(value['yaw_deg'],hist[-1]['yaw_deg']))<=2)
            if displaced(hist[-1]) and displaced(value) and nearby:
                last=hist[-1];hist.clear();hist.append(last);self.failures.pop(key,None)
        hist.append({**value, 'at': timestamp, 'frame_id': frame_id})
        # Retain one bracketing sample for the elapsed-time test; it is excluded
        # from precision counts/statistics if outside the actual last 3 seconds.
        while len(hist)>2 and timestamp-hist[1]['at'] >= self.duration:
            hist.popleft()

    def progress(self, key, now=None):
        now = time.monotonic() if now is None else now
        h = self.history.get(key, [])
        if not h or not -1e-9 <= now-h[-1]['at'] <= self.max_gap:
            return 0.
        return min(self.duration, h[-1]['at']-h[0]['at'])

    def assessment(self, key, now=None):
        now = time.monotonic() if now is None else now
        hist = self.history.get(key, [])
        seconds = self.progress(key, now)
        h = [v for v in hist if hist[-1]['at']-v['at']<=self.duration+1e-9]
        missing=[at for at,_ in self.failures.get(key,()) if now-self.duration<=at<=now]
        result={'seconds':seconds,'ready':False,'precise':0,'rejected':0,'missing':len(missing),'total':len(h)+len(missing),'value':None,'reasons':['실행용 좌표 측정 없음'],'inlier_fraction':0.}
        if not h:return result
        xy, angles = self._statistics(h)
        centre, angle = np.median(xy,axis=0), np.median(angles)
        inliers=(np.linalg.norm(xy-centre,axis=1)<=self.position_tolerance+1e-9)&(np.abs(angles-angle)<=1.+1e-9)
        accepted=[v for v,keep in zip(h,inliers) if keep]
        counts={q:sum(v.get('quality','precise')==q for v in accepted) for q in ('precise','medium')}
        result.update(precise=counts['precise'],rejected=len(h)-len(accepted))
        reasons=[]
        fraction=len(accepted)/result['total']
        if seconds+1e-9<self.duration:reasons.append('3초 측정 미완료')
        if counts['precise']<self.minimum_precise:reasons.append(f"유효 정밀 {counts['precise']}/{self.minimum_precise}회")
        if fraction<self.minimum_inlier_fraction:reasons.append(f'유효 측정 {fraction:.0%} / 필요 {self.minimum_inlier_fraction:.0%}')
        if missing and max(missing)>=h[-1]['at']:reasons.append('최근 프레임 미검출')
        if len(h)>=2 and not inliers[-1] and not inliers[-2]:reasons.append('최근 위치·각도 연속 이탈')
        if not accepted or now-accepted[-1]['at']>self.max_gap:reasons.append('새 유효 좌표 대기')
        result.update(reasons=reasons,inlier_fraction=fraction)
        if reasons:return result
        # Confidence is relative image evidence, not a calibrated accuracy probability.
        weights=np.array([(3. if v.get('quality','precise')=='precise' else 1.)*
                          float(np.clip(v.get('detection_confidence',1.),.1,1.))**2 for v in accepted])
        centre=np.average(xy[inliers],axis=0,weights=weights)
        # Angles were unwrapped around the circular medoid, so 179/-179 (or
        # 179/1 for undirected image angles) never average towards the wrong side.
        angle=float(np.average(angles[inliers],weights=weights))
        weight_sums={q:float(sum(w for v,w in zip(accepted,weights) if v.get('quality','precise')==q))
                     for q in ('precise','medium')}
        result['ready']=True
        result['value']={'x_mm':float(centre[0]),'y_mm':float(centre[1]),
            'yaw_deg':float(self._delta(angle,0.)), 'at':accepted[-1]['at'],
            'profile_signature':h[-1].get('profile_signature'),
            'stability_seconds':seconds, 'sample_count':len(accepted), 'quality_counts':counts,
            'rejected_count':result['rejected'], 'total_samples':result['total'], 'missing_count':len(missing), 'estimator':'quality_weighted_inlier_mean', 'quality_weight_sums':weight_sums}
        return result

    def status_text(self, key, now=None):
        a=self.assessment(key,now)
        return f"안정화 {a['seconds']:.1f}/3.0초 · 정밀 {a['precise']}/{self.minimum_precise}회 · 제외 {a['rejected']}개 · 누락 {a['missing']}회"

    def snapshot(self, keys, now=None):
        result={}
        for key in keys:
            a=self.assessment(key,now)
            if not a['ready']:
                raise ValueError(f"{key}: {self.status_text(key,now)} · {' / '.join(a['reasons'])}")
            result[key]=a['value']
        return result


def target_pose(target, observations):
    x, y, yaw = target['x_mm'], target['y_mm'], target['yaw_deg']
    if target['mode'] == 'marker':
        reference_key(target)
    if target['mode'] == 'rectangle':
        ref = observations.get(reference_key(target))
        if ref is None:raise ValueError('실행 전 지그 위치 측정값이 필요합니다.')
        a = math.radians(ref['yaw_deg'])
        x, y = ref['x_mm'] + math.cos(a)*x - math.sin(a)*y, ref['y_mm'] + math.sin(a)*x + math.cos(a)*y
        yaw += ref['yaw_deg']
    return np.array([x, y, target['z_mm']]), Rotation.from_euler(
        'xyz', [target.get('roll_deg', 180), target.get('pitch_deg', 180)-180, yaw], degrees=True).as_matrix()


class Kinematics:
    """URDF FK with calibrated joint zero and a TCP relative to gripper_frame_link."""
    def __init__(self, tool=None, *, trial_tolerance=None):
        # Explicit, per-call trial only. Normal teaching remains at 0.5 mm / 1°.
        self.position_tolerance, self.angle_tolerance = .5, 1.
        if trial_tolerance is not None:
            self.position_tolerance = number(trial_tolerance['position_mm'], .5, 4.)
            self.angle_tolerance = number(trial_tolerance['angle_deg'], 1., 6.)
        self.ik_errors = []
        self.tool = tool or default_station({})['tool']
        root = ET.parse(ROOT / 'assets/so101/so101_modified.urdf').getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain, link = [], 'gripper_frame_link'
        while link in by_child:
            j = by_child[link]
            o = j.find('origin')
            tf = np.eye(4)
            tf[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in o.get('rpy', '0 0 0').split()]).as_matrix()
            tf[:3, 3] = np.array([float(v) for v in o.get('xyz', '0 0 0').split()]) * 1000
            name = j.get('name')
            limits = j.find('limit')
            chain.append((name, tf, None if limits is None else
                          (math.degrees(float(limits.get('lower'))), math.degrees(float(limits.get('upper'))))))
            link = j.find('parent').get('link')
        self.chain = list(reversed(chain))
        from model_zero import model_offsets
        self.offsets = np.array(model_offsets(self.tool))
        limits = {name: lim for name, _, lim in self.chain if lim is not None}
        self.low = np.maximum(-180, np.array([limits[n][0] for n in JOINTS]) - self.offsets)
        self.high = np.minimum(180, np.array([limits[n][1] for n in JOINTS]) - self.offsets)
        if 'motor_limits_deg' in self.tool:
            motor_limits=np.array([vector(pair,2,-180,180) for pair in self.tool['motor_limits_deg']])
            if motor_limits.shape!=(5,2):raise ValueError('몸체 5개 모터의 보정 범위가 필요합니다.')
            self.low=np.maximum(self.low,motor_limits[:,0]);self.high=np.minimum(self.high,motor_limits[:,1])
            if np.any(self.low>=self.high):raise ValueError('모델과 모터의 관절 범위가 겹치지 않습니다.')
        self.tcp = np.eye(4)
        self.tcp[:3, 3] = vector(self.tool['xyz_mm'], 3, -200, 200)
        self.tcp[:3, :3] = Rotation.from_euler('xyz', vector(self.tool['rpy_deg'], 3, -180, 180), degrees=True).as_matrix()

    def fk(self, pose):
        q = np.array([pose[n + '.pos'] for n in JOINTS]) + self.offsets
        tf = np.eye(4)
        for name, origin, limit in self.chain:
            tf = tf @ origin
            if limit is not None:
                rot = np.eye(4)
                rot[:3, :3] = Rotation.from_euler('z', q[JOINTS.index(name)], degrees=True).as_matrix()
                tf = tf @ rot
        return tf @ self.tcp

    def ik(self, xyz, rotation, seed, gripper):
        def pose(q):
            return {**{n+'.pos': float(v) for n, v in zip(JOINTS, q)}, 'gripper.pos': float(gripper)}
        def residual(q):
            tf = self.fk(pose(q))
            return np.r_[tf[:3, 3] - xyz, Rotation.from_matrix(rotation.T @ tf[:3, :3]).as_rotvec() * 80]
        initial = np.clip([seed[n+'.pos'] for n in JOINTS], self.low + 1e-5, self.high - 1e-5)
        best = None
        for q0 in (initial, np.clip([0, -30, 50, 60, 0], self.low+1e-5, self.high-1e-5),
                   np.clip([0, 30, -50, -60, 0], self.low+1e-5, self.high-1e-5)):
            sol = least_squares(residual, q0, bounds=(self.low, self.high), max_nfev=100,
                                ftol=1e-8, xtol=1e-8, gtol=1e-8)
            p = pose(sol.x)
            tf = self.fk(p)
            position_error = float(np.linalg.norm(tf[:3, 3] - xyz))
            angle_error = math.degrees(np.linalg.norm(Rotation.from_matrix(rotation.T @ tf[:3, :3]).as_rotvec()))
            if best is None or position_error + angle_error < best[0]:
                best = (position_error + angle_error, position_error, angle_error, p)
            if position_error <= .5 and angle_error <= 1.:
                self.ik_errors.append((position_error,angle_error))
                return p
        if best[1] <= self.position_tolerance and best[2] <= self.angle_tolerance:
            self.ik_errors.append((best[1],best[2]))
            return best[3]
        raise ValueError(f'저장된 위치와 집게 방향을 함께 만족하는 자세를 계산하지 못했습니다. '
                         f'목표 X {xyz[0]:.1f} / Y {xyz[1]:.1f} / Z {xyz[2]:.1f}mm. '
                         f'위치 오차 {best[1]:.2f}mm, 방향 오차 {best[2]:.2f}°. '
                         '해당 스텝의 집기 위치·방향 또는 지그 배치를 조정하세요.')


def validate_motion(step, targets):
    motion = step.get('motion')
    if not motion:
        return
    if not isinstance(motion, dict) or motion.get('phase') not in PHASES+RETURN_PHASES+('waypoint',) or motion.get('target') not in targets:
        raise ValueError('카메라 스텝의 목표·동작 정보가 올바르지 않습니다.')
    if motion['phase'] in RETURN_PHASES and not targets[motion['target']].get('return_to_start'):
        raise ValueError('제자리 왕복 설정이 없는 목표입니다.')


def validate_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get('targets'), dict) or len(plan['targets']) > 100:
        raise ValueError('목표 목록이 올바르지 않습니다.')
    targets = {str(k): validate_target(v) for k, v in plan['targets'].items()}
    if not isinstance(plan.get('calibration_signature'), str):
        raise ValueError('보정 프로필 정보가 없습니다.')
    if 'preview_tool' in plan:
        tool = plan['preview_tool']
        from model_zero import model_offsets
        model_offsets(tool)
        vector(tool['xyz_mm'], 3, -200, 200)
        vector(tool['rpy_deg'], 3, -180, 180)
    from jig_detection import profile_signature
    geometry = {}
    for t in targets.values():
        if t['mode'] == 'rectangle':
            key = reference_key(t)
            sig = profile_signature(t)
            if key in geometry and geometry[key] != sig:
                raise ValueError('동일 지그의 제품·크기·높이·도킹 방향·검출 영역을 일치시키세요.')
            geometry[key] = sig
    return {**deepcopy(plan), 'targets': targets}


def make_steps(target_id, target, seed):
    op = '집기' if target['operation'] == 'pick' else '놓기'
    labels = list(PHASE_LABELS)
    labels[2] = '집게 닫기' if target['operation'] == 'pick' else '집게 열기'
    if target.get('approach_x_mm', 0) or target.get('approach_y_mm', 0):
        labels[1] = '내려서 집기점으로 접근'
    if target.get('lift_tilt_deg', 0):
        labels[3] = '위로 빼고 필요시 기울이기'
    phases = PHASES
    if target.get('return_to_start'):
        phases += RETURN_PHASES
        labels += list(RETURN_LABELS)
    return [{'name': f"{target['name']} · {op} {label}"[:80], 'pose': deepcopy(seed), 'captured_at': '',
             'motion': {'target': target_id, 'phase': phase}}
            for label, phase in zip(labels, phases)]


def approach_point(target, observations, xyz):
    """Offset in the carrier plane (or base plane for a fixed target)."""
    yaw = observations[reference_key(target)]['yaw_deg'] if target['mode'] == 'rectangle' else 0.
    offset = Rotation.from_euler('z', yaw, degrees=True).apply(
        [target['approach_x_mm'], target['approach_y_mm'], 0.])
    return xyz + offset


def lifting_poses(kin, target, xyz, rotation, seed, gripper):
    """Same shoulder-axis adaptive tilt as Test_teach, sampled at <=2 mm.

    Keep TCP XY fixed. Below clearance, keep the grasp orientation; above it,
    add only the tilt needed for IK and a 2 degree wrist-limit margin. Recompute
    the shoulder axis from this docking's grasp pose, never saved joint angles.
    """
    pan = math.radians(seed['shoulder_pan.pos'] + kin.offsets[0])
    axis = np.array([math.sin(pan), math.cos(pan), 0.])
    height, clearance, limit = target['retreat_mm'], target['lift_clearance_mm'], target['lift_tilt_deg']
    previous, previous_tilt = deepcopy(seed), 0.
    for z in np.linspace(0., height, max(1, math.ceil(height/2.))+1)[1:]:
        maximum = limit * np.clip((z-clearance)/max(1., height-clearance), 0., 1.)
        last_error = None
        for angle in np.linspace(previous_tilt, maximum, max(1, math.ceil(maximum-previous_tilt))+1):
            desired = Rotation.from_rotvec(-axis*math.radians(angle)).as_matrix() @ rotation
            try:
                p = kin.ik(xyz+[0., 0., z], desired, previous, gripper)
                if p['wrist_flex.pos'] > kin.high[3]-2 and (angle > 0 or maximum > 0):
                    raise ValueError('손목 한계 여유가 부족합니다.')
            except ValueError as error:
                last_error = error
                continue
            previous, previous_tilt = p, angle
            yield p
            break
        else:
            raise ValueError('상승 경로를 만들 수 없습니다: '+str(last_error))


def validate_execution_setup(plan, station, *, live=False):
    plan = validate_plan(plan)
    if plan['calibration_signature'] != calibration_signature(station):
        raise ValueError('카메라·로봇·TCP 보정이 바뀌었습니다. 목표 화면에서 경로를 다시 계산하고 저장하세요.')
    # Saved geometry is the execution source. Legacy verification flags are
    # provenance only; valid numeric data and kinematic checks remain required.
    vector(station['tool']['xyz_mm'],3,-200,200)
    vector(station['tool']['rpy_deg'],3,-180,180)
    from model_zero import model_offsets
    model_offsets(station['tool'])
    for t in plan['targets'].values():
        if t['mode'] == 'marker':
            reference_key(t)
    return plan


def compile_steps(steps, plan, station, observations, initial, *, live=False, single=False, trial_tolerance=None):
    from step_reference import used_plan
    plan = validate_execution_setup(used_plan(steps,plan), station, live=live)
    trial_tolerance=trial_tolerance or plan.get('execution_trial')
    kin = Kinematics(station['tool'],trial_tolerance=trial_tolerance)
    previous = deepcopy(initial)
    result, last_motion, lifted_paths = [], None, {}
    height_cycles = {}
    for source in steps:
        try:
            step = deepcopy(source)
            validate_motion(step, plan['targets'])
            motion = step.get('motion')
            if not motion:
                if step.get('gripper_only'):step['pose']={**previous,'gripper.pos':step['pose']['gripper.pos']}
                result.append(step); previous = step['pose']
                binding=step.get('follow_motion',{})
                last_motion=deepcopy(binding) if step.get('gripper_only') and binding.get('phase') in ('grip','release') else None
                continue
            if source.get('path'):step.setdefault('fixed_pose',deepcopy(source['pose']))
            t = plan['targets'][motion['target']]
            phase = motion['phase']
            if not single and phase not in ('approach','waypoint'):
                phases = PHASES+RETURN_PHASES
                expected = phases[phases.index(phase)-1]
                if last_motion != {'target': motion['target'], 'phase': expected}:
                    raise ValueError('카메라 동작은 상단 대기 → 하강 → 집기 → 상승 → 제자리 하강 → 놓기 → 후퇴 순서를 유지하세요.')
            xyz, rot = target_pose(t, observations)
            if t.get('upper_strategy')=='height_first' and phase!='waypoint':
                from height_planner import compile_height_cycle
                key=motion['target']
                if key not in height_cycles:
                    height_cycles[key]=compile_height_cycle(kin,t,xyz,rot,approach_point(t,observations,xyz),previous)
                cycle=height_cycles[key]
                if phase!='approach':
                    preceding=(PHASES+RETURN_PHASES)[(PHASES+RETURN_PHASES).index(phase)-1]
                    expected=cycle[preceding][-1]['pose']
                    if single and any(abs(expected[n+'.pos']-previous[n+'.pos'])>3 for n in JOINTS):
                        raise ValueError('앞 단계 자세에 도달하지 않았습니다. 상단 대기부터 실행하세요.')
                step['path']=deepcopy(cycle[phase]);step['pose']=deepcopy(step['path'][-1]['pose'])
                if trial_tolerance is not None:step['trial_ik']={'position_limit_mm':kin.position_tolerance,'angle_limit_deg':kin.angle_tolerance}
                step.setdefault('fixed_pose',deepcopy(source['pose']))
                result.append(step);previous,last_motion=step['pose'],motion
                continue
            entry = approach_point(t, observations, xyz)
            upper = entry + [0, 0, t['hover_mm']]
            retreat = xyz + [0, 0, t['retreat_mm']]
            before, after = (t['open'], t['closed']) if t['operation'] == 'pick' else (t['closed'], t['open'])
            goal = upper if phase in ('approach', 'withdraw') else retreat if phase == 'retreat' else xyz
            grip = after if phase in ('grip', 'retreat', 'return') else before
            path = []
            if phase in ('approach','waypoint'):
                if phase=='waypoint':goal=xyz;before=step.get('fixed_pose',step['pose'])['gripper.pos']
                p = kin.ik(goal, rot, previous, before)
                path.append({'pose': p, 'hold_s': t['wait_s'], 'duration_s': 0.})
            else:
                start = kin.fk(previous)
                required = upper if phase == 'descend' else xyz
                required_rot = rot
                if phase == 'return':
                    ascent = lifted_paths.get(motion['target'])
                    if ascent is None:
                        grasp = kin.ik(xyz, rot, previous, after)
                        ascent = [grasp, *lifting_poses(kin, t, xyz, rot, grasp, after)]
                    end = kin.fk(ascent[-1]); required, required_rot = end[:3, 3], end[:3, :3]
                if np.linalg.norm(start[:3, 3] - required) > max(1., kin.position_tolerance+.5) or np.linalg.norm(Rotation.from_matrix(required_rot.T @ start[:3, :3]).as_rotvec()) > math.radians(max(2., kin.angle_tolerance+1.)):
                    raise ValueError('선행 위치·방향에 도달하지 않았습니다. 상단 대기부터 실행하세요.')
                if abs(previous['gripper.pos'] - (after if phase in ('retreat', 'return', 'release') else before)) > .1:
                    raise ValueError('선행 그리퍼 동작이 맞지 않습니다.')
                if phase in ('grip', 'release'):
                    poses = [{**previous, 'gripper.pos': grip}]
                elif phase == 'return':
                    # Exact reverse of this execution's computed ascent, including
                    # undoing tilt above the jig; do not re-detect at the high pose.
                    poses = reversed(ascent[:-1])
                elif phase == 'retreat':
                    lifted_paths[motion['target']] = [deepcopy(previous)]
                    poses = lifting_poses(kin, t, xyz, rot, previous, grip)
                else:
                    # Descend beside the part first, then seat the fixed finger.
                    # The latter leg remains explicit Cartesian samples, not a
                    # diagonal shortcut through the part/jig during descent.
                    points, begin = [], start[:3, 3]
                    goals = [entry, goal] if phase == 'descend' and np.linalg.norm(entry-xyz)>1e-8 else [goal]
                    if phase == 'withdraw' and np.linalg.norm(entry-xyz)>1e-8:
                        goals = [entry, upper]
                    for end in goals:
                        count = max(1, int(math.ceil(np.linalg.norm(end-begin)/2.)))
                        points.extend(begin+(end-begin)*i/count for i in range(1, count+1))
                        begin = end
                    def linear_poses():
                        current = previous
                        for point in points:
                            current = kin.ik(point, rot, current, grip)
                            yield current
                    poses = linear_poses()
                for p in poses:
                    duration = float(np.linalg.norm(kin.fk(p)[:3, 3]-kin.fk(previous)[:3, 3])/t['speed_mm_s'])
                    path.append({'pose': p, 'hold_s': 0., 'duration_s': duration})
                    if phase == 'retreat':
                        lifted_paths[motion['target']].append(deepcopy(p))
                    previous = p
                path[-1]['hold_s'] = t['retreat_wait_s'] if phase == 'retreat' else t['wait_s']
            if trial_tolerance is not None:
                step['trial_ik']={'position_limit_mm':kin.position_tolerance,
                    'angle_limit_deg':kin.angle_tolerance,
                    'max_position_error_mm':max((e[0] for e in kin.ik_errors),default=0.),
                    'max_angle_error_deg':max((e[1] for e in kin.ik_errors),default=0.)}
            step['path'] = path
            step['pose'] = deepcopy(path[-1]['pose'])
            step.setdefault('fixed_pose',deepcopy(step['pose']))
            result.append(step)
            previous, last_motion = step['pose'], motion
        except ValueError as exc:
            raise ValueError(f"실행 불가 스텝: {source.get('name', '이름 없는 스텝')}\n{exc}") from exc
    return result


def expand_steps(steps, initial=None):
    expanded = []
    previous=deepcopy(initial) if initial is not None else (deepcopy(steps[0]['pose']) if steps else {})
    for step in steps:
        if not step.get('motion'):
            item=deepcopy(step)
            if step.get('gripper_only'):item['pose']={**previous,'gripper.pos':step['pose']['gripper.pos']}
            if step.get('safe_boundary'):item['precision']=True
            expanded.append(item);previous=item['pose']
            continue
        if not step.get('path'):
            raise ValueError('카메라 목표 경로를 먼저 계산하세요.')
        for node in step['path']:
            expanded.append({'name': step['name'], 'pose': node['pose'], 'hold_s': node['hold_s'],
                             'duration_s': node['duration_s'], 'cartesian': step['motion']['phase'] not in ('approach','waypoint'),
                             'precision': not bool(step.get('trial_ik'))})
            previous=node['pose']
    if len(expanded) > 10000:
        raise ValueError('계산된 경로가 너무 깁니다.')
    return expanded


def synchronized_approach(previous, goal, start, dt, duration):
    """Move every joint by the same fraction along a short Cartesian sample."""
    from teaching_settings import COMMAND_RATE, GRIPPER_COMMAND_RATE, MAX_COMMAND_DT
    total = {k: goal[k]-start[k] for k in goal}
    seconds = max(float(duration), max(abs(v)/(GRIPPER_COMMAND_RATE if k=='gripper.pos' else COMMAND_RATE)
                                      for k,v in total.items()), 1e-9)
    fraction = min(max(dt,0),MAX_COMMAND_DT)/seconds
    return {k: previous[k]+math.copysign(min(abs(goal[k]-previous[k]),abs(v)*fraction),goal[k]-previous[k])
            for k,v in total.items()}
