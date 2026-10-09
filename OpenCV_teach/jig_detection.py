"""Marker-free rectangular jig detection on a calibrated, known horizontal plane.

The operator selects a named profile; geometry does not identify the product.
Rectangle symmetry is resolved only by the configured docking direction prior.
All public coordinates are in the robot base frame, in millimetres/degrees.
"""
from copy import deepcopy
import math

import cv2
import numpy as np

class DetectionAmbiguity(ValueError):
    pass


PROFILE_DEFAULTS = dict(jig_name='하위 지그 220×148', long_mm=220., short_mm=148., surface_z_mm=0.,
    expected_yaw_deg=0., yaw_tolerance_deg=30., size_tolerance_pct=5.,
    appearance='white_frame', shape_model='ring', detection_method='hybrid', roi_x_pct=0., roi_y_pct=0., roi_w_pct=100., roi_h_pct=100., geometry_verified=False)
PROFILE_FIELDS = ('product', *PROFILE_DEFAULTS, 'jig_id', 'hole_layout')


def validate_profile(value):
    from vision_teaching import number
    result = {k: deepcopy(value[k]) for k in PROFILE_FIELDS if k in value}
    if 'jig_id' in result and (not isinstance(result['jig_id'],str) or not 1<=len(result['jig_id'])<=128):
        raise ValueError('지그 식별자가 올바르지 않습니다.')
    for key in ('jig_name', 'product'):
        if not isinstance(result.get(key), str) or not 1 <= len(result[key].strip()) <= 60:
            raise ValueError('제품 이름과 지그 프로필 이름을 1~60자로 입력하세요.')
        result[key] = result[key].strip()
    for k, lo, hi in (('long_mm',20,1500), ('short_mm',10,1500), ('surface_z_mm',-500,1000),
                     ('expected_yaw_deg',-360,360), ('yaw_tolerance_deg',1,60),
                     ('size_tolerance_pct',1,15), ('roi_x_pct',0,99), ('roi_y_pct',0,99),
                     ('roi_w_pct',1,100), ('roi_h_pct',1,100)):
        result[k] = number(value.get(k), lo, hi)
    if result['long_mm'] < result['short_mm']:
        raise ValueError('긴 변은 짧은 변보다 크거나 같아야 합니다.')
    if result['roi_x_pct'] + result['roi_w_pct'] > 100 or result['roi_y_pct'] + result['roi_h_pct'] > 100:
        raise ValueError('검출 영역은 영상의 0~100% 안에 있어야 합니다.')
    if not isinstance(value.get('geometry_verified'), bool):
        raise ValueError('지그 실측 확인 상태가 잘못됐습니다.')
    if value.get('appearance') not in ('white_frame', 'contrast'):
        raise ValueError('흰색 프레임 또는 명암 외곽 검출을 선택하세요.')
    shape=value.get('shape_model')
    if shape is None:
        shape='carrier_6jigs' if (value.get('jig_name')=='하위 지그 220×148' and value.get('appearance')=='white_frame' and result['long_mm']==220 and result['short_mm']==148) else 'rectangle'
    if shape=='carrier_6jigs':
        from jig_stl import legacy_holes
        result['hole_layout']=deepcopy(legacy_holes());shape='ring'
    if shape not in ('ring','rectangle'):
        raise ValueError('직사각형 또는 도넛형(내부 구멍 있음)을 선택하세요.')
    if 'hole_layout' in result:
        holes=result['hole_layout']
        if not isinstance(holes,list) or len(holes)>100:raise ValueError('STL 구멍 구조가 올바르지 않습니다.')
        for hole in holes:
            a=np.asarray(hole,float)
            if a.ndim!=2 or a.shape[1]!=2 or not 3<=len(a)<=1000 or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1):
                raise ValueError('STL 구멍 윤곽이 올바르지 않습니다.')
    method=value.get('detection_method','hybrid')
    if method not in ('hybrid','edge','white'):raise ValueError('색상+윤곽, 윤곽 중심, 흰색 중심 중 검출 방식을 선택하세요.')
    result['detection_method']=method
    result['shape_model']=shape
    result['appearance'] = value['appearance']
    result['geometry_verified'] = value['geometry_verified']
    return result


def profile_signature(profile):
    from vision_teaching import digest
    geometry=validate_profile(profile)
    geometry.pop('jig_id',None)  # Identity is checked separately by reference_key.
    return digest(geometry)


def roi_pixels(profile, size):
    w, h = size
    x, y = int(w*profile['roi_x_pct']/100), int(h*profile['roi_y_pct']/100)
    x2 = int(w*(profile['roi_x_pct']+profile['roi_w_pct'])/100)
    y2 = int(h*(profile['roi_y_pct']+profile['roi_h_pct'])/100)
    if x2-x < 16 or y2-y < 16:
        raise ValueError('검출 영역이 너무 작습니다.')
    return x, y, x2, y2


def rectangle_pose(corners, profile, station, size, *, enforce_heading=True):
    """Validate physical geometry AFTER undistortion and ray-plane projection."""
    from vision_teaching import pixels_on_plane, angle_delta
    p = validate_profile(profile)
    c = np.asarray(corners, float).reshape(4, 2)
    if not np.isfinite(c).all():
        raise ValueError('지그 모서리 좌표가 올바르지 않습니다.')
    world = pixels_on_plane(c, p['surface_z_mm'], station, size)[:, :2]
    # The detector supplies cyclic vertices, but winding and start corner are arbitrary.
    edges = np.roll(world, -1, axis=0)-world
    lengths = np.linalg.norm(edges, axis=1)
    if np.min(lengths) < 1:
        raise ValueError('지그 외곽이 너무 작습니다.')
    units = edges/lengths[:, None]
    if np.max(np.abs(np.sum(units*np.roll(units, -1, axis=0), axis=1))) > .12:
        raise ValueError('직각이 맞지 않습니다. 외곽 가림·윗면 높이·보정을 확인하세요.')
    pair = [float((lengths[0]+lengths[2])/2), float((lengths[1]+lengths[3])/2)]
    long_index = int(pair[1] > pair[0])
    expected = np.array([p['long_mm'] if i % 2 == long_index else p['short_mm'] for i in range(4)])
    errors = np.abs(lengths-expected)/expected
    if np.max(errors) > p['size_tolerance_pct']/100:
        raise ValueError('등록한 지그 크기와 외곽이 일치하지 않습니다.')
    axis = units[long_index]-units[(long_index+2) % 4]
    yaw = math.degrees(math.atan2(axis[1], axis[0]))
    # A rectangle is undirected. Pick the branch close to the operator's known heading.
    branches=(yaw,yaw+90,yaw+180,yaw+270) if p['long_mm']/p['short_mm']<1.15 else (yaw,yaw+180)
    yaw = min(branches, key=lambda a: abs(angle_delta(a, p['expected_yaw_deg'])))
    if enforce_heading and abs(angle_delta(yaw, p['expected_yaw_deg'])) > p['yaw_tolerance_deg']:
        raise ValueError(f'예상 방향 {p["expected_yaw_deg"]:.1f}° ±{p["yaw_tolerance_deg"]:.0f}° / 측정 방향 {angle_delta(yaw,0):.1f}° · 방향 설정을 확인하세요.')
    yaw = angle_delta(yaw, 0)
    center = world.mean(axis=0)
    return dict(x_mm=float(center[0]), y_mm=float(center[1]), yaw_deg=yaw,
                long_mm=max(pair), short_mm=min(pair), size_error_pct=float(errors.max()*100),
                profile_signature=profile_signature(p))


def outline_candidates(frame, roi=None, appearance='white_frame'):
    """Pixel-only inspection, also usable before metric calibration is available.

    White-frame masks suppress the warm wooden background in the supplied photos.
    Mild opening removes thin connections to clutter; no convex hull bridges gaps.
    Threshold variants accommodate white-balance changes, then merge the same rim.
    """
    size = frame.shape[1::-1]
    x, y, x2, y2 = roi or (0,0,*size)
    crop = frame[y:y2, x:x2]
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3,3), 0)
    _, light = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    masks = [light, cv2.bitwise_not(light)]
    edges = cv2.Canny(gray, 20, 60)
    masks.append(cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8)))
    if appearance == 'white_frame':
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:,:,1]
        for b_max in (124,126,128,130):
            mask = cv2.inRange(lab, np.array([150,115,100]), np.array([255,140,b_max]))
            mask[saturation > 60] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            masks.append(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8)))
    candidates = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:100]:
            area = cv2.contourArea(contour)
            if area < 400:
                continue
            polygon = cv2.approxPolyDP(contour, .015*cv2.arcLength(contour, True), True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            corners = polygon.reshape(4,2).astype(float)
            if (corners.min(axis=0) < 3).any() or (corners.max(axis=0) > [x2-x-4,y2-y-4]).any():
                continue
            if not .95 <= area/max(cv2.contourArea(polygon), 1) <= 1.05:
                continue
            corners += [x,y]
            duplicate = None
            for i, other in enumerate(candidates):
                # Merge only nearly identical complete outlines, not distinct objects.
                a1, a2 = cv2.contourArea(corners.astype(np.float32)), cv2.contourArea(other.astype(np.float32))
                intersection, _ = cv2.intersectConvexConvex(corners.astype(np.float32), other.astype(np.float32))
                if intersection/max(a1+a2-intersection, 1) > .94:
                    duplicate = i
                    break
            if duplicate is None:
                candidates.append(corners)
            elif cv2.contourArea(corners.astype(np.float32)) > cv2.contourArea(candidates[duplicate].astype(np.float32)):
                candidates[duplicate] = corners
    return sorted(candidates, key=lambda c: cv2.contourArea(c.astype(np.float32)), reverse=True)


def inspection_candidates(frame, profile, station=None):
    """Pixel hypotheses share the same model as execution; no area-based identity."""
    from carrier_detection import Candidate, detect_carriers
    p=validate_profile(profile);size=list(frame.shape[1::-1]);roi=roi_pixels(p,size)
    intrinsic=None
    if station:
        value=station.get('intrinsics')
        if value and value.get('camera')==station.get('camera') and value.get('size')==size:
            intrinsic=value
    if p['shape_model']=='ring':
        from ring_detection import detect_rings
        return detect_rings(frame,p,roi,intrinsic,station)
    found=[]
    appearance={'edge':'contrast','white':'white_frame'}.get(p['detection_method'],p['appearance'])
    for corners in outline_candidates(frame,roi,appearance):
        found.append(Candidate(corners.astype(np.float32),0.,True,reason='일반 사각형 후보'))
    return found


def detection_confidence(candidate, profile):
    """Relative structure/edge evidence; not a measured pose accuracy probability."""
    if profile.get('shape_model') not in ('carrier_6jigs','ring'):return 1.
    score=float(candidate.score)
    if candidate.edge_support:score=(score+min(candidate.edge_support))/2
    return float(np.clip(score,.1,1.)) if math.isfinite(score) else .1


def detect_rectangle(frame, profile, station, *, candidates=None, enforce_heading=True):
    """Return a geometric observation; motion requires separate 3 s consensus."""
    from vision_teaching import require_intrinsics, pixels_on_plane
    p = validate_profile(profile)
    size = list(frame.shape[1::-1])
    require_intrinsics(station, size)
    pixels_on_plane([[size[0]/2, size[1]/2]], p['surface_z_mm'], station, size)
    hypotheses=inspection_candidates(frame,p,station) if candidates is None else candidates
    candidates = [];reasons=[]
    for hypothesis in hypotheses:
        if hypothesis.quality not in ('precise', 'medium'):continue
        corners=hypothesis.corners
        try:
            value=rectangle_pose(corners,p,station,size,enforce_heading=enforce_heading)
            value['quality']=hypothesis.quality
            value['detection_confidence']=detection_confidence(hypothesis,p)
            candidates.append((corners,value))
        except ValueError as exc:
            reasons.append(str(exc))
    if not candidates:
        detail=' / '.join(dict.fromkeys(reasons))
        raise ValueError(('외곽 검출됨 · '+detail) if detail else '지그 미검출 · 외곽 전체, 실측 크기·높이, 검출 영역·도킹 방향을 확인하세요.')
    if len(candidates) != 1:
        raise DetectionAmbiguity('조건에 맞는 사각형이 여러 개입니다. 검출 영역이나 크기 허용 오차를 좁히세요.')
    return candidates[0]
