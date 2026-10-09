"""OpenCV's saved safe pose and its model frame, without hardware access."""
import hashlib
import json
import math
from pathlib import Path

from .robot import NAMES, pose_to_angles, profile


def load_safe_pose(path=None):
    source=Path(path or profile()['default_pose_program'])
    content=source.read_bytes();program=json.loads(content)
    steps=program['steps']
    boundaries=[(i,s['safe_boundary']) for i,s in enumerate(steps) if s.get('safe_boundary')]
    if (len(steps)<2 or boundaries!=[(0,'start'),(len(steps)-1,'end')] or
            steps[0]['pose']!=steps[-1]['pose']):
        raise ValueError('OpenCV 에피소드의 처음·끝 안전자세가 같아야 합니다.')
    pose={n+'.pos':program['steps'][0]['pose'][n+'.pos'] for n in NAMES}
    raw=pose_to_angles(pose)
    if not 0<=pose['gripper.pos']<=100:
        raise ValueError('안전자세의 그리퍼 벌림은 0~100%여야 합니다.')
    # Use the frame saved with these motor angles, not today's editable station.
    # Only the imported reference image uses this saved frame, never live IO.
    tool=program.get('vision_plan',{}).get('preview_tool',{})
    correction=tool.get('joint_offsets_deg',[0]*5)
    if not isinstance(correction,list) or len(correction)!=5 or any(
            type(v) not in (int,float) or not math.isfinite(v) or not -180<=v<=180 for v in correction):
        raise ValueError('OpenCV 안전자세에 저장된 모델 좌표를 확인하세요.')
    displayed=[q+math.radians(offset) for q,offset in zip(raw[:5],correction)]+[raw[5]]
    return dict(raw_angles=raw,model_angles=displayed,reference=dict(
        source=str(source),source_sha256=hashlib.sha256(content).hexdigest(),
        boundary='start',updated_at=program.get('updated_at'),pose=pose,
        model_offsets_deg=correction,display_only=True))
