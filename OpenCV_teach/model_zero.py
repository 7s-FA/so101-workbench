"""Map recorded motor middle ticks to the matching assembled-model posture.

LeRobot pose units and motor commands stay unchanged. This is the additive
model offset used by both FK/IK and preview, not a second EEPROM homing.
"""
from copy import deepcopy
from pathlib import Path
import json
import math

ROOT=Path(__file__).resolve().parent
JOINTS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')


def official_middle():
    return json.loads((ROOT/'calibration/model_zero/official_middle.json').read_text())


def aligned_tool(tool, calibration, middle_ticks, *, reference=None):
    from calibration_rebase import calibration_digest, motor_limits
    reference=reference or official_middle()
    if set(middle_ticks)!=set((*JOINTS,'gripper')):
        raise ValueError('6개 서보의 기록한 중간 틱이 필요합니다.')
    offsets=[]
    for name in JOINTS:
        entry=calibration[name];middle=middle_ticks[name]
        if entry['drive_mode']!=0 or not entry['range_min']<middle<entry['range_max']:
            raise ValueError(f'{name}: 영점 방향 또는 중간 틱 범위를 확인하세요.')
        displayed=(middle-(entry['range_min']+entry['range_max'])/2)*360/4095
        correction=reference['model_degrees'][name]-displayed
        if not math.isfinite(correction) or abs(correction)>180:
            raise ValueError(f'{name}: 모델 보정각이 범위를 벗어납니다.')
        offsets.append(correction)
    grip=calibration['gripper']
    if grip['drive_mode']!=0 or not grip['range_min']<middle_ticks['gripper']<grip['range_max']:
        raise ValueError('집게의 기록한 중간 틱과 범위를 확인하세요.')
    result=deepcopy(tool)
    result.update(joint_offsets_deg=[0.]*5,joint_reference_offsets_deg=offsets,
                  motor_limits_deg=motor_limits(calibration),verified=False)
    result['gripper_model_reference']={'middle_tick':middle_ticks['gripper'],
        'middle_deg':float(reference.get('gripper_middle_deg',90.)),
        'range_min':grip['range_min'],'range_max':grip['range_max'],
        'degrees_per_tick':360/4095,'calibration_sha256':calibration_digest(calibration)}
    result['motor_zero_reference']={'kind':'recorded_middle_to_official_model',
        'calibration_sha256':calibration_digest(calibration),'calibration':deepcopy(calibration),
        'middle_ticks':deepcopy(middle_ticks),'model_middle_deg':deepcopy(reference['model_degrees']),
        'model_offsets_deg':offsets.copy(),'reference_name':reference['name'],
        'physically_verified':False}
    return result


def carry_alignment(tool, old_calibration, new_calibration):
    """Preserve a known model mapping when importing another calibration file."""
    from calibration_rebase import calibration_digest,convert_value,motor_limits
    ref=tool.get('motor_zero_reference',{})
    if ref.get('calibration_sha256')!=calibration_digest(old_calibration):
        return None
    result=deepcopy(tool)
    field='joint_reference_offsets_deg' if 'joint_reference_offsets_deg' in tool else 'joint_offsets_deg'
    result[field]=[tool[field][i]-convert_value(n,0.,old_calibration,new_calibration)
                   for i,n in enumerate(JOINTS)]
    result['motor_limits_deg']=motor_limits(new_calibration);result['verified']=False
    if 'gripper_model_reference' in result:
        g=result['gripper_model_reference'];a=old_calibration['gripper'];b=new_calibration['gripper']
        g['middle_tick']=(g['middle_tick']+a['homing_offset']-b['homing_offset'])%4096
        g.update(range_min=b['range_min'],range_max=b['range_max'],calibration_sha256=calibration_digest(new_calibration))
    result['motor_zero_reference']={'kind':'preserved_encoder_model_mapping',
        'calibration_sha256':calibration_digest(new_calibration),'calibration':deepcopy(new_calibration),
        'source':deepcopy(ref),'physically_verified':False}
    return result


def saved_preview_tool(app):
    """Read the matching station even before opening the camera workspace."""
    from vision_teaching import digest
    path=Path(app.output)/'vision'/(digest(app.backend.identity)[:16]+'.json')
    try:
        stat=path.stat();key=(str(path),stat.st_mtime_ns,stat.st_size)
        cached=getattr(app,'_model_tool_cache',None)
        if cached and cached[0]==key:return cached[1]
        data=json.loads(path.read_text())
        if data.get('identity')!=app.backend.identity:return None
        tool=data['tool']
        app._model_tool_cache=(key,tool)
        return tool
    except (OSError,ValueError,KeyError):return None


def gripper_model_degrees(percent, tool):
    """Use the real encoder delta from the 90-degree assembly reference."""
    ref=(tool or {}).get('gripper_model_reference')
    if not ref:return float(percent)/100*math.degrees(1.74533)
    values=[float(ref[k]) for k in ('range_min','range_max','middle_tick','middle_deg','degrees_per_tick')]
    low,high,middle,angle,scale=values
    if not all(math.isfinite(x) for x in values) or not 0<=low<high<=4095 or not 0<=middle<=4095 or not 0<scale<1:
        raise ValueError('집게 모델 영점 또는 틱 변환값이 올바르지 않습니다.')
    tick=low+float(percent)/100*(high-low)
    return angle+(tick-middle)*scale


def model_offsets(tool):
    """Calibration coordinate conversion plus optional user corrections.

    Legacy files have only joint_offsets_deg and keep their previous meaning.
    New calibrations store the required reference separately, so zeroing manual
    corrections cannot erase the recorded motor middle.
    """
    tool=tool or {}
    def values(key):
        result=tool.get(key,[0.]*5)
        if not isinstance(result,(list,tuple)) or len(result)!=5:
            raise ValueError('모델 영점 변환은 관절 5개의 값이어야 합니다.')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or abs(x)>180 for x in result):
            raise ValueError('모델 영점 변환값이 올바르지 않습니다.')
        return result
    result=[a+b for a,b in zip(values('joint_reference_offsets_deg'),values('joint_offsets_deg'))]
    if any(abs(x)>180 for x in result):raise ValueError('모델 영점과 추가 보정각의 합이 범위를 벗어났습니다.')
    return result
