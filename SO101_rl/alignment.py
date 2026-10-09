"""A fixed simulation registration offset, separate from observed jig motion."""
from copy import deepcopy
import numpy as np


def carrier_transform(scene,offset,yaw,tilt=(0,0)):
    from scipy.spatial.transform import Rotation
    carrier=scene['carrier'];angle=np.radians(carrier['pose'][2]+yaw)
    tf=np.eye(4)
    tf[:3,:3]=Rotation.from_euler('xyz',[tilt[0],tilt[1],np.degrees(angle)],degrees=True).as_matrix()
    tf[:3,3]=np.r_[carrier['pose'][:2],carrier['bottom_z_mm']]+offset
    return tf


def aligned_scene(scene, reference, offset_mm, yaw_deg=None,tilt_deg=None):
    offset=np.asarray(offset_mm,float)
    if offset.shape!=(3,) or not np.isfinite(offset).all() or np.any(abs(offset)>30):
        raise ValueError('CAD 위치 보정은 축별 ±30mm 이내여야 합니다.')
    result=deepcopy(scene)
    old=np.asarray(scene['carrier'].get('cad_offset_mm',[0,0,0]),float)
    old_yaw=scene['carrier'].get('cad_yaw_deg',0.)
    yaw=old_yaw if yaw_deg is None else float(yaw_deg)
    if not np.isfinite(yaw) or abs(yaw)>12:raise ValueError('CAD 방향 보정은 ±12° 이내여야 합니다.')
    old_tilt=scene['carrier'].get('cad_tilt_deg',[0.,0.])
    tilt=np.asarray(old_tilt if tilt_deg is None else tilt_deg,float)
    if tilt.shape!=(2,) or not np.isfinite(tilt).all() or np.any(abs(tilt)>4):raise ValueError('CAD 기울기 보정은 축별 ±4° 이내여야 합니다.')
    delta=carrier_transform(scene,offset,yaw,tilt)@np.linalg.inv(carrier_transform(scene,old,old_yaw,old_tilt))
    for row in result['parts'].values():
        initial=delta@np.asarray(row['initial_pose_mm'],float)
        goal=reference.kin.fk(reference.steps[row['release_step']-1]['ticks'])@np.linalg.inv(
             reference.kin.fk(reference.steps[row['grasp_step']]['ticks']))@initial
        row['initial_pose_mm']=initial.tolist();row['target_pose_mm']=goal.tolist()
    if 'assembly' in result:
        for kind,row in result['parts'].items():row['target_pose_mm']=deepcopy(result['assembly']['target_poses_mm'][kind])
    result['carrier']['cad_offset_mm']=offset.tolist()
    result['carrier']['cad_yaw_deg']=yaw
    result['carrier']['cad_tilt_deg']=tilt.tolist()
    return result


def align_nests(scene, offsets_mm, output):
    """Move each complete fixture and its part together, in the arm2 frame."""
    if not offsets_mm:return scene
    import trimesh
    from core import TEACH,read_json
    from scene_assets import mesh,transform
    result=deepcopy(scene);carrier=result['carrier']
    world=carrier_transform(result,carrier['cad_offset_mm'],carrier['cad_yaw_deg'],carrier['cad_tilt_deg'])
    from scipy.spatial.transform import Rotation
    local_deltas={}
    for kind,value in offsets_mm.items():
        shift=np.asarray(value.get('offset_mm') if isinstance(value,dict) else value,float)
        angles=np.asarray(value.get('rotation_deg',[0,0,0]) if isinstance(value,dict) else [0,0,0],float)
        if kind not in result['parts'] or shift.shape!=(3,) or not np.isfinite(shift).all() or np.any(abs(shift)>20):
            raise ValueError('개별 받침 보정은 B 부품별 축 ±20mm 이내여야 합니다.')
        if angles.shape!=(3,) or not np.isfinite(angles).all() or np.any(abs(angles)>[4,4,12]):
            raise ValueError('개별 받침 회전 보정 범위를 초과했습니다.')
        tf=np.asarray(result['parts'][kind]['initial_pose_mm']);delta=np.eye(4)
        delta[:3,:3]=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
        delta[:3,3]=tf[:3,3]+shift-delta[:3,:3]@tf[:3,3]
        local_deltas['b_'+kind]=np.linalg.inv(world)@delta@world
        result['parts'][kind]['initial_pose_mm']=(delta@tf).tolist()
    assembly_path=TEACH/'assets/carrier_assembly/assembly.json'
    assembly=read_json(assembly_path);values=[]
    orient=transform([0,0,0],carrier.get('mesh_yaw_offset_deg',180))
    for part in assembly['parts']:
        value=mesh(assembly_path.parent/part['source'])
        value.apply_transform(orient@transform(part['translation_mm'],part['yaw_deg']))
        if part['name'] in local_deltas:value.apply_transform(local_deltas[part['name']])
        value.apply_scale(.001);values.append(value)
    path=output/'assets/static/aligned_nests.stl'
    trimesh.util.concatenate(values).export(path)
    row=next(row for row in result['static'] if row['name']=='registered_jig_0')
    row['mesh']=str(path)
    result['carrier']['nest_offsets_mm']=deepcopy(offsets_mm)
    return result
