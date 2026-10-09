"""CAD-derived pallet seating, nested B assembly and finite-range magnets."""
from copy import deepcopy
import numpy as np


def magnet_seats(product):
    top=float(product.bounds[1,2]);section=product.section(plane_origin=[0,0,top-.1],plane_normal=[0,0,1])
    seats=[]
    for loop in section.discrete:
        low,high=loop.min(0),loop.max(0);center=(low+high)/2;span=high-low
        if not (2<span[0]<10 and abs(span[0]-span[1])<.05 and min(abs(center[:2]))>10):continue
        hits,_,_=product.ray.intersects_location(np.array([[*center[:2],top+10]]),np.array([[0.,0.,-1.]]))
        floor=float(hits[:,2].max());depth=top-floor
        seats.append(dict(face_mm=[float(center[0]),float(center[1]),top],diameter_mm=float(span[0]),depth_mm=depth))
    if len(seats)!=4:raise ValueError('B 부품에서 네 자석 구멍을 확인할 수 없습니다.')
    return sorted(seats,key=lambda s:s['face_mm'][:2])


def build_assembly(scene,reference,settings):
    import trimesh
    from core import TEACH
    from scene_assets import mesh,seating_height,product_orientation,transform
    from so101_teach.workcell_scene import carriage_position_mm
    products={k:mesh(row['source']) for k,row in scene['parts'].items()}
    stage=scene['workcell']['linear_stage']
    fixture=mesh(TEACH/stage['a_fixture_asset'])
    seat=seating_height(products['housing'],fixture)
    base=transform(stage['position_mm'],stage['yaw_deg'])
    pallets=[base@transform([carriage_position_mm(stage)+x,0,9.4]) for x in (-40,40)]
    target=np.asarray(scene['parts']['housing']['target_pose_mm'])[:3,3]
    index=min(range(2),key=lambda i:np.linalg.norm(pallets[i][:2,3]-target[:2]))
    offset=np.asarray(settings['receiver_offset_mm'],float)
    if offset.shape!=(3,) or not np.isfinite(offset).all() or np.any(abs(offset)>30):raise ValueError('리니어 배치 보정은 축별 ±30mm 이내여야 합니다.')
    pallet=pallets[index].copy();pallet[:3,3]+=offset
    housing=pallet@transform([0,0,seat])
    tpu_z=seating_height(products['tpu'],products['housing'])
    nested=products['tpu'].copy();nested.apply_translation([0,0,tpu_z])
    orientation=product_orientation('cap',products['cap'])
    cap_mesh=products['cap'].copy();cap_mesh.apply_transform(orientation)
    cap_z=seating_height(cap_mesh,trimesh.util.concatenate([products['housing'],nested]))
    cap_local=transform([0,0,cap_z])@orientation
    targets={'housing':housing,'tpu':housing@transform([0,0,tpu_z]),'cap':housing@cap_local}
    return dict(pallet_index=index,pallet_pose_mm=pallet.tolist(),housing_seating_z_mm=seat,
                receiver_offset_mm=offset.tolist(),tpu_bottom_in_housing_mm=tpu_z,
                housing_top_mm=float(products['housing'].bounds[1,2]),
                cap_bottom_in_housing_mm=cap_z,cap_face_gap_mm=cap_z-float(products['housing'].bounds[1,2]),
                target_poses_mm={k:v.tolist() for k,v in targets.items()},
                magnet_seats={k:magnet_seats(products[k]) for k in ('housing','cap')},
                magnets=deepcopy(settings['magnets']),criteria=deepcopy(settings['criteria']),
                tpu_contact=deepcopy(settings.get('tpu_contact',{})),
                model='Rigid TPU seating; nonzero CAD cap gap; four finite-range magnet pairs; no snap constraint')


def magnetic_wrenches(housing,cap,housing_com,cap_com,assembly):
    """Equal/opposite point attractions; transforms/COM in metres, outputs N/Nm."""
    cfg=assembly['magnets'];zero=np.zeros(3)
    empty=(zero.copy(),zero.copy(),zero.copy(),zero.copy(),dict(active_pairs=0,max_pair_distance_mm=None,total_attraction_n=0.))
    if not cfg['enabled'] or -float(housing[:3,2]@cap[:3,2])<cfg['normal_alignment_min']:return empty
    def points(kind,tf):
        p=np.array([s['face_mm'] for s in assembly['magnet_seats'][kind]])/1000
        return p@tf[:3,:3].T+tf[:3,3]
    hp,cp=points('housing',housing),points('cap',cap)
    fh,th,fc,tc=zero.copy(),zero.copy(),zero.copy(),zero.copy();distances=[];total=0.
    radius=cfg['capture_distance_mm']/1000
    for p in hp:
        q=cp[np.argmin(np.linalg.norm(cp-p,axis=1))];delta=q-p;distance=float(np.linalg.norm(delta))
        distances.append(distance*1000)
        if not 1e-7<distance<radius:continue
        strength=cfg['max_pair_force_n']*(1-(distance/radius)**2)**2
        force=delta/distance*strength;fh+=force;fc-=force
        th+=np.cross(p-housing_com,force);tc+=np.cross(q-cap_com,-force);total+=strength
    return fh,th,fc,tc,dict(active_pairs=sum(0<d<cfg['capture_distance_mm'] for d in distances),
                            max_pair_distance_mm=max(distances),total_attraction_n=total)


def assembly_state(poses_mm,speeds,assembly):
    h,t,c=(poses_mm[k] for k in ('housing','tpu','cap'))
    target=np.asarray(assembly['target_poses_mm']['housing']);criteria=assembly['criteria']
    local_t=np.linalg.inv(h)@t;local_c=np.linalg.inv(h)@c
    housing_xy=float(np.linalg.norm(h[:2,3]-target[:2,3]));housing_z=float(abs(h[2,3]-target[2,3]))
    delta_yaw=np.degrees(np.arctan2((target[:3,:3].T@h[:3,:3])[1,0],(target[:3,:3].T@h[:3,:3])[0,0]))
    yaw_error=abs((delta_yaw+45)%90-45)
    seated=(housing_xy<=criteria['pallet_xy_mm'] and housing_z<=criteria['pallet_z_mm']
            and h[2,2]>np.cos(np.radians(5)) and yaw_error<5 and speeds['housing']<.02)
    inserted=(np.linalg.norm(local_t[:2,3])<=criteria['insert_xy_mm']
              and abs(local_t[2,3]-assembly['tpu_bottom_in_housing_mm'])<=criteria['insert_z_mm']
              and local_t[2,2]>np.cos(np.radians(8)) and speeds['tpu']<.02)
    face=np.mean([s['face_mm'] for s in assembly['magnet_seats']['cap']],axis=0)
    face=(local_c@np.r_[face,1])[:3];gap=float(face[2]-assembly['housing_top_mm'])
    coupled=(np.linalg.norm(face[:2])<=criteria['cap_xy_mm']
             and abs(gap-assembly['cap_face_gap_mm'])<=criteria['cap_gap_tolerance_mm']
             and -local_c[2,2]>np.cos(np.radians(5)) and speeds['cap']<.02)
    return dict(housing_on_pallet=bool(seated),tpu_in_housing=bool(inserted),cap_aligned=bool(coupled),
                housing_xy_error_mm=housing_xy,housing_z_error_mm=housing_z,
                tpu_offset_in_housing_mm=local_t[:3,3].tolist(),cap_face_gap_mm=gap,
                expected_cap_face_gap_mm=assembly['cap_face_gap_mm'],cap_face_xy_error_mm=float(np.linalg.norm(face[:2])))
