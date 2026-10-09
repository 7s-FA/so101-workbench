"""SO-101 tabletop teaching coordinates and validated persistence (millimetres)."""
from copy import deepcopy
from pathlib import Path
import json,math,sys,os,tempfile
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'OpenCV_teach'))
from vision_teaching import Kinematics,JOINTS

ORDER=('cap','ball','housing')

def transform(xyz,yaw=0.):
    tf=np.eye(4);tf[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix();tf[:3,3]=xyz;return tf

def checked(config):
    c=deepcopy(config)
    if c.get('robot')!='so101' or tuple(c.get('part_order',()))!=ORDER or set(c['parts'])!=set(ORDER):raise ValueError('SO-101 / 부품 A 3개 설정을 확인하세요.')
    def number(x,lo,hi):
        if isinstance(x,bool) or not math.isfinite(float(x)) or not lo<=float(x)<=hi:raise ValueError('입력값 범위를 확인하세요.')
    for x in c['carrier']['center_base_mm']:number(x,-600,600)
    number(c['carrier']['yaw_deg'],-360,360)
    for kind,p in c['parts'].items():
        u,v,z=p['jig_center_carrier_mm']
        number(u,-76,76);number(v,-40,40);number(z,0,30)
        for x in p['part_offset_jig_mm']:number(x,-100,100)
        for x in p['grasp_tcp_part_mm']:number(x,-100,150)
        for key in ('jig_yaw_deg','part_yaw_deg','grasp_yaw_deg'):number(p[key],-360,360)
        number(p['grasp_pitch_deg'],70,210)
        for key in ('hover_mm','lift_mm'):number(p[key],5,150)
        number(p.get('lift_tilt_deg',0.),0,30)
        number(p.get('lift_clearance_mm',40.),5,150)
        number(p.get('approach_clearance_mm',0.),0,10)
        number(p.get('hold_effort_nm',c['actuators'].get('gripper_hold_effort_nm',c['actuators']['gripper_effort_nm'])),.01,c['actuators']['gripper_effort_nm'])
        for key in ('open_percent','close_percent'):number(p[key],0,100)
        if p['close_percent']>=p['open_percent']:raise ValueError('닫힘 값은 열림 값보다 작아야 합니다.')
    number(c['contact']['static_friction'],.05,2.)
    number(c['contact']['dynamic_friction'],.05,c['contact']['static_friction'])
    number(c['contact']['thickness_mm'],.01,2.)
    return c


def load(path=None):return checked(json.loads((Path(path) if path else ROOT/'config/so101_scene.json').read_text()))

def frames(c,kind):
    p=c['parts'][kind];carrier=transform(c['carrier']['center_base_mm'],c['carrier']['yaw_deg'])
    jig=carrier@transform(p['jig_center_carrier_mm'],p['jig_yaw_deg'])
    part=jig@transform(p['part_offset_jig_mm'],p['part_yaw_deg'])
    return carrier,jig,part


def goal(c,kind,part_frame=None,approach=False):
    p=c['parts'][kind];part=frames(c,kind)[2] if part_frame is None else part_frame
    point=part[:3,3]+part[:3,:3]@np.asarray(p['grasp_tcp_part_mm'])
    if approach:point+=part[:3,0]*p.get('approach_clearance_mm',0.)
    yaw=math.degrees(math.atan2(part[1,0],part[0,0]))+p['grasp_yaw_deg']
    rotation=Rotation.from_euler('xyz',[180,p['grasp_pitch_deg']-180,yaw],degrees=True).as_matrix()
    return point,rotation


def measurements(c):
    rows={}
    for kind in ORDER:
        carrier,jig,part=frames(c,kind);point,_=goal(c,kind)
        center=part[:3,3]+part[:3,:3]@np.array([0,0,c['parts'][kind]['mesh_height_mm']/2])
        inv=np.linalg.inv(carrier)
        rows[kind]={'jig_center_carrier_mm':c['parts'][kind]['jig_center_carrier_mm'],
            'jig_origin_base_mm':jig[:3,3].tolist(),'part_origin_base_mm':part[:3,3].tolist(),
            'part_center_base_mm':center.tolist(),'part_center_carrier_mm':(inv[:3,:3]@center+inv[:3,3]).tolist(),
            'part_offset_jig_mm':c['parts'][kind]['part_offset_jig_mm'],
            'tcp_goal_base_mm':point.tolist(),'tcp_goal_carrier_mm':(inv[:3,:3]@point+inv[:3,3]).tolist(),
            'jig_distance_from_carrier_center_xy_mm':float(np.linalg.norm(jig[:2,3]-carrier[:2,3]))}
    return rows


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    fd,name=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(data)
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def save(c,path=None):atomic_json(path or ROOT/'config/so101_scene.json',checked(c))

def pose_from_angles(q):return {**{name+'.pos':math.degrees(float(q[i])) for i,name in enumerate(JOINTS)},'gripper.pos':float(q[5])/1.74533*100}

def angles_from_pose(p):return np.array([math.radians(p[name+'.pos']) for name in JOINTS]+[p['gripper.pos']/100*1.74533])


def motor_value(c,index,angle):
    """UI encoder coordinate predicted from the simulated joint, not telemetry."""
    if index==5:return float(angle)/1.74533*100
    calibration=c.get('motor_reference',{}).get('calibration')
    if not calibration:return math.degrees(angle)
    entry=calibration[JOINTS[index]]
    return round(math.degrees(angle)*4095/360+(entry['range_min']+entry['range_max'])/2)


def angle_from_motor(c,index,value):
    if index==5:return float(value)/100*1.74533
    calibration=c.get('motor_reference',{}).get('calibration')
    if not calibration:return math.radians(value)
    entry=calibration[JOINTS[index]]
    raw=round(value)
    if not entry['range_min']<=raw<=entry['range_max']:raise ValueError('새 모터 보정 범위를 벗어났습니다.')
    return math.radians((raw-(entry['range_min']+entry['range_max'])/2)*360/4095)


def joint_move_duration(start,goal,minimum=2.):
    """Cubic ease-in/out peak <=0.45 rad/s, below the simulator's 0.8 cap."""
    return max(minimum,1.5*float(np.max(np.abs(np.asarray(goal)-start)))/.45)


def solve(c,kind,seed,phase='grasp',part_frame=None):
    xyz,rotation=goal(c,kind,part_frame,approach=phase in ('hover','approach'));p=c['parts'][kind]
    if phase=='hover':xyz[2]+=p['hover_mm']
    if phase=='lift' and p.get('lift_tilt_deg',0):
        return lift_waypoints(c,kind,seed,part_frame)[-1]
    if phase=='lift':xyz[2]+=p['lift_mm']
    return angles_from_pose(Kinematics(c['tcp']).ik(xyz,rotation,pose_from_angles(seed),p['open_percent']))


def vertical_waypoints(c,kind,seed,start_height,end_height,part_frame=None,approach=False):
    """Keep target X/Y and orientation constant; at most 4 mm between IK samples."""
    xyz,rotation=goal(c,kind,part_frame,approach);kin=Kinematics(c['tcp']);q=np.array(seed,dtype=float)
    count=max(1,math.ceil(abs(end_height-start_height)/4.));result=[]
    for height in np.linspace(start_height,end_height,count+1)[1:]:
        point=xyz.copy();point[2]+=height
        q=angles_from_pose(kin.ik(point,rotation,pose_from_angles(q),seed[5]/1.74533*100))
        result.append(q)
    return result


def lift_waypoints(c,kind,seed,part_frame=None):
    """Clear the jig vertically, then tilt about the shoulder pitch axis.

    The TCP continues straight up. Tilting is independent of gripper jaw yaw;
    applying an Euler pitch in the part frame overconstrains this 5-DOF arm.
    """
    xyz,rotation=goal(c,kind,part_frame);p=c['parts'][kind]
    kin=Kinematics(c['tcp']);q=solve(c,kind,seed,'grasp',part_frame)
    q[5]=seed[5]
    lift=p['lift_mm'];clearance=p.get('lift_clearance_mm',40.)
    tilt=p.get('lift_tilt_deg',0.)
    pan=q[0]+math.radians(c['tcp']['joint_offsets_deg'][0])
    axis=np.array([math.sin(pan),math.cos(pan),0.])
    result=[];previous_tilt=0.
    for height in np.linspace(0,lift,max(1,math.ceil(lift/4.))+1)[1:]:
        blend=max(0.,min(1.,(height-clearance)/max(1.,lift-clearance)))
        maximum=tilt*blend
        point=xyz.copy();point[2]+=height
        last_error=None
        for angle in np.linspace(previous_tilt,maximum,max(1,math.ceil(maximum-previous_tilt))+1):
            desired=Rotation.from_rotvec(-axis*math.radians(angle)).as_matrix()@rotation
            try:
                candidate=angles_from_pose(kin.ik(point,desired,pose_from_angles(q),seed[5]/1.74533*100))
                # Leave 2 degrees before the hard wrist stop under load.
                if angle>0 and math.degrees(candidate[3])>kin.high[3]-2:
                    last_error=ValueError('손목 한계 여유가 부족합니다.');continue
                if angle==0 and math.degrees(candidate[3])>kin.high[3]-2 and maximum>0:continue
            except ValueError as error:last_error=error;continue
            q=candidate;previous_tilt=angle;break
        else:raise ValueError('상승 경로를 만들 수 없습니다: '+str(last_error))
        result.append(q)
    return result


def teaching_steps(c,kind,seed,part_frame=None):
    """Export usable joint candidates separately from the measured simulation pose."""
    kin=Kinematics(c['tcp']);p=c['parts'][kind]
    hover=solve(c,kind,seed,'hover',part_frame)
    grasp=solve(c,kind,hover,'grasp',part_frame)
    closed=grasp.copy();closed[5]=p['close_percent']/100*1.74533
    lifted=solve(c,kind,closed,'lift',part_frame);lifted[5]=closed[5]
    rows=[]
    for name,q in [('hover',hover),('descend',grasp),('close',closed),('lift',lifted)]:
        pose=pose_from_angles(q);tf=kin.fk(pose)
        rows.append({'step':name,'joint_pose':pose,'tcp_base_mm':tf[:3,3].tolist(),
                     'tcp_rpy_deg':Rotation.from_matrix(tf[:3,:3]).as_euler('xyz',degrees=True).tolist()})
    return rows
