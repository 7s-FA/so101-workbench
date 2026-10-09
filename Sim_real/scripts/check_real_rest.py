"""Static gravity estimate and start/end parking audit, not measured motor load."""
import hashlib,json,pickle,xml.etree.ElementTree as ET
import numpy as np
from arm_sim.common import ROOT
from arm_sim.demo_geometry import ArmGeometry
from arm_sim.conveyor_hardware import d435_assembly

REFERENCE={'arm_a':[0,-.8,.4,1.3,-1.5221168267948966,.65],
           'arm_c':[-.65,-.8,.4,1.3,-1.5221168267948966,.65],
           'arm_b':[-np.pi/2,-.5,-1,1.5,0,.65,-.65],
           'arm_d':[np.pi/2,-.5,-1,1.5,0,.65,-.65]}

def gravity_torque(k,q,extra_mass=0.):
    geometry=ArmGeometry(k);inertials=[]
    for link in ET.parse(ROOT/k.model['urdf']).findall('link'):
        node=link.find('inertial')
        if node is not None:
            inertials.append((link.get('name'),float(node.find('mass').get('value')),np.r_[np.fromstring(node.find('origin').get('xyz','0 0 0'),sep=' '),1.]))
    if k.so and extra_mass:
        _,camera=d435_assembly()
        inertials.append(('gripper_link',extra_mass,np.r_[camera[:3,3],1.]))
    def potential(joints):
        frames=geometry.transforms(joints)
        return sum(mass*(frames[link]@com)[2]*9.81 for link,mass,com in inertials)
    q=np.array(q,float);epsilon=1e-5
    return np.array([(potential(q+np.eye(len(q))[i]*epsilon)-potential(q-np.eye(len(q))[i]*epsilon))/(2*epsilon) for i in range(5)])

def check(plan):
    failures=[];arms={}
    for name,k in plan.ik.items():
        q=plan.safe_pose(name);old=REFERENCE[name];rows=[]
        for mass in ([0.,.10,.15] if k.so else [0.]):
            before=np.abs(gravity_torque(k,old,mass));after=np.abs(gravity_torque(k,q,mass))
            rows.append({'extra_camera_mount_mass_kg_assumed':mass,'before_Nm':before.tolist(),'after_Nm':after.tolist(),'peak_reduction_percent':float(100*(1-after.max()/before.max()))})
            if after.max()>=before.max():failures.append(name+'/gravity peak not improved')
        reach=float(np.linalg.norm(k.position(q)[:2]-k.base[:2,3]))
        if reach>.06:failures.append(name+'/tool projects more than 60mm from base')
        limits=np.asarray(k.model['limits']);margin=float(np.min(np.minimum(q[:5]-limits[:5,0],limits[:5,1]-q[:5])))
        if margin<.15:failures.append(name+'/rest too close to joint limit')
        if not np.allclose(plan.arms[name].at(0),q) or not np.allclose(plan.arms[name].at(plan.duration),q):failures.append(name+'/start or end not at rest')
        arms[name]={'joints_rad':q.tolist(),'tool_horizontal_offset_m':reach,'minimum_joint_limit_margin_rad':margin,'gravity_estimates':rows}
    parking=np.asarray(plan.cfg['forklift_parking_pose']);tr=plan.vehicles['waffle_1']
    for t in (0,plan.duration):
        pose=tr.at(t)
        if not np.allclose(pose[:2],parking[:2],atol=1e-8) or abs(np.sin((pose[2]-parking[2])/2))>1e-8:failures.append('forklift/start-end parking mismatch')
    if abs(parking[1]-plan.cfg['vehicle_lane_y'])<.4:failures.append('parking too near horizontal aisle')
    if plan.lift.at(plan.duration)[0]>1e-8:failures.append('fork not lowered on return')
    for index,(name,target) in enumerate(plan.cfg['burger_parking_poses'].items()):
        pose=plan.vehicles[name].at(plan.duration)
        if not np.allclose(pose[:2],target[:2],atol=1e-8) or abs(np.sin((pose[2]-target[2])/2))>1e-8:failures.append(name+'/final parking mismatch')
        bay=plan.cfg['traffic']['waiting_bays'][index]
        deck=plan.mobile_point(name,[-.032,0,0],plan.duration)[:2]
        if not np.allclose(deck,bay['center'],atol=1e-8):failures.append(name+'/deck not centred in parking outline')
        if (np.array(plan.cfg['burger']['deck_size'])>=np.array(bay['size'])).any():failures.append(name+'/parking outline too small')
    if plan.duration>600:failures.append('process exceeds 10 minutes')
    return {'status':'blocked' if failures else 'pass','failures':failures,'arms':arms,'parking_pose':parking.tolist(),'duration_s':plan.duration,'scope':'URDF static gravity estimate and camera mass sensitivity (0/100/150g hypothetical additional mass, not measured hardware mass); no motor-current, thermal, cable-force or physical load validation. Full path/traffic checks are separate.'}

if __name__=='__main__':
    out=ROOT/'outputs/sim_real_demo';blob=(out/'plan.pkl').read_bytes();r=check(pickle.loads(blob));r['plan_sha256']=hashlib.sha256(blob).hexdigest()
    (out/'rest_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(r['status']!='pass')
