"""Deterministic full-process choreography with explicit payload ownership.

Eighteen original meshes are processed in two nine-part batches. Assisted grips keep objects upright and
assembly uses the product CAD offsets. No Nav2, learned policy or hardware I/O.
"""
import bisect
import copy
import math
import json
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT
import numpy as np
from arm_sim.common import load_models, relative_target_limits
from arm_sim.demo_kinematics import ArmKinematics
from arm_sim.factory import warehouse_world_size, check_lidar_clearance

KINDS = ('housing', 'ball', 'cap')
ASSEMBLED_Z = {'housing': 0., 'ball': .003, 'cap': .0203}


def demo_layout(layout):
    cfg = copy.deepcopy(layout)
    cfg['table_height']=.23
    cfg['arms']['arm_a'].update(position=[-.95,.755],yaw=-math.pi/2)
    cfg['arms']['arm_d'].update(position=[1.30,.51],demo_pitch_degrees=[80.,30.],yaw=math.pi)
    right=cfg['arms']['arm_d']['position'][0]+.07
    left=right-.69
    bx=left+.07;cx=bx+.265
    cfg['arms']['arm_b'].update(position=[bx,.51],yaw=0.,demo_pitch_degrees=[80.,30.])
    cfg['arms']['arm_c'].update(position=[cx,.80])
    cfg['handoff']=[1.225,.655]
    cfg['jig']=[cx+.015,.54]
    cfg['map']['size']=[3.,2.]
    cfg['map']['floor_color']=[.045,.045,.045]
    cfg['show_labels']=False
    cfg['traffic']['obstacles']=[]
    cfg['finished']['center'][0]=.80
    cfg['finished']['enabled']=False
    cfg['tables']=[
        {'id':'supply_back','center':[-.95,.735],'size':[.60,.12]},
        {'id':'supply_left','center':[-1.148,.69],'size':[.19,.21]},
        {'id':'supply_right','center':[-.752,.69],'size':[.19,.21]},
        # D anchors the right corner; widen only toward B and move its work area.
        {'id':'assembly','center':[(left+right)/2,.6325],'size':[right-left,.415],
         'leg_inset':[.025,.08]}]
    cfg['clamps']={'arm_a':{'edge':'back','edge_value':.795,'table':'supply_back'},'arm_b':{'edge':'left','edge_value':left,'table':'assembly'},'arm_c':{'edge':'back','edge_value':.84,'table':'assembly'},'arm_d':{'edge':'right','edge_value':right,'table':'assembly'}}
    cfg['source'].update(count=6,pitch=.063)
    cfg['source_batches']=[]
    # Mirror the two square grids: the smaller balls fit the inner workspace,
    # while caps occupy the outer columns without staggering the rows.
    for bank,columns in enumerate(({'cap':-1.211,'housing':-1.148,'ball':-1.085},{'ball':-.815,'housing':-.752,'cap':-.689})):
        for kind,x in columns.items():
            cfg['source_batches'].append({'columns':{kind:x},'center_y':.69,'count':3,'pitch':.063,'carrier_width':.062,'start_index':bank*3})
    cfg['preparation'].update(columns={'housing':bx+.02,'ball':bx+.083,'cap':bx+.146},center_y=.6995,pitch=.063,carrier_width=.062,count=2)
    cfg['unload_dock']=[bx-.01125,.278]
    cfg['burger'].update(deck_size=[.20,.20],cargo_columns=3,cargo_pitch=.063,cargo_carrier_width=.062,
                         poses=[[-.95,.51,-math.pi/2],[-.60,.10,math.pi/2]])
    cfg['traffic']['routes']=[{'name':'MaterialU','points':[[-.95,.51],[-.95,-.35],[cfg['unload_dock'][0],-.35],cfg['unload_dock']]},
                             {'name':'ShippingU','points':[[1.22,.06],[1.22,-.35],[-.13,-.35],[-.13,.20]]}]
    cfg['traffic']['note']='Simple U-shaped shuttle with a shared lane and a waiting vehicle.'
    cfg['assumptions']=[a for a in cfg['assumptions'] if not a.startswith(('90 mm','Each mobile','Scene is'))]+['Demo: 18 initial parts in six 3-slot carriers; 63 mm pitch (13 mm gap between 50 mm parts); nine parts on each 200 mm cargo deck; zero added follower risers; assisted choreography with geometric guards.']
    # Separate the rack legs from the Waffle body and leave room beside output.
    cfg['warehouse']['center']=[-.13,.78]
    cfg['warehouse'].update(yaw=0.,entry_side='bottom')
    cfg['warehouse']['size'][0]=.40
    cfg['warehouse']['levels']=[.025,cfg['table_height']]
    cfg['warehouse']['post_height']=cfg['table_height']+.06
    cfg['warehouse']['fork_slots']={'width':.035,'center_spacing':.11,'levels':[0,1]}
    cfg['warehouse']['moving_trays']=True
    cfg['waffle']['pose']=[1.22,.06,math.pi/2]
    cfg['waffle'].update(fork_top=.024,mast_start=.013,mast_height=.28)
    cfg['docking_panels']=[
        {'id':'material_load','marker_id':10,'center':[-.95,.675,.09]},
        {'id':'material_unload','marker_id':11,'center':[cfg['unload_dock'][0],.448,.09]},
        {'id':'product_load','marker_id':12,'center':[cfg['waffle']['pose'][0],.448,.09]},
        {'id':'warehouse','marker_id':13,'center':[cfg['warehouse']['center'][0],.978,.09]}]
    for panel in cfg['docking_panels']:
        panel.update(size=[.18,.008,.18],marker_size=.07,marker_z=.12)
    check_lidar_clearance(cfg)
    return cfg


class Track:
    def __init__(self, value, smooth=False):
        self.smooth=smooth
        self.times = [0.]
        self.values = [np.array(value, dtype=float)]
        self.tangents = {}

    def add(self, t, value):
        if t < self.times[-1] - 1e-8:
            raise ValueError('Track time must not run backwards')
        value = np.asarray(value, dtype=float)
        if abs(t-self.times[-1]) < 1e-8:
            self.values[-1] = value.copy()
        else:
            self.times.append(float(t)); self.values.append(value.copy())

    def at(self, t):
        index = max(0, bisect.bisect_right(self.times, t)-1)
        if index == len(self.times)-1:
            return self.values[index].copy()
        span=getattr(self,'spans',{}).get(index)
        if span is not None:return span.at(t)
        u = np.clip((t-self.times[index])/(self.times[index+1]-self.times[index]), 0., 1.)
        tangents=getattr(self,'tangents',{})
        if index in tangents or index+1 in tangents:
            h=self.times[index+1]-self.times[index]
            a=h*tangents.get(index,np.zeros_like(self.values[index]))
            b=h*tangents.get(index+1,np.zeros_like(self.values[index]))
            d=self.values[index+1]-self.values[index]
            return self.values[index]+a*u+(10*d-6*a-4*b)*u**3+(-15*d+8*a+7*b)*u**4+(6*d-3*a-3*b)*u**5
        u = u**3*(10-15*u+6*u*u) if getattr(self,'smooth',False) else u*u*(3-2*u)
        return self.values[index]*(1-u)+self.values[index+1]*u

    def blend(self,first,last):
        """C2 continuation inside a transit; exact rest at action boundaries.

        A tangent never exceeds either adjacent secant and is zero when a
        joint reverses. This retains endpoint bounds without corner overshoot.
        """
        self.tangents.pop(first,None);self.tangents.pop(last,None)
        for i in range(first+1,last):
            before=(self.values[i]-self.values[i-1])/(self.times[i]-self.times[i-1])
            after=(self.values[i+1]-self.values[i])/(self.times[i+1]-self.times[i])
            self.tangents[i]=np.where(before*after>0,np.sign(before)*np.minimum(abs(before),abs(after)),0.)


class DemoPlan:
    def __init__(self, cfg, start_mode='preloaded'):
        if start_mode not in ('load','preloaded'):raise ValueError('Invalid demo start mode')
        self.start_mode=start_mode
        self.cfg = cfg
        self.motion_profiles=json.loads((ROOT/'config/demo_motion.json').read_text())
        models = load_models()
        self.ik = {name: ArmKinematics(models[p['model']], p, cfg['table_height']) for name,p in cfg['arms'].items()}
        self.arms = {name: Track(k.model['home'],smooth=True) for name,k in self.ik.items()}
        self.ready={}
        for name,k in self.ik.items():
            k.model=copy.deepcopy(k.model)
            bx,by=cfg['arms'][name]['position']
            rotation=Rotation.from_euler('z',math.pi/2).as_matrix()@np.diag([1.,-1.,-1.]) if k.so else np.array([[0,1.,0],[0,0,-1.],[-1.,0,0]])
            q=k.solve_pose([bx+(.05 if name=='arm_c' else 0),by-.20,.285] if k.so else [bx,by-.15,.285],rotation,k.model['home'])
            self.ready[name]=np.array([*q,*k.model['home'][5:]])
            k.model['home']=[0.,0.,-1.5,-1.5,0.,*k.model['home'][5:]] if k.so else [0.,-.5,-1.,1.5,0.,*k.model['home'][5:]]
            self.arms[name]=Track(k.model['home'],smooth=True)
        self.relative_limits = {name: np.asarray(limits) for name,k in self.ik.items()
                                if (limits := relative_target_limits(k.model)) is not None}
        self.vehicles = {f'burger_{i+1}': Track(p) for i,p in enumerate(cfg['burger']['poses'])}
        self.vehicles['waffle_1'] = Track(cfg['waffle']['pose'])
        self.lift = Track([0.])
        self.grasp_profiles=json.loads((ROOT/'config/demo_grasps.json').read_text())['profiles']
        self.grasp_events=[]
        self.rotations={}
        self.parts = {}
        for batch in cfg.get('source_batches',[cfg['source']]):
            for kind in batch['columns']:
                for i in range(batch['count']):
                    number=i+1+batch.get('start_index',0)
                    self.parts[f'{kind}_{number}']=[(0.,'world',np.array([batch['columns'][kind],batch['center_y']+(i-(batch['count']-1)/2)*batch['pitch'],cfg['table_height']+.005]))]
        self.rotations={n:[(0.,'world',np.eye(3))] for n in self.parts}
        # The first kit batch starts in the existing carrier slots. Reuse the
        # original part identities; their source nests are empty from frame 0.
        for kit in range(3) if start_mode=='preloaded' else ():
            for kind in KINDS:
                name=f'{kind}_{kit+1}'
                self.parts[name]=[(0.,'burger_1',self.cargo_offset(kind,kit))]
                self.rotations[name]=[(0.,'burger_1',Rotation.from_euler('z',math.pi).as_matrix())]
        rack_x,rack_y=cfg['warehouse']['center'];_,rack_depth=warehouse_world_size(cfg)
        inserted=rack_y-rack_depth/2+.093-(.15+cfg['waffle']['fork_length']/2)
        self.trays=[[(0.,'world',np.array([rack_x,inserted,math.pi/2,level-cfg['waffle']['fork_top']-.001]))]
                    for level in cfg['warehouse']['levels']]
        self.tray=self.trays[0]
        self.events = []
        self.activities=[];self.assembly_times=[];self.storage_time=float('inf');self.storage_times=[];self.milestones={}
        self.tasks=[];self.traffic_jobs=[];self.traffic_available=0.
        self.time = 1.
        self.completed = 0
        self.stored = 0
        self._build()
        self.events.sort(key=lambda e:e[0])
        self.duration = self.time + 3.
        self.events.append((self.time, '완료 · 원재료 18개 처리 / 완제품 6개 창고 적재', 6, 6))
        self.validate()

    def event(self, label):
        self.events.append((self.time, label, self.completed, self.stored))

    def activity(self,start,label):
        self.activities.append((start,self.time,label))

    def advance(self, t, dt, measured):
        """Retiming keeps arms, owned payloads and process events synchronized.

        Limit the next target against fresh simulator joint feedback. Clamping
        only a hand would detach its payload from the shared process timeline.
        """
        if not math.isfinite(dt) or dt < 0:
            raise ValueError('Invalid demo timestep')
        for name,limits in self.relative_limits.items():
            if name not in measured or np.shape(measured[name]) != limits.shape or not np.isfinite(measured[name]).all():
                raise ValueError(f'Missing or invalid joint feedback: {name}')
        def allowed(time):
            return all(np.all(np.abs(self.arms[name].at(time)-measured[name]) <= limits-1e-6)
                       for name,limits in self.relative_limits.items())
        end = min(self.duration, t+dt)
        if allowed(end):
            return end
        if not allowed(t):
            raise RuntimeError('Demo feedback diverged from current trajectory; restart required')
        lo,hi = t,end
        for _ in range(30):
            mid = (lo+hi)/2
            if allowed(mid):lo=mid
            else:hi=mid
        return lo

    def wait(self, seconds):
        self.time += seconds

    def motion_limits(self,arm,precision=False):
        held=any(self.entry(h,self.time)[1]==arm for h in self.parts.values())
        phase='precision' if precision else 'loaded' if held else 'empty'
        profile=self.motion_profiles[phase];grip=self.motion_profiles['gripper']
        count=len(self.arms[arm].values[-1])
        v=np.full(count,profile['velocity_rad_s']);a=np.full(count,profile['acceleration_rad_s2'])
        v[5:]=grip['velocity_rad_s'];a[5:]=grip['acceleration_rad_s2']
        return phase,v,a

    def retime_motion(self,arm,first,precision=False):
        from arm_sim.trajectory_timing import retime_tail
        phase,v,a=self.motion_limits(arm,precision)
        track=self.arms[arm]
        retime_tail(track,first,len(track.times)-1,max_velocity=v,max_acceleration=a,phase=phase)
        self.time=track.times[-1]

    def joints(self,arm,value,duration):
        track=self.arms[arm];track.add(self.time,track.at(self.time))
        delta=np.abs(np.asarray(value)-track.at(self.time))
        if np.max(delta)<1e-10:return
        _,v,a=self.motion_limits(arm)
        duration=max(.15,float(np.max(1.875*delta/v)),math.sqrt(float(np.max(5.774*delta/a))))
        self.time+=duration;track.add(self.time,value)

    def tool(self, arm, point):
        track = self.arms[arm]; k = self.ik[arm]
        start = k.position(track.at(self.time)); end = np.asarray(point)
        count = max(1, int(math.ceil(np.linalg.norm(end-start)/.04)))
        for target in np.linspace(start, end, count+1)[1:]:
            previous = track.at(self.time)
            q = k.solve(target, previous)
            value = previous.copy(); value[:5] = q
            self.joints(arm, value, max(.18, float(np.max(np.abs(value-previous)))*1.5/.8))

    def grip(self, arm, closed):
        value = self.arms[arm].at(self.time)
        if self.ik[arm].so:
            value[5] = .25 if closed else .9
        else:
            value[5:] = [.22, -.22] if closed else [.85, -.85]
        self.joints(arm, value, .5)

    def owner(self, name, owner, offset):
        history = self.parts[name]
        rotation=self.part_rotation(name,self.time)
        if owner in self.arms:
            r=self.ik[owner].transform(self.arms[owner].at(self.time))[:3,:3]
            offset=r.T@offset;rotation=r.T@rotation
        elif owner not in ('world','fall'):
            vehicle='waffle_1' if owner=='lift' else owner
            r=Rotation.from_euler('z',self.vehicles[vehicle].at(self.time)[2]).as_matrix()
            rotation=r.T@rotation
        self.rotations[name].append((self.time,owner,rotation))
        if self.time < history[-1][0]:
            raise ValueError('Payload history must be chronological')
        history.append((self.time, owner, np.asarray(offset, dtype=float)))

    @staticmethod
    def entry(history, t):
        index = bisect.bisect_right([e[0] for e in history], t)-1
        return history[max(0,index)]

    def mobile_point(self, vehicle, offset, t):
        x,y,yaw = self.vehicles[vehicle].at(t)
        dx,dy,z = offset
        return np.array([x+dx*math.cos(yaw)-dy*math.sin(yaw), y+dx*math.sin(yaw)+dy*math.cos(yaw), z])

    def payload(self, owner, offset, t):
        if owner == 'world': return offset.copy()
        if owner == 'fall':
            return np.array([offset[0],offset[1],max(offset[4],offset[2]-.5*9.81*max(0.,t-offset[3])**2)])
        if owner in self.arms:
            tf=self.ik[owner].transform(self.arms[owner].at(t))
            return tf[:3,3]+tf[:3,:3]@offset
        if owner == 'lift':
            return self.mobile_point('waffle_1', offset, t)+[0,0,float(self.lift.at(t)[0])]
        return self.mobile_point(owner, offset, t)

    def part_rotation(self,name,t):
        _,owner,rotation=self.entry(self.rotations[name],t)
        if owner in self.arms:return self.ik[owner].transform(self.arms[owner].at(t))[:3,:3]@rotation
        if owner not in ('world','fall'):
            vehicle='waffle_1' if owner=='lift' else owner
            return Rotation.from_euler('z',self.vehicles[vehicle].at(t)[2]).as_matrix()@rotation
        return rotation

    def part_at(self, name, t):
        _,owner,offset = self.entry(self.parts[name], t)
        return self.payload(owner, offset, t)

    def rotation_path(self,arm,start,rotation,point,midpoint=None):
        from scipy.spatial.transform import Slerp
        relative=Rotation.from_matrix(start[:3,:3].T@rotation).magnitude()
        if abs(relative-math.pi)<.001:
            k=self.ik[arm];previous=self.arms[arm].at(self.time)
            midpoint=(start[:3,3]+point)/2 if midpoint is None else np.asarray(midpoint)
            choices=[]
            for sign in (-1,1):
                middle=Rotation.from_euler('z',sign*math.pi/2).as_matrix()@start[:3,:3]
                try:
                    qmid=k.solve_pose(midpoint,middle,previous)
                    qend=k.solve_pose(point,rotation,qmid)
                except ValueError:continue
                if np.max(abs(qmid-previous[:5]))>math.pi+.01 or np.max(abs(qend-qmid))>math.pi+.01:continue
                score=max(abs(qmid[4]),abs(qend[4]))+.05*(np.linalg.norm(qmid-previous[:5])+np.linalg.norm(qend-qmid))
                choices.append((score,middle))
            if choices:
                middle=min(choices,key=lambda c:c[0])[1]
                return Slerp([0,.5,1],Rotation.from_matrix(np.array([start[:3,:3],middle,rotation])))
        return Slerp([0,1],Rotation.from_matrix(np.array([start[:3,:3],rotation])))

    def pose_move(self,arm,point,rotation,gripper=None,precision=False):
        from scipy.spatial.transform import Rotation,Slerp
        k=self.ik[arm];q=self.arms[arm].at(self.time);start=k.transform(q);point=np.asarray(point)
        turn=Rotation.from_matrix(start[:3,:3].T@rotation).magnitude()
        count=max(1,math.ceil(np.linalg.norm(point-start[:3,3])/.012),math.ceil(turn/.08))
        slerp=self.rotation_path(arm,start,rotation,point)
        initial_grip=q[5:].copy()
        track=self.arms[arm];track.add(self.time,q);first=len(track.times)-1
        for u in np.linspace(0,1,count+1)[1:]:
            previous=self.arms[arm].at(self.time);r=slerp(u).as_matrix()
            solved=k.solve_pose(start[:3,3]*(1-u)+point*u,r,previous)
            value=previous.copy();value[:5]=solved
            if gripper is not None:value[5:]=initial_grip*(1-u)+np.asarray(gripper)*u
            self.joints(arm,value,max(.08,float(np.max(np.abs(value-previous)))*1.5/.8))
        self.retime_motion(arm,first,precision)

    def traverse(self,arm,point,rotation):
        """Keep horizontal transfers outside the base's inner workspace."""
        track=self.arms[arm];track.add(self.time,track.at(self.time));first=len(track.times)-1
        k=self.ik[arm];tf=k.transform(self.arms[arm].at(self.time));point=np.asarray(point)
        center=k.base[:2,3].copy()
        if not k.so:center+=(k.base[:3,:3]@k.shoulder_offset)[:2]
        a=tf[:2,3]-center;b=point[:2]-center;delta=b-a
        u=np.clip(-np.dot(a,delta)/max(np.dot(delta,delta),1e-12),0,1)
        if np.linalg.norm(a+u*delta)>=.14:
            self.pose_move(arm,point,rotation);return
        radius=.18 if k.so else np.clip(min(np.linalg.norm(a),np.linalg.norm(b)),.16,.20)
        yaw=math.atan2(k.base[1,0],k.base[0,0])
        local=k.base[:2,:2].T
        al=local@a;bl=local@b
        start_local=math.atan2(al[1],al[0]);end_local=math.atan2(bl[1],bl[0])
        start=start_local+yaw;turn=end_local-start_local
        from scipy.spatial.transform import Slerp
        mid=center+radius*np.array([math.cos(start+turn/2),math.sin(start+turn/2)])
        rotations=self.rotation_path(arm,tf,rotation,point,[*mid,(tf[2,3]+point[2])/2])
        self.pose_move(arm,[*(center+radius*np.array([math.cos(start),math.sin(start)])),tf[2,3]],tf[:3,:3])
        # Sample the circular path itself. Joining coarse arc points with
        # separately interpolated chords made the elbow flex at every chord.
        for u in np.linspace(0,1,max(2,math.ceil(abs(turn)/.04))+1)[1:]:
            theta=start+turn*u;xy=center+radius*np.array([math.cos(theta),math.sin(theta)])
            previous=track.at(self.time);value=previous.copy()
            value[:5]=k.solve_pose([*xy,tf[2,3]*(1-u)+point[2]*u],rotations(u).as_matrix(),previous)
            self.joints(arm,value,.08)
        self.pose_move(arm,point,rotation)
        self.retime_motion(arm,first)

    def grasp_pose(self,arm,kind,base):
        from scipy.spatial.transform import Rotation
        k=self.ik[arm];profile=self.grasp_profiles['so101' if k.so else 'omx'][kind]
        canonical=np.diag([1.,-1.,-1.]) if k.so else np.array([[0,1.,0],[0,0,-1.],[-1.,0,0]])
        candidates=[];previous=self.arms[arm].at(self.time)
        dx=base[0]-k.base[0,3]
        threshold=.05 if arm=='arm_c' else .1
        preferred=(0. if dx<-threshold else math.pi if dx>threshold else math.pi/2) if k.so else 0.
        for angle in ((preferred,) if k.so else (0.,)):
            rotation=Rotation.from_euler('z',angle).as_matrix()@canonical
            point=np.asarray(base)-rotation@np.asarray(profile['part_base_in_tool'])
            try:
                q=k.solve_pose(point,rotation,previous)
                k.solve_pose(point+[0,0,.035 if arm!='arm_a' or kind=='ball' else .025],rotation,q)
            except ValueError:continue
            candidates.append((np.linalg.norm(q-previous[:5]),point,rotation,profile))
        if not candidates:raise ValueError(f'No reachable top-down grasp: {arm} {kind} {list(base)}; previous={previous.tolist()}')
        return min(candidates,key=lambda c:c[0])[1:]

    def set_grasp(self,arm,value):
        q=self.arms[arm].at(self.time);q[5:]=[value] if self.ik[arm].so else [value,-value]
        self.joints(arm,q,.35)

    def approach(self,arm,point,rotation,source_lane=None):
        track=self.arms[arm];track.add(self.time,track.at(self.time));first=len(track.times)-1
        current=self.ik[arm].transform(self.arms[arm].at(self.time))
        z=point[2]+(.025 if arm=='arm_a' else .035)
        if arm=='arm_a':
            # The low outer slots are reached along their own material column.
            # Cross-column transfers stay above the 30 mm balls.
            cruise=.288
            self.pose_move(arm,[*current[:2,3],cruise],current[:3,:3])
            bx,by=self.cfg['arms'][arm]['position']
            self.traverse(arm,[bx,by-.20,cruise],self.ik[arm].transform(self.ready[arm])[:3,:3])
            if source_lane is not None:
                self.traverse(arm,[point[0],source_lane,cruise],rotation)
                self.pose_move(arm,[point[0],source_lane,z],rotation)
                self.pose_move(arm,[*point[:2],z],rotation)
            else:self.traverse(arm,[*point[:2],cruise],rotation)
        else:
            if arm=='arm_c':
                # Both ends are cleared vertically. Interpolate their transit
                # heights: retaining the jig's higher clearance at the distant
                # preparation row exceeds SO-101 reach.
                self.traverse(arm,[*point[:2],z],rotation)
            else:
                if current[2,3]<z-1e-6:self.pose_move(arm,[*current[:2,3],z],current[:3,:3])
                self.traverse(arm,[*point[:2],z],rotation)
        self.retime_motion(arm,first)
        self.pose_move(arm,point,rotation,precision=True)

    def acquire(self,arm,names):
        if arm!='arm_d':self.prepare(arm)
        source=self.part_at(names[0],self.time);kind='product' if len(names)>1 else names[0].split('_')[0]
        point,rotation,profile=self.grasp_pose(arm,kind,source)
        self.set_grasp(arm,profile['open'])
        clearance=rotation[:,0]*.004 if self.ik[arm].so else np.zeros(3)
        lane=None
        if arm=='arm_a':
            number=int(names[0].rsplit('_',1)[1]);part_kind=names[0].split('_')[0]
            lane=next(c['center_y'] for c in self.cfg['source_batches'] if part_kind in c['columns'] and c.get('start_index',0)<number<=c.get('start_index',0)+c['count'])
        if arm=='arm_d' and np.allclose(self.arms[arm].at(self.time)[:5],self.ik[arm].model['home'][:5],atol=1e-5):
            # Unfold toward the actual pickup instead of first visiting the fork.
            q=self.arms[arm].at(self.time);q[:5]=self.ik[arm].solve_pose(point+[0,0,.035],rotation,q)
            self.joints(arm,q,max(.5,1.5*float(np.max(abs(q-self.arms[arm].at(self.time))))/.6))
            self.pose_move(arm,point,rotation,precision=True)
        else:self.approach(arm,point+clearance,rotation,source_lane=lane)
        closed=[profile['closed']] if self.ik[arm].so else [profile['closed'],-profile['closed']]
        self.pose_move(arm,point,rotation,closed,precision=True)
        tool=self.ik[arm].position(self.arms[arm].at(self.time))
        for name in names:self.owner(name,arm,self.part_at(name,self.time)-tool)
        self.grasp_events.append({'time':self.time,'arm':arm,'parts':list(names),'kind':names[0].split('_')[0],'profile':kind})
        lift=.035 if arm!='arm_a' or kind=='ball' else .025
        self.pose_move(arm,point+[0,0,lift],rotation,precision=True)
        if lane is not None:
            self.pose_move(arm,[point[0],lane,point[2]+lift],rotation)
            self.pose_move(arm,[point[0],lane,.288],rotation)

    def deposit(self,arm,names,destination,owner='world',offset=None):
        destination=np.asarray(destination).copy();kind='product' if len(names)>1 else names[0].split('_')[0]
        settled=None
        if arm=='arm_c' and kind=='ball' and owner=='world' and np.linalg.norm(destination[:2]-np.asarray(self.cfg['jig']))<1e-6:
            settled=destination.copy();destination[2]+=.016
        point,rotation,profile=self.grasp_pose(arm,kind,destination)
        self.approach(arm,point,rotation)
        relative={name:self.part_at(name,self.time)-self.part_at(names[0],self.time) for name in names}
        for name in names:
            self.owner(name,owner,(destination if owner=='world' else np.asarray(offset))+relative[name])
        if settled is not None:
            # Release over the housing opening; jaws never enter its rim. This
            # is an assisted free-fall trajectory, not contact-solver evidence.
            name=names[0];self.owner(name,'fall',np.r_[destination,self.time,settled[2]])
            landing=self.time+math.sqrt(2*(destination[2]-settled[2])/9.81)
            self.parts[name].append((landing,'world',settled))
            self.rotations[name].append((landing,'world',self.part_rotation(name,self.time)))
        clearance=rotation[:,0]*.004 if self.ik[arm].so else np.zeros(3)
        opened=[profile['open']] if self.ik[arm].so else [profile['open'],-profile['open']]
        self.pose_move(arm,point+clearance,rotation,opened,precision=True)
        self.pose_move(arm,point+clearance+[0,0,.025 if arm=='arm_a' else .035],rotation,precision=True)

    def transfer(self,arm,names,destination,owner='world',offset=None):
        self.acquire(arm,names)
        self.deposit(arm,names,destination,owner,offset)

    def prepare(self,arm):
        k=self.ik[arm];q=self.arms[arm].at(self.time)
        if np.allclose(q[:5],k.model['home'][:5],atol=1e-5):
            ready=self.ready[arm].copy();ready[5:]=q[5:]
            self.joints(arm,ready,max(.5,1.5*float(np.max(abs(ready-q)))/.6))

    def safe_pose(self,arm):
        held=any(self.entry(h,self.time)[1]==arm for h in self.parts.values())
        return self.ready[arm] if held else np.asarray(self.ik[arm].model['home'])

    def park(self,arm):
        k=self.ik[arm];ready=self.ready[arm];tf=k.transform(ready)
        self.traverse(arm,tf[:3,3],tf[:3,:3])
        value=self.arms[arm].at(self.time);target=ready.copy();target[5:]=value[5:]
        if np.max(abs(target[:5]-value[:5]))>math.pi:
            raise ValueError(f'Unnecessary full-turn return at {arm}: {value[:5].tolist()} -> {target[:5].tolist()}')
        self.joints(arm,target,max(.2,1.5*float(np.max(abs(target-value)))/.6))
        safe=self.safe_pose(arm).copy();safe[5:]=target[5:]
        if not np.allclose(safe[:5],target[:5]):
            self.joints(arm,safe,max(.5,1.5*float(np.max(abs(safe-target)))/.6))

    def pick(self,arm,names):
        self.acquire(arm,names);self.park(arm)

    def place(self,arm,names,destination):
        self.deposit(arm,names,destination);self.park(arm)

    def unload_kit(self,kit):
        """Retain the preparation zone through all three transfers; park once."""
        c=self.cfg['preparation']
        for kind in KINDS:
            destination=[c['columns'][kind],c['center_y']+(kit%2-.5)*c['pitch'],self.cfg['table_height']+.005]
            self.transfer('arm_b',[f'{kind}_{kit+1}'],destination)
        self.park('arm_b')

    def assemble_kit(self,kit):
        """Finish a product and hand it off before releasing the shared cell."""
        for kind in KINDS:
            jig=np.array([*self.cfg['jig'],self.cfg['table_height']+.006])+[0,0,ASSEMBLED_Z[kind]]
            self.transfer('arm_c',[f'{kind}_{kit+1}'],jig)
        self.assembly_times.append(self.time)
        self.transfer('arm_c',[f'{kind}_{kit+1}' for kind in KINDS],
                      [*self.cfg['handoff'],self.cfg['table_height']+.005])
        self.park('arm_c')

    def load_product(self,kit):
        c=self.cfg['waffle']
        offset=np.array([.15+c['fork_length']/2,(kit%3-1)*.09,c['fork_top']+.014])
        self.transfer('arm_d',[f'{kind}_{kit+1}' for kind in KINDS],
                      self.payload('lift',offset,self.time),'lift',offset)
        self.park('arm_d')

    def run_tasks(self,nodes):
        """Central workcell coordinator: dependencies + exclusive work zones.

        Every phase ends at a safe pose before its zone can be released.
        B and C retain the shared cell for a complete kit and wait empty.
        A's independent loading and D's direct loading may overlap with B.
        """
        finished={};available={n:self.arms[n].times[-1] for n in self.arms};zones={}
        while nodes:
            ready=[n for n in nodes if all(d in finished for d in n['deps'])]
            if not ready:raise ValueError('Cyclic workcell task dependencies')
            def start(n):return max(n.get('after',0),available[n['arm']],*(finished[d] for d in n['deps']),*(zones.get(z,0) for z in n['zones']))
            node=min(ready,key=lambda n:(start(n),{'arm_d':0,'arm_c':1,'arm_b':2}.get(n['arm'],3),n['id']))
            self.time=start(node);begin=self.time;node['run']()
            q=self.arms[node['arm']].at(self.time)
            if not np.allclose(q[:5],self.safe_pose(node['arm'])[:5],atol=1e-5):raise ValueError('Zone release requires verified safe pose')
            finished[node['id']]=self.time;available[node['arm']]=self.time
            for zone in node['zones']:zones[zone]=self.time
            self.tasks.append({k:node[k] for k in ('id','arm','zones','deps','label')}|{'start':begin,'end':self.time,'safe_joints':q.tolist()})
            self.activity(begin,node['label']);nodes.remove(node)
        return finished

    def drive(self, name, points):
        track = self.vehicles[name]
        track.add(self.time, track.at(self.time))
        for x,y in points:
            previous = track.at(self.time)
            distance = math.hypot(x-previous[0],y-previous[1])
            if distance < 1e-8:continue
            yaw = math.atan2(y-previous[1],x-previous[0])
            # A rearward dock departure uses reverse motion, avoiding a loaded
            # vehicle turning through a workbench beside the dock.
            delta = (yaw-previous[2]+math.pi)%(2*math.pi)-math.pi
            if abs(delta)>math.pi*.75:
                yaw += math.pi
                delta = (yaw-previous[2]+math.pi)%(2*math.pi)-math.pi
            yaw = previous[2]+delta
            self.time += max(.15, abs(delta)*1.5/.8)
            track.add(self.time,[previous[0],previous[1],yaw])
            self.time += max(.25,distance*1.5/.20)
            track.add(self.time,[x,y,yaw])

    def face(self,name,yaw):
        track=self.vehicles[name];p=track.at(self.time);track.add(self.time,p)
        angle=(yaw-p[2]+math.pi)%(2*math.pi)-math.pi
        self.time+=max(.25,abs(angle)*1.5/.8)
        track.add(self.time,[*p[:2],p[2]+angle])

    def raise_lift(self,height):
        self.lift.add(self.time,self.lift.at(self.time))
        self.time += max(.5,abs(height-float(self.lift.at(self.time)[0]))/.045)
        self.lift.add(self.time,[height])

    def cargo_offset(self,kind,kit):
        return np.array([-.032+(KINDS.index(kind)-1)*self.cfg['burger']['cargo_pitch'],(kit-1)*self.cfg['burger']['cargo_pitch'],self.cfg['burger']['deck_top']+.006])

    def traffic_job(self,vehicle,label,action):
        requested=self.time
        self.time=max(self.time,self.traffic_available)
        begin=self.time;initial=self.vehicles[vehicle].at(begin).tolist()
        action()
        self.traffic_jobs.append({'id':f'traffic_{len(self.traffic_jobs)}','vehicle':vehicle,
            'label':label,'requested':requested,'start':begin,'end':self.time,
            'start_pose':initial,'end_pose':self.vehicles[vehicle].at(self.time).tolist()})
        self.traffic_available=self.time;self.activity(begin,label)

    def loaded_route(self,vehicle):
        load=self.cfg['burger']['poses'][0][:2];dock=self.cfg['unload_dock']
        self.drive(vehicle,[(load[0],-.35),(dock[0],-.35),(dock[0],.05)])
        self.face(vehicle,-math.pi/2);self.drive(vehicle,[dock])

    def queue_route(self,vehicle):
        dock=self.cfg['unload_dock']
        self.drive(vehicle,[(dock[0],-.35),(-.60,-.35),self.cfg['burger']['poses'][1][:2]])

    def supply_route(self,vehicle):
        load=self.cfg['burger']['poses'][0][:2]
        self.drive(vehicle,[(-.60,-.15),(load[0],-.15)])
        self.face(vehicle,self.cfg['burger']['poses'][0][2]);self.drive(vehicle,[load])

    def process_batch(self,first,arrived,lift_ready):
        nodes=[];previous_b=None;previous_c=None;previous_d=None
        def add(ident,arm,deps,zones,label,run):
            nodes.append({'id':ident,'arm':arm,'deps':[d for d in deps if d],'zones':zones,
                          'label':label,'run':run,'after':lift_ready if arm=='arm_d' else arrived})
        for kit in range(first,first+3):
            batch=f'b_unload_{kit}'
            add(batch,'arm_b',[previous_b,previous_c],['bc_transfer'],
                f'B · 세트 {kit+1} 부품 3개 연속 하차 후 안전 복귀',lambda i=kit:self.unload_kit(i))
            previous_b=batch
            handoff=f'c_assemble_kit_{kit}';ds=f'd_load_{kit}'
            add(handoff,'arm_c',[batch,previous_c,previous_d],['central','bc_transfer'],
                f'C · 제품 {kit+1} 연속 조립·인계 후 안전 복귀',lambda i=kit:self.assemble_kit(i))
            add(ds,'arm_d',[handoff,previous_d],['central'],
                f'D · 완제품 {kit+1} 지게차 직접 상차 후 안전 복귀',lambda i=kit:self.load_product(i))
            previous_c=handoff;previous_d=ds
        completed=self.run_tasks(nodes)
        self.milestones.setdefault('kits_ready',[]).extend(completed[f'b_unload_{i}'] for i in range(first,first+3))
        self.milestones['b_windows']=[[t['start'],t['end']] for t in self.tasks if t['arm']=='arm_b']
        self.milestones['c_windows']=[[t['start'],t['end']] for t in self.tasks if t['arm']=='arm_c']
        return completed[previous_b],completed[previous_d]

    def _build(self):
        c=self.cfg;z=c['table_height']+.005;load=c['burger']['poses'][0][:2];dock=c['unload_dock']
        if self.start_mode=='load':
            start=self.time;self.event('A · 1번 차량에 원재료 9개 상차')
            for kit in range(3):
                for kind in KINDS:
                    offset=self.cargo_offset(kind,kit)
                    self.transfer('arm_a',[f'{kind}_{kit+1}'],self.mobile_point('burger_1',offset,self.time),'burger_1',offset)
            self.park('arm_a')
            self.milestones['first_loaded']=self.time
            self.activity(start,'A · 1번 차량 9개 상차')
        else:
            self.milestones['first_loaded']=0.
            self.events.append((0.,'운반로봇 1 · 재료 9개 사전 적재 완료',0,0))
        start=self.time;self.event('운반로봇 1 · 9개 적재 / U자 이동')
        self.traffic_job('burger_1','운반로봇 1 · U자 운반',lambda:self.loaded_route('burger_1'))
        arrived=self.time;self.milestones['first_arrived']=arrived

        # Start the forklift when the material carrier arrives. B/C can work
        # concurrently; D requires the retrieved tray at the loading dock.
        self.traffic_job('waffle_1','지게차 · 창고의 첫 빈 용기 회수',self.fetch_first_tray)
        self.event('지게차 · 직접 상차를 위한 용기 높이 맞춤')
        self.raise_lift(c['table_height']-c['waffle']['fork_top']-.005)
        lift_ready=self.time
        self.milestones['first_tray_ready']=lift_ready

        # A prepares the next full load while station 2 runs its work orders.
        self.time=arrived;start=self.time
        self.traffic_job('burger_2','운반로봇 2 · 상차 위치 진입',lambda:self.supply_route('burger_2'))
        start=self.time;self.event('A · 2번 차량에 다음 배치 9개 상차')
        for kit in range(3):
            for kind in KINDS:
                offset=self.cargo_offset(kind,kit)
                self.transfer('arm_a',[f'{kind}_{kit+4}'],self.mobile_point('burger_2',offset,self.time),'burger_2',offset)
        second_loaded=self.time;self.milestones['second_loaded']=second_loaded
        self.activity(start,'A · 다음 배치 9개 상차');self.milestones['a_second_window']=[start,self.time]
        self.park('arm_a')

        self.time=arrived
        b_finished,d_finished=self.process_batch(0,arrived,lift_ready)
        self.time=b_finished;start=self.time
        self.traffic_job('burger_1','운반로봇 1 · 빈 차량 복귀',lambda:self.queue_route('burger_1'))
        returned=self.time
        self.time=max(d_finished,returned);self.completed=3
        self.event('지게차 · 운반 높이로 하강 / 창고 이동')
        self.raise_lift(0.)
        self.traffic_job('waffle_1','지게차 · 1차 출하 / 빈 용기 회수·복귀',lambda:self.ship_batch(0,0,True))
        self.time=max(self.time,second_loaded,self.arms['arm_a'].times[-1])
        self.event('운반로봇 2 · 나머지 9개 운반 / 2차 공정 시작')
        self.traffic_job('burger_2','운반로봇 2 · 나머지 9개 U자 운반',lambda:self.loaded_route('burger_2'))
        arrived=self.time;self.milestones['second_arrived']=arrived
        self.raise_lift(c['table_height']-c['waffle']['fork_top']-.005)
        b_finished,d_finished=self.process_batch(3,arrived,self.time)
        self.time=b_finished
        self.traffic_job('burger_2','운반로봇 2 · 빈 차량 상차 위치 복귀',lambda:self.empty_supply_return('burger_2'))
        self.time=max(self.time,d_finished);self.completed=6
        self.event('지게차 · 2차 출하를 위한 운반 높이로 하강')
        self.raise_lift(0.)
        self.traffic_job('waffle_1','지게차 · 2차 출하 / 위층 적재·복귀',lambda:self.ship_batch(1,1))
        # All arms return to their initial joint poses for a repeatable restart.
        for arm,k in self.ik.items():
            self.joints(arm,k.model['home'],3.)

    def empty_supply_return(self,vehicle):
        dock=self.cfg['unload_dock'];load=self.cfg['burger']['poses'][0][:2]
        self.drive(vehicle,[(dock[0],-.35),(load[0],-.35)])
        self.face(vehicle,self.cfg['burger']['poses'][0][2]);self.drive(vehicle,[load])

    def fetch_first_tray(self):
        """Take the original lower-shelf empty tray after material arrival."""
        c=self.cfg;rack_x,rack_y=c['warehouse']['center']
        _,depth=warehouse_world_size(c);front=rack_y-depth/2
        approach=front-(.15+c['waffle']['fork_length'])-.04
        inserted=front+.093-(.15+c['waffle']['fork_length']/2)
        self.milestones['first_empty_pickup_start']=self.time
        self.event('지게차 · 아래층 빈 용기를 가지러 이동')
        self.drive('waffle_1',[(c['waffle']['pose'][0],-.35)])
        self.face('waffle_1',math.pi)
        self.drive('waffle_1',[(rack_x,-.35)])
        self.face('waffle_1',math.pi/2)
        self.drive('waffle_1',[(rack_x,approach)])
        height=c['warehouse']['levels'][0]-c['waffle']['fork_top']-.001
        self.raise_lift(height)
        self.drive('waffle_1',[(rack_x,inserted)])
        self.trays[0].append((self.time,'lift',np.zeros(3)))
        self.milestones['first_empty_tray_retrieved']=self.time
        self.raise_lift(height+.008)
        self.drive('waffle_1',[(rack_x,approach)])
        self.raise_lift(0.)
        self.event('지게차 · 빈 용기를 상차 위치로 운반')
        self.drive('waffle_1',[(rack_x,-.35)])
        self.face('waffle_1',math.pi)
        self.drive('waffle_1',[(c['waffle']['pose'][0],-.35)])
        self.face('waffle_1',math.pi/2)
        self.drive('waffle_1',[c['waffle']['pose'][:2]])

    def ship_batch(self,batch,level,retrieve_empty=False):
        c=self.cfg;rack_x,rack_y=c['warehouse']['center']
        _,depth=warehouse_world_size(c);front=rack_y-depth/2
        approach=front-(.15+c['waffle']['fork_length'])-.04
        inserted=front+.093-(.15+c['waffle']['fork_length']/2)
        self.drive('waffle_1',[(c['waffle']['pose'][0],-.35)])
        # Turn away from the right map edge before joining the common lane.
        self.drive('waffle_1',[(rack_x,-.35)])
        self.face('waffle_1',math.pi/2)
        self.drive('waffle_1',[(rack_x,approach)])
        shelf=c['warehouse']['levels'][level]
        elevated=shelf-c['waffle']['fork_top']-.001+.008
        self.raise_lift(elevated)
        self.drive('waffle_1',[(rack_x,inserted)])
        self.raise_lift(elevated-.008);self.wait(.7)
        for kind in KINDS:
            for i in range(1+batch*3,4+batch*3):
                name=f'{kind}_{i}';self.owner(name,'world',self.part_at(name,self.time))
        self.trays[batch].append((self.time,'world',np.array([*self.vehicles['waffle_1'].at(self.time),float(self.lift.at(self.time)[0])])))
        self.stored=(batch+1)*3;self.storage_time=self.time;self.storage_times.append(self.time)
        self.event('창고 적재 완료 · 포크 후진 및 복귀')
        self.drive('waffle_1',[(rack_x,approach)])
        self.raise_lift(0.)
        if retrieve_empty:
            self.event('지게차 · 위층 빈 용기 회수')
            pickup_height=c['warehouse']['levels'][1]-c['waffle']['fork_top']-.001
            self.raise_lift(pickup_height)
            self.drive('waffle_1',[(rack_x,inserted)])
            self.trays[1].append((self.time,'lift',np.zeros(3)))
            self.raise_lift(pickup_height+.008)
            self.drive('waffle_1',[(rack_x,approach)])
            self.raise_lift(0.)
            self.milestones['empty_tray_retrieved']=self.time
        self.drive('waffle_1',[(rack_x,-.35)])
        self.face('waffle_1',math.pi)
        self.drive('waffle_1',[(c['waffle']['pose'][0],-.35)])
        self.face('waffle_1',math.pi/2)
        self.drive('waffle_1',[c['waffle']['pose'][:2]])

    def sample(self,t):
        if not math.isfinite(t):raise ValueError('Invalid demo time')
        t=max(0.,min(float(t),self.duration))
        eligible=[e for e in self.events if e[0]<=t]
        event=eligible[-1] if eligible else (0.,'데모 준비',0,0)
        active=[label for start,end,label in self.activities if start<=t<end]
        tasks=[task for task in self.tasks if task['start']<=t<task['end']]
        zones={}
        for task in tasks:
            for zone in task['zones']:
                if zone in zones:raise ValueError('Shared work zone has multiple owners')
                zones[zone]=task['arm']
        states={}
        for name,tr in self.arms.items():
            task=next((task for task in tasks if task['arm']==name),None)
            held=any(self.entry(h,t)[1]==name for h in self.parts.values())
            expected=self.ready[name] if held else self.ik[name].model['home']
            safe=bool(np.allclose(tr.at(t)[:5],expected[:5],atol=1e-4))
            states[name]={'phase':task['label'] if task else (next((label for start,end,label in self.activities if start<=t<end and label.startswith(name[-1].upper()+' ·')),None) or (('부품 유지 대기' if held else '접은 자세 대기') if safe else '이동 중')),
                          'safe_pose':safe,'task_id':task['id'] if task else None}
        return {'time':t,'joints':{n:tr.at(t) for n,tr in self.arms.items()},
                'vehicles':{n:tr.at(t) for n,tr in self.vehicles.items()},
                'parts':{n:self.part_at(n,t) for n in self.parts},
                'part_rotations':{n:self.part_rotation(n,t) for n in self.parts},
                'owners':{n:self.entry(h,t)[1] for n,h in self.parts.items()},
                'lift':float(self.lift.at(t)[0]),'tray':self.entry(self.tray,t),
                'trays':[self.entry(h,t) for h in self.trays],
                'label':' | '.join(active) if active else event[1],
                'assembled':int(sum(time<=t for time in self.assembly_times)),'stored':int(3*sum(time<=t for time in self.storage_times)),'goal':6,
                'cargo':{n:sum(self.entry(h,t)[1]==n for h in self.parts.values()) for n in ('burger_1','burger_2')},
                'coordination':{'arms':states,'zones':zones,'tasks':self.tasks},
                'traffic':{'owner':next((j['vehicle'] for j in self.traffic_jobs if j['start']<=t<j['end']),None),'jobs':self.traffic_jobs,'waiting':[j['vehicle'] for j in self.traffic_jobs if j['requested']<=t<j['start']]},
                'progress':t/self.duration,'done':t>=self.duration}

    def validate(self):
        if len(self.parts)!=18:raise ValueError('Expected eighteen unique original parts')
        for arm,track in self.arms.items():
            limits=np.array(self.ik[arm].model['limits'])
            for span in {id(s):s for s in getattr(track,'spans',{}).values()}.values():
                if (span.lower<limits[:,0]-1e-6).any() or (span.upper>limits[:,1]+1e-6).any():
                    raise ValueError('Spline joint limit violation: '+arm)
            for q in track.values:
                if not np.isfinite(q).all() or (q<limits[:,0]-1e-6).any() or (q>limits[:,1]+1e-6).any():
                    raise ValueError('Joint limit violation: '+arm)
        final=self.sample(self.duration)
        if final['stored']!=6 or final['assembled']!=6 or any(final['cargo'].values()) or any(owner!='world' for owner in final['owners'].values()):
            raise ValueError('Incomplete warehouse delivery')
        for i in range(1,7):
            base=final['parts'][f'housing_{i}']
            for kind in KINDS:
                if np.linalg.norm(final['parts'][f'{kind}_{i}']-base-[0,0,ASSEMBLED_Z[kind]])>.0001:
                    raise ValueError('Assembly separation or wrong part identity')
        return {'unique_parts':18,'assembled':6,'stored':6,'next_batch_loaded':0,'shipping_trays':2,'duration_s':self.duration,
                'motion_profiles':self.motion_profiles,'milestones':self.milestones,
                'max_ik_error_m':{n:k.max_error for n,k in self.ik.items()},
                'assisted_grip':True,'navigation':'scripted waypoints; not Nav2',
                'physics_success_claim':False}
