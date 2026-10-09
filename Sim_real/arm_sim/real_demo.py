"""Revised four-arm assisted process. No hardware commands or contact-force claim."""
import copy,json,math
import numpy as np
from scipy.spatial.transform import Rotation
from arm_sim.common import ROOT,relative_target_limits
from arm_sim.conveyor_scene import preview_layout
from arm_sim.factory_demo import DemoPlan,Track,KINDS
from arm_sim.demo_kinematics import ArmKinematics

ASSEMBLY={'housing':0.,'ball':.003053,'cap':json.loads((ROOT/'config/conveyor_demo.json').read_text())['cap_orientation']['assembly_base_z_m']}

def real_layout():
    c=preview_layout()
    c['arms']['arm_a'].update(position=[-1.25453757,.49],yaw=-math.pi/4)
    c['arms']['arm_c']['position'][0]=.86
    # Keep the distal mass above the base; A/C share a south-facing alignment.
    for name in ('arm_a','arm_c'):
        c['rest_joints'][name][1:4]=[-.22,-1.14,-.69]
        c['rest_joints'][name][0]=-math.pi/4 if name=='arm_a' else 0.
    for name in ('arm_b','arm_d'):
        c['rest_joints'][name][1:4]=[-.30,-1.28,.17]
    c['arm_fixtures']['arm_a'].update(edge_inset=.00043023255,clamp_pad_size=.006,clamp_jaw_width=.008,clamp_upper_thickness=.004,custom_pads=[{'side':'left','position':[-1.28918580228,.46949390335]},{'side':'left','position':[-1.2410281088646993,.5289653052580165]}])
    c['rest_joints']['arm_b'][0]=-math.pi/2
    c['rest_joints']['arm_d'][0]=math.pi/2
    c['source_banks']=[[-1.065,.510]];c['source_pitch']=.070;c['fixture_clearance']=.002;c['cargo_table_clearance']=.009
    c['burger'].update(cargo_pitch=.070,deck_size=[.218,.218])
    # Turn the chassis toward the front camera's docking wall, keeping the
    # deck centre and each part's world pickup position unchanged.
    c['loading_dock']=[-1.20,.309];c['unload_dock'][1]=.314
    c['burger']['jig_yaw']=math.pi
    c['burger']['poses'][0]=[*c['loading_dock'],math.pi/2]
    # Bays mark the deck centre, which is 32mm behind the chassis origin.
    bay=c['traffic']['waiting_bays'][0]['center']
    c['burger_parking_poses']={'burger_1':[bay[0],bay[1]+.032,math.pi/2],
                               'burger_2':[*c['loading_dock'],math.pi/2]}
    c['traffic']['waiting_bays'].append({'center':[c['loading_dock'][0],c['loading_dock'][1]-.032],'size':[.240,.240]})
    c['dock_retreat']={'burger':.15,'waffle':.20}
    c['vehicle_lane_y']=-.42
    c['preparation_slots']=[{'kind':k,'kit':0,'position':xy} for k,xy in zip(KINDS,([.71,.640],[.64,.640],[.71,.570]))]
    c['handoff']=[1.09,.605];c['jig']=c['assembly_jig']
    c['trident_supports']=True;c['shipping_support_size']=[.238,.125];c['shipping_height']=.095
    c['fork_insertion_clearance']=.005
    c['vehicle_motion']={'linear_velocity':.225,'linear_acceleration':.128,'angular_velocity':.55,'angular_acceleration':.40,'dock_velocity':.0896,'dock_acceleration':.075264}
    c['lift_motion']={'velocity':.0392,'acceleration':.075264}
    c['finished_container']={'size':[.208,.072,.003],'pitch':.068,'jig':'housing_jig'}
    c['waffle'].update(fork_length=.140,tray_offset=.285,lift_stroke=.100,fork_top=.004,fork_thickness=.003,slider_offset=.048,fork_max_top=.104)
    for rack in c['warehouses']:rack.update(size=[.268,.125],levels=[.020,.095],post_height=.095,shelf_thickness=.005)
    c['shipping_vehicle_pose']=[c['shipping_dock'][0],c['shipping_dock'][1]-.285,math.pi/2]
    c['waffle']['pose']=[c['warehouses'][0]['center'][0],c['warehouses'][0]['center'][1]-.285-c['dock_retreat']['waffle'],math.pi/2]
    c['forklift_parking_pose']=list(c['waffle']['pose'])
    c['process_reference']['runtime_implemented']=True
    c['traffic']['mode']='exclusive scripted U routes'
    c['traffic']['routes'][0]['points']=[c['loading_dock'],[c['loading_dock'][0],-.42],[c['unload_dock'][0],-.42],c['unload_dock']]
    c['docking_panels'][0]['center'][0]=c['loading_dock'][0]
    c['traffic']['routes'][1]['points']=[[c['shipping_dock'][0],c['shipping_vehicle_pose'][1]],[c['shipping_dock'][0],c['vehicle_lane_y']],[-.01,c['vehicle_lane_y']]]
    return c

class RealPlan(DemoPlan):
    def __init__(self,cfg=None):
        self.cfg=cfg or real_layout();c=self.cfg;self.start_mode='preloaded'
        self.motion_profiles=json.loads((ROOT/'config/demo_motion.json').read_text())
        models=json.loads((ROOT/'assets/generated/conveyor_demo/models.json').read_text())
        self.ik={n:ArmKinematics(copy.deepcopy(models[p['model']]),p,c['table_height']) for n,p in c['arms'].items()}
        self.arms={};self.ready={}
        for n,k in self.ik.items():
            
            if not k.so:k.shoulder_offset=np.zeros(3)
            if not k.so:k.model['limits'][:5]=[[max(lo,-math.pi),min(hi,math.pi)] for lo,hi in k.model['limits'][:5]]
            k.model['home']=c['rest_joints'][n];self.ready[n]=np.array(k.model['home']);self.arms[n]=Track(k.model['home'],smooth=True)
        self.relative_limits={n:np.array(l) for n,k in self.ik.items() if (l:=relative_target_limits(k.model)) is not None}
        self.vehicle_moves=[];self.park_events=[]
        self.vehicles={f'burger_{i+1}':Track(p) for i,p in enumerate(c['burger']['poses'])};self.vehicles['waffle_1']=Track(c['waffle']['pose']);self.lift=Track([0.],smooth=True)
        for track in self.vehicles.values():track.smooth=True
        self.seating=json.loads((ROOT/'config/conveyor_demo.json').read_text())['jig_seating']
        profiles=json.loads((ROOT/'config/conveyor_grasps.json').read_text())['profiles'];profiles['ball']['open']=profiles['ball']['closed']+.08;profiles['product']=copy.deepcopy(profiles['housing']);profiles['product']['part_base_in_tool'][2]=.022
        self.grasp_profiles={'so101':profiles,'omx':json.loads((ROOT/'config/demo_grasps.json').read_text())['profiles']['omx']}
        self.grasp_profiles['omx']['cap']['closed']+=.006
        from arm_sim.grasp_geometry import GraspMeshes,triangle_model,stl_triangles
        from arm_sim.conveyor_hardware import d435_assembly,d435_vertices
        mount_tf,camera_tf=d435_assembly()
        self.camera_shapes=[(triangle_model(d435_vertices()),camera_tf),(triangle_model(stl_triangles(ROOT/'vendor/so_arm100/Optional/Wrist_Cam_Mount_RealSense_D435/stl/Wrist_cam_mount_RealSense_D435.stl')*.001),mount_tf)]
        self.grasp_meshes={n:GraspMeshes(k) for n,k in self.ik.items()}
        self.parts={};self.rotations={};self.grasp_events=[];self.phases=[]
        for kit in range(6):
            for col,kind in enumerate(KINDS):
                name=f'{kind}_{kit+1}'
                if kit<3:owner='burger_1';p=self.cargo_offset(kind,kit);r=Rotation.from_euler('z',-math.pi/2).as_matrix()
                else:
                    owner='world';x,y=c['source_banks'][0];p=self.seat(kind,[x+(col-1)*c['source_pitch'],y+(kit-4)*c['source_pitch']],c['table_height']);r=np.eye(3)
                self.parts[name]=[(0.,owner,p)];self.rotations[name]=[(0.,owner,r)]
        # Tray coordinates are centre XY and bottom Z, with world yaw.
        self.trays=[[(0.,'world',np.array([*r['center'],r['levels'][-1],0.]))] for r in c['warehouses']];self.tray=self.trays[0]
        self.events=[];self.activities=[];self.assembly_times=[];self.storage_times=[];self.storage_time=float('inf');self.milestones={};self.tasks=[];self.traffic_jobs=[];self.traffic_available=0.;self.time=1.;self.completed=0;self.stored=0
        self._build();del self.grasp_meshes;del self.camera_shapes;self.duration=self.time+1.;self.events.sort(key=lambda e:e[0]);self.validate()

    def seat(self,kind,xy,z):return np.r_[np.array(xy)+self.seating['offset_xy'],z+self.seating['contact_height'][kind]+self.seating['render_gap']]
    def cargo_offset(self,kind,kit):
        from arm_sim.conveyor_scene import cargo_jig_pose
        pos,yaw=cargo_jig_pose(self.cfg['burger'],KINDS.index(kind),kit)
        offset=Rotation.from_euler('z',yaw).apply([*self.seating['offset_xy'],0.])
        return np.array(pos)+offset+[0,0,self.seating['contact_height'][kind]+self.seating['render_gap']]
    def prepare(self,arm):pass
    def safe_pose(self,arm):return np.array(self.cfg['rest_joints'][arm])
    def park(self,arm):
        self.joints(arm,self.safe_pose(arm),1.);self.park_events.append({'arm':arm,'time':self.time})

    @staticmethod
    def solve_motion_pose(k,point,rotation,previous):
        try:return k.solve_pose(point,rotation,previous)
        except ValueError:
            # A five-axis arm cannot satisfy every full six-axis transit pose.
            # Keep position within 0.05mm and allow at most 3 degrees of tilt
            # during travel; pick/release endpoints still use exact grasp IK.
            from scipy.optimize import least_squares
            lim=np.asarray(k.model['limits'][:5]);point=np.asarray(point)
            def residual(q):
                tf=k.transform(q)
                return np.r_[1000*(tf[:3,3]-point),Rotation.from_matrix(rotation.T@tf[:3,:3]).as_rotvec()]
            sol=least_squares(residual,np.clip(previous[:5],lim[:,0]+1e-7,lim[:,1]-1e-7),bounds=(lim[:,0],lim[:,1]),max_nfev=180)
            tf=k.transform(sol.x)
            if np.linalg.norm(tf[:3,3]-point)>.00005 or Rotation.from_matrix(rotation.T@tf[:3,:3]).magnitude()>math.radians(3):raise ValueError('Transit exceeds position/3-degree orientation tolerance')
            return sol.x

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
            solved=self.solve_motion_pose(k,start[:3,3]*(1-u)+point*u,r,previous)
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
        radius=(.18 if arm=="arm_a" else .24) if k.so else np.clip(min(np.linalg.norm(a),np.linalg.norm(b)),.16,.20)
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
            value[:5]=self.solve_motion_pose(k,[*xy,tf[2,3]*(1-u)+point[2]*u],rotations(u).as_matrix(),previous)
            self.joints(arm,value,.08)
        self.pose_move(arm,point,rotation)
        self.retime_motion(arm,first)

    def grasp_pose(self,arm,kind,base):
        k=self.ik[arm];p=self.grasp_profiles['so101' if k.so else 'omx'][kind];q=self.arms[arm].at(self.time)
        canonical=np.diag([1.,-1.,-1.]) if k.so else np.array([[0,1.,0],[0,0,-1.],[-1.,0,0]])
        candidates=[]
        angles=(0,math.pi/2,-math.pi/2,math.pi)
        if kind=='ball' and k.so:
            # Ball cradles are open on one side; align the asymmetric jaw with it.
            on_cargo=np.asarray(base)[2]<self.cfg['table_height']-.005
            angles=(math.pi,) if on_cargo else ((math.pi/2,-math.pi/2) if np.asarray(base)[2]>self.cfg['table_height']+.02 else (-math.pi/2,))
        for angle in angles:
            r=Rotation.from_euler('z',angle).as_matrix()@canonical;point=np.asarray(base)-r@p['part_base_in_tool']
            try:
                at=k.solve_pose(point,r,q);above=k.solve_pose(point+[0,0,.035],r,at)
                if np.max(abs(above-q[:5]))>math.pi+.01:continue
                import fcl
                from itertools import combinations
                g=self.grasp_meshes[arm];probe=np.r_[at,[p['closed']] if k.so else [p['closed'],-p['closed']]];meshes=g.at(probe)
                if any(fcl.collide(a[2],b[2],fcl.CollisionRequest(),fcl.CollisionResult()) for a,b in combinations(meshes,2) if frozenset((a[0],b[0])) in g.self_pairs):continue
                if k.so:
                    from arm_sim.grasp_geometry import collision_object
                    from arm_sim.conveyor_hardware import clamp_boxes
                    clamp_obs=[collision_object(fcl.Box(*size),np.eye(3),pos) for _,pos,size in clamp_boxes(self.cfg,arm)]
                    collision=False;check_q=probe.copy()
                    for dz in np.linspace(.035,0,8):
                        check_q[:5]=k.solve_pose(point+[0,0,dz],r,check_q)
                        wrist=g.geometry.transforms(check_q)['gripper_link'];body=g.at(check_q)
                        if any(fcl.collide(obj,clamp,fcl.CollisionRequest(),fcl.CollisionResult()) for _,_,obj in body for clamp in clamp_obs):collision=True;break
                        for model,local in self.camera_shapes:
                            tf=wrist@local;camera=collision_object(model,tf[:3,:3],tf[:3,3])
                            if any(fcl.collide(camera,obj,fcl.CollisionRequest(),fcl.CollisionResult()) for _,_,obj in body):collision=True;break
                        if collision:break
                    if collision:continue
                turn=Rotation.from_matrix(k.transform(q)[:3,:3].T@r).magnitude() if arm=='arm_a' and any(self.entry(h,self.time)[1]==arm for h in self.parts.values()) else 0.
                candidates.append((np.linalg.norm(above-q[:5])+10*turn,point,r,p))
            except ValueError:continue
        if not candidates:raise ValueError(f'No vertical grasp/approach: {arm}/{kind} {np.asarray(base).tolist()}')
        return min(candidates,key=lambda x:x[0])[1:]

    def approach(self,arm,point,rotation,source_lane=None):
        q=self.arms[arm].at(self.time);above=np.asarray(point)+[0,0,.035]
        value=q.copy();value[:5]=self.ik[arm].solve_pose(above,rotation,q)
        if arm=='arm_a':
            from arm_sim.real_transfer import joint_route
            for waypoint in joint_route(self,arm,value):self.joints(arm,waypoint,1.)
        elif arm=='arm_d' and above[2]<self.cfg['table_height']+.04:
            edge=np.array([above[0],.330,self.cfg['table_height']+.055]);v=q.copy();v[:5]=self.ik[arm].solve_pose(edge,rotation,q)
            self.joints(arm,v,1.)
            self.pose_move(arm,above,rotation)
        else:
            current=self.ik[arm].transform(q)
            axis=2 if self.ik[arm].so else 0
            if np.linalg.norm(current[:3,axis]-[0,0,-1])<.01:
                # Lift before crossing neighboring parts; do not descend diagonally.
                transit_z=max(current[2,3],self.cfg["table_height"]+.05);raised=current[:3,3].copy();raised[2]=transit_z
                self.pose_move(arm,raised,current[:3,:3])
                self.traverse(arm,np.r_[above[:2],transit_z],rotation)
                self.pose_move(arm,above,rotation)
            else:self.joints(arm,value,1.)
        self.pose_move(arm,point,rotation,precision=True)

    def acquire(self,arm,names):
        kind='product' if len(names)>1 else names[0].split('_')[0];source=self.part_at(names[0],self.time)
        point,r,p=self.grasp_pose(arm,kind,source);self.set_grasp(arm,p['open']);start=self.time
        self.approach(arm,point,r);self.set_grasp(arm,p['closed'])
        tip=self.ik[arm].position(self.arms[arm].at(self.time))
        for n in names:self.owner(n,arm,self.part_at(n,self.time)-tip)
        self.grasp_events.append({'time':self.time,'arm':arm,'parts':names,'kind':kind,'profile':kind})
        self.pose_move(arm,point+[0,0,.035],r,precision=True)
        self.phases.append({'arm':arm,'action':'pick','start':start,'end':self.time,'parts':names})

    def deposit(self,arm,names,destination,owner='world',offset=None):
        kind='product' if len(names)>1 else names[0].split('_')[0];destination=np.asarray(destination).copy()
        settled=None
        if arm=='arm_c' and kind=='ball' and np.linalg.norm(destination[:2]-self.seat('housing',self.cfg['jig'],self.cfg['table_height'])[:2])<1e-5:
            settled=destination.copy();destination[2]+=.032
        point,r,p=self.grasp_pose(arm,kind,destination);start=self.time
        self.approach(arm,point,r)
        tip=self.part_at(names[0],self.time)
        if np.linalg.norm(tip-destination)>.0003:raise ValueError('Release would teleport payload')
        for n in names:
            relative=self.part_at(n,self.time)-tip
            self.owner(n,owner,(destination if owner=='world' else np.asarray(offset))+relative)
        if settled is not None:
            n=names[0];rpart=self.part_rotation(n,self.time);self.owner(n,'fall',np.r_[destination,self.time,settled[2]])
            landing=self.time+math.sqrt(2*(destination[2]-settled[2])/9.81);self.parts[n].append((landing,'world',settled));self.rotations[n].append((landing,'world',rpart))
        self.set_grasp(arm,p['open']);self.pose_move(arm,point+[0,0,.035],r,precision=True)
        
        if arm=='arm_d' and destination[2]<self.cfg['table_height']:
            self.pose_move(arm,[point[0],.330,self.cfg['table_height']+.055],r)
        self.phases.append({'arm':arm,'action':'place','start':start,'end':self.time,'parts':names})

    def transfer(self,arm,names,*args,**kwargs):
        try:return super().transfer(arm,names,*args,**kwargs)
        except ValueError as error:raise ValueError(f'{arm} / {names}: {error}') from error

    def unload_kit(self,kit):
        for s in self.cfg['preparation_slots']:
            self.transfer('arm_b',[f"{s['kind']}_{kit+1}"],self.seat(s['kind'],s['position'],self.cfg['table_height']))
        self.park('arm_b')
    def assemble_kit(self,kit):
        base=self.seat('housing',self.cfg['jig'],self.cfg['table_height'])
        for kind in KINDS:self.transfer('arm_c',[f'{kind}_{kit+1}'],base+[0,0,ASSEMBLY[kind]])
        self.assembly_times.append(self.time)
        self.transfer('arm_c',[f'{k}_{kit+1}' for k in KINDS],self.seat('housing',self.cfg['handoff'],self.cfg['table_height']))
        self.park('arm_c')
    def load_product(self,kit):
        c=self.cfg;xy=np.array(c['shipping_dock'])+[(kit%3-1)*c['finished_container']['pitch'],0]
        self.transfer('arm_d',[f'{k}_{kit+1}' for k in KINDS],self.seat('housing',xy,c['shipping_height']+c['finished_container']['size'][2]))
        self.park('arm_d')
    def face(self,name,yaw):
        from arm_sim.real_vehicle import rest_duration
        track=self.vehicles[name];p=track.at(self.time);angle=(yaw-p[2]+math.pi)%(2*math.pi)-math.pi
        if abs(angle)<1e-7:return
        # A 180-degree turn beside the right map edge must sweep inward.
        if name=='waffle_1' and p[0]>1.0 and abs(abs(angle)-math.pi)<1e-6:
            angle=math.pi if math.sin(p[2])>0 else -math.pi
        v=self.cfg['vehicle_motion'];start=self.time;track.add(start,p)
        self.time+=rest_duration(angle,v['angular_velocity'],v['angular_acceleration']);track.add(self.time,[*p[:2],p[2]+angle])
        self.vehicle_moves.append({'vehicle':name,'start':start,'end':self.time,'mode':'turn'})

    def drive(self,name,points,*,reverse=None,docking=True):
        from arm_sim.real_vehicle import rest_duration
        track=self.vehicles[name];v=self.cfg['vehicle_motion']
        for x,y in points:
            p=track.at(self.time);distance=math.hypot(x-p[0],y-p[1])
            if distance<1e-8:continue
            heading=math.atan2(y-p[1],x-p[0]);delta=(heading-p[2]+math.pi)%(2*math.pi)-math.pi
            backing=(distance<=.21 and abs(abs(delta)-math.pi)<.02) if reverse is None else reverse
            target=heading+math.pi if backing else heading
            self.face(name,target)
            p=track.at(self.time);start=self.time;track.add(start,p)
            velocity=v['dock_velocity'] if docking else v['linear_velocity']
            acceleration=v['dock_acceleration'] if docking else v['linear_acceleration']
            self.time+=rest_duration(distance,velocity,acceleration);track.add(self.time,[x,y,p[2]])
            mode=('dock_reverse' if backing else 'dock_forward') if docking else ('reverse_transit' if backing else 'forward_transit')
            self.vehicle_moves.append({'vehicle':name,'start':start,'end':self.time,'mode':mode,'distance':distance})

    def loaded_route(self,vehicle):
        self.route(vehicle,self.cfg['unload_dock'],math.pi/2)

    def route(self,vehicle,destination,yaw):
        start=self.vehicles[vehicle].at(self.time);destination=np.asarray(destination)
        if np.linalg.norm(start[:2]-destination)<1e-6:
            self.face(vehicle,yaw);return
        # Retreat all the way to the aisle with the nose still facing the dock.
        # Each straight segment rests at both ends; only face() changes yaw.
        lane=self.cfg['vehicle_lane_y']
        if abs(start[1]-lane)>1e-8:
            self.drive(vehicle,[[start[0],lane]],reverse=math.sin(start[2])>0,docking=False)
        # Forklift always faces left in the horizontal aisle: drive toward the
        # warehouses, reverse toward the worktable so the fork stays inward.
        reverse_across=vehicle=='waffle_1' and destination[0]>start[0]
        self.drive(vehicle,[[destination[0],lane]],reverse=reverse_across,docking=False)
        self.face(vehicle,yaw)
        self.drive(vehicle,[destination],reverse=False,docking=False)

    def raise_lift(self,height):
        from arm_sim.real_vehicle import rest_duration
        if not -1e-9<=height<=self.cfg['waffle']['lift_stroke']+1e-9:raise ValueError('Lift target exceeds actual 100mm stroke')
        current=float(self.lift.at(self.time)[0]);delta=height-current
        if abs(delta)<1e-9:return
        v=self.cfg['lift_motion'];self.lift.add(self.time,[current])
        self.time+=rest_duration(delta,v['velocity'],v['acceleration']);self.lift.add(self.time,[height])

    def tray_pose(self,index,t):
        _,owner,value=self.entry(self.trays[index],t)
        if owner=='world':return value.copy()
        p=self.vehicles['waffle_1'].at(t);xy=self.mobile_point('waffle_1',[self.cfg['waffle']['tray_offset'],0,0],t)
        return np.array([*xy[:2],self.cfg['waffle']['fork_top']+self.lift.at(t)[0],p[2]-math.pi/2])
    def rack_dock(self,index):
        x,y=self.cfg['warehouses'][index]['center'];return [x,y-self.cfg['waffle']['tray_offset']]
    def tray_event(self,index,owner):
        p=self.tray_pose(index,self.time);self.trays[index].append((self.time,owner,p))
    def fetch_tray(self,index):
        c=self.cfg;dock=self.rack_dock(index)
        pose=self.vehicles['waffle_1'].at(self.time)
        initial_approach=(not any(e['vehicle']=='waffle_1' for e in self.vehicle_moves) and abs(pose[0]-dock[0])<1e-8 and c['vehicle_lane_y']<pose[1]<dock[1] and abs(math.sin(pose[2])-1)<1e-8)
        if not initial_approach:self.route('waffle_1',[dock[0],c['vehicle_lane_y']],math.pi/2)
        self.raise_lift(c['warehouses'][index]['levels'][-1]-c['waffle']['fork_top']-c['fork_insertion_clearance']);self.drive('waffle_1',[dock],reverse=False,docking=False);self.raise_lift(c['warehouses'][index]['levels'][-1]-c['waffle']['fork_top'])
        self.tray_event(index,'lift');self.raise_lift(c['warehouses'][index]['levels'][-1]+.005-c['waffle']['fork_top']);self.drive('waffle_1',[[dock[0],c['vehicle_lane_y']]],reverse=True,docking=False);self.raise_lift(.015)
        p=c['shipping_vehicle_pose'];self.route('waffle_1',[p[0],c['vehicle_lane_y']],math.pi/2)
        self.raise_lift(c['shipping_height']-c['waffle']['fork_top']+.005);self.drive('waffle_1',[[c['shipping_dock'][0],c['shipping_dock'][1]-c['waffle']['tray_offset']]],reverse=False,docking=False)
        self.raise_lift(c['shipping_height']-c['waffle']['fork_top']);self.tray_event(index,'world');self.raise_lift(c['shipping_height']-c['waffle']['fork_top']-c['fork_insertion_clearance'])
        self.drive('waffle_1',[[p[0],c['vehicle_lane_y']]],reverse=True,docking=False)  # Keep empty forks ready for the next pickup.
    def ship(self,index,return_home=True):
        c=self.cfg;p=c['shipping_vehicle_pose'];self.raise_lift(c['shipping_height']-c['waffle']['fork_top']-c['fork_insertion_clearance'])
        self.drive('waffle_1',[[c['shipping_dock'][0],c['shipping_dock'][1]-c['waffle']['tray_offset']]],reverse=False,docking=False);self.raise_lift(c['shipping_height']-c['waffle']['fork_top']);self.tray_event(index,'lift')
        for kit in range(index*3,index*3+3):
            for kind in KINDS:
                n=f'{kind}_{kit+1}';pos=self.part_at(n,self.time);pose=self.vehicles['waffle_1'].at(self.time);r=Rotation.from_euler('z',pose[2]).as_matrix();v=r.T@(pos-[*pose[:2],self.lift.at(self.time)[0]])
                self.owner(n,'lift',v)
        self.raise_lift(c['shipping_height']-c['waffle']['fork_top']+.005);self.drive('waffle_1',[[p[0],c['vehicle_lane_y']]],reverse=True,docking=False);self.raise_lift(.015)
        dock=self.rack_dock(index);self.route('waffle_1',[dock[0],c['vehicle_lane_y']],math.pi/2)
        self.raise_lift(c['warehouses'][index]['levels'][-1]-c['waffle']['fork_top']+.005);self.drive('waffle_1',[dock],reverse=False,docking=False);self.raise_lift(c['warehouses'][index]['levels'][-1]-c['waffle']['fork_top']);self.tray_event(index,'world')
        for kit in range(index*3,index*3+3):
            for kind in KINDS:
                n=f'{kind}_{kit+1}';self.owner(n,'world',self.part_at(n,self.time))
        self.storage_times.append(self.time);self.stored=(index+1)*3
        self.raise_lift(c['warehouses'][index]['levels'][-1]-c['waffle']['fork_top']-c['fork_insertion_clearance']);self.drive('waffle_1',[[dock[0],c['vehicle_lane_y']]],reverse=True,docking=False)
        if index==1:self.raise_lift(0.)  # Lower only after the final storage operation.
        if return_home:self.route('waffle_1',[p[0],c['vehicle_lane_y']],math.pi/2)
    def _build(self):
        c=self.cfg;self.milestones['first_loaded']=0.
        self.traffic_job('burger_1','운반로봇 1 · 재료 9개 운반',lambda:self.loaded_route('burger_1'));arrived=self.time;self.milestones['first_arrived']=arrived
        # Stationary fork preparation can overlap Burger 2's road travel.
        self.raise_lift(c['warehouses'][0]['levels'][-1]-c['waffle']['fork_top']-c['fork_insertion_clearance'])
        self.activity(arrived,'지게차 · 정지 상태에서 빈 용기 회수 높이 준비');self.time=arrived
        self.traffic_job('burger_2','운반로봇 2 · 상차 위치 이동',lambda:self.route('burger_2',c['loading_dock'],math.pi/2));start=self.time
        self.traffic_job('waffle_1','지게차 · 첫 빈 용기 회수·작업대 배치',lambda:self.fetch_tray(0));tray_ready=self.time
        self.milestones['first_tray_ready']=tray_ready
        self.time=start
        for kit in range(3):
            for kind in KINDS:
                off=self.cargo_offset(kind,kit);self.transfer('arm_a',[f'{kind}_{kit+4}'],self.mobile_point('burger_2',off,self.time),'burger_2',off)
        self.park('arm_a');loaded=self.time;self.activity(start,'A · 다음 배치 9개 상차');self.milestones['second_loaded']=loaded
        self.time=arrived;bend,dend=self.process_batch(0,arrived,tray_ready)
        self.time=max(bend,loaded);self.traffic_job('burger_1','운반로봇 1 · 대기 구역 주차',lambda:self.route('burger_1',c['burger_parking_poses']['burger_1'][:2],math.pi/2))
        # Deliver the next material batch before the forklift's storage trip.
        # B/C may then work while the forklift stores and fetches containers.
        self.traffic_job('burger_2','운반로봇 2 · 재료 9개 운반',lambda:self.loaded_route('burger_2'));arrived=self.time;self.milestones['second_arrived']=arrived
        self.time=max(self.time,dend);self.traffic_job('waffle_1','지게차 · 첫 완제품 3개 보관',lambda:self.ship(0,False))
        self.traffic_job('waffle_1','지게차 · 두 번째 빈 용기 회수·배치',lambda:self.fetch_tray(1));tray_ready=self.time
        self.time=arrived;bend,dend=self.process_batch(3,arrived,tray_ready)
        # Vacate the unload dock as soon as B releases the final kit; C/D keep working.
        self.time=bend
        self.traffic_job('burger_2','운반로봇 2 · 상차 대기 위치 복귀',lambda:self.route('burger_2',c['burger_parking_poses']['burger_2'][:2],math.pi/2))
        self.time=max(self.time,dend);self.traffic_job('waffle_1','지게차 · 두 번째 완제품 3개 보관',lambda:self.ship(1,False))
        self.traffic_job('waffle_1','지게차 · 전용 대기 위치 복귀',lambda:self.route('waffle_1',c['forklift_parking_pose'][:2],c['forklift_parking_pose'][2]))
    def validate(self):
        final=self.sample(self.duration)
        if len(self.parts)!=18 or final['assembled']!=6 or final['stored']!=6:raise ValueError('Incomplete process')
        if any(o!='world' for o in final['owners'].values()):raise ValueError('Payload left on vehicle or arm')
        if not np.allclose(final['vehicles']['waffle_1'][:2],self.cfg['forklift_parking_pose'][:2],atol=1e-8):raise ValueError('Forklift did not return to parking')
        for name,parking in self.cfg['burger_parking_poses'].items():
            pose=final['vehicles'][name]
            if not np.allclose(pose[:2],parking[:2],atol=1e-8) or abs(math.sin((pose[2]-parking[2])/2))>1e-8:raise ValueError('Carrier did not park: '+name)
        for arm,tr in self.arms.items():
            lim=np.array(self.ik[arm].model['limits']);qs=np.array(tr.values)
            if not np.isfinite(qs).all() or (qs<lim[:,0]-1e-6).any() or (qs>lim[:,1]+1e-6).any():raise ValueError('Joint bounds '+arm)
        for kit in range(1,7):
            for kind in KINDS:
                if np.linalg.norm(final['parts'][f'{kind}_{kit}']-final['parts'][f'housing_{kit}']-[0,0,ASSEMBLY[kind]])>.0003:raise ValueError('Assembly mismatch')
        return {'unique_parts':18,'assembled':6,'stored':6,'duration_s':self.duration,'assisted_grip':True,'physics_success_claim':False,'motion_profiles':self.motion_profiles,'milestones':self.milestones}
