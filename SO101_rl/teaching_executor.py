"""Run the current teaching motor worker against an in-memory simulation bus.

No worker thread, serial port, or hardware activation is started. The original
MotionSession source is loaded into a private module so only its clock changes;
its contact, arrival, compensation and episode sequencing code stays identical.
"""
import importlib.util
from types import SimpleNamespace
import numpy as np
from core import JOINTS, TEACH


class SimClock:
    def __init__(self):self.now=100.
    def monotonic(self):return self.now
    def time(self):return self.now
    def time_ns(self):return round(self.now*1e9)


class MemoryBus:
    def __init__(self, ticks, gate):
        self.motion_gate=gate
        self.values={n:{'Present_Position':v,'Goal_Position':v,'Torque_Enable':1,
                       'Status':0,'Operating_Mode':0,'Phase':4} for n,v in ticks.items()}
    def read(self,register,name,**kwargs):return self.values[name].get(register,0)
    def write(self,register,name,value,**kwargs):self.values[name][register]=value
    def sync_write(self,register,values,**kwargs):
        for name,value in values.items():self.write(register,name,value)


def estimated_load_raw(effort, full_scale_effort, *, gripper_error_ticks=None, gripper_model=None,
                       gripper_closing=True, joint_error_ticks=None, arm_model=None,
                       joint_velocity_ticks=None):
    """Explicit proxy, NOT a calibrated STS Present_Load-to-torque conversion."""
    effort=np.asarray(effort,float);scale=np.asarray(full_scale_effort,float)
    if effort.shape!=(6,) or scale.shape!=(6,) or not np.isfinite(effort).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('시뮬 부하 추정 입력 오류')
    magnitude=np.rint(np.clip(abs(effort)/scale,0,1)*1023).astype(int)
    raw=magnitude+np.where(effort<0,1024,0)
    if arm_model is not None:
        errors=np.asarray(joint_error_ticks,float)
        velocities=np.asarray(joint_velocity_ticks,float)
        if errors.shape!=(6,) or not np.isfinite(errors).all():raise ValueError('관절 내부 목표 오차가 필요합니다.')
        if velocities.shape!=(6,) or not np.isfinite(velocities).all():raise ValueError('관절 속도 피드백이 필요합니다.')
        for i,name in enumerate(JOINTS[:5]):
            model=arm_model['joints'][name];error=abs(errors[i])
            if error<=model['error_ticks'][-1] and abs(velocities[i])<=5:
                raw[i]=round(float(np.interp(error,model['error_ticks'],model['load_magnitude'])))+(1024 if effort[i]<0 else 0)
    if gripper_model is not None:
        if gripper_error_ticks is None or not np.isfinite(gripper_error_ticks):raise ValueError('집게 내부 목표 오차가 필요합니다.')
        # These samples model commanded closing against resistance. An arm
        # collision that pushes an otherwise held/open jaw outward is outside
        # that model, and must not fabricate a closing-contact sensor reading.
        raw[5]=(round(float(np.interp(max(0,gripper_error_ticks),gripper_model['error_ticks'],gripper_model['load_magnitude'])))
                if gripper_closing else 0)
    return raw


class TeachingExecutor:
    def __init__(self, reference, initial_angles):
        self.ref=reference;self.clock=SimClock();self.elapsed=0.;self.time=0.
        self.index=0;self.fraction=0.;self.wait_s=0.;self.failure=''
        self.finished=False;self.finished_at=None;self.next_poll=0.
        self.contact_events=[];self.arrivals=[];self.waiting=False
        previous=reference.joint_path.segments[0].start['gripper']
        self.closing_steps=[]
        for step in reference.steps:
            requested=step['ticks']['gripper']
            self.closing_steps.append(requested<previous)
            previous=requested
        # Relative imports still resolve in so101_teach. No global time module
        # or live application's classes are monkey-patched.
        spec=importlib.util.spec_from_file_location('so101_teach._isaac_motion',TEACH/'so101_teach/motion.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.time=self.clock
        initial=self.ticks(initial_angles)
        self.bus=MemoryBus(initial,module.MotionGate(reference.calibration))
        self.session=module.MotionSession('isaac-memory-only',reference.calibration,
            bus_factory=lambda *args:self.bus,rate_ticks_s=reference.rate,
            grip_contact_load=reference.profile.get('grip_contact_load',module.DEFAULT_CONTACT_LOAD))
        # PhysX already holds the measured reset pose. Skip the serial-specific
        # activation sequence; all subsequent episode execution uses on_snapshot.
        self.session.running=True;self.session.state='HOLD'
        self.session.last_goals=initial.copy();self.session.enabled=initial.copy()
        self.request=self.session.request('play',[s['ticks'] for s in reference.steps])
        self.nominal=np.asarray(reference.mapping.angles(initial))
        self.last_load_raw=dict.fromkeys(JOINTS,0)

    def ticks(self,angles):
        return {n:round(v) for n,v in self.tick_values(angles).items()}

    def tick_values(self,angles):
        # Feedback must retain physical overshoot, unlike validated command
        # conversion which rejects positions outside the calibration endpoints.
        mapping=self.ref.calibration.angle_mapping;model=self.ref.mapping
        return {n:(mapping.ticks(n,np.degrees(float(a)-model.radians[n])) if mapping else
                        model.middle[n]-model.trims[n]+(float(a)-model.radians[n])*4096/(2*np.pi))
                for n,a in zip(JOINTS,angles)}

    def velocity_ticks(self,angles,velocity):
        before=self.tick_values(angles);after=self.tick_values(np.asarray(angles)+np.asarray(velocity)*.001)
        return {n:round((after[n]-before[n])/.001) for n in JOINTS}

    def advance(self,dt,actual,velocity,load_raw):
        self.elapsed+=dt;self.clock.now=100.+self.elapsed
        if self.elapsed+1e-9<self.next_poll or self.failure or self.finished:
            return self.nominal.copy(),self.index
        self.next_poll=max(self.next_poll+self.session.poll_seconds,self.elapsed)
        ticks=self.ticks(actual);speed=self.velocity_ticks(actual,velocity)
        health={}
        for i,n in enumerate(JOINTS):
            self.bus.values[n]['Present_Position']=ticks[n]
            health[n]={'torque':1,'status':0,'packet_alarm':0,'temperature_c':25,
                       'goal_ticks':self.bus.read('Goal_Position',n),'load_raw':int(load_raw[i]),
                       'velocity_signed_raw':speed[n]}
        self.last_load_raw={n:h['load_raw'] for n,h in health.items()}
        stamp=self.clock.now
        snapshot=SimpleNamespace(ticks=ticks,telemetry=health,monotonic=stamp,
            calibration_matches=True,fresh=lambda now=None:0<=(self.clock.now if now is None else now)-stamp<=.5)
        self.session.heartbeat=self.clock.now
        before_contacts=len(self.session.report.get('gripper_contacts',[]))
        active_index=self.session.index
        try:self.session.on_snapshot(self.bus,snapshot)
        except (RuntimeError,ValueError) as exc:self.failure=str(exc)
        contacts=self.session.report.get('gripper_contacts',[])
        for event in contacts[before_contacts:]:
            self.contact_events.append({**event,'step_index':active_index,'simulation_time_s':self.elapsed})
        self.arrivals=self.session.report.get('step_arrivals',[])
        pauses=self.session.report.get('pauses',[])
        if pauses:self.failure=pauses[-1]['detail']
        self.index=min(self.session.index,len(self.ref.steps)-1)
        self.finished=self.session.completed_request_id==self.request
        if self.finished:self.finished_at=self.elapsed;self.fraction=1.
        elif self.session.segment:
            _,_,began,duration=self.session.segment
            self.fraction=min(1.,max(0.,(self.clock.now-began)/duration))
            self.wait_s=max(0.,self.clock.now-began-duration)
        self.waiting=bool(not self.finished and (self.fraction>=1. or self.session.grip_contact.pending is not None))
        start=0. if self.index==0 else self.ref.ends[self.index-1]
        progress=start+self.fraction*(self.ref.ends[self.index]-start)
        self.time=max(self.time,float(progress))
        self.nominal=np.asarray(self.ref.mapping.angles(self.session.last_goals))
        return self.nominal.copy(),self.index

    @property
    def hold_tick(self):return self.session.grip_contact.hold_tick
    @property
    def pending_contact(self):return self.session.grip_contact.pending is not None
    @property
    def phase(self):return (self.index+self.fraction)/len(self.ref.steps)
    @property
    def gripper_closing(self):return self.closing_steps[self.index]
