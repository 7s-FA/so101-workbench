"""One client, no queue, latched faults; run in the existing IO Python environment."""
import argparse
import json
import math
import select
import signal
import sys
import time
from pathlib import Path
from .mapping import Mapping, vector, START_POSE_TOLERANCE_RAD
from .bus import Bus
from .temperature import TemperatureGuard
from .tracking import Tracking, ReferenceTracking
from .compensation import PoseMap,identity,CONTEXTS,PHASES
from .motion_timing import VELOCITY_RAD_S
from .entry import CurrentPoseEntry, validate_entry, EntryPoseChanged, reference_entry_ready, ENTRY_SPEED, entry_duration


class Guard:
    def __init__(self, mapping):
        self.mapping = mapping
        self.last = None
        self.last_time = None

    def check(self, target, measured, now, reference=False):
        target=vector(target);measured=vector(measured)
        ticks=self.mapping.to_ticks(target)
        if not reference and max(abs(a-b) for a,b in zip(target[:5],measured[:5]))>.18:
            raise ValueError('Arm tracking error exceeds 0.18 rad')
        # Gripper error is expected during contact; bound current in Bus.enable.
        if self.last is not None:
            dt=now-self.last_time
            if not 0 < dt <= .3:
                raise ValueError('Stale command; restart required')
            if not reference and any(abs(a-b)>limit*dt+.003 for a,b,limit in zip(target,self.last,VELOCITY_RAD_S)):
                raise ValueError('Target moves too fast; no queued catch-up is allowed')
        self.last=target;self.last_time=now
        return ticks


class StartAlignment:
    """Bridge a start pose within 10 degrees per joint, at <=0.1 rad/s."""
    def __init__(self, measured, home, now, reference=False):
        self.reference=reference
        self.home=vector(home);self.value=vector(measured);self.time=now;self.started=now
        self.last_error=None
        if max(abs(a-b) for a,b in zip(self.home,self.value))>START_POSE_TOLERANCE_RAD+1e-12:
            raise ValueError('Match all six physical start angles within 10 deg before enabling')

    def advance(self, requested, now):
        if max(abs(a-b) for a,b in zip(vector(requested),self.home))>1e-6:
            raise ValueError('Wait for start alignment before starting motion')
        if now-self.started>20:
            detail=f'; ID {self.last_error[0]} remaining error {self.last_error[1]:.2f} deg' if self.last_error else ''
            raise ValueError('Start alignment did not settle within 20 s'+detail+'; requires stable feedback within 3 deg and low output; inspect load or pose mapping')
        dt=now-self.time
        if not 0<dt<=.3:raise ValueError('Stale start alignment')
        step=.1*min(dt,.05)
        self.value=[a+max(-step,min(step,b-a)) for a,b in zip(self.value,self.home)]
        self.time=now
        return self.value

    def reached(self, measured, *, stationary=False, effort=math.inf):
        if self.reference:
            return (max(abs(a-b) for a,b in zip(self.value,self.home))<1e-6 and
                    reference_entry_ready(measured,self.home,stationary=stationary,effort=effort))
        errors=[abs(a-b) for a,b in zip(vector(measured),self.home)]
        worst=max(range(6),key=errors.__getitem__)
        self.last_error=(11+worst,math.degrees(errors[worst]))
        # Use the same residual bound as Tracking's recovery threshold. Position
        # servos can settle with a small load/deadband error; do not demand 1.43 deg
        # while the normal controller permits recovery at 3 deg. Stationary comes
        # from the existing 0.5 s encoder window and <=1.5 deg/s velocity check.
        return (max(abs(a-b) for a,b in zip(self.value,self.home))<1e-6 and
                errors[worst]<=math.radians(3)+1e-12 and stationary and effort<60)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--visual-confirmed',action='store_true')
    parser.add_argument('--fast-feedback',action='store_true')
    parser.add_argument('--reference-control',action='store_true')
    parser.add_argument('--sustained-temperature',action='store_true')
    parser.add_argument('--display-only-scene',action='store_true')
    parser.add_argument('--entry-speed',type=float,default=ENTRY_SPEED)
    args=parser.parse_args()
    if args.sustained_temperature and not args.reference_control:
        parser.error('--sustained-temperature requires --reference-control')
    if args.display_only_scene and not args.reference_control:
        parser.error('--display-only-scene requires --reference-control')
    entry_duration([0]*6,[0]*6,args.entry_speed)
    if args.entry_speed != ENTRY_SPEED and not args.reference_control:
        parser.error('--entry-speed requires --reference-control')
    mapping=Mapping(json.loads(args.calibration.read_text()),allow_visual_confirmation=args.visual_confirmed)
    if str(Path(args.port).resolve())!=str(Path(mapping.cfg['port']).resolve()):
        raise ValueError('Calibration belongs to another USB port')
    root=Path(__file__).resolve().parents[1]
    fingerprint=identity(mapping,root/'assets/omx_teach.urdf')
    map_path=root/'outputs/follower/pose_compensation.json'
    pose_map=None
    bus=None;release_on_interrupt=False;tracking=None;telemetry=None;last_log=0.;last_reply={}
    temperature_guard=TemperatureGuard() if args.sustained_temperature else None
    def sample_health(active):
        options=dict(detailed=True,packed=args.fast_feedback)
        if temperature_guard:options.update(temperature_guard=temperature_guard,active=active)
        return bus.sample(**options)
    def interrupted(signum,*_):
        nonlocal release_on_interrupt
        release_on_interrupt=signum==signal.SIGINT
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        bus=Bus(args.port);info=bus.inspect()
        log_dir=Path(__file__).resolve().parents[1]/'outputs/follower'
        log_dir.mkdir(parents=True,exist_ok=True)
        log_path=log_dir/f'tracking_{time.time_ns()}.jsonl'
        telemetry=log_path.open('w')
        if [{k:e[k] for k in ('id','model','mode','drive','offset')} for e in info] != mapping.cfg['motor_config']:
            raise ValueError('Motor configuration changed since calibration')
        print(json.dumps({'ok':True,'state':'connected_read_only','motors':info}),flush=True)
        armed=False;guard=Guard(mapping);alignment=None
        while True:
            ready,_,_=select.select([sys.stdin],[],[],.25 if armed else 30)
            if not ready:
                if armed:raise RuntimeError('Simulation heartbeat lost')
                continue
            line=sys.stdin.readline()
            if not line:break
            message=json.loads(line)
            age=time.monotonic()-message['sent']
            if not 0 <= age <= .2:raise ValueError('Stale input packet')
            if message['op']=='close':break
            if message['op']=='hold':
                bus.hold()
                print(json.dumps({'ok':True,'state':'stopped_restart_required'}),flush=True)
                break
            sample=sample_health(armed)
            measured=mapping.to_angles([x['ticks'] for x in sample])
            torque_on=any(bus.read(motor,64,1) for motor in mapping.cfg['ids']) if not armed else True
            release_required=(torque_on or any(bus.read(motor,98,1) for motor in mapping.cfg['ids'])) if not armed else False
            if message['op']=='arm':
                if armed:raise ValueError('Already armed')
                target=vector(message['target'])
                # Re-read configuration immediately before any motor writes.
                info=bus.inspect()
                if [{k:e[k] for k in ('id','model','mode','drive','offset')} for e in info]!=mapping.cfg['motor_config']:
                    raise ValueError('Motor configuration changed before enable')
                sample=sample_health(False);measured=mapping.to_angles([x['ticks'] for x in sample])
                if args.reference_control:
                    mapping.validate_observation(measured)
                    target=mapping.clamp_target(target)
                else:mapping.to_ticks(measured)
                mapping.to_ticks(target)
                entry=message.get('entry')
                if entry is not None:
                    try:validate_entry(entry,measured,target,mapping,allow_start_residual=args.reference_control,
                                       require_collision_check=not args.display_only_scene)
                    except EntryPoseChanged as error:
                        # No enable/write has happened. Keep this read-only USB
                        # session available for a fresh audit instead of latching.
                        print(json.dumps(dict(ok=False,error=str(error),code='entry_pose_changed',
                            angles=measured,recoverable=True)),flush=True)
                        continue
                    # Begin at the fresh encoder pose, never jump back to the
                    # audit snapshot. validate_entry bounds drift to 0.5 deg.
                    alignment=CurrentPoseEntry(measured,target,time.monotonic(),reference=args.reference_control,
                                               peak_speed=args.entry_speed)
                else:
                    alignment=StartAlignment(measured,target,time.monotonic(),reference=args.reference_control)
                if not message['path']:raise ValueError('A checked motion path is required')
                for q in message['path']:mapping.to_ticks(q)
                enabled=message.get('compensation_enabled',True)
                reuse=message.get('reuse_compensation',False)
                evaluation=message.get('evaluation',True)
                if type(enabled) is not bool or type(reuse) is not bool:raise ValueError('Compensation options must be boolean')
                if type(evaluation) is not bool:raise ValueError('Evaluation option must be boolean')
                pose_map=PoseMap()
                if args.reference_control:bus.enable(info,[x['ticks'] for x in sample],reference=True)
                else:bus.enable(info,[x['ticks'] for x in sample])
                # Learned offsets from the former soft gains are not valid for
                # the new output profile (including a lower installed cap).
                fingerprint=identity(mapping,root/'assets/omx_teach.urdf')+':'+json.dumps(getattr(bus,'power_profile',{}),sort_keys=True)
                if enabled and reuse:pose_map.load(map_path,fingerprint)
                armed=True
                guard.last=measured;guard.last_time=time.monotonic()
                alignment.time=guard.last_time
                tracking=(ReferenceTracking(mapping,measured,guard.last_time) if args.reference_control else
                          Tracking(mapping,measured,guard.last_time,enabled,pose_map,evaluation=evaluation))
            elif message['op']=='target':
                if not armed:raise ValueError('Enable follower first')
                trial_active=message.get('trial_active',False)
                if type(trial_active) is not bool:raise ValueError('trial_active must be boolean')
                context=message.get('context','unknown');phase=message.get('phase','unknown')
                if context not in CONTEXTS or phase not in PHASES:raise ValueError('Unknown compensation phase/context')
                now=time.monotonic();target=message['target']
                reference_motion=args.reference_control and alignment is None
                if reference_motion and not isinstance(tracking,ReferenceTracking):
                    tracking=ReferenceTracking(mapping,measured,guard.last_time)
                if alignment is not None:
                    if isinstance(alignment,CurrentPoseEntry):
                        target=alignment.advance(target,now,pace=tracking.status.get('pace',1.))
                    else:target=alignment.advance(target,now)
                    context,phase='empty','alignment'
                if args.reference_control:target=mapping.clamp_target(target)
                target,tracking_status=tracking.update(target,measured,sample,now,correction=alignment is None or max(abs(a-b) for a,b in zip(target,alignment.home))<1e-6,context=context,phase=phase)
                if alignment is not None and alignment.reached(measured,
                        stationary=tracking_status['compensation']['stationary'],
                        effort=tracking_status['pwm_percent']):
                    alignment=None
                ticks=guard.check(target,measured,now,reference=args.reference_control)
                bus.positions(ticks)
                if trial_active and alignment is None and phase!='alignment':tracking.record_motion_error()
            elif message['op']!='read':raise ValueError('Unknown operation')
            last_reply={'ok':True,'state':'armed' if armed else 'read_only',
                'angles':measured,'motors':sample,'aligned':armed and alignment is None,
                'torque_on':armed or torque_on,'release_required':not armed and release_required,
                'tracking':tracking.status if tracking else {},'telemetry_file':str(log_path),
                'power_profile':getattr(bus,'power_profile',{}),
                'motion_error':tracking.error_summary() if tracking else {},
                'received':time.monotonic()}
            if temperature_guard:last_reply['temperature']=temperature_guard.status()
            transitions=last_reply.get('temperature',{}).get('transitions',[])
            if time.monotonic()-last_log>=.2 or transitions:
                telemetry.write(json.dumps(dict(time=time.time(),**last_reply))+'\n');telemetry.flush();last_log=time.monotonic()
            print(json.dumps(last_reply),flush=True)
            if temperature_guard:temperature_guard.clear_transitions()
    except Exception as e:
        temperature=temperature_guard.status() if temperature_guard else None
        if telemetry:
            telemetry.write(json.dumps(dict(time=time.time(),fault=str(e),last=last_reply,temperature=temperature))+'\n');telemetry.flush()
        print(json.dumps({'ok':False,'error':str(e),'temperature':temperature}),flush=True)
        return 2
    finally:
        if bus:
            try:
                if release_on_interrupt and bus.owned:
                    bus.release()
                    print('FOLLOWER EXIT: torque OFF confirmed',file=sys.stderr,flush=True)
            finally:bus.close()
        if telemetry:
            telemetry.close()
            if tracking:
                try:
                    log_path.with_suffix('.summary.json').write_text(json.dumps(dict(
                        telemetry_file=str(log_path),profile=tracking.status.get('profile','evaluation' if tracking.evaluation else 'adaptive'),
                        motion_error=tracking.error_summary()),indent=2)+'\n')
                except OSError as e:print('Motion summary save failed: '+str(e),file=sys.stderr,flush=True)
        if pose_map is not None:
            try:pose_map.save(map_path,fingerprint)
            except (OSError,ValueError) as e:print('Compensation map save failed: '+str(e),file=sys.stderr,flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
