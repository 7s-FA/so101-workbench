"""One SO-101 USB owner, using OpenCV teaching's activation and health checks."""
import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import select
import signal
import sys
import time

from .config import TEACH
from .robot import OCV,NAMES,Mapping,angles_to_pose,pose_to_angles,profile,reference_config
sys.path[:0]=[str(OCV),str(TEACH)]
from teach_left_arm import Hardware as TeachingHardware,TrackingProgress
from teaching_settings import HEALTH_INTERVAL,STALL_SECONDS
from teach_io.entry import CurrentPoseEntry,validate_entry,EntryPoseChanged
from .reference import ReferenceCodec


class Session:
    def __init__(self,port,leader_port=None,backend=None,clock=time.monotonic):
        self.clock=clock;self.mapping=Mapping();self.cfg=profile();self.codec=ReferenceCodec()
        self.armed=False;self.error=None;self.entry=None;self.leader=None
        self.owned=False;self.goal=None;self.sent=None;self.last_health=0.;self.last_sample=None
        self.progress=[TrackingProgress() for _ in NAMES[:5]]
        if Path(port).resolve()!=Path(self.cfg['follower_port']).resolve():raise ValueError('최신 SO-101 팔로워 보정에 등록된 USB를 선택하세요.')
        if leader_port and (not self.cfg.get('leader_calibration') or Path(leader_port).resolve()!=Path(self.cfg['leader_port']).resolve()):
            raise ValueError('장치 연결에서 SO-101 리더 USB와 보정 파일을 먼저 등록하세요.')
        self.backend=backend or TeachingHardware(self.cfg['follower_profile'],follower_only=True)
        # LeRobot's default does not add a per-send relative-target ceiling.
        # Absolute calibrated bounds and hardware health checks remain active.
        if backend is None:self.backend.follower.config.max_relative_target=None
        try:
            self.backend.connect()
            if leader_port:
                from lerobot.teleoperators.so_leader import SO101Leader,SO101LeaderConfig
                from diagnose_shoulder_lift import ensure_ports_available,protect_port,compare_calibration
                ensure_ports_available([leader_port])
                file=Path(self.cfg['leader_calibration'])
                self.leader=SO101Leader(SO101LeaderConfig(port=leader_port,id=file.stem,calibration_dir=file.parent,use_degrees=True))
                protect_port(self.leader.bus);self.leader.bus.connect()
                if not compare_calibration(self.leader)['matches']:raise ValueError('SO-101 리더의 보정 파일과 모터값이 다릅니다.')
                if any(self.leader.bus.read('Torque_Enable',n,normalize=False) for n in NAMES):raise ValueError('조작용 팔의 모터 힘을 먼저 꺼 주세요.')
            self.read()
        except BaseException:
            self.close(release=False);raise

    def health(self,force=False):
        self.backend.check_temperature_watchdog()
        now=self.clock()
        if force or now-self.last_health>=HEALTH_INTERVAL:
            self.backend.health(active=self.armed);self.last_health=self.clock()

    def read(self):
        self.health()
        _,p=self.backend.read();q=pose_to_angles(p)
        ticks=[];leader_target=None;leader_angles=None
        if self.leader:
            raw=self.leader.bus.sync_read('Present_Position',normalize=False)
            ticks=[raw[n] for n in NAMES]
            leader_angles=pose_to_angles(self.codec.normalize(ticks,'leader'))
            leader_target=self.codec.leader_target(ticks,self.mapping)
        now=self.clock();stationary=False
        if self.last_sample:
            old,at=self.last_sample
            if max(abs(a-b) for a,b in zip(q[:5],old[:5]))>math.radians(.3):self.last_sample=(q,now)
            else:stationary=now-at>=.3
        else:self.last_sample=(q,now)
        guard=self.backend.temperature_guard
        sample=self.backend.temperature_sample or {}
        pending=guard.pending()
        motors=[dict(id=i+1,joint=n,temperature_raw=sample.get('temperatures_raw',{}).get(n)) for i,n in enumerate(NAMES)]
        result=dict(ok=True,angles=q,leader_ticks=ticks,leader_target=leader_target,leader_angles=leader_angles,
            motors=motors,torque_on=self.armed,release_required=self.owned,
            aligned=self.entry is None,
            temperature=dict(unit='raw',limit_raw=guard.limit,duration_s=guard.duration,
                max_gap_s=guard.max_gap,pending=pending,state='fault' if guard.fault else 'high' if pending else 'normal',fault=guard.fault.detail if guard.fault else None),
            tracking=dict(desired_angles=self.sent or q,command_angles=self.sent or q,
                profile='lerobot_so101',pace=1.,compensation=dict(stationary=stationary)),
            power_profile=dict(robot='so101',driver='Feetech STS3215',source='OpenCV_teach',gravity_compensation=False))
        self.latest=result
        return result

    def arm(self,target,entry):
        if self.error:raise RuntimeError(self.error)
        current=self.read()['angles'];target=self.mapping.clamp_target(target)
        validate_entry(entry,current,target,self.mapping,allow_start_residual=True,require_collision_check=False)
        self.health(force=True)
        self.owned=True
        current=pose_to_angles(self.backend.activate())
        self.armed=True;self.sent=current;self.goal=target
        self.entry=CurrentPoseEntry(current,target,self.clock(),reference=True,peak_speed=.5)
        self.progress=[TrackingProgress() for _ in NAMES[:5]]
        return self.read()

    def update(self,target):
        if self.error:raise RuntimeError(self.error)
        if not self.armed:raise ValueError('모터 조작을 먼저 시작하세요.')
        result=self.read();target=self.mapping.clamp_target(target);now=self.clock()
        if self.entry:
            target=self.entry.advance(self.goal,now)
            # Finish the commanded ramp, not a second arbitrary position-error gate.
            # Continuing targets still use calibration bounds and progress monitoring.
            if self.entry.elapsed>=self.entry.duration:
                self.entry=None
        requested=angles_to_pose(self.mapping.clamp_target(target))
        sent=self.backend.send(requested)
        self.sent=pose_to_angles(sent)
        # Preserve the reference controller's bounded target, not its uncapped leader request.
        result['tracking'].update(desired_angles=self.sent,command_angles=self.sent)
        result['aligned']=self.entry is None
        if not self.entry:
            measured=angles_to_pose(result['angles'])
            for i,n in enumerate(NAMES[:5]):
                key=n+'.pos';status=self.progress[i].update(now,measured[key],requested[key],sent[key])
                if status['blocked']:raise RuntimeError(f'{n}: {STALL_SECONDS:g}초 동안 진행이 없습니다. 부하·장애물을 확인하세요.')
        return result

    def hold(self):
        self.entry=None
        if self.armed:
            _,pose=self.backend.read()
            self.sent=pose_to_angles(self.backend.send(self.backend.project_target(pose)))

    def release(self):
        # Explicit release works even after a latched monitoring fault.
        errors=[];bus=self.backend.follower.bus
        for name in NAMES:
            try:bus.write('Torque_Enable',name,0,normalize=False,num_retry=1)
            except Exception as e:errors.append(str(e))
            try:
                if bus.read('Torque_Enable',name,normalize=False,num_retry=1)!=0:raise RuntimeError(name+': 힘 해제 확인 실패')
            except Exception as e:errors.append(str(e))
        if errors:raise RuntimeError('; '.join(errors))
        self.armed=False;self.owned=False;self.backend.motion_attempted=False

    def close(self,release=True):
        if self.leader and self.leader.bus.is_connected:self.leader.bus.disconnect(disable_torque=False)
        if getattr(self,'backend',None):
            if release and (self.owned or self.backend.motion_attempted):
                errors=self.backend.close()
                if errors:raise RuntimeError('; '.join(errors))
                self.owned=False;self.armed=False
            elif self.backend.follower.bus.is_connected:self.backend.follower.bus.disconnect(disable_torque=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',required=True);parser.add_argument('--leader');parser.add_argument('--release-only',action='store_true')
    args=parser.parse_args();protocol=sys.stdout;session=None;stopping=False
    def send(result):protocol.write(json.dumps(result,allow_nan=False)+'\n');protocol.flush()
    def stop(*_):
        nonlocal stopping
        stopping=True
    for s in (signal.SIGINT,signal.SIGTERM):signal.signal(s,stop)
    try:
        with redirect_stdout(sys.stderr):
            if args.release_only:
                from .hardware import release_port
                if not release_port(args.port):raise RuntimeError('모터 힘 해제 확인 실패')
                return
            session=Session(args.port,args.leader)
        send(session.latest);last=time.monotonic()
        while not stopping:
            if not select.select([sys.stdin],[],[],.1)[0]:
                if session.armed and not session.error and time.monotonic()-last>.5:
                    with redirect_stdout(sys.stderr):session.hold()
                    session.error='팔 제어 명령 수신이 중단되었습니다. 다시 연결해 주세요.'
                continue
            line=sys.stdin.readline()
            if not line:break
            try:
                item=json.loads(line);op=item['op'];now=time.monotonic();last=now
                if op not in ('release','close') and (now-item.get('sent',0)>.5 or item.get('sent',now)>now+.01):raise RuntimeError('오래된 팔 제어 명령입니다.')
                with redirect_stdout(sys.stderr):
                    if op=='close':session.close(release=item.get('release',True));session=None;send(dict(ok=True));break
                    if op=='release':session.release();result=dict(ok=True,angles=session.latest['angles'],torque_on=False,release_required=False)
                    elif session.error:raise RuntimeError(session.error)
                    elif op=='read':result=session.read()
                    elif op=='arm':result=session.arm(item['target'],item['entry'])
                    elif op=='target':result=session.update(item['target'])
                    elif op=='hold':session.hold();result=session.read()
                    else:raise ValueError('알 수 없는 USB 명령입니다.')
                send(result)
            except EntryPoseChanged as error:
                # The arm moved between two read-only samples, before activation.
                # Let the owner refresh the plan without poisoning the USB session.
                send(dict(session.latest,ok=False,retryable='entry_pose_changed',error=str(error)))
            except Exception as error:
                session.error=str(error)
                try:
                    with redirect_stdout(sys.stderr):session.hold()
                except Exception:pass
                send(dict(session.latest,ok=False,error=str(error)))
    except Exception as error:
        send(dict(ok=False,error=str(error)));raise SystemExit(1)
    finally:
        if session:
            with redirect_stdout(sys.stderr):session.close(release=True)

if __name__=='__main__':main()
