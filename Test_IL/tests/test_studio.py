import copy
import json
import math
import time

import pytest
from fastapi.testclient import TestClient

from studio.config import CAMERAS, fingerprint, load
from studio.store import Store, validate_samples
from studio.engine import Engine, interpolate, validate_replay
from studio.hardware import mapping
from studio.server import create_app


def frames(t=100.):
    return {c:dict(jpeg=b'fixture',acquired=t,sequence=1,shape=[480,640,3]) for c in CAMERAS}


def sample(t,q=None):
    q = q or [0.,0.,0.,0.,0.,.5]
    return dict(t=t,state=q.copy(),action=q.copy(),command=q.copy())


def episode(store,synthetic=True):
    store.begin(dict(fingerprint=fingerprint(),synthetic=synthetic))
    for i in range(4):store.append(sample(i/30),frames())
    store.finish()
    return store.save('Success')


def test_discard_does_not_consume_number_or_delete_saved(tmp_path):
    s=Store(tmp_path)
    assert episode(s)['id']==1
    s.begin({});s.append(sample(0),frames());s.discard()
    assert episode(s)['id']==2
    assert [x['id'] for x in Store(tmp_path).list()]==[1,2]


def test_saved_frames_and_failures_persist(tmp_path):
    s=Store(tmp_path);s.begin({})
    for i in range(3):s.append(sample(i/30),frames(i))
    assert s.save('Failure')['included'] is False
    meta,rows=s.episode(1)
    assert len(rows)==3 and meta['duration']==2/30
    assert (s.path(1)/'gripper'/'0000000.jpg').read_bytes()==b'fixture'
    s.include(1,True)
    assert s.episode(1)[0]['included'] is True


def test_interrupted_draft_is_preserved(tmp_path):
    s=Store(tmp_path);s.begin({});s.append(sample(0),frames());s.finish()
    assert len(Store(tmp_path).abandoned())==1
    assert Store(tmp_path).list()==[]


def test_invalid_sample_and_id_are_rejected(tmp_path):
    s=Store(tmp_path);episode(s)
    _,rows=s.episode(1)
    rows[1]['t']=float('nan')
    with pytest.raises(ValueError):validate_samples(rows)
    with pytest.raises(ValueError):s.path('../config')
    with pytest.raises(ValueError):s.include(1,'yes')


def test_original_timeline_interpolation():
    rows=[sample(0,[0]*6),sample(1,[1]*6)]
    assert interpolate(rows,.25,'action')[0]==[.25]*6
    assert interpolate(rows,5,'state')[0]==[1]*6


def test_replay_requires_same_calibration_and_real_data(tmp_path):
    s=Store(tmp_path);episode(s)
    meta,rows=s.episode(1)
    with pytest.raises(ValueError,match='fixtures'):validate_replay(meta,rows,fingerprint(),mapping())
    meta['synthetic']=False
    validate_replay(meta,rows,fingerprint(),mapping())
    with pytest.raises(ValueError,match='calibration'):validate_replay(meta,rows,'changed',mapping())
    rows[1]['action'][0]=.5
    with pytest.raises(ValueError,match='velocity'):validate_replay(meta,rows,fingerprint(),mapping())


def test_long_gap_cannot_be_replayed(tmp_path):
    s=Store(tmp_path);episode(s,False)
    meta,rows=s.episode(1)
    rows[-1]['t']=1
    with pytest.raises(ValueError,match='gap'):validate_replay(meta,rows,fingerprint(),mapping())


@pytest.mark.parametrize('playing',[False,True])
def test_seek_preserves_playback_and_rebases_clock(tmp_path,monkeypatch,playing):
    e=Engine(load(),tmp_path);episode(e.store)
    try:
        e.select(1);goal=e.goal.copy()
        monkeypatch.setattr('studio.engine.time.monotonic',lambda:100.)
        if playing:e.operation('preview',{})
        e.operation('seek',{'t':.03})
        assert e.previewing is playing
        e.update(100.02,.02)
        assert e.play_t==pytest.approx(.05 if playing else .03)
        assert e.goal==goal and e.hardware is None
        # Preserve the state at drag start, even if playback ended during the drag.
        e.previewing=False
        e.operation('seek',{'t':.02,'playing':True})
        assert e.previewing
        e.operation('seek',{'t':999,'playing':True})
        assert not e.previewing and e.play_t==e.samples[-1]['t']
        with pytest.raises(ValueError):e.operation('seek',{'t':0,'playing':'true'})
        e.mode='teleop'
        with pytest.raises(ValueError):e.operation('seek',{'t':0,'playing':True})
    finally:e.close()


def test_offline_preview_does_not_change_robot_target(tmp_path):
    e=Engine(load(),tmp_path);episode(e.store)
    original=e.goal.copy()
    e.operation('select',{'id':1});e.operation('preview',{})
    e.update(time.monotonic()+.04,.04);e.publish()
    assert e.mode=='preview' and e.goal==original and e.hardware is None
    assert e.sim_state()['display_source']=='recorded_measured'
    e.close()


def test_no_recording_without_teleop_or_cameras(tmp_path):
    e=Engine(load(),tmp_path)
    with pytest.raises(ValueError,match='Enable leader'):e.operation('record',{})
    with pytest.raises(ValueError,match='Both'):e.frames()
    e.close()


def test_commands_cannot_overlap(tmp_path):
    e=Engine(load(),tmp_path);episode(e.store);e.operation('select',{'id':1})
    for mode in ('recording','replay','preparing','checking','teleop'):
        e.mode=mode
        with pytest.raises(ValueError):e.operation('select',{'id':1})
        with pytest.raises(ValueError):e.operation('preview',{})
    e.mode='idle';e.close()


def test_api_local_origin_and_test_mode(tmp_path):
    app=create_app(data_root=tmp_path,launch_sim=False,test_mode=True)
    headers={'x-studio-command':'1'}
    with TestClient(app) as client:
        assert client.get('/').status_code==200
        assert client.get('/api/status').json()['connected'] is False
        assert client.post('/api/command/stop',json={}).status_code==403
        assert client.post('/api/command/stop',json={},headers={**headers,'origin':'https://evil.example'}).status_code==403
        assert client.post('/api/command/connect',json={},headers=headers).status_code==409
        assert client.post('/api/heartbeat',json={},headers=headers).status_code==200
        assert client.post('/api/command/stop',json={},headers=headers).status_code==200
        assert client.get('/api/image/sim').status_code==204


def test_fault_stops_recording_and_does_not_auto_reconnect(tmp_path):
    e=Engine(load(),tmp_path)
    e.mode='fault'
    e.start()
    time.sleep(.06)
    assert e.public['connected'] is False
    e.close()


class FakeFollower:
    armed=True
    process=None


class FakeHardware:
    def __init__(self,q):
        self.q=list(q);self.follower=FakeFollower();self.leader=True
        self.leader_cfg={};self.anchor=list(q);self.origin=[2048]*6
        self.commands=[]
    def target(self):return self.q.copy(),[2048]*6
    def update(self,q,active=False):
        self.q=list(q);self.commands.append(list(q))
        return dict(angles=self.q.copy(),aligned=True,motors=[],tracking=dict(pace=1.,desired_angles=list(q),command_angles=list(q)))
    def close(self,release=True):pass


class FakeCamera:
    device='fixture';error=None;frame=None
    def snapshot(self):return frames(time.monotonic())['gripper']
    def close(self):pass


def test_record_save_replay_keep_original_and_write_evaluation(tmp_path,monkeypatch):
    e=Engine(load(),tmp_path)
    monkeypatch.setattr(e.dataset_writer,'request',lambda:None)
    e.hardware=FakeHardware([0.,0.,0.,0.,0.,.5])
    e.measured=e.hardware.q.copy();e.goal=e.measured.copy()
    e.cameras={name:FakeCamera() for name in CAMERAS}
    e.latest=e.hardware.update(e.goal)
    e.mode='teleop';e.ping();e.operation('record',{})
    started=e.record_start
    for i in range(5):e.update(started+(i+1)/30,1/30)
    e.operation('stop_record',{})
    saved=e.operation('save',{'result':'Success'})
    assert saved['id']==1 and saved['synthetic'] is False
    e.hold();e.operation('select',{'id':1})
    original=(e.store.path(1)/'samples.jsonl').read_bytes()
    e.entry=dict(start=e.measured,goal=e.measured,began=time.monotonic(),duration=2)
    e.initial_arm=True;e.after_prepare='replay';e.mode='preparing'
    now=time.monotonic();e.update(now,1/30)
    assert e.mode=='replay'
    for i in range(1,7):e.update(now+i/30,1/30)
    assert e.mode=='holding'
    assert (e.eval_path/'capture/gripper/0000000.jpg').exists()
    assert json.loads((e.eval_path/'metadata.json').read_text())['outcome']=='command_sequence_completed'
    assert (e.store.path(1)/'samples.jsonl').read_bytes()==original
    assert len(e.store.list())==1
    e.close()


def test_browser_loss_blocks_motor_update(tmp_path):
    e=Engine(load(),tmp_path)
    e.hardware=FakeHardware([0]*5+[.5]);e.mode='teleop'
    with pytest.raises(RuntimeError,match='Browser'):e.update(time.monotonic(),.03)
    assert e.hardware.commands==[]
    e.close()


def test_save_keeps_leader_control_for_successive_episodes(tmp_path,monkeypatch):
    e=Engine(load(),tmp_path)
    monkeypatch.setattr(e.dataset_writer,'request',lambda:None)
    e.hardware=FakeHardware(e.measured)
    e.cameras={name:FakeCamera() for name in CAMERAS}
    e.mode='teleop';e.ping()
    task=e.learning.tasks.save('연속 녹화 시험');e.task_id=task['id']
    first_content=None
    try:
        for number in (1,2):
            e.operation('record',{})
            for i in range(3):e.update(e.record_start+(i+1)/30,1/30)
            if number==1:e.operation('stop_record',{})
            goal=e.goal.copy()
            saved=e.operation('save',{'result':'Success'})
            e.publish()
            assert saved['id']==number and saved['task_id']==task['id']
            assert e.mode=='teleop' and e.goal==goal and e.hardware.follower.armed
            assert e.public['recording_seconds']==0 and e.store.draft is None
            before=len(e.hardware.commands)
            e.update(time.monotonic(),.02)
            assert len(e.hardware.commands)==before+1
            assert e.store.count==0  # Repositioning is not recorded automatically.
            if number==1:first_content=(e.store.path(1)/'samples.jsonl').read_bytes()
        assert (e.store.path(1)/'samples.jsonl').read_bytes()==first_content
    finally:e.close()


@pytest.mark.parametrize('mode',['holding','fault'])
def test_save_does_not_resume_previously_stopped_control(tmp_path,monkeypatch,mode):
    e=Engine(load(),tmp_path)
    monkeypatch.setattr(e.dataset_writer,'request',lambda:None)
    e.store.begin({})
    for i in range(3):e.store.append(sample(i/30),frames())
    e.store.finish();e.mode=mode
    try:
        e.operation('save',{'result':'Success'})
        assert e.mode==mode
    finally:e.close()


@pytest.mark.parametrize('op',['discard','reset'])
def test_discard_current_recording_never_moves_arm_or_removes_saved_episode(tmp_path,monkeypatch,op):
    e=Engine(load(),tmp_path)
    e.hardware=FakeHardware(e.measured)
    e.cameras={name:FakeCamera() for name in CAMERAS}
    episode(e.store)
    original=(e.store.path(1)/'samples.jsonl').read_bytes()
    e.mode='teleop';e.ping();e.operation('record',{})
    try:
        for i in range(3):e.update(e.record_start+(i+1)/30,1/30)
        goal=e.goal.copy();commands=len(e.hardware.commands)
        monkeypatch.setattr(e,'prepare',lambda *a:pytest.fail('Discard must not move the arm'))
        e.operation(op,{})
        assert e.mode=='teleop' and e.hardware.follower.armed
        assert e.goal==goal and len(e.hardware.commands)==commands
        assert e.store.draft is None and e.last_record_t is None
        assert (e.store.path(1)/'samples.jsonl').read_bytes()==original
        e.operation('record',{})
        assert e.mode=='recording' and e.store.count==0
    finally:e.close()


def test_camera_staleness_and_timestamp_skew_are_not_silent(tmp_path):
    e=Engine(load(),tmp_path)
    class OffsetCamera(FakeCamera):
        def __init__(self,offset):self.offset=offset
        def snapshot(self):return frames(time.monotonic()+self.offset)['table']
    e.cameras={'gripper':OffsetCamera(0),'table':OffsetCamera(-.15)}
    with pytest.raises(RuntimeError,match='timestamps'):e.frames()
    from studio.cameras import Camera
    c=Camera.__new__(Camera);c.frame=frames(time.monotonic()-1)['gripper'];c.error=None
    with pytest.raises(RuntimeError,match='stale'):c.snapshot()
    e.close()


def test_fault_state_does_not_keep_sending_motor_commands(tmp_path):
    e=Engine(load(),tmp_path)
    e.hardware=FakeHardware(e.measured);e.mode='fault'
    e.update(time.monotonic(),.03)
    assert not e.hardware.commands
    e.close()


def test_teleop_saturates_each_joint_independently(tmp_path):
    e=Engine(load(),tmp_path)
    e.hardware=FakeHardware(e.measured);e.mode='teleop';e.ping()
    target=e.measured.copy();target[0]=100.;target[1]+=.01
    e.hardware.target=lambda:(target,[2048]*6)
    e.update(time.monotonic(),.03)
    assert e.mapping.to_ticks(e.goal)[0]==e.mapping.cfg['tick_limits'][0][1]  # direct target is clipped to the encoder limit
    assert e.goal[1]==pytest.approx(target[1])
    assert e.goal[0]<math.pi
    e.close()


def test_absolute_target_is_independent_of_follower_start():
    from studio.reference import ReferenceCodec
    from studio.robot import reference_config
    cfg=reference_config();cfg['leader']=copy.deepcopy(cfg['follower'])  # test-only identical calibration
    codec=ReferenceCodec(cfg);m=mapping()
    ticks=[2200,1900,2150,2400,2048,2312]
    q=codec.leader_target(ticks,m)
    # With the workshop's identical arm ranges, absolute arm encoders mirror 1:1.
    assert m.to_ticks(q)[:5]==pytest.approx(ticks[:5],abs=1)
    assert q==codec.leader_target(ticks,m)


def test_teleop_entry_targets_leader_not_current_follower(tmp_path,monkeypatch):
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured);e.ping()
    leader=e.measured.copy();leader[0]+=.2
    e.hardware.target=lambda:(leader,[2048]*6)
    prepared=[]
    monkeypatch.setattr(e,'prepare',lambda q,mode:prepared.append((q,mode)))
    e.operation('teleop',{'workspace_clear':True})
    assert prepared==[(leader,'teleop')]
    e.close()


def test_leader_motion_during_entry_does_not_abort_or_change_worker_goal(tmp_path):
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured);e.ping()
    e.mode='checking';e.after_prepare='teleop';e.entry={'goal':e.measured.copy()}
    q=e.measured.copy();q[0]+=.2;e.hardware.target=lambda:(q,[2048]*6)
    e.update(time.monotonic(),.03)
    assert e.goal==e.entry['goal']
    assert e.mode=='checking' and e.leader_goal==q
    e.close()


def test_moving_leader_is_caught_up_without_jump_or_fixed_entry_timeout(tmp_path):
    from studio.hardware import IL_ENTRY_SPEED
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured)
    base=e.measured.copy();leader=base.copy();leader[0]+=.6
    e.hardware.target=lambda:(leader.copy(),[2048]*6)
    now=time.monotonic();e.heartbeat=now
    e.mode='preparing';e.after_prepare='teleop';e.initial_arm=True
    e.entry=dict(goal=base.copy(),began=now,duration=2)
    try:
        for i in range(1,250):
            now+=.02;e.heartbeat=now;before=e.goal.copy()
            if i==20:leader[0]+=.1  # Move again while catching up.
            e.update(now,.02)
            assert max(abs(a-b) for a,b in zip(before,e.goal))<=IL_ENTRY_SPEED*.02+1e-10
            if i==30:e.entry['began']-=100  # A moving leader is not a settle failure.
            if e.mode=='teleop':break
        assert e.mode=='teleop' and e.goal==pytest.approx(leader)
        assert len(e.hardware.commands)>50  # Heartbeat/feedback continued.
    finally:e.close()


def test_stop_after_fault_does_not_reopen_dead_command_pipe(tmp_path):
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured)
    e.mode='fault';e.error='Original control fault'
    try:
        for _ in range(3):
            e.operation('stop',{});e.update(time.monotonic(),.02)
        assert e.mode=='fault' and e.error=='Original control fault'
        assert e.hardware.commands==[]
    finally:e.close()


def test_alignment_completion_does_not_reanchor_to_measured_pose(tmp_path):
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured);e.ping()
    goal=e.measured.copy();goal[0]+=.02
    e.hardware.target=lambda:(goal,[2048]*6)
    e.mode='preparing';e.after_prepare='teleop';e.initial_arm=True
    e.entry=dict(goal=goal,began=time.monotonic(),duration=2)
    e.update(time.monotonic(),.03)
    assert e.mode=='teleop' and e.goal==goal
    e.close()
