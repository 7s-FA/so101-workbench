import math
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from studio import cameras
from studio.cameras import Camera, CameraUnavailable, camera_status
from studio.config import load
from studio.engine import Engine
from studio.hardware import IL_ENTRY_SPEED
from teach_io.entry import CurrentPoseEntry, entry_duration
from test_studio import FakeHardware, FakeCamera


def test_failed_read_reopens_camera_and_keeps_monotonic_observations(monkeypatch):
    recovered=threading.Event();caps=[];events=[]
    class Capture:
        def __init__(self,*_):self.count=0;self.number=len(caps);self.released=False;caps.append(self)
        def isOpened(self):return True
        def set(self,*_):return True
        def get(self,prop):return 1196444237 if prop==cameras.cv2.CAP_PROP_FOURCC else 30
        def read(self):
            self.count+=1;time.sleep(.005)
            if self.number==0 and self.count==2:return False,None
            return True,np.zeros((4,4,3),dtype=np.uint8)
        def release(self):self.released=True
    monkeypatch.setattr(cameras.cv2,'VideoCapture',Capture)
    monkeypatch.setattr(cameras.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0,stdout='exposure_dynamic_framerate: 0',stderr=''))
    def log(self,event,**detail):
        events.append((event,detail))
        if event=='recovered':recovered.set()
    monkeypatch.setattr(Camera,'log',log)
    c=Camera('/dev/fake')
    try:
        assert recovered.wait(3),'Camera did not reopen after read failure'
        assert c.snapshot()['sequence']>=2
        assert c.failures==1 and c.reconnects==1
        assert camera_status(c)['connected']
        assert c.actual['fixed_frame_rate']
        assert [x[0] for x in events][:3]==['stream_started','capture_error','recovered']
    finally:c.close()
    assert all(cap.released for cap in caps)
    assert not c.thread.is_alive()


def test_single_read_failure_can_use_recent_real_frame_but_not_stale_data():
    c=Camera.__new__(Camera);c.device='/dev/fake';c.error='Camera frame read failed'
    c.frame=dict(acquired=time.monotonic(),sequence=10,jpeg=b'actual')
    assert c.snapshot() is c.frame
    assert camera_status(c)['state']=='recovering'
    original_time=c.frame['acquired'];c.frame=dict(c.frame,acquired=original_time-1)
    with pytest.raises(CameraUnavailable):c.snapshot()
    assert c.frame['sequence']==10 and c.frame['acquired']==original_time-1


def test_camera_loss_ends_recording_but_keeps_manual_motor_control(tmp_path,monkeypatch):
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured)
    e.cameras={name:FakeCamera() for name in ('gripper','table')}
    e.task_id=e.learning.tasks.save('SO101 test')['id']
    e.mode='teleop';e.ping();e.operation('record',{})
    try:
        for i in range(3):e.update(e.record_start+(i+1)/30,1/30)
        e.camera_interrupted(CameraUnavailable('temporary camera failure'))
        assert e.mode=='teleop' and e.error is None
        job=e.storage_job
        job.future.result(timeout=2)
        e.finish_storage(job)
        assert e.store.draft is not None and e.store.count==3
        before=len(e.hardware.commands);e.update(time.monotonic(),.02)
        assert len(e.hardware.commands)==before+1
        assert e.store.count==3  # No invalid frame was saved; no automatic recording restart.
        e.publish()
        assert e.public['recording_interruption']['reason']=='temporary camera failure'
        assert json.loads((e.store.draft/'metadata.json').read_text())['interruption']
        for result in ('Success','Failure'):
            with pytest.raises(ValueError,match='저장할 수 없습니다'):
                e.operation('save',{'result':result})
            with pytest.raises(ValueError,match='저장할 수 없습니다'):
                e.store.save(result)
        assert not e.store.list()
        e.operation('discard',{})
        e.operation('record',{})
        e.publish()
        assert e.public['recording_interruption'] is None
        for i in range(3):e.update(e.record_start+(i+1)/30,1/30)
        e.dataset_writer.request=lambda:None
        saved=e.operation('save',{'result':'Success'})
        assert saved['id']==1 and not saved.get('interruption')
    finally:e.close()


def test_camera_loss_holds_model_without_closing_motor_connection(tmp_path):
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured);e.mode='model'
    try:
        e.camera_interrupted(CameraUnavailable('lost'))
        assert e.mode=='holding' and e.hardware.follower.armed
        e.update(time.monotonic(),.02)
        assert e.hardware.commands[-1]==e.limit_goal(e.measured)
    finally:e.close()


def test_initial_move_is_five_times_faster_with_smooth_endpoints():
    start=[0.]*6;goal=[math.pi/2]*6
    legacy=CurrentPoseEntry(start,goal,0)
    fast=CurrentPoseEntry(start,goal,0,reference=True,peak_speed=IL_ENTRY_SPEED)
    assert legacy.duration/fast.duration==pytest.approx(5)
    assert fast.duration==pytest.approx(entry_duration(start,goal,IL_ENTRY_SPEED))
    previous=start;velocities=[]
    for i in range(1,400):
        q=fast.advance(goal,i*.02)
        velocities.append(max(abs(a-b)/.02 for a,b in zip(q,previous)));previous=q
        if fast.elapsed>=fast.duration:break
    assert max(velocities)<=IL_ENTRY_SPEED+1e-6
    assert velocities[0]<.001 and velocities[-1]<.001
    assert previous==pytest.approx(goal)
