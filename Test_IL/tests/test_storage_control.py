import concurrent.futures
import threading
import time
import pytest
from studio.config import load, CAMERAS
from studio.engine import Engine
from test_studio import FakeHardware, FakeCamera, sample, frames


def make_engine(tmp_path, monkeypatch):
    e=Engine(load(),tmp_path)
    monkeypatch.setattr(e.dataset_writer,'request',lambda:None)
    e.hardware=FakeHardware(e.measured)
    e.cameras={name:FakeCamera() for name in CAMERAS}
    e.task_id=e.learning.tasks.save('SO101 test')['id']
    e.mode='teleop';e.ping()
    e.operation('record',{})
    for i in range(4):e.recorder.append(sample(i/30),frames())
    return e


@pytest.mark.parametrize('op',['save','discard','stop_record'])
def test_slow_storage_does_not_interrupt_motor_updates_or_stop(tmp_path,monkeypatch,op):
    e=make_engine(tmp_path,monkeypatch)
    entered=threading.Event();release=threading.Event();progress=threading.Event()
    finish=e.recorder.finish;timestamps=[]
    def slow_finish():
        entered.set()
        assert release.wait(3)
        finish()
    monkeypatch.setattr(e.recorder,'finish',slow_finish)
    update=e.hardware.update
    def motor_update(*args,**kwargs):
        timestamps.append(time.monotonic())
        if entered.is_set() and timestamps[-1]-timestamps[0]>.4:progress.set()
        return update(*args,**kwargs)
    monkeypatch.setattr(e.hardware,'update',motor_update)
    reply=concurrent.futures.Future()
    e.commands.put((op,{'result':'Success'},reply));e.start()
    try:
        assert entered.wait(1)
        assert progress.wait(1.5), e.public
        assert len(timestamps)>10 and max(b-a for a,b in zip(timestamps,timestamps[1:]))<.25
        assert not reply.done() and e.mode=='teleop'
        with pytest.raises(ValueError,match='처리'):e.call('record')
        assert e.call('stop')['ok']
        assert e.mode=='holding'
        release.set();reply.result(timeout=2)
        assert e.mode=='holding'  # Saving must not undo Stop.
        assert e.store.draft is None if op!='stop_record' else e.store.draft is not None
        if op=='save':assert e.store.episode(1)[0]['samples']==4
    finally:
        release.set();e.close()


def test_save_disk_error_keeps_draft_and_motor_session(tmp_path,monkeypatch):
    e=make_engine(tmp_path,monkeypatch)
    def fail(*args):raise OSError('disk failure fixture')
    monkeypatch.setattr(e.store,'save',fail)
    reply=concurrent.futures.Future()
    e.commands.put(('save',{'result':'Success'},reply));e.start()
    try:
        with pytest.raises(OSError,match='disk failure'):reply.result(timeout=2)
        assert e.mode=='teleop' and e.store.draft is not None
        assert e.storage_job is None and e.hardware.follower.armed
        assert e.call('stop')['ok']
    finally:e.close()
