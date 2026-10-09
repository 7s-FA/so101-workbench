import fcntl
import pytest
from studio.engine import Engine
from studio.config import load
from studio.store import Store
from studio.learning import Tasks
from test_studio import sample,frames


def test_unselected_folder_fails_before_camera_or_record_mutation(tmp_path):
    e=Engine(load(),tmp_path);e.mode='teleop'
    e.frames=lambda:pytest.fail('No frames should be read without a folder')
    try:
        with pytest.raises(ValueError,match='폴더'):e.operation('record',{})
        assert e.mode=='teleop' and e.store.draft is None
    finally:e.mode='idle';e.close()


def test_create_select_clear_folder_during_teleop(tmp_path):
    e=Engine(load(),tmp_path);e.mode='teleop'
    try:
        task=e.operation('task_save',{'name':'SO101 새 작업'})
        e.operation('task_select',{'id':task['id']})
        assert e.task_id==task['id'] and e.mode=='teleop'
        with pytest.raises(ValueError,match='멈춘'):e.operation('task_save',{'id':task['id'],'name':'다른 이름'})
        e.operation('task_select',{'id':None})
        assert e.task_id is None and e.mode=='teleop'
    finally:e.mode='idle';e.close()


def test_move_during_export_preserves_open_frames_and_updates_catalog(tmp_path):
    s=Store(tmp_path);t=Tasks(s);task=t.save('SO101 이동')
    s.begin({'robot_type':'so101'})
    for i in range(3):s.append(sample(i/30),frames())
    s.save('Success')
    with s.export_episode(1) as (meta,rows,open_frame):
        filename=rows[0]['cameras']['gripper']['file']
        with open_frame('gripper',filename) as f:before=f.read()
        with (tmp_path/'.lerobot-write.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            t.assign(1,task['id'])
        with open_frame('gripper',filename) as f:assert f.read()==before
    assert s.list()[0]['task_id']==task['id']
    assert s.path(1).parent.parent.name=='SO101 이동'


def test_continuous_record_save_discard_and_switch_folder(tmp_path,monkeypatch):
    from test_studio import FakeHardware,FakeCamera
    e=Engine(load(),tmp_path);e.hardware=FakeHardware(e.measured)
    e.cameras={k:FakeCamera() for k in ('gripper','table')};e.mode='teleop'
    monkeypatch.setattr(e.dataset_writer,'request',lambda:None)
    try:
        tasks=[e.operation('task_save',{'name':f'SO101 작업 {i}'}) for i in range(2)]
        saved=[]
        for i in range(20):
            task=tasks[i%2];e.operation('task_select',{'id':task['id']});e.ping()
            e.operation('record',{})
            for n in range(3):e.recorder.append(sample(n/30),frames())
            if i%4==3:e.operation('discard',{})
            else:saved.append(e.operation('save',{'result':'Success'}))
            assert e.mode=='teleop' and e.hardware.follower.armed
            assert e.store.draft is None and e.storage_job is None
        assert [m['id'] for m in saved]==list(range(1,16))
        assert len(e.store.list())==15
        for m in e.store.list():assert e.store.path(m['id']).parent.parent==e.store.task_folder(m['task_id'])
    finally:e.close()
