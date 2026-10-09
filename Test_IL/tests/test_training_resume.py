import json
from types import SimpleNamespace
import pytest
from studio.learning import Learning
from studio.store import Store,atomic_json
from studio.engine import Engine
from studio.config import load
from test_studio import FakeHardware


def learning_source(tmp_path):
    learning=Learning(Store(tmp_path));dataset=tmp_path/'dataset';(dataset/'meta').mkdir(parents=True)
    atomic_json(dataset/'meta/info.json',{})
    params=dict(steps=100000,batch_size=8,learning_rate=1e-5,chunk_size=100,n_action_steps=20,save_freq=1000,seed=1000,num_workers=2,device='cuda',pretrained_backbone=True)
    job=dict(id='parent',state='cancelled',name='ball',task={'id':'task','name':'Ball'},params=params,created_at=1,
             dataset=str(dataset),repo_id='local/test',episodes=[0],studio_ids=[1],fingerprint='same',calibration={'zero':1},
             lerobot_calibration={'a':1},camera_shapes={'gripper':[480,640,3],'table':[480,640,3]},fps=30,start_pose=[0]*6,synthetic=False)
    (learning.root/'parent').mkdir()
    atomic_json(learning.root/'parent/job.json',job)
    ck=learning.root/'parent/output/checkpoints/014000';(ck/'pretrained_model').mkdir(parents=True)
    for name in ['model.safetensors','policy_preprocessor.json','policy_postprocessor.json']:(ck/'pretrained_model'/name).touch()
    (ck/'training_state').mkdir()
    for name in ['optimizer_param_groups.json','optimizer_state.safetensors','rng_state.safetensors']:(ck/'training_state'/name).touch()
    atomic_json(ck/'training_state/training_step.json',{'step':14000});atomic_json(ck/'studio_ready.json',{'step':14000})
    return learning,job,ck


def test_resume_uses_saved_dataset_and_additional_steps(tmp_path,monkeypatch):
    learning,parent,ck=learning_source(tmp_path)
    monkeypatch.setattr(learning,'launch',lambda job,name:dict(job,name=name))
    monkeypatch.setattr(learning,'snapshot',lambda task:pytest.fail('Same-data resume must not depend on mutable source episodes'))
    job=learning.resume('parent:014000',86000)
    assert job['params']['steps']==100000 and job['start_step']==14000
    assert job['dataset']==parent['dataset'] and job['studio_ids']==[1]
    assert job['params']['n_action_steps']==20 and job['normalization']=='lerobot_native'
    assert job['resume_checkpoint']==str(ck/'pretrained_model')
    assert json.loads((learning.root/'parent/job.json').read_text())==parent


def test_resume_new_data_checks_compatibility_and_pins_snapshot(tmp_path,monkeypatch):
    learning,parent,ck=learning_source(tmp_path)
    fields=('dataset','repo_id','episodes','studio_ids','fingerprint','calibration','lerobot_calibration','camera_shapes','fps','start_pose','synthetic')
    source={k:parent[k] for k in fields};source.update(episodes=[0,1],studio_ids=[1,2])
    monkeypatch.setattr(learning,'snapshot',lambda task:source.copy())
    monkeypatch.setattr(learning,'launch',lambda job,name:job)
    assert learning.resume('parent:014000',100,True)['studio_ids']==[1,2]
    source['fps']=15
    with pytest.raises(ValueError,match='카메라·보정'):learning.resume('parent:014000',100,True)


@pytest.mark.parametrize('steps',[0,True,1.5,1000001])
def test_resume_rejects_bad_additional_steps(tmp_path,steps):
    learning,parent,ck=learning_source(tmp_path)
    with pytest.raises(ValueError):learning.resume('parent:014000',steps)


def test_incomplete_training_state_not_resumable(tmp_path):
    learning,parent,ck=learning_source(tmp_path)
    (ck/'training_state/optimizer_state.safetensors').unlink()
    assert not learning.models()[0]['resumable']
    with pytest.raises(ValueError,match='재개 상태'):learning.resume('parent:014000',1)


def test_stop_requests_checkpoint_instead_of_killing_process(tmp_path):
    learning,parent,ck=learning_source(tmp_path)
    learning.active='parent';learning.process=SimpleNamespace(poll=lambda:None)
    parent['state']='running';atomic_json(learning.root/'parent/job.json',parent)
    learning.stop()
    assert (learning.root/'parent/stop_requested').exists()
    assert learning.jobs()[0]['state']=='stopping'
    assert json.loads((learning.root/'parent/job.json').read_text())['state']=='running'


def test_training_does_not_block_teleop_record_save(tmp_path,monkeypatch):
    from test_studio import frames
    engine=Engine(load(),tmp_path);engine.learning.process=SimpleNamespace(poll=lambda:None)
    engine.hardware=FakeHardware(engine.measured);engine.hardware.leader_cfg={}
    engine.latest={'power_profile':{}}
    # Capture the admission path without physically enabling a motor.
    monkeypatch.setattr(engine,'prepare',lambda goal,mode:setattr(engine,'mode',mode))
    monkeypatch.setattr(engine,'frames',frames)
    monkeypatch.setattr(engine.dataset_writer,'request',lambda:None)
    engine.ping()
    try:
        engine.operation('teleop',{'workspace_clear':True})
        assert engine.mode=='teleop'
        engine.task_id=engine.learning.tasks.save('SO101 test')['id']
        engine.operation('record',{})
        for t in [.01,.05]:
            engine.recorder.append(dict(t=t,acquired=t,state=engine.measured,action=engine.measured,command=engine.measured),frames())
        result=engine.operation('save',{'result':'Success'})
        assert result['id']==1 and engine.mode=='teleop'
        assert engine.store.list()[0]['samples']==2
    finally:
        engine.learning.process=None;engine.close()
