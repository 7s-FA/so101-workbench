import json
import math
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from studio.learning import Tasks,Learning,parameters
from studio.store import Store,atomic_json
from studio.engine import Engine
from studio.config import load
from studio.server import create_app
from test_studio import episode


def test_tasks_are_user_named_and_stable_on_rename(tmp_path):
    store=Store(tmp_path);episode(store);tasks=Tasks(store)
    assert tasks.list()==[]
    first=tasks.save('컵 옮기기');tasks.assign(1,first['id'])
    tasks.save('컵 정리',first['id'])
    assert Tasks(store).get(first['id'])['name']=='컵 정리'
    assert store.episode(1)[0]['task_id']==first['id']
    tasks.assign(1,None);assert store.episode(1)[0]['task_id'] is None
    for name in ['', ' '*5, 'a'*81]:
        with pytest.raises(ValueError):tasks.save(name)
    with pytest.raises(ValueError):tasks.save('컵 정리')
    with pytest.raises(ValueError):tasks.assign(1,'unknown')


@pytest.mark.parametrize('values',[{'steps':0},{'steps':True},{'learning_rate':math.nan},
    {'n_action_steps':101,'chunk_size':100},{'device':'shell'},{'extra':1},{'batch_size':129}])
def test_invalid_training_parameters(values):
    with pytest.raises(ValueError):parameters(values)


def test_tasks_and_model_hardware_guard_over_api(tmp_path):
    app=create_app(data_root=tmp_path,launch_sim=False,test_mode=True)
    engine=app.state.engine;episode(engine.store)
    with TestClient(app) as client:
        def post(op,body):return client.post('/api/command/'+op,json=body,headers={'x-studio-command':'1'})
        assert client.get('/api/learning').json()['tasks']==[]
        task=post('task_save',{'name':'원하는 작업'}).json()
        assert post('task_assign',{'id':1,'task_id':task['id']}).status_code==200
        assert post('task_select',{'id':task['id']}).status_code==200
        assert post('model_run',{'workspace_clear':True}).status_code==409
        assert post('train_start',{'task_id':task['id'],'params':{},'name':'first'}).status_code==409
        assert not engine.hardware


def test_only_complete_checkpoints_are_selectable(tmp_path):
    learning=Learning(Store(tmp_path));directory=tmp_path/'training/job';directory.mkdir()
    atomic_json(directory/'job.json',dict(id='job',state='completed',task={'id':'t','name':'작업'},
        name='model',synthetic=True,created_at=1))
    checkpoint=directory/'output/checkpoints/000002/pretrained_model';checkpoint.mkdir(parents=True)
    for name in ['model.safetensors','policy_preprocessor.json','policy_postprocessor.json']:(checkpoint/name).touch()
    assert learning.models()==[]
    atomic_json(checkpoint.parent/'studio_ready.json',{'step':2})
    assert len(learning.models())==1 and learning.models()[0]['physical_allowed'] is False
    with pytest.raises(ValueError):learning.model('../escape')
    assert learning.model('job:000002')[0]==checkpoint


def test_model_admission_and_start_use_existing_entry(tmp_path,monkeypatch):
    engine=Engine(load(),tmp_path)
    task=engine.learning.tasks.save('이동')
    worker=SimpleNamespace(ready=True,error=None,close=lambda:None)
    engine.policy_worker=worker;engine.policy_job={'synthetic':True}
    with pytest.raises(ValueError,match='가상 시험'):engine.operation('model_run',{'workspace_clear':True})
    engine.policy_job.update(synthetic=False,fingerprint='wrong')
    with pytest.raises(ValueError,match='보정'):engine.operation('model_run',{'workspace_clear':True})
    engine.policy_job.update(fingerprint=engine.fingerprint,camera_shapes={'gripper':[48,64,3],'table':[48,64,3]},
        lerobot_calibration=json.loads((__import__('studio.config',fromlist=['ROOT']).ROOT/'config/lerobot_calibration.json').read_text()),start_pose=[0]*6)
    monkeypatch.setattr(engine,'frames',lambda:{k:{'shape':v} for k,v in engine.policy_job['camera_shapes'].items()})
    entry=[];monkeypatch.setattr(engine,'prepare',lambda goal,mode:entry.append((goal,mode)))
    with pytest.raises(ValueError,match='확인'):engine.operation('model_run',{})
    engine.operation('model_run',{'workspace_clear':True,'seconds':20})
    assert entry==[([0]*6,'model')] and engine.model_seconds==20
    engine.close()


def test_model_stale_output_cannot_reach_robot(tmp_path):
    engine=Engine(load(),tmp_path);engine.model_clock=100;engine.model_seconds=30
    engine.policy_worker=SimpleNamespace(error=None,result={'acquired':100,'action':[100]*6},close=lambda:None)
    engine.frames=lambda:{}
    with pytest.raises(RuntimeError,match='늦어'):engine.update_model(101,.03)
    engine.close()


def test_model_loop_mirrors_measured_pose_and_saves_then_holds(tmp_path):
    from studio.reference import ReferenceCodec
    from test_studio import frames
    engine=Engine(load(),tmp_path)
    q=[0.,0.,0.,0.,0.,.2]
    closed=[];requests=[];targets=[]
    worker=SimpleNamespace(ready=True,error=None,result=None,busy=False,
        close=lambda:closed.append(True),submit=lambda *args:requests.append(args))
    engine.policy_worker=worker;engine.model_id='job:2';engine.model_codec=ReferenceCodec()
    engine.policy_job={'fps':30};engine.model_seconds=1.;engine.measured=q.copy();engine.goal=q.copy()
    engine.hardware=SimpleNamespace(follower=SimpleNamespace(armed=True),leader=None,
        update=lambda goal,**kwargs:targets.append(list(goal)) or dict(angles=q.copy(),aligned=True,tracking={}),close=lambda **kwargs:None)
    engine.mode='preparing';engine.after_prepare='model';engine.initial_arm=True
    engine.entry={'began':100.,'duration':2.};engine.heartbeat=100.;engine.frames=lambda:frames()
    engine.update(100.,.03)
    assert engine.mode=='model'
    worker.result={'acquired':100.01,'action':[20.,0.,0.,0.,0.,50.],'elapsed':.02}
    engine.update(100.04,.04);engine.publish()
    assert engine.public['display_angles']==q and engine.public['display_source']=='live_measured'
    expected=engine.limit_goal(engine.mapping.to_angles(ReferenceCodec().follower_ticks([20.,0.,0.,0.,0.,50.])))
    assert engine.goal==expected and targets[-1]==expected
    assert len(requests)==1
    engine.update(101.05,.03)
    assert engine.mode=='holding' and engine.goal==engine.limit_goal(q) and closed
    report=json.loads((engine.eval_path/'metadata.json').read_text())
    assert report['outcome']=='time_limit_completed' and report['model_id']=='job:2'
    assert report['physical_grasp_success'] is None
    assert (engine.eval_path/'capture/samples.jsonl').exists()
    engine.close()


def test_training_pins_included_task_and_dataset_generation(tmp_path,monkeypatch):
    from studio.config import ROOT,fingerprint
    import studio.learning as module
    store=Store(tmp_path);catalog=Tasks(store)
    task=catalog.save('하나');other=catalog.save('둘')
    for _ in range(3):episode(store)
    catalog.assign(1,task['id']);catalog.assign(2,other['id']);catalog.assign(3,task['id']);store.include(3,False)
    for record in store.list():
        record.update(calibration=json.loads((ROOT/'config/follower_calibration.json').read_text()),
            lerobot_calibration=json.loads((ROOT/'config/lerobot_calibration.json').read_text()))
        atomic_json(store.path(record['id'])/'metadata.json',record)
    generation=tmp_path/'.lerobot_versions/fixed';generation.mkdir(parents=True)
    atomic_json(generation/'studio_manifest.json',{'repo_id':'local/test','episodes':[
        {'studio_id':n,'episode_index':n-1,'source_fingerprint':fingerprint()} for n in (1,2,3)]})
    (store.task_folder(task['id'])/'lerobot').symlink_to(generation)
    commands=[]
    class Process:
        def __init__(self,args,**kwargs):commands.append((args,kwargs))
        def wait(self,**kwargs):return 1
        def poll(self):return 1
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    learning=Learning(store);job=learning.start(task['id'],{'steps':2},'시험')
    learning.thread.join()
    assert job['studio_ids']==[1] and job['episodes']==[0] and job['dataset']==str(generation)
    assert commands[0][1]['start_new_session'] is True
    catalog.save('이름 변경',task['id']);catalog.assign(1,other['id'])
    assert job['task']['name']=='하나' and job['studio_ids']==[1]
    learning.close()
