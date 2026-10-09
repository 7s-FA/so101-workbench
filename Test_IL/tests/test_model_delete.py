import copy
import json
import shutil
import pytest
from fastapi.testclient import TestClient
from studio.server import create_app
from studio.store import atomic_json
from test_training_resume import learning_source


def another_checkpoint(source,step):
    other=source.parent/f'{step:06d}';shutil.copytree(source,other)
    atomic_json(other/'training_state/training_step.json',{'step':step})
    atomic_json(other/'studio_ready.json',{'step':step})
    return other


def test_delete_requires_explicit_confirmation(tmp_path):
    learning,job,ck=learning_source(tmp_path)
    for confirmed in [False,None,1,'true']:
        with pytest.raises(ValueError,match='확인'):learning.delete_model('parent:014000',confirmed)
        assert ck.exists()


def test_only_selected_checkpoint_removed_and_latest_alias_repaired(tmp_path):
    learning,job,ck=learning_source(tmp_path);old=another_checkpoint(ck,13000)
    (ck.parent/'last').symlink_to(ck.name)
    result=learning.delete_model('parent:014000',True)
    assert result['history_deleted'] is False
    assert not ck.exists() and old.exists()
    assert (ck.parent/'last').resolve()==old.resolve()
    assert (learning.root/'parent/job.json').exists()
    assert (tmp_path/'dataset/meta/info.json').exists()
    assert [m['id'] for m in learning.models()]==['parent:013000']
    result=learning.delete_model('parent:013000',True)
    assert result['history_deleted'] is True
    assert not (ck.parent/'last').is_symlink() and not learning.models()
    assert learning.jobs()==[] and not (learning.root/'parent').exists()


@pytest.mark.parametrize('state',['running','starting','stopping'])
def test_training_and_its_resume_source_are_protected(tmp_path,state):
    learning,job,ck=learning_source(tmp_path)
    job['state']=state;atomic_json(learning.root/'parent/job.json',job)
    with pytest.raises(ValueError,match='학습에 사용'):learning.delete_model('parent:014000',True)
    job['state']='cancelled';atomic_json(learning.root/'parent/job.json',job)
    child=copy.deepcopy(job);child.update(id='child',state=state,resume_checkpoint=str(ck/'pretrained_model'))
    (learning.root/'child').mkdir();atomic_json(learning.root/'child/job.json',child)
    with pytest.raises(ValueError,match='학습에 사용'):learning.delete_model('parent:014000',True)
    assert ck.exists()


def test_unknown_or_path_ids_never_delete(tmp_path):
    learning,job,ck=learning_source(tmp_path)
    for value in ['parent:999999','../parent:014000',str(ck),None]:
        with pytest.raises(ValueError):learning.delete_model(value,True)
        assert ck.exists()


def test_api_confirmation_loaded_model_and_second_deletion(tmp_path):
    learning,job,ck=learning_source(tmp_path);old=another_checkpoint(ck,13000)
    app=create_app(data_root=tmp_path,launch_sim=False,test_mode=True)
    engine=app.state.engine
    with TestClient(app) as client:
        def delete(model_id,**extra):
            return client.post('/api/command/model_delete',json={'id':model_id,**extra},headers={'x-studio-command':'1'})
        assert delete('parent:014000').status_code==409
        engine.model_id='parent:014000'
        assert delete('parent:014000',confirmed=True).status_code==409
        assert ck.exists()
        engine.model_id=None
        assert delete('parent:014000',confirmed=True).status_code==200
        assert delete('parent:013000',confirmed=True).status_code==200
        assert client.get('/api/learning').json()['models']==[]
        assert client.get('/api/learning').json()['jobs']==[]
        assert not ck.exists() and not old.exists()


def test_last_model_removes_logs_but_preserves_episodes_and_other_training(tmp_path):
    learning,job,ck=learning_source(tmp_path)
    for name in ['train.log','progress.json','inference.log']:(learning.root/'parent'/name).write_text('{}')
    child=copy.deepcopy(job);child.update(id='child',state='completed')
    (learning.root/'child').mkdir();atomic_json(learning.root/'child/job.json',child)
    external=tmp_path/'episodes';external.mkdir();(external/'sample').write_text('keep')
    (learning.root/'parent/episode_reference').symlink_to(external,target_is_directory=True)
    learning.delete_model('parent:014000',True)
    assert not (learning.root/'parent').exists()
    assert [j['id'] for j in learning.jobs()]==['child']
    assert (external/'sample').read_text()=='keep'
    assert (tmp_path/'dataset/meta/info.json').exists()


def test_failed_cleanup_does_not_republish_deleted_training(tmp_path,monkeypatch):
    learning,job,ck=learning_source(tmp_path)
    def fail(path):raise OSError('simulated disk failure')
    monkeypatch.setattr('studio.learning.shutil.rmtree',fail)
    with pytest.raises(ValueError,match='파일 정리'):learning.delete_model('parent:014000',True)
    assert learning.jobs()==[] and learning.models()==[]
    assert list((learning.root/'.deleted').glob('*/job.json'))
