import json
from pathlib import Path
import pytest
from studio.checkpoints import prune_previous_checkpoints
from studio.learning import parameters
from studio.store import atomic_json
from test_training_resume import learning_source


def checkpoint(root, step):
    p=root/f'{step:06d}'
    for name in ['model.safetensors','train_config.json','policy_preprocessor.json','policy_postprocessor.json']:
        f=p/'pretrained_model'/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('data')
    for name in ['optimizer_state.safetensors','optimizer_param_groups.json','rng_state.safetensors']:
        f=p/'training_state'/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('data')
    atomic_json(p/'training_state/training_step.json',{'step':step})
    atomic_json(p/'studio_ready.json',{'step':step})
    return p


def test_keep_latest_removes_only_previous_completed_in_same_run(tmp_path):
    root=tmp_path/'run/checkpoints'
    old=checkpoint(root,1);current=checkpoint(root,2);newer=checkpoint(root,3)
    incomplete=root/'000000';incomplete.mkdir()
    other=checkpoint(tmp_path/'parent/checkpoints',1)
    (root/'000004').symlink_to(other,target_is_directory=True)
    (root/'last').symlink_to(current.name,target_is_directory=True)
    removed=prune_previous_checkpoints(current)
    assert removed==['000001'] and not old.exists()
    assert all(p.exists() for p in (current,newer,incomplete,other,root/'last',root/'000004'))


@pytest.mark.parametrize('missing',['studio_ready.json','pretrained_model/model.safetensors','training_state/optimizer_state.safetensors'])
def test_incomplete_new_model_never_removes_previous(tmp_path,missing):
    old=checkpoint(tmp_path,1);new=checkpoint(tmp_path,2)
    (new/missing).unlink()
    with pytest.raises(ValueError,match='incomplete'):prune_previous_checkpoints(new)
    assert (old/'studio_ready.json').exists()


def test_inconsistent_new_model_preserves_previous(tmp_path):
    old=checkpoint(tmp_path,1);new=checkpoint(tmp_path,2)
    atomic_json(new/'training_state/training_step.json',{'step':1})
    with pytest.raises(ValueError,match='mismatch'):prune_previous_checkpoints(new)
    assert old.exists()


def test_retention_defaults_on_and_accepts_explicit_opt_out():
    assert parameters({})['keep_latest_only'] is True
    assert parameters({'keep_latest_only':False})['keep_latest_only'] is False
    assert parameters({'keep_latest_only':True})['keep_latest_only'] is True
    for value in ['true',1,None]:
        with pytest.raises(ValueError):parameters({'keep_latest_only':value})


def test_resume_retention_only_affects_new_run(tmp_path,monkeypatch):
    learning,parent,source=learning_source(tmp_path)
    monkeypatch.setattr(learning,'launch',lambda job,name:job)
    new=learning.resume('parent:014000',10,keep_latest_only=True)
    assert new['params']['keep_latest_only'] is True
    assert 'keep_latest_only' not in json.loads((learning.root/'parent/job.json').read_text())['params']
    assert source.exists()
    with pytest.raises(ValueError):learning.resume('parent:014000',1,keep_latest_only='true')
