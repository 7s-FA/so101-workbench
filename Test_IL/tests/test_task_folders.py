import fcntl
import json
from pathlib import Path
import pytest
from studio.store import Store, atomic_json
from studio.learning import Tasks
from studio.engine import Engine
from studio.config import load
from test_studio import episode, sample, frames


def test_record_move_rename_and_delete_keep_payload_and_number(tmp_path):
    store=Store(tmp_path);tasks=Tasks(store)
    task=tasks.save('볼 옮기기')
    store.begin({'task_id':task['id']})
    for i in range(3):store.append(sample(i/30),frames())
    store.save('Success')
    old=store.path(1)
    assert old==tmp_path/'tasks/볼 옮기기/episodes/episode_000001'
    payload={p.relative_to(old):p.read_bytes() for p in old.rglob('*') if p.is_file() and p.name!='metadata.json'}
    generation=tmp_path/'.lerobot_versions/fixed';generation.mkdir(parents=True)
    (old.parent.parent/'lerobot').symlink_to('../../.lerobot_versions/fixed')
    tasks.save('공 옮기기',task['id'])
    assert not old.exists()
    assert (store.task_folder(task['id'])/'lerobot').resolve()==generation
    tasks.assign(1,None)
    assert store.path(1).parent.parent.name=='미분류'
    assert all((store.path(1)/p).read_bytes()==data for p,data in payload.items())
    store.delete(1)
    assert episode(store)['id']==2


def test_legacy_migration_is_repeatable_and_preserves_bytes(tmp_path):
    store=Store(tmp_path);episode(store)
    legacy=tmp_path/'episodes';legacy.mkdir()
    original=store.path(1);raw=(original/'samples.jsonl').read_bytes()
    original.rename(legacy/original.name)
    for _ in range(2):
        reopened=Store(tmp_path)
        assert [m['id'] for m in reopened.list()]==[1]
        assert (reopened.path(1)/'samples.jsonl').read_bytes()==raw
    assert not legacy.exists()


def test_folder_edits_lock_export_and_reject_paths(tmp_path):
    store=Store(tmp_path);tasks=Tasks(store);task=tasks.save('처음');episode(store)
    for name in ('../escape','a/b','a\\b','.hidden',' .hidden',' 미분류 ','미분류','\x00'):
        with pytest.raises(ValueError):tasks.save(name)
    with (tmp_path/'.lerobot-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with pytest.raises(ValueError):tasks.save('다음',task['id'])
        assert tasks.assign(1,task['id'])['task_id']==task['id']
    assert store.path(1).exists() and tasks.get(task['id'])['name']=='처음'


def test_select_folder_between_recordings_keeps_teleop(tmp_path):
    engine=Engine(load(),tmp_path)
    task=engine.learning.tasks.save('하우징')
    engine.mode='teleop'
    engine.operation('task_select',{'id':task['id']})
    assert engine.mode=='teleop' and engine.task_id==task['id']
    engine.store.begin({'task_id':task['id']})
    with pytest.raises(ValueError):engine.operation('task_select',{'id':None})
    engine.store.discard();engine.mode='idle';engine.close()
