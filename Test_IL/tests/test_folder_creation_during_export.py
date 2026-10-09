import concurrent.futures
import fcntl
import json
import threading
import pytest
from fastapi.testclient import TestClient
from studio.config import load
from studio.server import create_app
from studio.store import Store
from studio.learning import Tasks


def test_new_folder_and_selection_work_during_export_but_rename_is_protected(tmp_path):
    app=create_app(load(),tmp_path,launch_sim=False,test_mode=True)
    headers={'X-Studio-Command':'1'}
    with TestClient(app) as client:
        with (tmp_path/'.lerobot-write.lock').open('a') as locked:
            fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=client.post('/api/command/task_save',headers=headers,json={'name':'새 SO101 작업'})
            assert result.status_code==200,result.text
            task=result.json()
            assert (tmp_path/'tasks/새 SO101 작업').is_dir()
            assert client.post('/api/command/task_select',headers=headers,json={'id':task['id']}).status_code==200
            rename=client.post('/api/command/task_save',headers=headers,json={'id':task['id'],'name':'변경'})
            assert rename.status_code==409
            assert not (tmp_path/'tasks/변경').exists()
            duplicate=client.post('/api/command/task_save',headers=headers,json={'name':'새 SO101 작업'})
            assert duplicate.status_code==409 and '같은 작업 이름' in duplicate.text
        assert client.post('/api/command/task_save',headers=headers,json={'id':task['id'],'name':'변경'}).status_code==200


def test_concurrent_creators_preserve_both_catalog_entries(tmp_path):
    stores=[Store(tmp_path),Store(tmp_path)];barrier=threading.Barrier(2)
    def create(i):barrier.wait(timeout=2);return Tasks(stores[i]).save('작업 '+str(i))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(create,i) for i in range(2)]
        results=[future.result(timeout=3) for future in futures]
    assert len({v['id'] for v in results})==2
    assert {v['name'] for v in Tasks(stores[0]).list()}=={'작업 0','작업 1'}


def test_catalog_failure_rolls_back_only_new_empty_folder(tmp_path,monkeypatch):
    import studio.learning as module
    tasks=Tasks(Store(tmp_path));original=tasks.save('기존')
    before=(tmp_path/'tasks.json').read_bytes()
    def fail(*args):raise OSError('disk failure')
    monkeypatch.setattr(module,'atomic_json',fail)
    with pytest.raises(OSError):tasks.save('새 폴더')
    assert not (tmp_path/'tasks/새 폴더').exists()
    assert (tmp_path/'tasks.json').read_bytes()==before
    assert tasks.get(original['id'])['name']=='기존'
