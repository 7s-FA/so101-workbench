import fcntl
import pytest
from fastapi.testclient import TestClient
from studio.store import Store
from studio.engine import Engine
from studio.config import load
from studio.server import create_app
from test_studio import episode


def test_result_change_keeps_raw_data_and_updates_inclusion(tmp_path):
    s=Store(tmp_path);episode(s)
    raw=(s.path(1)/'samples.jsonl').read_bytes()
    assert s.change_result(1,'Failure')['included'] is False
    assert s.episode(1)[0]['result']=='Failure'
    assert (s.path(1)/'samples.jsonl').read_bytes()==raw
    assert s.change_result(1,'Success')['included'] is True
    s.include(1,False);assert not s.episode(1)[0]['included']
    with pytest.raises(ValueError):s.change_result(1,'Other')


def test_deleted_ids_not_reused_even_when_all_deleted(tmp_path):
    s=Store(tmp_path);episode(s);episode(s)
    raw=(s.path(2)/'samples.jsonl').read_bytes()
    destination=s.delete(2)
    assert (destination/'samples.jsonl').read_bytes()==raw
    assert [e['id'] for e in s.list()]==[1]
    s.delete(1)
    assert episode(Store(tmp_path))['id']==3


def test_individual_edits_continue_during_export(tmp_path):
    s=Store(tmp_path);episode(s)
    with (tmp_path/'.lerobot-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        s.change_result(1,'Failure')
        s.include(1,True)
        assert s.episode(1)[0]['included'] is True
        s.delete(1)
    assert s.list()==[]



def test_delete_preview_resets_selection_and_requires_confirmation(tmp_path,monkeypatch):
    e=Engine(load(),tmp_path);episode(e.store);episode(e.store);requests=[]
    monkeypatch.setattr(e.dataset_writer,'request',lambda:requests.append(True))
    e.select(1);e.operation('preview',{})
    with pytest.raises(ValueError):e.operation('delete_episode',{'id':1})
    e.operation('delete_episode',{'id':1,'confirmed':True})
    assert not e.previewing and e.selected==2 and e.mode=='idle'
    e.operation('delete_episode',{'id':2,'confirmed':True})
    assert e.selected is None and e.samples==[] and e.play_t==0
    assert len(requests)==2
    e.close()


def test_management_api_does_not_operate_hardware(tmp_path,monkeypatch):
    app=create_app(data_root=tmp_path,launch_sim=False,test_mode=True)
    e=app.state.engine;episode(e.store);requests=[]
    monkeypatch.setattr(e.dataset_writer,'request',lambda:requests.append(True))
    with TestClient(app) as client:
        headers={'x-studio-command':'1'}
        def post(op,body):return client.post('/api/command/'+op,json=body,headers=headers)
        assert post('change_result',{'id':1,'result':'Failure'}).status_code==200
        assert client.get('/api/episodes').json()[0]['result']=='Failure'
        assert post('delete_episode',{'id':1}).status_code==409
        assert post('delete_episode',{'id':1,'confirmed':True}).status_code==200
        assert client.get('/api/episodes').json()==[]
        assert client.get('/api/status').json()['connected'] is False
    assert len(requests)==2


def test_no_edits_during_robot_operations(tmp_path,monkeypatch):
    e=Engine(load(),tmp_path);episode(e.store)
    monkeypatch.setattr(e.dataset_writer,'request',lambda:None)
    for mode in ('recording','replay','preparing','checking','teleop'):
        e.mode=mode
        for op,data in [('delete_episode',{'id':1,'confirmed':True}),('change_result',{'id':1,'result':'Failure'}),('include',{'id':1,'included':False})]:
            with pytest.raises(ValueError,match='작업을 멈춘'):e.operation(op,data)
    e.mode='idle';e.close()


def test_export_frame_handles_survive_individual_delete(tmp_path):
    s=Store(tmp_path);episode(s);episode(s)
    original=(s.path(2)/'metadata.json').read_bytes()
    with s.export_episode(1) as (meta,rows,open_frame):
        s.delete(1)
        with open_frame('gripper',rows[0]['cameras']['gripper']['file']) as frame:
            assert frame.read()==b'fixture'
    assert (s.path(2)/'metadata.json').read_bytes()==original
    assert [e['id'] for e in s.list()]==[2]
    # A not-yet-encoded snapshot can still open its now-archived source.
    with s.export_episode(1) as (meta,rows,open_frame):assert meta['id']==1


def test_individual_api_edits_and_repeated_delete_while_encoding(tmp_path,monkeypatch):
    app=create_app(data_root=tmp_path,launch_sim=False,test_mode=True)
    e=app.state.engine
    for _ in range(3):episode(e.store)
    original=(e.store.path(3)/'metadata.json').read_bytes()
    def writing():e.dataset_writer.status={'state':'writing','message':'fixture'}
    monkeypatch.setattr(e.dataset_writer,'request',writing)
    with TestClient(app) as client, (tmp_path/'.lerobot-write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        writing()
        def post(op,data):return client.post('/api/command/'+op,json=data,headers={'x-studio-command':'1'})
        assert post('change_result',{'id':1,'result':'Failure'}).status_code==200
        assert (e.store.path(3)/'metadata.json').read_bytes()==original
        for number in (1,2):
            assert post('delete_episode',{'id':number,'confirmed':True}).status_code==200
        assert [r['id'] for r in client.get('/api/episodes').json()]==[3]
        assert (e.store.path(3)/'metadata.json').read_bytes()==original
