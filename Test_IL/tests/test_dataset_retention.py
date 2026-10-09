import fcntl
import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
import threading

import pytest

from studio.dataset_retention import prune_dataset_versions, training_reference_guard
from studio.learning import Learning
from studio.store import Store, atomic_json


def generation(root, number):
    path=root/'.lerobot_versions'/f'{number:032x}'
    (path/'meta').mkdir(parents=True)
    atomic_json(path/'meta/info.json',{})
    (path/'payload').write_bytes(b'test dataset\n'*100)
    return path


def link_to(root, name, target):
    link=root/'tasks'/name/'lerobot';link.parent.mkdir(parents=True,exist_ok=True)
    link.symlink_to(target)
    return link


def job(root, name, dataset, state='completed'):
    path=root/'training'/name/'job.json';path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,dict(dataset=str(dataset),state=state))
    return path


def test_only_unreferenced_generations_are_removed(tmp_path):
    current=generation(tmp_path,1);link_to(tmp_path,'A',current)
    other=generation(tmp_path,2);link_to(tmp_path,'B',other)
    legacy=generation(tmp_path,3);(tmp_path/'lerobot').symlink_to(legacy)
    expected={current.name,other.name,legacy.name}
    for n,state in enumerate(['running','starting','stopping','completed','cancelled','interrupted','error'],4):
        dataset=generation(tmp_path,n);job(tmp_path,state,dataset,state);expected.add(dataset.name)
    checkpoint_source=generation(tmp_path,11);expected.add(checkpoint_source.name)
    config=tmp_path/'training/completed/output/checkpoints/001000/pretrained_model/train_config.json'
    config.parent.mkdir(parents=True);atomic_json(config,{'dataset':{'root':str(checkpoint_source)}})
    unused=generation(tmp_path,12)
    raw=tmp_path/'tasks/A/episodes/original';raw.parent.mkdir();raw.write_text('keep raw')
    model=config.parent/'model.safetensors';model.write_text('keep model')
    report=prune_dataset_versions(tmp_path)
    assert report['status']=='cleaned' and not report['errors']
    assert set(report['kept'])==expected and report['removed']==[unused.name]
    assert report['removed_bytes']>0 and not unused.exists()
    assert raw.read_text()=='keep raw' and model.read_text()=='keep model'
    assert all((tmp_path/'.lerobot_versions'/name).exists() for name in expected)


@pytest.mark.parametrize('corruption',['job_json','dataset_missing','checkpoint_json','checkpoint_type','broken_current'])
def test_uncertain_references_abort_before_any_deletion(tmp_path,corruption):
    current=generation(tmp_path,1);old=generation(tmp_path,2)
    link=link_to(tmp_path,'A',current);path=job(tmp_path,'run',current)
    if corruption=='job_json':path.write_text('{')
    elif corruption=='dataset_missing':atomic_json(path,{'dataset':str(tmp_path/'.lerobot_versions'/('f'*32))})
    elif corruption.startswith('checkpoint'):
        config=path.parent/'output/checkpoints/1/pretrained_model/train_config.json'
        config.parent.mkdir(parents=True);config.write_text('{' if corruption=='checkpoint_json' else '[]')
    else:
        link.unlink();link.symlink_to(tmp_path/'.lerobot_versions'/('f'*32))
    report=prune_dataset_versions(tmp_path)
    assert report['status']=='skipped' and report['errors'] and not report['removed']
    assert current.exists() and old.exists()


def test_dry_run_and_writer_lock_preserve_in_progress_files(tmp_path):
    old=generation(tmp_path,1)
    report=prune_dataset_versions(tmp_path,dry_run=True)
    assert report['status']=='dry_run' and report['candidates']==[old.name]
    assert not report['removed'] and old.exists()
    with (tmp_path/'.lerobot-write.lock').open('a') as writer:
        fcntl.flock(writer,fcntl.LOCK_EX)
        assert prune_dataset_versions(tmp_path)['status']=='busy'
        assert old.exists()
    assert prune_dataset_versions(tmp_path)['removed']==[old.name]


def test_cleanup_never_traverses_symlinks_or_unknown_directories(tmp_path):
    unused=generation(tmp_path,1)
    outside=tmp_path/'outside';outside.mkdir();(outside/'keep').write_text('keep')
    (unused/'linked').symlink_to(outside,target_is_directory=True)
    versions=unused.parent
    (versions/('f'*32)).symlink_to(outside,target_is_directory=True)
    unknown=versions/'manual-copy';unknown.mkdir()
    report=prune_dataset_versions(tmp_path)
    assert report['removed']==[unused.name]
    assert (outside/'keep').read_text()=='keep' and unknown.exists()
    assert (versions/('f'*32)).is_symlink()


def test_versions_directory_cannot_redirect_cleanup(tmp_path):
    outside=tmp_path/'outside';outside.mkdir()
    (tmp_path/'.lerobot_versions').symlink_to(outside,target_is_directory=True)
    assert prune_dataset_versions(tmp_path)['status']=='skipped'


@pytest.mark.parametrize('mode',['start','resume'])
def test_job_publication_is_protected_from_concurrent_cleanup(tmp_path,monkeypatch,mode):
    import studio.dataset_retention as retention
    from test_training_resume import learning_source
    if mode=='resume':
        learning,parent,_=learning_source(tmp_path)
        fields=('dataset','repo_id','episodes','studio_ids','fingerprint','calibration',
                'lerobot_calibration','camera_shapes','fps','start_pose','synthetic')
        source={key:parent[key] for key in fields}
    else:
        learning=Learning(Store(tmp_path));source={}
    old=generation(tmp_path,1);new=generation(tmp_path,2);unused=generation(tmp_path,3)
    link=link_to(tmp_path,'Current',old)
    source['dataset']=str(old)
    monkeypatch.setattr(learning,'snapshot',lambda task:source.copy())
    monkeypatch.setattr(learning.tasks,'get',lambda task:{'id':task,'name':'Current'})
    entered=threading.Event();release=threading.Event();collector_waiting=threading.Event()
    outcomes=[];failures=[]

    def launch(data,name):
        entered.set()
        assert release.wait(5)
        job(tmp_path,'new',old,'starting')
        return data
    monkeypatch.setattr(learning,'launch',launch)

    @contextmanager
    def observed_guard(root):
        collector_waiting.set()
        with training_reference_guard(root):yield
    monkeypatch.setattr(retention,'training_reference_guard',observed_guard)

    def start():
        try:
            if mode=='start':learning.start('task',{},'test')
            else:learning.resume('parent:014000',10,include_new=True)
        except BaseException as error:failures.append(error)
    training=threading.Thread(target=start);training.start()
    assert entered.wait(5)
    # Reproduce a newly published export between snapshot and job publication.
    link.unlink();link.symlink_to(new)
    collector=threading.Thread(target=lambda:outcomes.append(prune_dataset_versions(tmp_path)))
    collector.start()
    try:
        assert collector_waiting.wait(5)
        assert not outcomes and old.exists()
    finally:
        release.set();training.join(5);collector.join(5)
    assert not training.is_alive() and not collector.is_alive() and not failures
    assert outcomes[0]['removed']==[unused.name]
    assert old.exists() and new.exists()


@pytest.mark.parametrize('outcome',['success','stale','failure'])
def test_export_only_prunes_after_successful_publication(tmp_path,monkeypatch,capsys,outcome):
    path=Path(__file__).resolve().parents[1]/'scripts/write_dataset.py'
    spec=importlib.util.spec_from_file_location('retention_writer_test',path)
    writer=importlib.util.module_from_spec(spec);spec.loader.exec_module(writer)
    old=generation(tmp_path,1)
    def export(store,task_id):
        if outcome=='failure':raise ValueError('encoding failed')
        if outcome=='stale':return None
        new=generation(tmp_path,2);link_to(tmp_path,'Current',new)
        return 1
    monkeypatch.setattr(writer,'export_task',export)
    if outcome=='failure':
        with pytest.raises(ValueError,match='encoding failed'):writer.write_dataset(tmp_path)
    else:
        writer.write_dataset(tmp_path)
        result=json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        if outcome=='success':assert result['cleanup']['removed']==[old.name]
        else:assert result['cleanup'] is None
    assert old.exists()==(outcome!='success')
