import json

import pytest

from studio.dataset import DatasetWriter
from studio.engine import Engine
from studio.config import load


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def saved_task(root):
    folder=root/'tasks'/'Ball'
    save_json(folder/'episodes/episode_000001/metadata.json',
              dict(id=1,result='Success',included=True,fingerprint='same'))
    save_json(folder/'lerobot/studio_manifest.json',
              dict(task_name='Ball',episodes=[dict(studio_id=1,result='Success',included=True,source_fingerprint='same')]))
    return folder


def test_restart_recognizes_matching_export(tmp_path):
    saved_task(tmp_path)
    assert DatasetWriter(tmp_path).status['state']=='ready'


@pytest.mark.parametrize('change',['added','deleted','label','missing','renamed','broken'])
def test_restart_detects_stale_export(tmp_path,change):
    folder=saved_task(tmp_path)
    raw=folder/'episodes/episode_000001/metadata.json'
    manifest=folder/'lerobot/studio_manifest.json'
    if change=='added':
        save_json(folder/'episodes/episode_000002/metadata.json',
                  dict(id=2,result='Success',included=True,fingerprint='same'))
    elif change=='deleted':raw.unlink()
    elif change=='label':
        meta=json.loads(raw.read_text());meta.update(result='Failure',included=False);save_json(raw,meta)
    elif change=='missing':manifest.unlink()
    elif change=='renamed':folder.rename(folder.with_name('New name'))
    else:manifest.write_text('{')
    assert DatasetWriter(tmp_path).status['state']=='stale'


def test_start_resumes_interrupted_export_without_starting_training(tmp_path,monkeypatch):
    engine=Engine(load(),tmp_path);calls=[]
    engine.dataset_writer.status={'state':'stale'}
    monkeypatch.setattr(engine.dataset_writer,'request',lambda:calls.append('export'))
    monkeypatch.setattr(engine.learning,'start',lambda *args:pytest.fail('Must not start training'))
    engine.start()
    try:
        assert calls==['export']
        assert engine.hardware is None
    finally:engine.close()


def test_empty_dataset_needs_no_export(tmp_path):
    assert DatasetWriter(tmp_path).status['state']=='idle'
