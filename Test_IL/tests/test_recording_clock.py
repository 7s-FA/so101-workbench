import concurrent.futures
import queue
from types import SimpleNamespace
import pytest
from studio.engine import Engine
from studio.store import Store
from test_studio import sample,frames


def test_control_tick_is_sampled_after_record_command(monkeypatch):
    import studio.engine as module
    clock=[10.0];observed=[];e=Engine.__new__(Engine)
    class Quit:
        done=False
        def is_set(self):return self.done
        def wait(self,_):self.done=True
    e.quit=Quit();e.commands=queue.Queue();e.commands.put(('record',{},concurrent.futures.Future()))
    e.cfg={'record_fps':30};e.mode='teleop';e.storage_job=None
    def record(*_,**kwargs):
        clock[0]=10.00002;e.record_start=clock[0];e.mode='recording'
    e.operation=record;e.update=lambda now,dt:observed.append((now-e.record_start,dt))
    e.publish=lambda:None
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    e.run()
    assert observed[0][0]>=0
    assert observed[0][1]==pytest.approx(.00002)


@pytest.mark.parametrize('bad',[-.00002,float('nan'),float('inf')])
def test_store_rejects_invalid_time_before_writing_images(tmp_path,bad):
    store=Store(tmp_path);store.begin({})
    try:
        with pytest.raises(ValueError):store.append(sample(bad),frames())
        assert store.count==0
        assert not list(store.draft.glob('*/*.jpg'))
        assert (store.draft/'samples.jsonl').read_text()==''
    finally:store.finish()
