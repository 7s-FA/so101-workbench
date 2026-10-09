import pytest
from studio.execution import options,ChunkTimeline,GoalFilter
from test_training_resume import learning_source


def test_config_uses_model_length_and_ensemble_requires_one():
    assert options({}, {'n_action_steps':100,'chunk_size':100})['action_steps']==100
    assert options({'mode':'ensemble','action_steps':20})['action_steps']==1
    for bad in [{'action_steps':101},{'filter_ms':float('nan')},{'velocity_scale':2},
                {'acceleration_scale':-1},{'gripper_filter':1},{'mode':'unknown'},{'unknown':1}]:
        with pytest.raises(ValueError):options(bad)


def test_execution_options_round_trip_does_not_modify_training(tmp_path):
    learning,job,ck=learning_source(tmp_path)
    before=(learning.root/'parent/job.json').read_bytes()
    config=learning.save_execution_options('parent:014000',{'mode':'prefetch','filter_ms':30,'action_steps':10})
    model=learning.models()[0]
    assert model['execution_options']==config and model['chunk_size']==100
    assert config['mode']=='prefetch'
    assert (learning.root/'parent/job.json').read_bytes()==before
    with pytest.raises(ValueError):learning.save_execution_options('parent:014000',{'velocity_scale':10})
    assert learning.models()[0]['execution_options']==config


def test_prefetch_drops_elapsed_prefix_and_executes_once_by_time():
    timeline=ChunkTimeline(10,options({'mode':'prefetch','action_steps':10}))
    actions=[[i]*6 for i in range(10)]
    assert timeline.accept({'acquired':10.,'actions':actions},10.21)
    assert timeline.take(10.21)==(None,False)
    assert timeline.take(10.31)==([3.]*6,True)
    assert timeline.take(10.32)==(None,False)
    # Missed control iterations skip obsolete targets rather than speeding through them.
    assert timeline.take(10.75)==([7.]*6,False)
    assert timeline.take(11.5)==(None,False)
    assert timeline.metrics()['buffer_waits']==1
    timeline.take(11.6);assert timeline.waits==1


def test_prefetch_replaces_only_overlapping_future_targets():
    timeline=ChunkTimeline(10,options({'mode':'prefetch','action_steps':10}))
    timeline.accept({'acquired':0.,'actions':[[1]*6]*10},0.)
    timeline.take(0.)
    timeline.accept({'acquired':.4,'actions':[[2]*6]*10},.51)
    assert timeline.take(.52)==([1.]*6,False)
    assert timeline.take(.61)==([2.]*6,True)
    assert not timeline.accept({'acquired':.3,'actions':[[3]*6]*10},.7)
    assert timeline.take(.71)==([2.]*6,False)


def test_stale_chunk_does_not_replace_valid_queue_and_auto_trigger():
    timeline=ChunkTimeline(30,options({'mode':'prefetch','action_steps':20}),warm_ms=100)
    assert timeline.should_request()
    timeline.accept({'acquired':1.,'actions':[[0]*6]*100},1.02)
    assert not timeline.should_request()
    before=list(timeline.queue)
    assert not timeline.accept({'acquired':1.1,'actions':[[1]*6]*100},1.7)
    assert list(timeline.queue)==before and timeline.dropped==1
    timeline.take(1.51);assert timeline.should_request()


@pytest.mark.parametrize('dt',[0.,.001,1/30,.1])
def test_default_target_passes_through_without_motion_ceiling(dt):
    f=GoalFilter([0]*6,options())
    target=[.7,-.5,.6,1.2,-.8,.9]
    assert f.apply(target,1.,dt,True)==target
    next_target=[-.2,.4,.1,.8,.6,.2]
    assert f.apply(next_target,2.,dt,True)==next_target
    assert f.apply(next_target,3.,dt,False)==next_target


def test_legacy_saved_limits_cannot_restore_the_old_motion_ceiling():
    cfg=options({'velocity_scale':.1,'acceleration_scale':1,'action_steps':100})
    assert 'velocity_scale' not in cfg and 'acceleration_scale' not in cfg
    assert cfg['action_steps']==100
    assert GoalFilter([0]*6,cfg).apply([1]*6,1.,.001)==[1]*6


def test_filter_and_blend_exempt_gripper_until_enabled():
    cfg=options({'filter_ms':100,'blend_ms':80})
    f=GoalFilter([0]*6,cfg)
    q=f.apply([1]*6,1.,.02,True)
    assert q[:5]==[0]*5 and q[5]==1
    q=f.apply([1]*6,1.04,.02)
    assert 0<q[0]<.5 and q[5]==1
    f=GoalFilter([0]*6,options(dict(cfg,gripper_filter=True)))
    assert f.apply([1]*6,1.,.02,True)==[0]*6


def test_direct_targets_still_require_six_finite_values():
    f=GoalFilter([0]*6,options())
    with pytest.raises(ValueError):f.apply([float('nan')]*6,1.,.02)
    with pytest.raises(ValueError):f.apply([float('inf')]*6,1.,.02)
    with pytest.raises(ValueError):f.apply([0]*5,1.,.02)


@pytest.mark.parametrize('mode',['original','prefetch','ensemble'])
@pytest.mark.parametrize('values',[[20,-20,30,30,-20,70],[250,-250,250,200,-200,150]])
def test_engine_transmits_model_target_directly_but_keeps_joint_ranges(tmp_path,mode,values):
    from types import SimpleNamespace
    from studio.engine import Engine
    from studio.config import load
    from studio.reference import ReferenceCodec
    engine=Engine(load(),tmp_path)
    engine.frames=lambda:{};engine.policy_job={'fps':30}
    engine.model_clock=10.;engine.model_seconds=30.;engine.model_submit_at=10.
    engine.model_options=options({'mode':mode,'action_steps':10})
    engine.model_timeline=ChunkTimeline(30,engine.model_options)
    engine.model_codec=ReferenceCodec();engine.goal=[0,0,0,0,0,.2]
    result={'acquired':10.,'action':values,'actions':[values]*10}
    engine.policy_worker=SimpleNamespace(error=None,result=result,busy=False,close=lambda:None)
    from studio.robot import pose_to_angles
    expected=engine.mapping.clamp_target(pose_to_angles(values))
    try:
        engine.update_model(10.,.001)
        assert engine.goal==pytest.approx(expected,abs=.002)
        engine.mapping.to_ticks(engine.goal)
    finally:engine.close()


def test_prefetch_rejects_one_step_and_ensemble_avoids_double_blending():
    with pytest.raises(ValueError,match='2 이상'):options({'mode':'prefetch','action_steps':1})
    assert options({'mode':'ensemble','blend_ms':80})['blend_ms']==0


def test_runtime_mode_does_not_mutate_model_or_training_config(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from studio.engine import Engine
    from studio.config import load
    import studio.engine as module
    engine=Engine(load(),tmp_path/'engine')
    learning,job,ck=learning_source(tmp_path/'models')
    engine.learning=learning
    job['robot_type']='so101'
    __import__('studio.store',fromlist=['atomic_json']).atomic_json(learning.root/'parent/job.json',job)
    calls=[]
    monkeypatch.setattr(module,'PolicyWorker',lambda *args: calls.append(args) or SimpleNamespace(close=lambda:None))
    engine.operation('model_load',{'id':'parent:014000','device':'cpu','options':{'mode':'prefetch','action_steps':10}})
    assert calls[0][0]==ck/'pretrained_model' and calls[0][3]['mode']=='prefetch'
    assert engine.model_options==learning.models()[0]['execution_options']
    assert engine.model_metrics=={}
    assert learning.models()[0]['default_execution_options']['mode']=='original'
    engine.mode='model'
    with pytest.raises(ValueError,match='멈춘 뒤'):engine.operation('model_settings',{'id':'parent:014000','options':{}})
    engine.mode='idle';engine.close()


def test_engine_prefetch_schedules_once_holds_empty_buffer_and_resumes(tmp_path):
    from types import SimpleNamespace
    from studio.engine import Engine
    from studio.config import load
    from studio.reference import ReferenceCodec
    engine=Engine(load(),tmp_path)
    engine.frames=lambda:{};engine.policy_job={'fps':30}
    engine.model_clock=10.;engine.model_seconds=30.;engine.model_submit_at=10.
    engine.model_options=options({'mode':'prefetch','action_steps':10})
    engine.model_timeline=ChunkTimeline(30,engine.model_options)
    engine.model_codec=ReferenceCodec();engine.goal=[0,0,0,0,0,.2]
    engine.measured=engine.goal.copy()
    requests=[]
    worker=SimpleNamespace(error=None,result=None,busy=False,close=lambda:None,submit=lambda *a:requests.append(a))
    engine.policy_worker=worker
    worker.result={'acquired':10.,'observation_at':9.99,'actions':[[i]*5+[50] for i in range(10)],'computed_chunk':True,'compute_ms':20}
    engine.update_model(10.025,.025)
    initial=engine.goal.copy()
    assert engine.model_command_count==0  # Time 0 already passed; sample 1 is not yet due.
    engine.update_model(10.04,.015)
    assert engine.model_command_count==1 and engine.goal!=initial
    assert engine.model_prediction_count==1  # Reading a result twice is not another model prediction.
    engine.update_model(10.35,.03)
    engine.update_model(10.45,.03)
    held=engine.goal.copy()
    engine.update_model(10.50,.03)
    assert engine.goal==held and engine.model_metrics['buffer_waits']==1
    worker.result={'acquired':10.49,'actions':[[15]*5+[50]]*10,'computed_chunk':True,'compute_ms':20}
    engine.update_model(10.51,.01)
    assert engine.goal==held
    engine.update_model(10.54,.03)
    assert engine.goal!=held and engine.model_prediction_count==2 and requests
    assert engine.measured==[0,0,0,0,0,.2]  # Scheduler never overwrites actual feedback.
    engine.close()
