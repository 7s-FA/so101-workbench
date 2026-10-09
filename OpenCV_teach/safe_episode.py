"""Fixed safe endpoints and a cancellable safe-pose -> fresh vision handoff."""
from copy import deepcopy
import time
import uuid

from vision_teaching import JOINTS, calibration_signature, digest, vector


def model_safe_pose(config, tool):
    q=vector(config.get('safe_deg',config['home_deg']),6,-180,180)
    from model_zero import model_offsets
    source=model_offsets(config['tcp'])
    offsets=model_offsets(tool)
    from teach_left_arm import pose
    return pose({**{n+'.pos':q[i]+source[i]-offsets[i] for i,n in enumerate(JOINTS)},'gripper.pos':q[5]})


def with_safe_endpoints(steps, safe_pose):
    middle=[deepcopy(s) for s in steps if not s.get('safe_boundary')]
    return [{'name':'시작 · 안전 자세','pose':deepcopy(safe_pose),'captured_at':'','safe_boundary':'start'},
            *middle,
            {'name':'종료 · 안전 자세','pose':deepcopy(safe_pose),'captured_at':'','safe_boundary':'end'}]


def validate_boundaries(steps):
    marked=[(i,s['safe_boundary']) for i,s in enumerate(steps) if 'safe_boundary' in s]
    if not marked:return False
    if (len(steps)<3 or marked!=[(0,'start'),(len(steps)-1,'end')] or
            steps[0].get('motion') or steps[-1].get('motion') or steps[0]['pose']!=steps[-1]['pose']):
        raise ValueError('안전 자세는 처음과 끝에 한 개씩, 같은 관절값으로 배치하세요.')
    return True


class SafeEpisodeExecution:
    """No serial I/O: the existing Session executes the entry joint step."""
    measurement_timeout=3.

    def __init__(self, view):
        self.view=view
        self.pending=None

    def cancel(self, reason):
        if self.pending is None:return
        self.pending=None
        self.view.app.vision_busy=False
        self.view.app.refresh_buttons()
        self.view.measurement_failure=reason
        self.view.status.set(reason)

    def begin(self):
        from tkinter import messagebox
        from teach_left_arm import validate_sequence
        from vision_teaching import validate_execution_setup
        view=self.view;app=view.app
        view.idle()
        if app.state!='HOLD' or not app.latest or time.monotonic()-app.latest['sample_time']>.6:
            raise ValueError('팔로워를 연결하고 최신 관절값이 표시될 때 실행하세요.')
        if not validate_boundaries(app.steps):raise ValueError('시작·종료 안전 자세가 필요합니다.')
        from step_reference import used_plan
        plan=used_plan(app.steps,app.vision_plan)
        # Fixed safe joints do not depend on TCP geometry. Collect the jig
        # reference first; execution of the camera path still validates setup.
        from vision_teaching import validate_plan
        validate_plan(plan)
        if any(t['mode']=='rectangle' for t in plan['targets'].values()) and (
                not view.frame_live or time.monotonic()-view.frame_at>.6):
            raise ValueError('상단 카메라를 먼저 연결하세요. 안전 자세 도착 후 지그 측정을 시작합니다.')
        safe=deepcopy(app.steps[0]);sample=deepcopy(app.latest)
        revision=digest([app.steps,app.vision_plan]);calibration=calibration_signature(view.station)
        validate_sequence(app.backend,[safe],sample['follower'])
        if not messagebox.askokcancel('에피소드 실행',
                f'{app.teaching_title}\n안전 자세 → 지그 측정 → {len(app.steps)-2}개 작업 스텝 → 안전 자세\n시작할까요?',parent=view.window):return
        if app.state!='HOLD' or app.session.stop.is_set():return
        if revision!=digest([app.steps,app.vision_plan]) or calibration!=calibration_signature(view.station):
            raise ValueError('확인 중 스텝·보정이 바뀌었습니다. 다시 실행하세요.')
        # Never use images from before the safe entry, even if a latch exists.
        view.detector_choice.set('현재 에피소드')
        view.clear_jig_reference('시작 안전 자세 도착 후 지그를 새로 측정합니다.')
        token=uuid.uuid4().hex
        self.pending={'token':token,'phase':'entry','started':time.monotonic(),
            'revision':revision,'calibration':calibration,
            'safe':safe['pose']}
        app.vision_busy=True;app.refresh_buttons()
        view.status.set('시작 안전 자세로 이동합니다. 도착 후 지그를 새로 측정합니다.')
        # Tagged completion is emitted only after the controller has arrived.
        sent=app.send('play',{'steps':[safe],'title':app.teaching_title+' · 시작 안전 자세',
                            'follower':sample['follower'],'request_id':token})
        if sent is False:self.cancel('명령 대기열이 가득 찼습니다. 다시 실행하세요.')

    def completed(self, result):
        p=self.pending
        if not p or result.get('token')!=p['token']:return
        if not result.get('ok'):
            self.cancel('안전 자세 이동 취소: '+result.get('error','이동 실패'))
            return
        p['phase']='settling';p['arrival_at']=result['sample_time']

    def poll(self):
        from teach_left_arm import within_arrival_tolerance
        from teaching_settings import MOVE_TIMEOUT
        p=self.pending
        if p is None:return
        view=self.view;app=view.app;now=time.monotonic()
        if app.closing or app.session.stop.is_set() or app.state in ('DISCONNECTED','STOPPING'):
            self.cancel('안전 자세 실행을 취소했습니다.');return
        if p['revision']!=digest([app.steps,app.vision_plan]) or p['calibration']!=calibration_signature(view.station):
            self.cancel('스텝·보정 변경으로 실행을 취소했습니다.');app.send('pause');return
        if p['phase']=='entry':
            if now-p['started']>MOVE_TIMEOUT+2:
                self.cancel('안전 자세 도착 확인 시간이 초과됐습니다.');app.send('pause')
            return
        sample=app.latest
        if (not sample or sample['sample_time']<p['arrival_at'] or now-sample['sample_time']>.6 or app.state!='HOLD'):
            if now-p['arrival_at']>2:self.cancel('안전 자세의 최신 관절값을 확인할 수 없습니다.');app.send('pause')
            return
        if not within_arrival_tolerance(p['safe'],sample['follower'],'MOVE'):
            self.cancel('안전 자세가 바뀌어 측정을 취소했습니다.');return
        if p['phase']=='settling':
            p['phase']='measuring';p['measure_started']=now
            view.measurement_not_before=now
            view.select_tab(3)
            view.reset_detection()
        final=now-p['measure_started']>=self.measurement_timeout
        try:
            from step_reference import used_plan
            plan=used_plan(app.steps,app.vision_plan)
            observations=view.observations_for(plan,final=final)
            if any(v['at']<p['measure_started'] for v in observations.values()):return
        except ValueError as exc:
            if final:self.cancel('3초 측정 종료 · '+str(exc))
            else:view.status.set('안전 자세에서 측정 중 · '+str(exc))
            return
        # The remaining steps include the fixed end safe pose. The original
        # episode stays unchanged for saving and offline preview.
        from vision_teaching import reference_key
        view.jig_reference.capture_many(plan['targets'].values(),observations,view.station)
        self.pending=None;app.vision_busy=False;app.refresh_buttons()
        app.navigation.home()
        if app.preview is not None:app.preview.show_simulation()
        try:view.execute_motion(prepared=True)
        except Exception as exc:view.status.set(str(exc));app.note(str(exc))
