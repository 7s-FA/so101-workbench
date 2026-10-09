"""Fresh bounded acquisition, separated from the fixed teaching reference."""
from copy import deepcopy
import time
from vision_teaching import reference_key,calibration_signature,digest

class JigAcquisition:
    timeout=3.
    def __init__(self,view):self.view=view;self.pending=None
    def cancel(self,reason):
        if self.pending is None:return
        self.view.measurement_failure=reason
        p=self.pending;self.pending=None;self.view.app.vision_busy=False;self.view.app.refresh_buttons();self.view.status.set(reason)
        if p.get('on_cancel'):p['on_cancel'](reason)
    def begin(self,plan,callback=None,*,teaching=False,show=True,heading_only=False,on_cancel=None):
        view=self.view;view.idle()
        targets=[t for t in plan['targets'].values() if t['mode']=='rectangle']
        if not targets:raise ValueError('측정할 지그를 선택하세요.')
        if self.pending is not None:return
        if not view.frame_live or time.monotonic()-view.frame_at>.6:raise ValueError('상단 카메라의 실시간 영상을 먼저 연결하세요.')
        view.detector_choice.set('현재 에피소드')
        if not heading_only:view.clear_jig_reference('새 지그 위치를 읽습니다.')
        view.reset_detection();view.measurement_not_before=time.monotonic()
        self.pending={'started':time.monotonic(),'plan':deepcopy(plan),'target':deepcopy(targets[0]),
                      'revision':digest([view.app.steps,view.app.vision_plan]),
                      'calibration':calibration_signature(view.station),'callback':callback,'teaching':teaching,'heading_only':heading_only,'on_cancel':on_cancel,'return_to_episode':bool(show and callback and not teaching and not heading_only)}
        if show:view.select_tab(3)
        view.app.vision_busy=True;view.app.refresh_buttons();self.poll()
    def poll(self):
        p=self.pending
        if p is None:return
        view=self.view;app=view.app
        if app.closing or app.state not in ('DISCONNECTED','READY','HOLD'):
            self.cancel('로봇 상태가 바뀌어 지그 측정을 취소했습니다.');return
        if p['calibration']!=calibration_signature(view.station) or p['revision']!=digest([app.steps,app.vision_plan]):
            self.cancel('스텝·보정 변경으로 지그 측정을 취소했습니다. 다시 실행하세요.');return
        final=time.monotonic()-p['started']>=self.timeout
        try:observations=view.observations_for(p['plan'],final=final)
        except ValueError as exc:
            if final:self.cancel('3초 측정 종료 · '+str(exc))
            else:view.status.set('지그 위치 읽기 · '+str(exc))
            return
        targets=list(p['plan']['targets'].values())
        if p.get('heading_only'):
            self.pending=None;app.vision_busy=False;app.refresh_buttons()
            if p['callback']:p['callback'](observations)
            return
        view.jig_reference.capture_many(targets,observations,view.station)
        if p['teaching']:view.teaching_reference.capture_many(targets,observations,view.station)
        self.pending=None;app.vision_busy=False;app.refresh_buttons();view.update_jig_reference()
        view.status.set('티칭 기준 저장 완료 · 이 에피소드의 스텝 저장에 유지합니다.' if p['teaching'] else '실행 기준 저장 완료')
        if p.get('return_to_episode'):
            app.navigation.home()
            if app.preview is not None:app.preview.show_simulation()
        if p['callback']:
            try:p['callback']()
            except Exception as exc:view.status.set('지그 기준 저장 완료 · '+str(exc));app.note(str(exc))
