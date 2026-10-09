"""Small interactive entry point for the existing B contact simulation."""
import time
import numpy as np
from core import atomic_json

class PlaybackControl:
    def __init__(self,env):
        self.env=env;self.playing=False;self.finished=False;self.reset_requested=False
        self.message='재생을 누르면 완제품 B 에피소드를 시작합니다.'
        self.last_info={}
        self.pending_jig=None;self.jig_revision=0
    def play(self):
        try:
            if not self.env.carrier_matches_plan():
                self.request_jig(play_after=True);return
        except ValueError as error:
            self.pause();self.message=str(error);return
        if self.finished:self.reset_requested=True
        self.playing=True;self.env.sim.play()
    def pause(self):
        self.env.sim.forward();self.playing=False;self.env.sim.pause()
    def restart(self):self.reset_requested=True;self.playing=False;self.env.sim.pause()
    def request_jig(self,pose=None,*,play_after=False):
        self.pause();self.pending_jig={'pose':pose,'play_after':play_after}
        self.message='지그 위치로 티칭 경로를 계산합니다.'
    def apply_jig(self,pose=None):
        self.pause()
        self.message='지그 위치로 티칭 경로를 계산합니다.'
        try:record=self.env.apply_jig_pose(pose)
        except (ValueError,RuntimeError) as error:
            self.env._write_carrier_pose(self.env.ref.jig_pose)
            self.message='적용 실패 · '+str(error)
            return False
        self.finished=False;self.reset_requested=False;self.last_info={}
        self.jig_revision+=1
        self.message=f"위치 적용 · 경로 계산 완료 (최대 {record['max_position_error_mm']:.2f}mm) · 재생을 누르세요."
        return True
    def update(self):
        if self.pending_jig is not None:
            request=self.pending_jig;self.pending_jig=None
            if self.apply_jig(request['pose']) and request['play_after']:self.play()
            return
        try:
            if not self.env.carrier_matches_plan():
                self.pause();self.message='장면의 지그 위치가 바뀌었습니다. 재생을 누르면 현재 위치로 다시 계산합니다.'
                return
        except ValueError as error:
            self.pause();self.message=str(error);return
        # Native timeline controls and the small panel share one play state.
        if self.env.sim.is_playing() and not self.playing:
            if self.finished:self.reset_requested=True
            self.playing=True
        elif self.playing and not self.env.sim.is_playing():
            self.playing=False
        if self.reset_requested:
            self.env.sim.play();self.env.reset(seed=42,options={'randomized':False})
            self.reset_requested=False;self.finished=False;self.last_info={}
            self.message='처음 위치로 되돌렸습니다.'
        if not self.playing:
            self.env.sim.pause()
            return
        self.env.sim.play()
        _,_,terminated,truncated,self.last_info=self.env.step(np.zeros(6))
        self.message=self.env.ref.steps[self.env.step_index]['name']
        if self.last_info['grasp_waiting']:
            self.message+=f" · 도착·집게 접촉 확인 중 ({self.last_info['grasp_wait_s']:.1f}초)"
        if terminated or truncated:
            self.finished=True;self.pause()
            self.message=('재생 완료 · B 조립 성공' if self.last_info['is_success'] else
                          '중단 · '+(self.last_info['failure_reason'] or '물리 상태 확인 필요') if self.last_info['failure'] else
                          '재생 완료 · B 조립 미완료')
            atomic_json(self.env.output/'viewer_result.json',self.last_info)

def run_viewer(app,env,*,smoke=False,capture_only=False,jig_render=False):
    import omni.ui as ui
    from camera_view import top_camera
    top_camera(env,select=True)
    control=PlaybackControl(env)
    panel=ui.Window('완제품 B 시뮬',width=400,height=820)
    panel.frame.style={'font_size':18,'Label':{'color':0xffeeeeee},'Button':{'height':40}}
    with panel.frame:
        with ui.VStack(spacing=8,margin=12):
            ui.Label('완제품 B · Isaac Sim',height=30,style={'font_size':23})
            ui.Label('복사본 32스텝 · 실물 연결 없음',height=25)
            ui.Button('상단 카메라 시점',height=34,clicked_fn=lambda:top_camera(env,select=True))
            with ui.HStack(spacing=8,height=42):
                ui.Button('재생',clicked_fn=control.play)
                ui.Button('일시정지',clicked_fn=control.pause)
                ui.Button('처음으로',clicked_fn=control.restart)
            progress=ui.Label('0.0초 · 1 / 32',height=24)
            status=ui.Label(control.message,height=55,word_wrap=True)
            load_status=ui.Label('',height=65,word_wrap=True)
            assembly_status=ui.Label('',height=50,word_wrap=True)
            carrier=env.cfg['carrier']
            offset=carrier.get('cad_offset_mm',[0,0,0])
            tilt=carrier.get('cad_tilt_deg',[0,0])
            ui.Label(f"CAD 임시 보정 {offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}mm\n"
                     f"방향 {carrier.get('cad_yaw_deg',0):.1f}° · 기울기 {tilt[0]:.1f}, {tilt[1]:.1f}°\n"
                     f"개별 받침 보정 {len(carrier.get('nest_offsets_mm',{}))}개 · 자석 물성 임시",
                     height=65,word_wrap=True)
            ui.Separator(height=4)
            ui.Label('운반 지그 위치 · 로봇팔2 기준',height=28)
            fields=[]
            for label,value in zip(('X (mm)','Y (mm)','방향 (°)'),env.carrier_pose()):
                with ui.HStack(height=28,spacing=8):
                    ui.Label(label,width=95)
                    entry=ui.FloatDrag(min=-1000 if len(fields)<2 else -360,max=1000 if len(fields)<2 else 360,step=.5)
                    entry.model.set_value(value);fields.append(entry.model)
            def apply_pose(pose=None):
                control.request_jig(pose)
            ui.Button('지그 위치 적용 · 경로 재계산',height=38,
                      clicked_fn=lambda:apply_pose([m.get_value_as_float() for m in fields]))
            with ui.HStack(height=38,spacing=8):
                ui.Button('장면 위치로 계산',clicked_fn=lambda:apply_pose())
                ui.Button('저장 위치 복원',clicked_fn=lambda:apply_pose(env.ref.taught_jig_references[env.ref.jig_id]['pose']))
            ui.Label('위치를 적용하면 부품도 함께 배치하고 처음부터 준비합니다.',height=40,word_wrap=True)
            ui.Label('마우스로 시점을 회전·확대할 수 있습니다.',height=28,word_wrap=True)
    for _ in range(10):app.update()
    for window in ui.Workspace.get_windows():
        if window.title not in ('Viewport',panel.title):window.visible=False
    viewport=ui.Workspace.get_window('Viewport')
    if viewport:
        viewport.visible=True
        panel.dock_in(viewport,ui.DockPosition.RIGHT,.27)
        ui.Workspace.set_dock_id_width(panel.dock_id,400)
    control.play()
    if capture_only:control.pause()
    if jig_render:
        saved=env.ref.jig_pose
        control.apply_jig([saved[0]+2,saved[1]+1,saved[2]+.5])
        control.apply_jig(saved);control.play()
    initial_joints=env.q().copy()
    frames=0;paused_at=None;smoke_checks={};shown_jig_revision=0;render_ready=None
    while app.is_running():
        if jig_render:
            if render_ready is None and env.step_index>=3 and env.playback.fraction>=.9:
                control.pause();render_ready=frames
                atomic_json(env.output/'viewer_render_state.json',env.diagnostic_state())
            if render_ready is not None and frames>=render_ready+5:
                (env.output/'viewer_ready').write_text('ready')
            if (env.output/'viewer_captured').exists():break
            if frames>1500:raise RuntimeError('지그 복원 후 렌더링 검증 시간 초과')
        if capture_only:
            if frames==10:
                from omni.kit.viewport.utility import get_active_viewport
                viewport=get_active_viewport()
                if str(viewport.camera_path)!='/World/TopCamera' or tuple(viewport.resolution)!=(1280,720):
                    raise RuntimeError('상단 카메라/해상도 적용 실패')
                atomic_json(env.output/'viewer_camera_check.json',dict(active_camera='/World/TopCamera',
                    resolution=list(viewport.resolution),lighting=env.lighting))
                (env.output/'viewer_ready').write_text('ready')
            if frames>10 and (env.output/'viewer_captured').exists():break
            if frames>1500:raise RuntimeError('상단 카메라 캡처 응답 시간 초과')
        if smoke:
            if frames==10:
                smoke_checks['autoplay_advances_time']=env.time>0.
            if frames==18:
                # The teaching worker first confirms the safe pose for 0.2 s.
                # Observe motion after that intentional arrival dwell.
                smoke_checks['autoplay_moves_robot_joints']=bool(np.max(np.abs(env.q()-initial_joints))>np.radians(.1))
                control.pause();paused_at=env.time
            if frames==24:
                smoke_checks['pause_freezes_time']=env.time==paused_at
                control.restart()
            if frames==25:
                smoke_checks['reset_returns_to_start']=env.time==0.
                smoke_checks['play_advances_time']=paused_at is not None and paused_at>0.
            if frames==28:
                pose=env.carrier_pose()
                control.request_jig([pose[0]+2,pose[1]+1,pose[2]+.5])
            if frames==29:
                smoke_checks['apply_shifted_jig']=control.jig_revision==1
                smoke_checks['scene_and_plan_match']=env.carrier_matches_plan()
            if frames==30:control.play()
            if frames==36:control.pause()
            if frames==38:
                pose=env.carrier_pose();env._write_carrier_pose([pose[0]+1,pose[1],pose[2]])
            if frames==39:smoke_checks['viewport_edit_pauses_playback']=not control.playing and not env.carrier_matches_plan()
            if frames==40:control.play()
            if frames==45:
                smoke_checks['play_recalculates_viewport_pose']=env.carrier_matches_plan() and env.time>0
                control.pause()
            if frames==48:
                control.request_jig(env.ref.taught_jig_references[env.ref.jig_id]['pose'])
            if frames==49:
                smoke_checks['restore_saved_pose']=np.allclose(env.ref.jig_pose,env.ref.taught_jig_references[env.ref.jig_id]['pose'])
            if frames==52:control.play()
            if frames>60 and (control.finished or env.step_index>=5) and not (env.output/'viewer_ready').exists():
                smoke_checks['shared_teaching_worker']=env.playback.session.on_snapshot.__code__.co_filename.endswith('so101_teach/motion.py')
                smoke_checks['arm2_load_threshold']=env.playback.session.grip_contact.threshold==env.ref.profile['grip_contact_load']
                smoke_checks['no_lift_without_grasp_arrival']=env.step_index<5 or any(a['step_index']==5 for a in env.playback.arrivals)
                smoke_checks['estimated_load_visible']='추정' in load_status.text
                smoke_checks['gripper_target_and_feedback_visible']='목표' in load_status.text and '실제' in load_status.text
                from run import capture
                capture(env,env.output/'viewer_grasp.png',view='grasp')
                atomic_json(env.output/'viewer_render_state.json',env.diagnostic_state())
                control.pause()
                if not all(smoke_checks.values()):raise RuntimeError(str(smoke_checks))
                atomic_json(env.output/'viewer_smoke.json',smoke_checks)
                (env.output/'viewer_ready').write_text('ready')
            if frames>60 and (env.output/'viewer_captured').exists():break
            if frames>1500:raise RuntimeError('숨긴 창 캡처 응답 시간 초과')
        control.update()
        if shown_jig_revision!=control.jig_revision:
            for model,value in zip(fields,env.carrier_pose()):model.set_value(value)
            shown_jig_revision=control.jig_revision
        progress.text=f'{env.playback.elapsed:.1f}초 · {env.step_index+1} / {len(env.ref.steps)}'
        goal=env.ref.steps[env.step_index]['ticks']['gripper'];actual=env.playback.ticks(env.q())['gripper']
        load_status.text=(f"집게 목표 {goal} / 실제 {actual}틱\n"
                          +f"부하 추정 {env.playback.last_load_raw['gripper']&1023} / 기준 {env.playback.session.grip_contact.threshold} raw"
                          +(' · 접촉 유지' if env.playback.hold_tick is not None else '')+'\n실측 기록 기반 · 토크 환산 미보정')
        status.text=control.message if not (not control.playing and not control.finished and env.time>0) else '일시정지 · '+control.message
        assembly=control.last_info.get('assembly',{})
        magnets=control.last_info.get('magnets',{})
        assembly_status.text=('하우징 안착 '+('완료' if assembly.get('housing_on_pallet') else '대기')
                              +' · TPU 삽입 '+('완료' if assembly.get('tpu_in_housing') else '대기')
                              +f"\n자석 {magnets.get('active_pairs',0)} / 4 · 캡 간격 "
                              +(f"{assembly['cap_face_gap_mm']:.2f}mm" if assembly.get('cap_aligned') else '대기'))
        env.render()
        frames+=1
        if not control.playing:time.sleep(.02)
    atomic_json(env.output/'viewer_session.json',dict(closed=True,simulation_time_s=env.time,
                finished=control.finished,last_result=control.last_info,hardware_commands=False))
