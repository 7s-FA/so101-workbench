"""Read-only SO-101 pose preview. MuJoCo/OpenGL live only in a child process.

No robot backend or command queue is passed to the renderer. Simulation reads the shared episode without issuing motion commands. Importing this module needs no
third-party packages; the base teaching GUI remains usable without MuJoCo.
"""

import math
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time

from teaching_ui import (CHROME, CHROME_TEXT, JOINT_LABELS, state_label, message_ko)
from episode_preview import EpisodePlayer

JOINTS = tuple(JOINT_LABELS)
KEYS = tuple(name + ".pos" for name in JOINTS)
SCENE = Path(__file__).resolve().parent / "assets" / "so101" / "preview_scene.xml"
WIDTH, HEIGHT = 640, 400
PERIOD_MS = 1000 / 30  # Target 30 frames/s; unchanged poses are not rendered again.
AUTO_RETURN_SECONDS = 3.
CURRENT, TARGET, SIMULATION = "로봇 현재 자세", "선택 스텝", "시뮬레이션"
DEFAULT_CAMERA = (135., -25., .95)


def pose_to_radians(pose, tool=None):
    """Apply the saved assembly reference without changing motor command units.

    Calibrated grippers use encoder delta from their recorded 90-degree middle.
    Legacy/demo files retain the 0..100 -> 0..100-degree display convention.
    Do not clamp measured angles to CAD limits.
    """
    result = []
    for key in KEYS:
        try:
            value = float(pose[key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{JOINT_LABELS[key[:-4]]}: 숫자를 입력하세요.") from exc
        if not math.isfinite(value):
            raise ValueError(f"{JOINT_LABELS[key[:-4]]}: 유효한 숫자를 입력하세요.")
        if key == "gripper.pos" and not 0 <= value <= 100:
            raise ValueError("그리퍼: 0~100 사이로 입력하세요.")
        if key != "gripper.pos" and not -180 <= value <= 180:
            raise ValueError(f"{JOINT_LABELS[key[:-4]]}: −180~180° 사이로 입력하세요.")
        if key == 'gripper.pos':
            from model_zero import gripper_model_degrees
            result.append(math.radians(gripper_model_degrees(value,tool)))
        else:
            from model_zero import model_offsets
            offset=model_offsets(tool)[len(result)]
            result.append(math.radians(value+float(offset)))
    return tuple(result)


def preview_tool(app):
    """Current robot uses its current calibration; foreign recordings keep theirs."""
    view = getattr(app, 'vision', None)
    if view is not None and not getattr(app, 'preview_only', False):
        return view.station['tool']
    if not getattr(app,'preview_only',False):
        from model_zero import saved_preview_tool
        tool=saved_preview_tool(app)
        if tool is not None:return tool
    return getattr(app, 'vision_plan', {}).get('preview_tool') or getattr(app,'episode_model_tool',None)


def choose_pose(app, mode, now=None):
    """Return (pose, source token, description). A missing pose must hide the image."""
    from teaching_settings import DISPLAY_STALE_SECONDS
    if mode == TARGET:
        editor = app.pose_editor
        if editor is not None:
            return ({key: value.get() for key, value in editor.values.items()},
                    (TARGET, "draft", id(editor.step)), f"편집 중 · {editor.step['name']} · 적용 전")
        index = app.selected()
        if index is None or not 0 <= index < len(app.steps):
            return None, None, "에피소드에서 스텝을 선택하세요."
        step = app.steps[index]
        return step["pose"].copy(), (TARGET, id(step)), f"목표 자세 · {step['name']}"
    if app.state in ("DISCONNECTED", "CONNECTING", "STOPPING") or not app.latest:
        return None, None, "연결 후 팔로워의 현재 자세를 표시합니다."
    now = time.monotonic() if now is None else now
    sample_time = app.latest.get("sample_time")
    if not isinstance(sample_time, (int, float)) or not math.isfinite(sample_time) or not 0 <= now - sample_time <= DISPLAY_STALE_SECONDS:
        return None, None, "관절값 갱신 중단 · 새 측정값을 기다립니다."
    label = "데모 팔로워" if app.backend.identity["demo"] else "실물 팔로워"
    return app.latest["follower"].copy(), (CURRENT,), f"{label} · {state_label(app.state)}"


def _put_latest(mailbox, item):
    """Bounded, nonblocking latest-value delivery (safe to drop a display frame)."""
    try:
        mailbox.put_nowait(item)
        return True
    except queue.Full:
        try:
            mailbox.get_nowait()
        except queue.Empty:
            pass
        try:
            mailbox.put_nowait(item)
            return True
        except queue.Full:
            return False


def draw_carrier_overlays(mujoco,scene,carriers):
    """Measured carrier rims/grid in robot-base coordinates (display only)."""
    import numpy as np
    for c in carriers:
        a=math.radians(c['yaw']);rot=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1.]])
        centre=np.array([c['x'],c['y'],c['z']])/1000.
        length,width=c['long']/1000.,c['short']/1000.
        boxes=[([0,-width/2,0],[length/2,.001,.001]),([0,width/2,0],[length/2,.001,.001]),
               ([-length/2,0,0],[.001,width/2,.001]),([length/2,0,0],[.001,width/2,.001]),
               ]
        if c.get('shape_model','carrier_6jigs')=='carrier_6jigs':
            boxes.extend([([0,0,0],[length/2,.001,.001]),([-length/6,0,0],[.001,width/2,.001]),([length/6,0,0],[.001,width/2,.001])])
        if c.get('mesh_asset'):
            centre[2]+=c.get('height',0.)/1000.+.002
        for offset,size in boxes:
            if scene.ngeom>=scene.maxgeom:return
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_BOX,np.array(size),
                               centre+rot@np.array(offset),rot.ravel(),np.array([.1,.75,.8,1.],dtype=np.float32))
            scene.ngeom+=1

    # Robot axes at the base origin, plus local long-side direction for each STL.
    for c in [dict(x=0,y=0,z=0,yaw=0),*[c for c in carriers if c.get('mesh_asset')]]:
        a=math.radians(c['yaw']);rotation=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1.]])
        centre=np.array([c['x'],c['y'],c['z']+c.get('height',0)+3])/1000.
        for axis,color in [(0,[1,.2,.15,1]),(1,[.2,1,.3,1])]:
            if scene.ngeom>=scene.maxgeom:return
            offset=np.zeros(3);offset[axis]=.025
            size=np.full(3,.0015);size[axis]=.025
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_BOX,size,centre+rotation@offset,rotation.ravel(),np.array(color,dtype=np.float32))
            scene.ngeom+=1


def render_worker(commands, frames, stopping, scene_path):
    """Spawn entry point. Kinematic rendering only: never call mj_step."""
    renderer = None
    # A closed parent must not leave the child waiting for an unread large frame.
    frames.cancel_join_thread()
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        addresses = [model.joint(name).qposadr[0] for name in JOINTS]
        ranges = [model.joint(name).range.copy() for name in JOINTS]
        camera = mujoco.MjvCamera()
        camera.lookat[:] = (.12, -.025, .24)
        option = mujoco.MjvOption()
        option.geomgroup[3] = 0  # Hide duplicate collision hulls, retain visual meshes.
        renderer_size = None
        mesh_key=()
        while not stopping.is_set():
            try:
                token, angles, view, *dimensions = commands.get(timeout=.2)
            except queue.Empty:
                continue
            size = dimensions[0] if dimensions else (WIDTH, HEIGHT)
            scenes=dimensions[1] if len(dimensions)>1 else []
            meshes=[c for c in scenes if c.get('mesh_asset')]
            new_key=tuple(c['mesh_asset'] for c in meshes)
            if new_key!=mesh_key:
                try:
                    from jig_stl import model_with_meshes
                    candidate=model_with_meshes(scene_path,meshes) if meshes else mujoco.MjModel.from_xml_path(scene_path)
                except Exception as exc:
                    _put_latest(frames,('asset_error',token,str(exc)));continue
                if renderer is not None:renderer.close();renderer=None
                model=candidate;data=mujoco.MjData(model);renderer_size=None;mesh_key=new_key
                addresses=[model.joint(name).qposadr[0] for name in JOINTS]
                ranges=[model.joint(name).range.copy() for name in JOINTS]
            if meshes:
                from jig_stl import place_meshes
                place_meshes(model,meshes)
            if size != renderer_size:
                if renderer is not None:
                    renderer.close()
                # The framebuffer must grow with the actual Tk viewport; the
                # scene's initial offscreen size is not a display size limit.
                model.vis.global_.offwidth, model.vis.global_.offheight = size
                renderer = mujoco.Renderer(model, width=size[0], height=size[1])
                renderer_size = size
            for address, angle in zip(addresses, angles):
                data.qpos[address] = angle
            mujoco.mj_forward(model, data)
            camera.azimuth, camera.elevation, camera.distance = view
            # Preserve the horizontal field of view in a narrow, tall panel.
            # The stored distance still controls the user's relative zoom.
            camera.distance *= max(1., 1.35 * size[1] / size[0])
            renderer.update_scene(data, camera=camera, scene_option=option)
            if len(dimensions)>1:
                draw_carrier_overlays(mujoco,renderer.scene,dimensions[1])
            pixels = renderer.render()
            outside = tuple(name for name, angle, (low, high) in zip(JOINTS, angles, ranges)
                            if angle < low - 1e-5 or angle > high + 1e-5)
            ppm = f"P6\n{size[0]} {size[1]}\n255\n".encode("ascii") + pixels.tobytes()
            _put_latest(frames, ("frame", token, angles, outside, ppm, size))
    except Exception as exc:
        # Keep a failure report available until the parent has consumed it.
        error = ("error", f"{type(exc).__name__}: {exc}")
        while not stopping.is_set():
            _put_latest(frames, error)
            stopping.wait(.2)
    finally:
        if renderer is not None:
            renderer.close()


class RenderClient:
    """IPC reader keeps large pipe reads off Tk's event/STOP/heartbeat thread."""
    def __init__(self, scene_path=SCENE):
        ctx = mp.get_context("spawn")
        self.commands, self.frames = ctx.Queue(1), ctx.Queue(1)
        self.stopping = ctx.Event()
        self.inbox = queue.Queue(1)
        self.closed = False
        self.finished = threading.Event()
        self.process = ctx.Process(target=render_worker,
                                   args=(self.commands, self.frames, self.stopping, str(scene_path)),
                                   daemon=True, name="SO101-3D-preview")
        self.process.start()
        self.reader = threading.Thread(target=self._read, daemon=True, name="SO101-frame-reader")
        self.reader.start()

    def _read(self):
        while not self.stopping.is_set():
            try:
                item = self.frames.get(timeout=.2)
            except queue.Empty:
                if not self.process.is_alive():
                    _put_latest(self.inbox, ("error", "3D 렌더러가 종료되었습니다. 프로그램을 다시 실행하세요."))
                    return
                continue
            except (EOFError, OSError, ValueError):
                return
            _put_latest(self.inbox, item)

    def submit(self, item):
        return not self.closed and _put_latest(self.commands, item)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stopping.set()
        # No join on Tk's event thread: a stuck GL driver cannot delay STOP.
        threading.Thread(target=self._reap, daemon=True, name="SO101-preview-cleanup").start()

    def _reap(self):
        self.process.join(1.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(1.)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(1.)
        self.reader.join(.5)
        for mailbox in (self.commands, self.frames):
            mailbox.cancel_join_thread()
            mailbox.close()
        self.finished.set()


class PreviewPanel:
    def __init__(self, app, parent):
        import tkinter as tk
        from tkinter import ttk
        self.app = app
        self.frame = ttk.Frame(parent, style="Panel.TFrame", padding=1)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(2, weight=1)
        self.settings_visible=False
        self.bad_mesh_assets=set()
        self.player = EpisodePlayer()
        self.simulation_active = False
        self.auto_return_at = None
        self.simulation_was_running = False
        self.previous_mode = None
        self.setting_timeline = False
        self.closed, self.job, self.client = False, None, None
        self.viewport, self.viewport_job = (WIDTH, HEIGHT), None
        self.camera = DEFAULT_CAMERA
        self.last_request, self.token, self.photo = None, None, None
        self.last_frame_angles, self.last_frame_at, self.pending_since = None, 0., None
        self.render_error = None
        self.mode = tk.StringVar(value=TARGET if app.state == "DISCONNECTED" else CURRENT)
        self.mode.trace_add("write", self.on_mode_changed)
        self.status = tk.StringVar(value="3D 모델을 불러오는 중…")
        self.detail = tk.StringVar(value="")
        top = ttk.Frame(self.frame, style="Band.TFrame", padding=(8, 5))
        top.grid(row=0, column=0, sticky="ew")
        self.heading=ttk.Label(top, text="3D 미리보기", style="Section.TLabel");self.heading.pack(side="left")
        toolbar = ttk.Frame(top, style="Band.TFrame")
        toolbar.pack(side="right")
        self.camera_reset = ttk.Button(toolbar, text="시점 복원", command=self.reset_camera, style="Compact.TButton")
        self.camera_reset.pack(side="right", padx=(6, 0))
        self.settings_toggle=ttk.Button(toolbar,text='스텝 설정',command=self.toggle_step_settings,style='Compact.TButton')
        self.settings_toggle.pack(side='right',padx=(6,0))
        self.content=ttk.Frame(self.frame,style="PanelBody.TFrame")
        self.content.grid(row=2,column=0,sticky='nsew')
        self.content.columnconfigure(0,weight=1);self.content.columnconfigure(1,weight=0,minsize=0);self.content.rowconfigure(0,weight=1)
        self.canvas = tk.Canvas(self.content, width=240, height=HEIGHT, background=CHROME,
                                highlightthickness=1, highlightbackground=CHROME, takefocus=True)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(8,0))
        self.image_item = self.canvas.create_image(1, 1, anchor="nw")
        self.overlay = self.canvas.create_text(WIDTH // 2, HEIGHT // 2, text="3D 모델을 불러오는 중…",
                                              fill=CHROME_TEXT, font="TkDefaultFont", width=WIDTH-60, justify="center")
        self.canvas.bind("<Configure>", self.resize_canvas)
        self.canvas.bind("<ButtonPress-1>", self.begin_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<Button-4>", lambda _: self.zoom(.9))
        self.canvas.bind("<Button-5>", lambda _: self.zoom(1.1))
        self.canvas.bind("<MouseWheel>", lambda event: self.zoom(.9 if event.delta > 0 else 1.1))
        for key, da, de in (("Left", -8, 0), ("Right", 8, 0), ("Up", 0, 8), ("Down", 0, -8)):
            self.canvas.bind("<" + key + ">", lambda event, a=da, e=de: self.rotate(a, e))
        self.canvas.bind("<plus>", lambda _: self.zoom(.9))
        self.canvas.bind("<minus>", lambda _: self.zoom(1.1))
        self.info = info = ttk.Frame(self.frame, style="PanelBody.TFrame", padding=(8, 3))
        info.grid(row=3, column=0, sticky="ew")
        info.columnconfigure(0, weight=1)
        self.status_label = ttk.Label(info, textvariable=self.status, style="PreviewStatus.TLabel", width=1)
        self.status_label.grid(row=0, column=0, sticky="ew")
        self.detail_label = ttk.Label(info, textvariable=self.detail, style="PanelMuted.TLabel", width=1)
        self.detail_label.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        from step_reference import InlineStepSettings
        self.step_settings=InlineStepSettings(app,self.content)
        self.step_settings.window.grid(row=0,column=0,sticky='nsew')
        self.step_settings.window.grid_remove()
        self.build_transport()
        try:
            self.client = RenderClient()
        except Exception as exc:
            self.render_error = str(exc)
        self.tick()

    def toggle_step_settings(self):
        if self.settings_visible:self.show_simulation()
        else:self.show_step_settings()

    def show_step_settings(self):
        self.settings_visible=True
        self.canvas.grid_remove();self.info.grid_remove();self.transport.grid_remove()
        self.step_settings.window.grid();self.step_settings.refresh()
        self.heading.configure(text='선택 스텝 설정')
        self.settings_toggle.configure(text='시뮬 보기',style='Compact.Primary.TButton')
        self.camera_reset.pack_forget()

    def show_simulation(self):
        self.settings_visible=False
        self.step_settings.window.grid_remove();self.canvas.grid();self.info.grid()
        self.heading.configure(text='3D 미리보기')
        self.settings_toggle.configure(text='스텝 설정',style='Compact.TButton')
        self.camera_reset.pack(side='right',padx=(6,0));self.refresh_transport()

    def build_transport(self):
        import tkinter as tk
        from tkinter import ttk
        self.transport = box = ttk.Frame(self.frame, style="PanelBody.TFrame", padding=(8, 4))
        box.grid(row=4, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        self.timeline = ttk.Scale(box, from_=0, to=1, command=self.seek_simulation)
        self.timeline.grid(row=0, column=0, sticky="ew")
        self.sim_clock = tk.StringVar(value="0.0 / 0.0초")
        ttk.Label(box, textvariable=self.sim_clock, style="PanelMuted.TLabel").grid(row=0, column=1, padx=(8, 0))
        actions = ttk.Frame(box, style="PanelBody.TFrame")
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        actions.columnconfigure(2, weight=1)
        self.sim_pause = ttk.Button(actions, text="일시정지", command=self.toggle_pause, style="Compact.TButton")
        self.sim_pause.grid(row=0, column=0, padx=(0, 4))
        self.sim_reset = ttk.Button(actions, text="처음으로", command=self.reset_simulation, style="Compact.TButton")
        self.sim_reset.grid(row=0, column=1, padx=(0, 6))
        self.sim_speed = ttk.Combobox(actions, state="readonly", values=("0.5배", "1배", "2배"), width=4)
        self.sim_speed.set("1배")
        self.sim_speed.grid(row=0, column=3)
        self.sim_speed.bind("<<ComboboxSelected>>", lambda _: self.player.set_speed(float(self.sim_speed.get().removesuffix("배"))))
        self.refresh_transport()

    def play_simulation(self):
        if (self.render_error or not self.app.steps or self.app.closing
                or self.app.pose_editor is not None or self.app.name_editor is not None):return
        from copy import deepcopy
        from step_reference import needs_measurement,used_plan
        from vision_teaching import calibration_signature,digest,reference_key
        from planning_worker import compile_steps_isolated as compile_steps
        document=deepcopy(self.app.document())
        if not any(step.get('motion') for step in document['steps']):
            self.start_simulation_document(document,'고정 위치 경로');return
        if self.app.preview_only:
            self.start_simulation_document(document,'다른 로봇의 저장 경로 · 현재 지그 보정 없음');return
        view=self.app.open_vision(show=False)
        if view is None:return
        if view.job or view.reference_acquisition.pending or self.app.state not in ('DISCONNECTED','READY','HOLD'):
            self.app.note('진행 중인 동작·측정이 끝난 뒤 시뮬레이션을 실행하세요.');return
        revision=digest([self.app.steps,self.app.vision_plan]);signature=calibration_signature(view.station)
        plan=used_plan(document['steps'],document.get('vision_plan',{}))
        def calculate():
            try:
                observations=view.reference_observations(plan) if needs_measurement(document['steps'],plan) else {}
                station=deepcopy(view.station)
                def work():
                    if any(step.get('motion') for step in document['steps']):
                        document['steps']=compile_steps(document['steps'],plan,station,observations,document['steps'][0]['pose'])
                    return document
                def ready(result):
                    if self.closed or self.app.closing:return
                    if revision!=digest([self.app.steps,self.app.vision_plan]) or signature!=calibration_signature(view.station):
                        self.app.note('스텝·보정이 변경되어 시뮬레이션 계산을 취소했습니다.');return
                    from jig_stl import observed_scenes
                    scene=observed_scenes(plan.get('targets',{}).values(),view.station,observations)
                    view.status.set('지그 보정 경로 계산 완료 · 시뮬레이션 재생 · 모터 명령 없음')
                    self.start_simulation_document(result,('시험 오차 허용 · ' if plan.get('execution_trial') else '')+('현재 지그 측정 보정 적용' if scene else '고정 위치 경로'),scene)
                view.run_job(work,ready,error_prefix='시뮬레이션 경로 계산 실패: ')
            except Exception as exc:self.app.note('시뮬레이션 실행 실패: '+str(exc))
        try:
            if needs_measurement(document['steps'],plan):
                if not view.frame_live:
                    self.app.note('상단 카메라를 연결하고 지그가 보이는 상태에서 시뮬레이션을 실행하세요.');return
                self.status.set('시뮬레이션용 지그 위치 읽는 중 · 로봇 구동 없음')
                view.reference_acquisition.begin(plan,calculate,show=False)
            else:calculate()
        except Exception as exc:self.app.note('시뮬레이션 실행 실패: '+str(exc))

    def start_simulation_document(self,document,label,scene=None):
        self.show_simulation()
        try:self.player.load(document)
        except (ValueError,TypeError,KeyError) as exc:
            self.app.note('시뮬레이션 실행 실패: '+message_ko(exc));return
        self.simulation_scene=scene or []
        self.simulation_reference_label=label
        self.timeline.configure(to=self.player.duration)
        self.mode.set(SIMULATION);self.player.play();self.simulation_active=True
        self.simulation_was_running=self.player.running
        self.app.clear_step_selection();self.app.refresh_buttons();self.refresh_transport()

    def on_mode_changed(self, *_):
        self.cancel_auto_return()
        if self.mode.get() != SIMULATION:
            self.simulation_active = False

    def cancel_auto_return(self, *_):
        self.auto_return_at = None
        self.simulation_was_running = False

    def invalidate_episode(self):
        self.cancel_auto_return()
        self.simulation_active = False
        self.player = EpisodePlayer()
        self.player.set_speed(float(self.sim_speed.get().removesuffix("배")))
        if self.mode.get() == SIMULATION:
            self.mode.set(TARGET)
        self.last_request = None
        self.refresh_transport()

    def stop_simulation(self):
        self.player.reset()
        self.mode.set(CURRENT)
        self.refresh_transport()

    def pause_simulation(self):
        self.cancel_auto_return()
        self.player.pause()
        if hasattr(self, "sim_pause"):
            self.refresh_transport()

    def toggle_pause(self):
        self.cancel_auto_return()
        if self.player.running:
            self.player.pause()
        elif self.player.steps and not self.render_error:
            self.mode.set(SIMULATION)
            self.player.play()
            self.simulation_active = True
            self.simulation_was_running = self.player.running
        self.refresh_transport()

    def reset_simulation(self):
        self.player.reset()
        self.mode.set(SIMULATION)
        self.simulation_active = bool(self.player.steps)
        self.refresh_transport()

    def seek_simulation(self, value):
        if not self.setting_timeline and self.player.steps:
            self.player.seek(float(value))
            self.mode.set(SIMULATION)
            self.simulation_active = True

    def refresh_transport(self):
        if self.settings_visible:
            self.transport.grid_remove();return
        if self.player.steps and self.mode.get() == SIMULATION:
            self.transport.grid()
        else:
            self.transport.grid_remove()
        ready = bool(self.player.steps) and self.photo is not None and not self.render_error
        self.sim_pause.configure(state="normal" if ready else "disabled",
                                 text="일시정지" if self.player.running or not self.player.steps else "계속 재생")
        self.sim_reset.configure(state="normal" if ready else "disabled")
        self.timeline.configure(state="normal" if ready else "disabled")
        self.setting_timeline = True
        self.timeline.set(self.player.position)
        self.setting_timeline = False
        self.sim_clock.set(f"{self.player.position:.1f} / {self.player.duration:.1f}초")

    def resize_canvas(self, event):
        self.canvas.coords(self.overlay, event.width // 2, event.height // 2)
        self.canvas.itemconfigure(self.overlay, width=max(100, event.width-60))
        self.detail_label.configure(wraplength=max(200, event.width-4))
        if (event.width-2, event.height-2) != self.viewport:
            self.show_message("화면 크기 조정 중…")
        if self.viewport_job is not None:
            self.frame.after_cancel(self.viewport_job)
        self.viewport_job = self.frame.after(150, self.fit_viewport)

    def fit_viewport(self):
        self.viewport_job = None
        if self.canvas.winfo_ismapped():
            self.viewport = (max(1, self.canvas.winfo_width()-2),
                             max(1, self.canvas.winfo_height()-2))
            # Also refresh when a rapid resize returns to the previous size:
            # the old image was hidden while the user dragged the window.
            self.last_request = None

    def reset_camera(self):
        self.camera = DEFAULT_CAMERA

    def begin_drag(self, event):
        self.canvas.focus_set()
        self.drag_position = (event.x, event.y)

    def drag(self, event):
        x, y = self.drag_position
        self.rotate((x-event.x)*.5, (y-event.y)*.5)
        self.drag_position = (event.x, event.y)

    def rotate(self, azimuth, elevation):
        a, e, d = self.camera
        self.camera = ((a+azimuth) % 360, max(-85, min(15, e+elevation)), d)
        return "break"

    def zoom(self, factor):
        a, e, d = self.camera
        self.camera = (a, e, max(.35, min(2., d*factor)))
        return "break"

    def show_message(self, text):
        self.canvas.itemconfigure(self.image_item, state="hidden")
        self.canvas.itemconfigure(self.overlay, text=text, state="normal")

    def confirmed_jig_scenes(self):
        view=getattr(self.app,'vision',None)
        if view is None or self.app.preview_only:return []
        from vision_teaching import reference_key,calibration_signature
        from jig_detection import profile_signature
        from jig_stl import observed_scenes
        references=view.jig_reference.values
        targets=[t for t in self.app.vision_plan.get('targets',{}).values() if t.get('mode')=='rectangle']
        signature=calibration_signature(view.station)
        token=(signature,tuple((k,v['id']) for k,v in references.items()),
               tuple((reference_key(t),profile_signature(t),repr(view.station.get('jigs',{}).get(reference_key(t),{}).get('preview_model'))) for t in targets))
        if token!=getattr(self,'confirmed_scene_token',None):
            self.confirmed_scene_token=token
            observations={}
            for target in targets:
                key=reference_key(target);value=references.get(key)
                if value and value['calibration_signature']==signature and value['profile_signature']==profile_signature(target):
                    observations[key]={**value['observation'],'reference_id':value['id']}
            self.confirmed_scene=observed_scenes(targets,view.station,observations)
        return getattr(self,'confirmed_scene',[])

    def tick(self):
        import tkinter as tk
        if self.closed:
            return
        now = time.monotonic()
        mode = self.mode.get()
        if self.app.state in ('PLAY','MOVE','SYNCING') and mode!=CURRENT:
            self.pause_simulation();self.mode.set(CURRENT);self.show_simulation();mode=CURRENT
        if (mode == TARGET and self.app.selected() is None and self.app.pose_editor is None
                and self.app.state not in ("DISCONNECTED", "CONNECTING", "STOPPING")):
            self.mode.set(CURRENT)
            mode = CURRENT
        if mode != SIMULATION and self.previous_mode == SIMULATION:
            self.player.pause(now)
        self.previous_mode = mode
        if mode == SIMULATION:
            pose, index, phase = self.player.sample(now)
            if (self.simulation_was_running and not self.player.running
                    and self.player.position >= self.player.duration):
                self.auto_return_at = now + AUTO_RETURN_SECONDS
                self.simulation_active = False
            self.simulation_was_running = self.player.running
            if self.auto_return_at is not None and now >= self.auto_return_at:
                self.mode.set(CURRENT)
                mode = CURRENT
        if mode == SIMULATION:
            token = (SIMULATION, self.player.revision)
            if pose is None:
                pose = dict.fromkeys(KEYS, 0.)
                description = "기준 자세 · 스텝을 선택하세요."
            else:
                description = getattr(self,"simulation_reference_label","")+" · "+f"시뮬레이션 {phase} · {index+1}/{len(self.player.steps)} · {self.player.steps[index]['name']}"
                if self.auto_return_at is not None:
                    description += f" · {math.ceil(self.auto_return_at-now)}초 후 현재 자세"
        else:
            pose, token, description = choose_pose(self.app, mode, now)
            if pose is None and mode == TARGET and self.app.selected() is None:
                pose, token, description = dict.fromkeys(KEYS, 0.), (TARGET, "reference"), "기준 자세 · 스텝을 선택하세요."
        try:
            tool = preview_tool(self.app)
            angles = pose_to_radians(pose,tool) if pose is not None else None
        except ValueError as exc:
            angles, token, description = None, None, str(exc)
        if token != self.token:
            self.token, self.last_request, self.pending_since = token, None, None
            self.last_frame_angles = None
            self.show_message("자세를 불러오는 중…")
        catalog=getattr(getattr(self.app,'vision',None),'catalog',None)
        scenes=self.confirmed_jig_scenes()
        if mode==SIMULATION:
            current_ids={c.get('reference_id') for c in scenes}
            scenes=[c for c in getattr(self,'simulation_scene',[]) if c.get('reference_id') and c['reference_id'] in current_ids]
        if scenes:description='카메라 확정 위치·각도 적용 · '+description
        if any(c.get('model_warning') for c in scenes):description='STL 대신 외곽 표시 · '+description
        scenes=[{k:v for k,v in c.items() if k!='mesh_asset'} if c.get('mesh_asset') in self.bad_mesh_assets else c for c in scenes]
        self.status.set(description)
        if angles is None:
            self.last_request = None
            self.show_message(description)
            self.detail.set("저장된 자세를 보려면 목록에서 스텝을 클릭하세요.")
        else:
            request = (token, angles, self.camera, self.viewport, scenes)
            # Display frames may be dropped by either bounded IPC queue. Retry a
            # still-unacknowledged stationary pose; otherwise the final edit can
            # remain blank forever after a resize or rapid pose changes.
            retry=self.last_frame_angles!=angles and now-getattr(self,'last_submit_at',0.)>.5
            if not self.render_error and (request != self.last_request or retry) and self.client:
                if self.client.submit(request):
                    self.last_request = request
                    self.last_submit_at = now
                    if self.pending_since is None:
                        self.pending_since = now
        if self.client:
            try:
                item = self.client.inbox.get_nowait()
            except queue.Empty:
                item = None
            if item and item[0] == "error":
                self.render_error = item[1]
                self.client.close()
                self.pause_simulation()
            elif item and item[0]=='asset_error':
                self.bad_mesh_assets.update(c['mesh_asset'] for c in (self.last_request[4] if self.last_request else []) if c.get('mesh_asset'))
                self.last_request=None
                if catalog is not None:catalog.stl.status.set('STL 표시 실패 · '+item[2])
            elif (item and item[0]=='frame' and angles is not None and item[1] == token
                  and item[5] == self.viewport
                  and item[5] == (self.canvas.winfo_width()-2, self.canvas.winfo_height()-2)):
                _, _, rendered_angles, outside, ppm, _ = item
                self.photo = tk.PhotoImage(master=self.frame, data=ppm, format="PPM")
                self.canvas.itemconfigure(self.image_item, image=self.photo, state="normal")
                self.canvas.itemconfigure(self.overlay, state="hidden")
                self.last_frame_angles, self.last_frame_at = rendered_angles, now
                if rendered_angles==angles:self.pending_since=None
                self.detail.set("모델 가동범위 밖: " + ", ".join(JOINT_LABELS[j] for j in outside)
                                if outside else "로봇 구동 없음 · 전체 스텝을 순서대로 재생합니다." if mode == SIMULATION else "로봇에서 측정한 현재 자세입니다." if mode == CURRENT else "드래그: 회전 · 휠: 확대·축소 · 로봇 구동 없음")
        if angles is not None and self.pending_since is not None and now-self.pending_since > 2:
            self.show_message("3D 화면 갱신 대기 중…\n로봇 상태는 메인 GUI에서 확인하세요.")
        if self.render_error:
            self.status.set("3D 화면을 열 수 없습니다.")
            self.detail.set(self.render_error)
            self.show_message("3D 환경 설치 후 다시 실행하세요.\n\n데모: bash install_preview.sh\n실행: bash run_preview.sh\n\n실물: bash install_preview.sh --hardware\n실행: bash run.sh\n\n설치되어 있다면 OpenGL·디스플레이 연결을 확인하세요.")
        self.refresh_transport()
        # Include image transfer/display work in the frame interval, so its cost
        # does not lower the target rate. Always yield to input and heartbeat jobs.
        delay_ms = max(1, math.ceil(PERIOD_MS - (time.monotonic() - now) * 1000))
        self.job = self.frame.after(delay_ms, self.tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.player.pause()
        if self.job is not None:
            self.frame.after_cancel(self.job)
            self.job = None
        if self.viewport_job is not None:
            self.frame.after_cancel(self.viewport_job)
            self.viewport_job = None
        if self.client:
            self.client.close()
        if self.app.preview is self:
            self.app.preview = None
        self.frame.destroy()
        self.app.layout_workspace()
