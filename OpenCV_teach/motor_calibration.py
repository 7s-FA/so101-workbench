"""Button-driven SO-101 calibration, with a recorded six-servo midpoint and measured endpoints.

The worker exclusively owns one serial port. No Goal_Position/torque-on writes
are made: the operator moves the arm, including its gripper, by hand.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import queue
import threading
import time
import json

GUIDE = 'https://huggingface.co/docs/lerobot/v0.4.4/en/so101#calibrate'
VIDEO = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot/calibrate_so101_2.mp4'
LABELS = {'shoulder_pan':'베이스 회전', 'shoulder_lift':'어깨 들기', 'elbow_flex':'팔꿈치',
          'wrist_flex':'손목 굽힘', 'wrist_roll':'집게 회전', 'gripper':'집게 벌림'}
MIDPOINT_TICK = 2047  # LeRobot half-turn homing uses floor(4095 / 2).
MIDPOINT_TOLERANCE = 5


def serializable(calibration):
    return {name: asdict(entry) for name, entry in calibration.items()}


def checked_calibration(data):
    from lerobot.motors import MotorCalibration
    if not isinstance(data,dict) or set(data)!=set(LABELS):
        raise ValueError('몸체 5관절과 집게 벌림이 포함된 SO-101 보정 JSON을 선택하세요.')
    result={}
    for ident,name in enumerate(LABELS,1):
        row=data[name]
        keys=('id','drive_mode','homing_offset','range_min','range_max')
        if not isinstance(row,dict) or any(type(row.get(k)) is not int for k in keys):
            raise ValueError(f'{LABELS[name]}: 정수 보정값이 필요합니다.')
        if (row['id']!=ident or row['drive_mode']!=0 or not -2047<=row['homing_offset']<=2047
                or not 0<=row['range_min']<row['range_max']<=4095):
            raise ValueError(f'{LABELS[name]}: SO-101 모터 ID·영점·범위를 확인하세요.')
        result[name]=MotorCalibration(**{k:row[k] for k in keys})
    return result


def read_calibration_file(path):
    return serializable(checked_calibration(json.loads(Path(path).read_text())))


def build_calibration(motors, offsets, mins, maxes, midpoints=None):
    from lerobot.motors import MotorCalibration
    result = {}
    for name, motor in motors.items():
        low, high = mins.get(name), maxes.get(name)
        if low is None or high is None:
            raise ValueError(f'{LABELS[name]}의 최소·최대 틱을 먼저 측정하세요.')
        if not 0 <= low < high <= 4095:
            raise ValueError(f'{LABELS[name]}의 양 끝을 움직여 주세요. 현재 범위 {low}~{high}')
        if midpoints is not None and not low < midpoints[name] < high:
            raise ValueError(f'{LABELS[name]}: 기록한 중간 틱 {midpoints[name]}의 양쪽 끝을 측정하세요.')
        result[name] = MotorCalibration(id=motor.id, drive_mode=0,
            homing_offset=int(offsets[name]), range_min=int(low), range_max=int(high))
    return result


def make_arm(backend, role, port, arm_id):
    import re
    if Path(port).parent != Path('/dev/serial/by-id'):
        raise ValueError('등록된 USB 포트를 선택하세요.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', arm_id):
        raise ValueError('팔 이름은 영문·숫자·밑줄·하이픈을 사용하세요.')
    from diagnose_shoulder_lift import ensure_ports_available, protect_port
    ensure_ports_available([port])
    if role == 'leader':
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
        arm = SO101Leader(SO101LeaderConfig(port=port, id=arm_id,
            calibration_dir=backend.args.calibration_dir, use_degrees=True))
    else:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        arm = SO101Follower(SO101FollowerConfig(port=port, id=arm_id,
            calibration_dir=backend.args.calibration_dir, use_degrees=True, cameras={}))
    protect_port(arm.bus)
    return arm


def carry_station_after_calibration(view):
    """Keep camera/TCP geometry under the new identity, never retag old episodes."""
    from teach_left_arm import Hardware, atomic_json
    from vision_teaching import digest
    backend=view.app.backend
    fresh=Hardware(backend.profile_path,follower_only=backend.follower_only)
    if fresh.identity==backend.identity:return
    station=deepcopy(view.station);station['identity']=deepcopy(fresh.identity)
    follower_id=backend.follower.id
    if fresh.identity['calibration_sha256'].get(follower_id)!=backend.identity['calibration_sha256'].get(follower_id):
        from model_zero import aligned_tool,carry_alignment
        window=getattr(view,'calibration_window',None)
        worker=getattr(window,'worker',None)
        old_cal=serializable(backend.follower.calibration);new_cal=serializable(fresh.follower.calibration)
        if worker is not None and worker.midpoints:
            station['tool']=aligned_tool(station['tool'],new_cal,worker.midpoints)
        else:
            carried=carry_alignment(station['tool'],old_cal,new_cal)
            if carried is not None:station['tool']=carried
            else:
                station['tool']['joint_offsets_deg']=[0.]*5
                station['tool'].pop('motor_zero_reference',None)
                station['tool'].pop('joint_reference_offsets_deg',None)
                station['tool'].pop('gripper_model_reference',None)
        station['tool']['motor_limits_deg']=[list(fresh.bounds[n+'.pos']) for n in LABELS if n!='gripper']
        station['tool']['verified']=False
        if station.get('extrinsics'):station['extrinsics']['verified']=False
    station.pop('joint_zero',None)
    destination=view.app.output/'vision'/(digest(fresh.identity)[:16]+'.json')
    atomic_json(destination,station)
    view.clear_jig_reference('모터 보정이 바뀌었습니다. 앱을 다시 실행하고 지그 기준을 읽으세요.')


class CalibrationWorker:
    def __init__(self, factory, recovery_path, *, preset=None):
        self.factory, self.recovery_path = factory, Path(recovery_path)
        self.commands = queue.Queue(); self.events = queue.Queue()
        self.cancel = threading.Event(); self.finished = threading.Event()
        self.state = 'CONNECTING'; self.saved = False; self.rollback_error = None
        self.preset=deepcopy(preset)
        self.middle_before = {}; self.midpoints = {}; self.evidence = {}
        self.thread = threading.Thread(target=self.run, daemon=False)

    def emit(self, kind, value):
        self.events.put((kind, value))

    def transition(self, state):
        self.state = state; self.emit('state', state)

    def commit(self, arm, draft):
        from teach_left_arm import atomic_json
        self.transition('SAVING')
        arm.bus.write_calibration(draft)
        if arm.bus.read_calibration()!=draft:raise RuntimeError('모터에 저장한 보정값이 읽은 값과 다릅니다.')
        if self.cancel.is_set():return
        # The recovery record also keeps the measured reference, without adding
        # unsupported fields to LeRobot's MotorCalibration JSON.
        self.evidence.update(status='verified_motor_values', calibration=serializable(draft),
                             midpoint_before=self.middle_before, midpoint_after=self.midpoints)
        atomic_json(self.recovery_path,self.evidence)
        atomic_json(arm.calibration_fpath,serializable(draft))
        arm.calibration=draft;self.saved=True
        self.emit('saved',{'path':str(arm.calibration_fpath),'calibration':serializable(draft),'reference_path':str(self.recovery_path)})

    def run(self):
        import teach_left_arm as teaching
        from teach_left_arm import stamp
        arm = None; previous = None; modes = {}; changed = False
        offsets = {}; mins = {}; maxes = {}; draft = None
        try:
            imported=checked_calibration(self.preset) if self.preset is not None else None
            if self.cancel.is_set():return
            arm = self.factory(); bus = arm.bus
            bus.connect()
            previous = deepcopy(bus.read_calibration())
            modes = {n: bus.read('Operating_Mode', n, normalize=False) for n in bus.motors}
            self.evidence = {'time':stamp(), 'arm':arm.id,
                'calibration_file':str(arm.calibration_fpath), 'previous_eeprom':serializable(previous),
                'previous_json':arm.calibration_fpath.read_text() if arm.calibration_fpath.exists() else None,
                'operating_modes':modes, 'status':'started','source_mode':'file' if imported else 'measure'}
            teaching.atomic_json(self.recovery_path,self.evidence)
            if self.cancel.is_set():return
            changed = True
            # Same preparation as SO101Follower/Leader.calibrate().
            bus.disable_torque()
            if imported is not None:
                if set(bus.motors)!=set(imported) or any(bus.motors[n].id!=entry.id for n,entry in imported.items()):
                    raise ValueError('선택한 보정과 연결된 모터 ID가 다릅니다.')
                self.commit(arm,imported)
                return
            from lerobot.motors.feetech import OperatingMode
            for name in bus.motors:
                bus.write('Operating_Mode', name, OperatingMode.POSITION.value)
            self.transition('MIDPOINT')
            while not self.cancel.is_set():
                positions = bus.sync_read('Present_Position', normalize=False)
                if set(positions) != set(bus.motors) or any(not 0 <= v <= 4095 for v in positions.values()):
                    raise ValueError('모터 위치가 0~4095 범위를 벗어났습니다. 중간 자세를 다시 확인하세요.')
                if self.state == 'RANGE':
                    for name in bus.motors:
                        if abs(positions[name]-last_positions[name]) > 2048:
                            raise ValueError(f'{LABELS[name]}: 0/4095 경계를 넘었습니다. 중간 자세를 다시 잡고 사용 범위를 측정하세요.')
                        mins[name] = min(mins.get(name, positions[name]), positions[name])
                        maxes[name] = max(maxes.get(name, positions[name]), positions[name])
                    last_positions = dict(positions)
                self.emit('positions', {'current':positions, 'mins':dict(mins), 'maxes':dict(maxes),
                                        'offsets':dict(offsets), 'middle':dict(self.midpoints), 'at':time.monotonic()})
                try: action = self.commands.get_nowait()
                except queue.Empty: action = None
                if self.cancel.is_set(): break
                if action == 'middle' and self.state == 'MIDPOINT':
                    self.middle_before = dict(positions)
                    offsets = bus.set_half_turn_homings()
                    measured = bus.sync_read('Present_Position', normalize=False)
                    if set(measured)!=set(bus.motors):
                        raise ValueError('중간 자세 기록 후 일부 서보의 위치를 읽지 못했습니다.')
                    bad = [LABELS[n] for n,v in measured.items() if abs(v-MIDPOINT_TICK)>MIDPOINT_TOLERANCE]
                    if bad:
                        raise ValueError('중간 틱 적용 확인 실패: '+', '.join(bad)+'. 자세를 유지한 채 다시 보정하세요.')
                    self.midpoints = {n:int(v) for n,v in measured.items()}
                    self.evidence.update(status='midpoint_recorded',midpoint_before=self.middle_before,
                                         midpoint_after=self.midpoints,offsets=dict(offsets))
                    teaching.atomic_json(self.recovery_path,self.evidence)
                    mins = dict(self.midpoints); maxes = dict(self.midpoints)
                    last_positions = dict(self.midpoints)
                    self.transition('RANGE')
                elif action == 'review' and self.state == 'RANGE':
                    try: draft = build_calibration(bus.motors, offsets, mins, maxes, self.midpoints)
                    except ValueError as exc: self.emit('notice', str(exc))
                    else: self.transition('REVIEW')
                elif action == 'resume' and self.state == 'REVIEW':
                    self.transition('RANGE')
                elif action == 'save' and self.state == 'REVIEW':
                    self.commit(arm,draft)
                    break
                self.cancel.wait(.05)
        except Exception as exc:
            self.emit('error', str(exc))
        finally:
            if arm is not None and arm.bus.is_connected:
                try:
                    arm.bus.disable_torque()
                    if changed and not self.saved and previous is not None:
                        arm.bus.write_calibration(previous)
                        for name, mode in modes.items():arm.bus.write('Operating_Mode', name, mode)
                        if arm.bus.read_calibration() != previous:
                            raise RuntimeError('이전 모터 보정값 복원 확인에 실패했습니다.')
                        arm.calibration = previous
                        self.emit('notice', '취소한 보정은 되돌렸습니다. 기존 JSON은 유지했습니다.')
                except Exception as exc:
                    self.rollback_error = str(exc)
                    self.emit('error', f'모터 정리/복원 확인 실패: {exc}. 복원 자료: {self.recovery_path}')
                finally:
                    try:arm.bus.port_handler.closePort()
                    except Exception as exc:
                        self.rollback_error = str(exc); self.emit('error', str(exc))
            self.transition('DONE' if self.saved and not self.rollback_error else 'CLOSED')
            self.finished.set()


class MotorCalibrationWindow:
    def __init__(self, view):
        import tkinter as tk
        from tkinter import ttk
        from app_choices import AppChoice
        self.view, self.app = view, view.app
        self.worker = None; self.closed = False; self.pending_start = None; self.closing = False
        self.carried=False
        self.selected_calibration=None
        from app_navigation import AppPage
        self.window = AppPage(self.app); self.window.title('SO-101 모터 캘리브레이션 · 영점과 집게 범위 저장')
        self.window.geometry('980x720'); self.window.minsize(900,680)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        frame=ttk.Frame(self.window,padding=18);frame.pack(fill='both',expand=True);frame.columnconfigure(1,weight=1)
        ttk.Label(frame,text='① 팔 선택 → ② 전체 서보 중간 자세 기록 → ③ 최소·최대 틱 측정 → ④ 저장',style='PanelHeading.TLabel').grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,15))
        self.role=tk.StringVar(value='팔로워암');self.port=tk.StringVar();self.arm_id=tk.StringVar()
        self.role_choice=AppChoice(frame,textvariable=self.role,values=('팔로워암','리더암'),state='readonly',width=15)
        self.role_choice.grid(row=1,column=0,sticky='w');self.role_choice.bind('<<ComboboxSelected>>',lambda _:self.role_changed())
        ports=tuple(str(p) for p in sorted(Path('/dev/serial/by-id').glob('*')))
        self.port_choice=AppChoice(frame,textvariable=self.port,values=ports,width=55)
        self.port_choice.grid(row=1,column=1,columnspan=2,sticky='ew',padx=(10,0))
        ttk.Label(frame,text='보정 파일 이름').grid(row=2,column=0,sticky='w',pady=8)
        self.id_entry=ttk.Entry(frame,textvariable=self.arm_id);self.id_entry.grid(row=2,column=1,sticky='ew',padx=10)
        ttk.Button(frame,text='공식 캘리브레이션 영상',command=self.video).grid(row=2,column=2)
        self.instructions=tk.StringVar(value='팔을 받친 뒤 시작하세요. 모터 힘을 풀고 손으로 자세와 집게 벌림을 맞춥니다.\n공식 절차와 같이 보정 중에는 모터를 구동하는 ± 버튼을 사용하지 않습니다.')
        reference=ttk.Frame(frame);reference.grid(row=3,column=0,columnspan=3,sticky='ew',pady=10)
        from PIL import Image, ImageTk
        photo=Image.open(Path(__file__).parent/'assets/so101/calibration_middle.jpg')
        photo.thumbnail((190,165));self.photo=ImageTk.PhotoImage(photo,master=self.window)
        ttk.Label(reference,image=self.photo).pack(side='left',padx=(0,15))
        reference_text=ttk.Frame(reference);reference_text.pack(side='left',fill='x',expand=True)
        files=ttk.Frame(reference_text);files.pack(fill='x',pady=(0,8))
        self.choose_button=ttk.Button(files,text='기존 보정 JSON 선택',command=self.choose_file)
        self.choose_button.pack(side='left')
        self.apply_file_button=ttk.Button(files,text='선택한 보정 적용',command=lambda:self.start(use_file=True),state='disabled')
        self.apply_file_button.pack(side='left',padx=8)
        self.source_label=tk.StringVar(value='선택한 파일 없음')
        ttk.Label(reference_text,textvariable=self.source_label,wraplength=590,style='Muted.TLabel').pack(fill='x',pady=(0,8))
        ttk.Label(reference_text,textvariable=self.instructions,wraplength=590,justify='left').pack(fill='x')
        self.table=ttk.Treeview(frame,columns=('name','min','middle','current','max','offset','status'),show='headings',height=6)
        for key,label,width in [('name','관절',115),('min','최소 틱',85),('middle','기록한 중간 틱',115),('current','현재 틱',85),('max','최대 틱',85),('offset','영점 오프셋',105),('status','측정',160)]:
            self.table.heading(key,text=label);self.table.column(key,width=width,anchor='center',stretch=key in ('name','status'))
        for name,label in LABELS.items():self.table.insert('','end',iid=name,values=(label,'—','—','—','—','—',''))
        self.table.grid(row=4,column=0,columnspan=3,sticky='nsew');frame.rowconfigure(4,weight=1,minsize=230)
        actions=ttk.Frame(self.window.footer);actions.pack(fill='x',pady=4)
        self.start_button=ttk.Button(actions,text='팔을 받쳤습니다 · 보정 시작',command=self.start)
        self.start_button.pack(side='left')
        self.next_button=ttk.Button(actions,text='전체 서보 중간 자세 기록',command=self.next,state='disabled',style='Primary.TButton')
        self.next_button.pack(side='left',padx=8)
        self.resume_button=ttk.Button(actions,text='범위 더 측정',command=lambda:self.command('resume'),state='disabled')
        self.resume_button.pack(side='left')
        ttk.Button(actions,text='취소 / 닫기',command=self.close).pack(side='right')
        self.status=tk.StringVar(value='모터와 파일은 시작 버튼을 누르기 전에는 변경되지 않습니다.')
        ttk.Label(frame,textvariable=self.status,wraplength=920).grid(row=6,column=0,columnspan=3,sticky='ew',pady=8)
        self.file_label=tk.StringVar();ttk.Label(frame,textvariable=self.file_label,wraplength=920,style='Muted.TLabel').grid(row=7,column=0,columnspan=3,sticky='ew',pady=8)
        self.role_changed();self.job=self.window.after(80,self.poll)

    def role_changed(self):
        self.selected_calibration=None;self.source_label.set('선택한 파일 없음');self.apply_file_button.configure(state='disabled')
        role='leader' if self.role.get()=='리더암' else 'follower'
        arm=getattr(self.app.backend,role,None);args=getattr(self.app.backend,'args',None)
        self.port.set(getattr(args,role+'_port',None) or '')
        self.arm_id.set(getattr(arm,'id',role+'_'+getattr(args,'side','right')))
        directory=getattr(args,'calibration_dir',Path('calibration'))
        self.file_label.set(f'저장 폴더: {directory}\n손목 회전·그리퍼를 포함한 6개 서보의 최소·최대 틱을 직접 측정합니다.')

    def video(self):
        import webbrowser
        webbrowser.open(VIDEO)

    def choose_file(self):
        if self.worker or self.pending_start:return
        from app_choices import askopenfilename
        folder=getattr(getattr(self.app.backend,'args',None),'calibration_dir',Path(__file__).parent/'calibration')
        path=askopenfilename(parent=self.window,title='SO-101 보정 JSON 선택',initialdir=folder,filetypes=[('SO-101 보정 JSON','*.json')])
        if path:self.load_file(path)

    def load_file(self,path):
        if self.worker or self.pending_start:return
        try:data=read_calibration_file(path)
        except Exception as exc:
            self.selected_calibration=None;self.apply_file_button.configure(state='disabled')
            self.source_label.set('선택 실패');self.status.set(str(exc));return
        self.selected_calibration=data;self.source_label.set(str(Path(path).resolve()))
        self.apply_file_button.configure(state='normal')
        for name,label in LABELS.items():
            row=data[name];self.table.item(name,values=(label,row['range_min'],'미기록','—',row['range_max'],row['homing_offset'],'파일에서 읽음'))
        self.instructions.set('선택한 파일의 영점과 범위를 표에서 확인하세요.\n팔을 받친 뒤 파일 적용을 누르면 모터 힘을 풀고 이 값을 기록합니다.')
        self.status.set('파일만 읽었습니다. 모터와 현재 팔의 보정 파일은 아직 바뀌지 않았습니다.')

    def start(self, *, use_file=False):
        if self.worker or self.pending_start:return
        if self.app.backend.identity.get('demo'):
            self.status.set('실물 프로필에서 보정하세요. 데모는 모터에 저장하지 않습니다.');return
        if getattr(self.app,'calibration_reload_required',False):
            self.status.set('새 보정값을 사용하려면 앱을 다시 실행하세요.');return
        role='leader' if self.role.get()=='리더암' else 'follower'
        port=self.port.get().strip();arm_id=self.arm_id.get().strip()
        if not port or not arm_id:self.status.set('USB 포트와 보정 파일 이름을 입력하세요.');return
        if use_file and self.selected_calibration is None:self.status.set('먼저 보정 JSON을 선택하세요.');return
        self.app.stop()
        self.pending_start=(role,port,arm_id,deepcopy(self.selected_calibration) if use_file else None)
        self.app.calibration_busy=True
        self.start_button.configure(state='disabled');self.role_choice.configure(state='disabled')
        self.port_choice.configure(state='disabled');self.id_entry.configure(state='disabled')
        self.choose_button.configure(state='disabled');self.apply_file_button.configure(state='disabled')
        if not use_file:
            self.instructions.set('사진처럼 상완은 위로, 전완·집게는 앞으로 놓고 가동 집게가 위로 벌어지게 맞춘 뒤 영점을 기록하세요.')
            for name,label in LABELS.items():self.table.item(name,values=(label,'—','—','—','—','—',''))
        self.status.set('티칭 연결을 닫은 뒤 선택한 팔의 보정을 시작합니다.')

    def command(self, action):
        if self.worker and not self.worker.finished.is_set():
            self.worker.commands.put(action)
            # A rejected range review must remain retryable.
            if action!='review':self.next_button.configure(state='disabled')

    def next(self):
        if self.worker:
            self.command({'MIDPOINT':'middle','RANGE':'review','REVIEW':'save'}.get(self.worker.state))

    def poll(self):
        if self.closed:return
        if self.pending_start and self.app.state=='DISCONNECTED' and not (self.app.session.thread and self.app.session.thread.is_alive()):
            role,port,arm_id,preset=self.pending_start;self.pending_start=None
            if self.app.cleanup_fault:
                self.status.set('이전 연결의 종료 확인에 실패했습니다. 연결 상태를 확인한 뒤 앱을 다시 여세요.')
            else:
                from datetime import datetime
                path=self.app.output/'motor-diagnostics'/f'calibration_recovery_{arm_id}_{datetime.now():%Y%m%d_%H%M%S_%f}.json'
                self.worker=CalibrationWorker(lambda:make_arm(self.app.backend,role,port,arm_id),path,preset=preset)
                self.worker.thread.start()
        if self.worker:
            try:
                while True:
                    kind,value=self.worker.events.get_nowait()
                    if kind=='positions':self.show_positions(value)
                    elif kind=='state':self.show_state(value)
                    elif kind=='saved':
                        self.app.calibration_reload_required=True
                        self.status.set('저장·읽기 검증 완료: '+value['path']+'\n중간 자세 측정 기록: '+value['reference_path']+'\n앱을 다시 실행하면 새 모터 보정값을 사용합니다. 이전 티칭 자세는 다시 확인·저장하세요.')
                    else:self.status.set(value)
            except queue.Empty:pass
            if self.worker.finished.is_set():
                self.app.calibration_busy=False
                if self.worker.rollback_error:self.app.cleanup_fault=True
                if self.worker.saved and not self.carried:
                    self.carried=True
                    try:carry_station_after_calibration(self.view)
                    except Exception as exc:self.status.set('모터 보정 저장 완료. 카메라 설정 이어받기는 실패했습니다: '+str(exc))
                if self.closing:self.destroy();return
        elif self.closing and not self.pending_start:self.destroy();return
        self.job=self.window.after(80,self.poll)

    def show_state(self, state):
        messages={
            'MIDPOINT':('전체 서보 중간 자세 기록','그리퍼를 포함한 6개 서보를 사진의 중간 자세에 손으로 맞추세요. 고정 집게는 아래, 가동 집게는 위로 벌립니다.\n버튼을 누르면 그 자세가 약 2048틱(라이브러리 기준 2047)으로 읽히는지 확인·기록합니다.'),
            'RANGE':('범위 측정 완료','6개 서보를 하나씩 중간 자세의 양쪽 사용 범위 끝까지 움직이세요. 손목 회전도 측정합니다.\n집게는 두 끝이 처음 닿는 닫힘 위치와 사용할 최대 열림까지 움직이세요. 기록한 중간 틱은 유지됩니다.'),
            'REVIEW':('모터와 JSON에 저장','각 서보의 최소·중간·최대 틱을 확인하세요. 범위 더 측정으로 계속 측정할 수 있습니다.\n저장하면 모터·보정 JSON을 갱신하고 중간 자세의 보정 전후 틱도 측정 기록에 남깁니다.'),
            'SAVING':('저장 중…','모터에 기록하고 다시 읽어 확인합니다.'),
            'DONE':('저장 완료','실제 모터 보정이 끝났습니다. 앱을 다시 실행하세요.'),
            'CLOSED':('보정 종료','보정 연결이 종료되었습니다.')}
        if state in messages:
            label,text=messages[state];self.instructions.set(text)
            self.next_button.configure(text=label,state='normal' if state in ('MIDPOINT','RANGE','REVIEW') else 'disabled')
        self.resume_button.configure(state='normal' if state=='REVIEW' else 'disabled')

    def show_positions(self, data):
        for name,label in LABELS.items():
            low,high=data['mins'].get(name),data['maxes'].get(name)
            middle=data.get('middle',{}).get(name)
            status=('양쪽 범위 측정됨' if middle is not None and low is not None and low<middle<high
                    else '양쪽 끝을 측정하세요' if middle is not None else '중간 자세를 맞추세요')
            self.table.item(name,values=(label,low if low is not None else '—',middle if middle is not None else '—',data['current'][name],high if high is not None else '—',data['offsets'].get(name,'—'),status))

    def close(self):
        self.closing=True;self.pending_start=None
        if self.worker and not self.worker.finished.is_set():
            self.worker.cancel.set();self.status.set('기존 보정값 복원·연결 종료 중…')
        else:self.destroy()

    def destroy(self):
        if self.closed:return
        self.closed=True;self.app.calibration_busy=False
        if self.job:self.window.after_cancel(self.job)
        self.window.destroy();self.view.calibration_window=None
