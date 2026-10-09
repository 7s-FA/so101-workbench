"""Overhead-camera workspace. Tk owns widgets; workers never send motor commands."""
from app_choices import AppChoice

from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
import uuid

MAX_ANALYSIS_AGE = 1.2  # processing age; raw camera feed still expires after 0.6 s


def camera_worker(source, width, height, frames, stop):
    import cv2
    camera = None
    frames.cancel_join_thread()
    try:
        camera = cv2.VideoCapture(int(source) if source.isdecimal() else source)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not camera.isOpened():
            raise ValueError('상단 카메라를 열 수 없습니다. 장치 경로와 다른 앱의 사용 여부를 확인하세요.')
        frame_id = 0
        while not stop.is_set():
            ok, frame = camera.read()
            if not ok:
                raise ValueError('상단 카메라 영상 수신이 끊겼습니다.')
            frame_id += 1
            item = ('frame', time.monotonic(), frame_id, frame)
            try:
                frames.put_nowait(item)
            except queue.Full:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    pass
                try:
                    frames.put_nowait(item)
                except queue.Full:
                    pass
            stop.wait(.03)
    except Exception as exc:
        try:
            frames.put(('error', str(exc)), timeout=.2)
        except queue.Full:
            pass
    finally:
        if camera is not None:
            camera.release()


class VisionWorkspace:
    def __init__(self, app):
        import tkinter as tk
        from tkinter import ttk
        from vision_teaching import default_station, digest
        self.app = app
        self.calibration_window = None
        self.profile_path = app.output / 'vision' / (digest(app.backend.identity)[:16] + '.json')
        self.station = default_station(app.backend.identity)
        self.load_error = ''
        if self.profile_path.exists():
            try:
                from episode_preview import read_episode
                data = read_episode(self.profile_path)
                if data['schema'] not in (1,2) or data['identity'] != app.backend.identity:
                    raise ValueError('보정 프로필의 로봇 정보가 다릅니다.')
                for key in self.station:
                    if key not in data:
                        raise ValueError('보정 프로필 항목이 누락됐습니다.')
                if data['schema'] == 1:
                    data = {**data, 'schema':2, 'jigs':{}}
                    self.load_error = '기존 카메라·TCP 보정은 유지했습니다. 직사각형 지그를 새로 등록하세요.'
                self.station = data
            except Exception as exc:
                self.load_error = str(exc)
        if not self.load_error:
            from camera_profile_recovery import recover_camera
            recovered,source=recover_camera(self.station,self.profile_path.parent)
            if source:
                from teach_left_arm import atomic_json
                atomic_json(self.profile_path,recovered);self.station=recovered
                self.load_error='같은 카메라의 기존 렌즈·모델 좌표 보정을 복원했습니다. 관절 영점·TCP는 유지합니다.'
        from quick_reference import QuickJigs
        self.tracker = QuickJigs(max_gap=MAX_ANALYSIS_AGE)
        from jig_reference import JigReferences
        self.jig_reference = JigReferences()
        self.teaching_reference = JigReferences()
        self.teaching_auto_pending = False
        self.execution_error = ''
        self.frame = None
        self.frame_at = 0.
        self.frame_id = 0
        self.frame_live = False
        self.process = None
        self.stopping = None
        self.frames = None
        self.job = None
        self.completed = queue.Queue()
        self.detection_completed = queue.Queue()
        self.detection_busy = False
        self.detection_generation = 0
        self.detection_submitted = None
        self.detection_displayed_at = 0.
        self.samples, self.sample_size = [], None
        self.sample_board = None
        self.observations = {}
        from app_navigation import AppPage
        self.window = AppPage(app,show=False,persistent=True,fluid=True)
        self.window.title('OpenCV · 상단 카메라 · 지그와 집기 목표')
        self.window.geometry('1180x800')
        self.window.minsize(1000, 720)
        self.window.protocol('WM_DELETE_WINDOW', self.hide)
        self.window.bind('<Escape>', lambda _: app.stop())
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(1, weight=1)
        self.window.bind('<F1>', lambda _: app.open_help())
        self.camera_pane = left = ttk.Frame(self.window, padding=(12, 0, 8, 8))
        left.grid(row=1, column=0, sticky='nsew')
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, background='#182432', highlightthickness=0, width=490, height=320)
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.canvas.create_text(245, 150, text='상단 카메라를 연결하세요\n로봇 연결 없이 보정·인식을 확인할 수 있습니다.', fill='#ecf2f8', justify='center', tags='empty')
        self.camera_status = tk.StringVar(value='카메라 연결 안 됨')
        ttk.Label(left, textvariable=self.camera_status, wraplength=480).grid(row=1, column=0, sticky='ew', pady=8)
        reference_box=ttk.LabelFrame(left,text='도킹한 지그의 기준',padding=7)
        reference_box.grid(row=2,column=0,sticky='ew',pady=(2,6))
        ttk.Label(reference_box,text='에피소드 선택 시 티칭 기준 자동 저장 · 티칭 중 유지. 스텝·에피소드 실행 전에는 새 위치를 최대 3초간 읽습니다.',wraplength=465).pack(anchor='w')
        bar=ttk.Frame(reference_box);bar.pack(fill='x',pady=4)
        self.reference_capture=ttk.Button(bar,text='티칭 기준 다시 읽기',command=lambda:self.guarded(self.capture_jig_reference),style='Compact.TButton')
        self.reference_capture.pack(side='left')
        ttk.Button(bar,text='작업 끝 · 기준 해제',command=self.end_teaching_reference,style='Compact.TButton').pack(side='left',padx=6)
        self.reference_status=tk.StringVar(value='기준 없음 · 위치와 각도를 먼저 읽어 주세요.')
        ttk.Label(reference_box,textvariable=self.reference_status,wraplength=465).pack(anchor='w')
        ttk.Label(left, text='선택한 지그 감지', style='PanelHeading.TLabel').grid(row=3, column=0, sticky='w')
        self.detected = ttk.Treeview(left, columns=('id', 'product', 'position'), show='headings', height=3)
        for col, label, width in [('id', '지그 이름', 105), ('product', '선택한 제품', 100), ('position', '로봇 기준 X · Y · θ', 220)]:
            self.detected.heading(col, text=label); self.detected.column(col, width=width, minwidth=60)
        self.detector_choice = tk.StringVar(value='현재 에피소드')
        self.detector_list = AppChoice(left, textvariable=self.detector_choice, state='readonly', values=('현재 에피소드',))
        self.detector_list.grid(row=4, column=0, sticky='ew', pady=4)
        self.detector_list.bind('<<ComboboxSelected>>', lambda _: self.reset_detection())
        self.detected.grid(row=5, column=0, sticky='ew', pady=6)
        self.notebook = ttk.Notebook(self.window)
        self.notebook.grid(row=1, column=1, sticky='nsew', padx=(0, 12))
        self.settings_window=AppPage(app,show=False,persistent=True)
        self.settings_window.title('장비 설정 · 처음 연결하거나 설치를 바꿀 때')
        self.settings_window.geometry('750x800');self.settings_window.minsize(650,650)
        self.settings_window.protocol('WM_DELETE_WINDOW',self.settings_window.withdraw)
        self.settings_window.bind('<Escape>',lambda _:app.stop())
        self.settings_notebook=ttk.Notebook(self.settings_window);self.settings_notebook.pack(fill='both',expand=True,padx=10)
        robot=ttk.Frame(self.settings_notebook,padding=16);self.settings_notebook.add(robot,text='로봇 연결·영점')
        ttk.Label(robot,text='저장된 로봇·카메라·지그 설정을 사용합니다.',wraplength=650,style='PanelHeading.TLabel').pack(anchor='w',pady=12)
        ttk.Label(robot,text='평소에는 에피소드에서 스텝을 선택하고, 실물 조정에서 위치를 맞추세요.',wraplength=650).pack(anchor='w',pady=12)
        ttk.Label(robot,text='팔로워 보정: '+str(app.backend.identity.get('follower_id',app.backend.identity.get('pair_name','데모'))),wraplength=650).pack(anchor='w',pady=12)
        from launch_mode import open_mode_settings
        ttk.Button(robot,text='실행 모드 · 팔로워 단독 / 리더·팔로워 함께',command=lambda:open_mode_settings(app)).pack(anchor='w',pady=6)
        self.monitor_button=ttk.Button(robot,text='모터 상태 · 전압·온도',command=app.open_monitor)
        self.monitor_button.pack(anchor='w',pady=6)
        ttk.Button(robot,text='그림 설명서',command=app.open_help).pack(anchor='w',pady=6)
        ttk.Button(robot,text='SO-101 모터 캘리브레이션 · 영점·집게',command=self.open_motor_calibration).pack(anchor='w',pady=6)
        ttk.Button(robot,text='실물을 보며 자세 맞추기',command=lambda:self.select_tab(4)).pack(anchor='w',pady=6)
        follower=getattr(app.backend,'follower',None)
        calibration_file=str(getattr(follower,'calibration_fpath','데모 · 보정 파일 없음'))
        ttk.Label(robot,text='현재 팔의 보정 파일\n'+calibration_file+'\n\n다른 팔은 각자의 보정 JSON으로 등록해야 합니다.\n등록 변경: 앱 종료 후 opencv-teach --setup\n저장된 관절 영점·TCP 값은 유지합니다.',wraplength=650).pack(anchor='w',pady=12)
        self.settings_window.withdraw()
        self.tabs = []
        self.pages = []
        self.scroll_canvases = []
        for i,label in enumerate(('카메라 보정', '지그 등록·선택', '고급 · 집게 끝', '감지 확인', '실물 조정')):
            notebook=self.settings_notebook if i in (0,1,2) else self.notebook
            page = ttk.Frame(notebook)
            page.columnconfigure(0, weight=1); page.rowconfigure(0, weight=1)
            canvas = tk.Canvas(page, background='#dde3e7', highlightthickness=0, width=500)
            scroll = ttk.Scrollbar(page, orient='vertical', command=canvas.yview)
            canvas.configure(yscrollcommand=scroll.set)
            canvas.grid(row=0,column=0,sticky='nsew');scroll.grid(row=0,column=1,sticky='ns')
            tab = ttk.Frame(canvas, padding=10)
            item = canvas.create_window(0,0,window=tab,anchor='nw')
            canvas.bind('<Configure>',lambda e,c=canvas,i=item:c.itemconfigure(i,width=e.width))
            tab.bind('<Configure>',lambda e,c=canvas:c.configure(scrollregion=c.bbox('all')))
            notebook.add(page, text=label); self.tabs.append(tab); self.pages.append(page)
            self.scroll_canvases.append(canvas)
        # Keep logical page IDs stable; present the inspection task first.
        # Camera/manual operation and initial settings are separate pages in the same window.
        self.window.section='vision'
        self.notebook.select(self.pages[3])
        self.notebook.bind('<<NotebookTabChanged>>',self.operation_tab_changed)
        self.vars = {}
        self.status = tk.StringVar(value=self.load_error or '에피소드를 선택하면 티칭 기준을 자동으로 읽습니다. 설치를 바꿀 때만 장비 설정을 수정하세요.')
        ttk.Label(self.window, textvariable=self.status, wraplength=1100, padding=12).grid(row=2, column=0, columnspan=2, sticky='ew')
        ttk.Label(self.settings_window,textvariable=self.status,wraplength=690,padding=10).pack(fill='x')
        self.build_camera()
        self.build_targets()
        self.build_tool()
        self.build_detection_check()
        from manual_teaching import ManualTeachingPanel
        self.manual=ManualTeachingPanel(self,self.tabs[4])
        from safe_episode import SafeEpisodeExecution
        self.safe_execution=SafeEpisodeExecution(self)
        from jig_acquisition import JigAcquisition
        self.reference_acquisition=JigAcquisition(self)
        self.execution_revision=0
        self.measurement_not_before=0.
        for form,canvas in zip(self.tabs,self.scroll_canvases):
            self.bind_scrolling(form,canvas)
        self.refresh_target_list()
        if self.target_ids:
            self.target_id=self.target_ids[0];self.refresh_target_list();self.load_target()
        if self.station.get('intrinsics'):
            self.sample_status.set('렌즈 보정 저장됨 · %d장 · RMS %.3fpx' % (len(self.station['intrinsics'].get('errors_px',[])),self.station['intrinsics'].get('rms_px',0)))
            self.status.set(self.station['intrinsics'].get('quality_note') or '저장된 카메라 설정을 사용합니다. 에피소드를 선택하세요.')
        self.closed = False
        self.settings_notebook.bind('<<NotebookTabChanged>>',self.settings_tab_changed)
        self.refresh_setup_status()
        self.poll()

    def show_settings(self):
        self.settings_window.deiconify();self.settings_window.lift()

    def select_tab(self, logical_index):
        if logical_index in (0,1,2):
            self.settings_notebook.select(self.pages[logical_index]);self.settings_tab_changed();self.show_settings()
        else:
            self.layout_operation(logical_index==4)
            self.window.section='manual' if logical_index==4 else 'vision'
            self.window.title('실물 조정' if logical_index==4 else '카메라 · 지그 측정')
            self.window.deiconify();self.notebook.select(self.pages[logical_index]);self.window.lift()
            if self.frame is not None:self.draw_frame()

    def operation_tab_changed(self,_event=None):
        manual=self.notebook.select()==str(self.pages[4])
        self.layout_operation(manual)
        if manual and hasattr(self,'image_roi') and self.image_roi.active:self.image_roi.cancel()
        self.window.section='manual' if manual else 'vision'
        self.window.title('실물 조정' if manual else '카메라 · 지그 측정')
        if self.app.navigation.current is self.window:
            self.app.navigation.activate(self.window.section)

    def settings_tab_changed(self,_event=None):
        catalog_active=self.settings_notebook.select()==str(self.pages[1])
        self.settings_window.dock_preview=catalog_active
        self.settings_window.fluid=catalog_active
        if catalog_active:
            self.app.show_preview();self.app.preview.show_simulation()
        if self.app.navigation.current is self.settings_window:
            self.app.navigation.position(self.settings_window)

    def layout_operation(self,manual):
        self.window.dock_preview=manual
        if manual:
            self.camera_pane.grid_remove()
            self.window.columnconfigure(0,weight=1);self.window.columnconfigure(1,weight=0)
            self.notebook.grid_configure(column=0,columnspan=2,padx=8)
        else:
            self.camera_pane.grid()
            self.window.columnconfigure(0,weight=1);self.window.columnconfigure(1,weight=1)
            self.notebook.grid_configure(column=1,columnspan=1,padx=(0,12))
        if self.app.navigation.current is self.window:self.app.navigation.position(self.window)

    def checking(self):
        return self.notebook.select() == str(self.pages[3])

    def section(self, parent, row, title, *, expanded=False):
        from tkinter import ttk
        shell=ttk.Frame(parent)
        shell.grid(row=row,column=0,columnspan=2,sticky='ew',pady=(6,2))
        shell.columnconfigure(0,weight=1)
        body=ttk.Frame(shell,padding=(4,6))
        def toggle():
            if body.winfo_manager():body.grid_remove();button.configure(text='＋ '+title)
            else:body.grid(row=1,column=0,sticky='ew');button.configure(text='− '+title)
        button=ttk.Button(shell,text=('− ' if expanded else '＋ ')+title,command=toggle)
        button.grid(row=0,column=0,sticky='ew')
        if expanded:body.grid(row=1,column=0,sticky='ew')
        return body

    def refresh_setup_status(self):
        intrinsic=self.station.get('intrinsics')
        if not intrinsic:lens='미설정'
        elif intrinsic.get('usable_for_metric') is False:lens='사진 추가 필요'
        else:lens=f"저장됨 · {len(intrinsic.get('errors_px',[]))}장 · {intrinsic.get('rms_px',0):.3f}px"
        from model_camera import calibration_label
        robot=calibration_label(self.station)
        tool='저장된 모델값 사용' if self.station['tool'].get('reference',{}).get('kind')=='model_fixed_finger_tip' else '저장값 사용'
        self.setup_status.set(f'렌즈 보정: {lens}\n로봇 좌표: {robot}  /  집게 끝: {tool}')

    def bind_scrolling(self, widget, canvas):
        def wheel(event):
            direction=-1 if getattr(event,'num',None)==4 or getattr(event,'delta',0)>0 else 1
            canvas.yview_scroll(direction*2,'units')
            return 'break'
        def focus(event):
            self.window.update_idletasks()
            y=event.widget.winfo_rooty()-canvas.winfo_rooty()
            if y<0 or y+event.widget.winfo_height()>canvas.winfo_height():
                total=max(1,canvas.bbox('all')[3])
                current=canvas.canvasy(0)
                canvas.yview_moveto(max(0,(current+y-50)/total))
        for sequence in ('<Button-4>','<Button-5>','<MouseWheel>'):widget.bind(sequence,wheel,add='+')
        widget.bind('<FocusIn>',focus,add='+')
        for child in widget.winfo_children():self.bind_scrolling(child,canvas)

    def show(self):
        self.window.deiconify(); self.window.lift(); self.refresh_target_list()
        self.refresh_setup_status()

    def hide(self):
        self.window.withdraw()

    def field(self, parent, row, label, key, value, choices=None, increment=None):
        import tkinter as tk
        from tkinter import ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=(0, 8), pady=3)
        var = tk.StringVar(value=str(value)); self.vars[key] = var
        slider_range=(0,100,.1) if key in ('t_open','t_closed') else (-180,180,.1) if key.startswith('offset_') else None
        cell=ttk.Frame(parent) if slider_range else parent
        widget = (AppChoice(cell, textvariable=var, values=choices, state='readonly', width=18) if choices else
                  ttk.Spinbox(cell, textvariable=var, from_=-1000, to=1000, increment=increment, width=18) if increment else
                  ttk.Entry(cell, textvariable=var, width=18))
        if slider_range:
            from motor_slider import ValueSlider
            cell.grid(row=row,column=1,sticky='ew',pady=3);widget.pack(fill='x')
            if not hasattr(self,'servo_sliders'):self.servo_sliders={}
            self.servo_sliders[key]=ValueSlider(cell,var,*slider_range)
            self.servo_sliders[key].scale.pack(fill='x')
        else:widget.grid(row=row, column=1, sticky='ew', pady=3)
        parent.columnconfigure(1, weight=1)
        return widget

    def button_row(self, parent, row, buttons):
        from tkinter import ttk
        bar = ttk.Frame(parent)
        bar.grid(row=row, column=0, columnspan=2, sticky='ew', pady=6)
        for label, action in buttons:
            ttk.Button(bar, text=label, command=lambda f=action: self.guarded(f), style='Compact.TButton').pack(side='left', padx=(0, 5))
        return bar

    def build_camera(self):
        import tkinter as tk
        from tkinter import ttk
        tab = self.tabs[0]
        tab.columnconfigure(0,weight=1)
        camera, board = self.station['camera'], self.station['board']
        ttk.Label(tab,text='렌즈 보정 → 로봇 좌표 연결',style='PanelHeading.TLabel').grid(row=0,column=0,columnspan=2,sticky='w',pady=6)
        ttk.Label(tab,text='감지 박스와 픽셀 중심 확인은 보정 없이도 가능합니다.\n로봇을 지그 위치로 움직일 때 아래 두 보정이 필요합니다.',wraplength=440).grid(row=1,column=0,columnspan=2,sticky='ew',pady=6)
        device=self.section(tab,2,'카메라 장치 · 해상도 변경')
        sources=sorted(str(p) for p in Path('/dev/v4l/by-id').glob('*video-index0'))
        sources=list(dict.fromkeys([str(camera['source']),*sources,'0']))
        self.field(device,0,'상단 카메라','source',camera['source'],sources)
        self.field(device,1,'영상 너비 px','width',camera['width'])
        self.field(device,2,'영상 높이 px','height',camera['height'])
        self.button_row(device,3,[('연결',self.start_camera),('연결 해제',self.stop_camera)])
        lens=ttk.LabelFrame(tab,text='1. 렌즈 오차 보정',padding=10)
        lens.grid(row=3,column=0,columnspan=2,sticky='ew',pady=8)
        self.sample_status=tk.StringVar(value='저장된 보정 없음 · 폴더 사진으로 자동 보정하세요.')
        ttk.Label(lens,textvariable=self.sample_status,wraplength=440).grid(row=0,column=0,columnspan=2,sticky='ew',pady=4)
        self.field(lens,1,'보정 사진 폴더','photo_dir',str(Path(__file__).resolve().parent/'calibration'/'camera_images'))
        ttk.Label(lens,text='기본 체커: 가로 13 × 세로 9 내부 코너 · 간격 20mm',wraplength=440,style='Muted.TLabel').grid(row=2,column=0,columnspan=2,sticky='w',pady=4)
        self.button_row(lens,3,[('폴더 사진으로 자동 보정',lambda:self.capture_folder(auto_calibrate=True))])
        advanced=self.section(lens,4,'체커 설정 · 사진별 오차 · 수동 수집')
        for row,(label,key) in enumerate((('내부 코너 가로 개수','cols'),('내부 코너 세로 개수','rows'),('코너 간격 mm','square_mm'))):
            self.field(advanced,row,label,key,board[key])
        self.button_row(advanced,3,[('현재 사진 수집',self.capture_board),('보정 계산',self.calibrate)])
        self.sample_list=ttk.Treeview(advanced,columns=('index','error'),show='headings',height=3)
        self.sample_list.heading('index',text='사진');self.sample_list.heading('error',text='재투영 오차 px')
        self.sample_list.grid(row=4,column=0,columnspan=2,sticky='ew',pady=5)
        self.button_row(advanced,5,[('선택 제외',self.remove_sample),('수집 초기화',self.clear_samples)])
        self.button_row(advanced,6,[('폴더 사진 수집만',self.capture_folder),('보정 파일 불러오기',self.import_intrinsics)])
        ext=self.station.get('extrinsics') or {}
        model=self.section(tab,4,'2. 모델 위치로 좌표 연결 · 추정값')
        ttk.Label(model,text='왼쪽 SO-101·오른쪽 카메라 마운트 STL을 사용합니다.\n광학 중심은 PCB 면 중앙(0mm)으로 적용합니다.\n렌즈 돌출 20mm는 더하지 않습니다. 실측 보정과 구분합니다.',wraplength=420).grid(row=0,column=0,columnspan=2,sticky='ew',pady=6)
        self.field(model,1,'PCB 앞 광학 중심 거리 mm','model_optical_offset',ext.get('optical_offset_mm',0))
        self.field(model,2,'영상 축 회전 °','model_image_roll',ext.get('image_roll_deg',0),('0','90','180','270'))
        self.button_row(model,3,[('모델 위치로 보정 적용',self.apply_model_camera)])
        coord=self.section(tab,5,'선택 · 설치 변경 시 체커보드로 좌표 다시 연결')
        ttk.Label(coord,text='카메라의 mm 위치를 로봇 베이스 좌표로 바꿉니다.\n수평 체커보드의 0번 코너 위치와 축 방향을 실측하세요.\n기본값 0은 측정값이 아닙니다.',wraplength=420).grid(row=0,column=0,columnspan=2,sticky='ew',pady=6)
        self.button_row(coord,1,[('기준판 방향 보기',self.show_board)])
        for i,(axis,value) in enumerate(zip('XYZ',ext.get('origin_mm',[0,0,0])),2):
            self.field(coord,i,f'0번 코너 {axis} mm','origin_'+axis,value)
        self.field(coord,5,'코너 열 방향 θ °','board_yaw',ext.get('yaw_deg',0))
        self.field(coord,6,'코너 행 진행 방향','board_row_sign','-Y' if ext.get('row_sign',-1)==-1 else '+Y',('-Y','+Y'))
        self.axes_confirm=tk.BooleanVar(value=False)
        ttk.Checkbutton(coord,text='0번 코너·열(+X)·행(±Y)을 실측하여 입력함',variable=self.axes_confirm).grid(row=7,column=0,columnspan=2,sticky='w',pady=6)
        self.button_row(coord,8,[('로봇 좌표 보정 저장',self.extrinsics)])
        ttk.Label(tab,text='배경·조명만 바뀌면 렌즈 보정을 반복할 필요가 없습니다.\n카메라를 옮기면 로봇 좌표 연결을 다시 설정하세요.\n렌즈·초점·해상도가 바뀌면 렌즈 보정을 확인하세요.',wraplength=440,style='Muted.TLabel').grid(row=6,column=0,columnspan=2,sticky='ew',pady=8)

    def build_targets(self):
        from tkinter import ttk
        tab = self.tabs[1]
        tab.columnconfigure(0,weight=1)
        self.target_list = AppChoice(tab, state='readonly')
        self.target_list.grid(row=0,column=0,columnspan=2,sticky='ew')
        self.target_list.bind('<<ComboboxSelected>>', lambda _: self.load_target())
        self.target_id = None
        defaults = [('목표 이름','name','용기 1번 칸',None), ('제품 이름 (직접 선택)','product','제품 A',None),
                    ('위치 기준','mode','지그를 따라가기',('지그를 따라가기','이 자리에 고정')),
                    ('동작','operation','집기',('집기','놓기')), ('지그 이름','jig_name','하위 지그 220×148',None),
                    ('동작 구성','return_to_start','잡기·놓기 4단계',('잡기·놓기 4단계','집고 제자리에 놓기')),
                    ('외곽 검출 방식','appearance','흰색 프레임',('흰색 프레임','명암 외곽')),
                    ('지그 형태','shape_model','도넛형 (내부 구멍 있음)',('도넛형 (내부 구멍 있음)','직사각형')),
                    ('검출 방법','detection_method','색상+윤곽 (기본)',('색상+윤곽 (기본)','윤곽 중심','흰색 중심')),
                    ('긴 변 mm (STL 220)','long_mm',220,None), ('짧은 변 mm (STL 148)','short_mm',148,None),
                    ('지그 테두리 높이 Z mm','surface_z_mm',0,None),
                    ('지그가 놓인 방향 °','expected_yaw_deg',0,None), ('지그 방향 허용 범위 ±°','yaw_tolerance_deg',30,None),
                    ('외곽 크기 허용 오차 %','size_tolerance_pct',5,None),
                    ('검출 영역 왼쪽 %','roi_x_pct',0,None), ('검출 영역 위쪽 %','roi_y_pct',0,None),
                    ('검출 영역 너비 %','roi_w_pct',100,None), ('검출 영역 높이 %','roi_h_pct',100,None),
                    ('잡기·놓기 위치 X mm','x_mm',0,None), ('잡기·놓기 위치 Y mm','y_mm',0,None),
                    ('잡기·놓기 높이 mm','z_mm',30,None), ('집게 방향 ° (지그 기준)','yaw_deg',180,None),
                    ('집게 기울기 ° (180=아래)','pitch_deg',180,None),
                    ('집게 좌우 기울기 °','roll_deg',180,None),
                    ('잡기 전 띄울 높이 mm','hover_mm',20,None), ('잡은 뒤 올릴 높이 mm','retreat_mm',20,None),
                    ('하강 여유 X mm (지그 기준)','approach_x_mm',0,None),
                    ('하강 여유 Y mm (지그 기준)','approach_y_mm',0,None),
                    ('기울이기 전 올릴 높이 mm','lift_clearance_mm',40,None),
                    ('상승 중 최대 기울임 ° (0=고정)','lift_tilt_deg',0,None),
                    ('상승 후 유지 시간 s','retreat_wait_s',.4,None),
                    ('집게 열림값 0~100','open',80,None),
                    ('집게 닫힘값 0~100','closed',20,None), ('위아래 이동 속도 mm/s','speed_mm_s',10,None),
                    ('각 단계 도착 후 대기 s','wait_s',.4,None)]
        jig=self.section(tab,2,'지그 크기·높이·검출 방법',expanded=True)
        advanced=self.section(tab,3,'검출 범위·도킹 방향')
        basic=self.section(tab,4,'고급 · 기존 목표 좌표 직접 편집')
        motion=self.section(tab,5,'고급 · 기존 자동 동작의 집게·속도')
        groups={basic:{'name','mode','operation','return_to_start','x_mm','y_mm','z_mm','yaw_deg','pitch_deg','hover_mm'},
                jig:{'product','jig_name','shape_model','detection_method','long_mm','short_mm','surface_z_mm'},
                motion:{'open','closed','speed_mm_s','wait_s','retreat_mm','roll_deg',
                        'approach_x_mm','approach_y_mm','lift_clearance_mm','lift_tilt_deg','retreat_wait_s'}}
        counts={group:0 for group in (basic,jig,motion,advanced)}
        for label,key,value,choices in defaults:
            group=next((g for g,keys in groups.items() if key in keys),advanced)
            self.field(group,counts[group],label,'t_'+key,value,choices,increment=1 if key in ('x_mm','y_mm','z_mm','yaw_deg') else None)
            counts[group]+=1
        import tkinter as tk
        ttk.Label(jig,text='외곽 테두리 높이 = 로봇 기준 지그 바닥 높이 + 4.8mm.\n직사각형 외곽은 180° 반전을 구별하지 못하므로\n운반로봇이 멈추는 방향을 일정하게 맞추세요.',wraplength=420).grid(row=counts[jig],column=0,columnspan=2,sticky='ew',pady=6)
        actions=ttk.Frame(self.pages[1],padding=(10,4))
        actions.grid(row=1,column=0,columnspan=2,sticky='ew')
        self.button_row(actions,0,[('새 지그 기준',self.new_target),('지그 설정 저장',self.save_target)])
        self.button_row(actions,1,[('실물을 보며 위치 맞추기',lambda:self.select_tab(4))])
        self.button_row(basic,counts[basic],[('기존 동작 스텝 추가',self.add_steps),('경로 계산·3D 반영',self.preview_plan)])
        sim=self.section(tab,1,'시뮬레이션의 집기값 가져오기')
        ttk.Label(sim,text='Test_teach에서 저장한 배치·집기점·안전 자세를 가져옵니다.\n시작 안전 자세 → 측정 → 집기·제자리 놓기 → 종료 안전 자세,\n상단·구·하단별 9단계 후보를 각각 저장합니다.\n하강 여유·상승 중 기울임도 포함합니다.',wraplength=420).grid(row=0,column=0,columnspan=2,sticky='ew',pady=4)
        self.sim_part=AppChoice(sim,state='readonly',values=('상단','구','하단'))
        self.sim_part.grid(row=1,column=0,columnspan=2,sticky='ew');self.sim_part.current(0)
        self.button_row(sim,2,[('후보 3개 저장 · 선택 부품 열기',self.import_simulation)])
        ttk.Label(sim,text='집게 끝·카메라·모터 보정은 현재 설정을 사용합니다.\n질량·마찰·유지 토크는 참고 정보로만 저장합니다.\n도킹 후 실행 → 안전 자세 도착 → 새 3초 측정.\n집기·제자리 놓기를 마치면 안전 자세로 돌아옵니다.',wraplength=420,style='Muted.TLabel').grid(row=3,column=0,columnspan=2,sticky='ew',pady=4)
        ttk.Label(tab,text='여기서는 카메라가 읽을 지그만 설정합니다.\n잡기·놓기 위치는 메인 화면의 스텝에서 맞추세요.\n기존 숫자 입력·자동 동작 생성은 고급 설정에 남겨두었습니다.',wraplength=440,style='Muted.TLabel').grid(row=7,column=0,columnspan=2,sticky='ew',pady=8)
        self.button_row(advanced,counts[advanced],[('사진 외곽 점검',self.inspect_outline)])
        # Keep legacy variables/load compatibility; normal setup edits only jig profiles.
        for child in tab.winfo_children():child.grid_remove()
        actions.grid_remove()
        from jig_catalog import CatalogPanel
        self.catalog=CatalogPanel(self,tab)

    def build_detection_check(self):
        import tkinter as tk
        from tkinter import ttk
        tab=self.tabs[3]
        ttk.Label(tab,text='지그 감지 확인',style='PanelHeading.TLabel').grid(row=0,column=0,columnspan=2,sticky='w',pady=8)
        ttk.Label(tab,text='① 카메라 연결 → ② 흰색 지그 전체를 화면에 배치\n③ 지그 테두리·중심 십자·방향 화살표를 확인하세요.\n기본값은 220 × 148mm · 3×2 칸 지그입니다.',wraplength=440).grid(row=1,column=0,columnspan=2,sticky='ew',pady=6)
        self.button_row(tab,2,[('카메라 연결',self.start_camera),('영상 중지',self.stop_camera),('사진 열기',self.open_image)])
        method_bar=ttk.LabelFrame(tab,text='검출 방법',padding=6)
        method_bar.grid(row=3,column=0,columnspan=2,sticky='ew',pady=6)
        ttk.Label(method_bar,text='범위를 설정할 지그').pack(anchor='w')
        self.roi_jig_picker=AppChoice(method_bar,state='readonly')
        self.roi_jig_picker.pack(fill='x',pady=(2,6))
        self.roi_jig_picker.bind('<<ComboboxSelected>>',self.select_roi_jig)
        method=AppChoice(method_bar,textvariable=self.vars['t_detection_method'],values=('색상+윤곽 (기본)','윤곽 중심','흰색 중심'),state='readonly')
        method.pack(fill='x')
        ttk.Button(method_bar,text='선택 방법 적용',command=lambda:self.guarded(self.save_detection_method)).pack(anchor='e',pady=4)
        ttk.Label(method_bar,text='기본은 색상+윤곽입니다. 윤곽/흰색 중심은 빠르지만 배경에 따라 놓칠 수 있습니다. 모든 방법에서 지그 구조·크기를 확인합니다.',wraplength=420).pack(anchor='w')
        area=ttk.Frame(method_bar);area.pack(fill='x',pady=4)
        self.roi_button=ttk.Button(area,text='감지 범위 직접 지정',command=lambda:self.guarded(self.image_roi.toggle))
        self.roi_button.pack(side='left')
        ttk.Button(area,text='전체 범위로',command=lambda:self.guarded(self.reset_detection_roi)).pack(side='left',padx=4)
        self.roi_hint=tk.StringVar(master=tab,value='직접 지정 버튼을 누르고 영상에서 사각형을 끌어 주세요. 검출 전에도 지정할 수 있습니다.')
        ttk.Label(method_bar,textvariable=self.roi_hint,wraplength=420).pack(fill='x',pady=4)
        from image_roi import ImageROI
        self.image_roi=ImageROI(self)
        self.detection_timing=tk.StringVar(value='검출 처리 시간: 측정 전')
        ttk.Label(tab,textvariable=self.detection_timing,wraplength=440).grid(row=4,column=0,columnspan=2,sticky='w')
        self.check_status=tk.StringVar(value='카메라를 연결하거나 지그 사진을 여세요.')
        self.check_center=tk.StringVar(value='—')
        self.check_angle=tk.StringVar(value='—')
        self.check_world=tk.StringVar(value='로봇 좌표 보정 후 표시')
        for i,(label,var) in enumerate((('감지 상태',self.check_status),('영상 중심 X · Y (px)',self.check_center),
                                       ('영상 긴 변 방향 (°)',self.check_angle),('로봇 X · Y (mm) · θ (°)',self.check_world)),3):
            ttk.Label(tab,text=label,style='PanelHeading.TLabel').grid(row=i*2,column=0,columnspan=2,sticky='w',pady=(10,2))
            ttk.Label(tab,textvariable=var,wraplength=440).grid(row=i*2+1,column=0,columnspan=2,sticky='ew')
        ttk.Label(tab,text='초록: 정밀 / 주황: 중간 / 회색: 낮음·참고용\n최대 3초 수집 · 정밀 포함 2개면 즉시 확정합니다.\n3초 시점에는 정밀 1개 또는 일관된 중간 2개도 사용합니다.\n이상값 제외(3mm·각도폭 5°) 후 정밀 우선 가중평균을 씁니다.\n짧은 누락은 유지하고, 여러 후보·긴 중단·이동 시 다시 셉니다.\n영상 각도: 오른쪽 0°, 시계방향 +, 180° 앞뒤 미구분.',wraplength=440,style='Muted.TLabel').grid(row=15,column=0,columnspan=2,sticky='ew',pady=14)
        self.setup_status=tk.StringVar()
        ttk.Separator(tab).grid(row=16,column=0,columnspan=2,sticky='ew',pady=6)
        ttk.Label(tab,textvariable=self.setup_status,wraplength=440).grid(row=17,column=0,columnspan=2,sticky='ew',pady=6)
        self.button_row(tab,18,[('카메라 보정 열기',lambda:self.select_tab(0)),('집기·놓기 설정',lambda:self.select_tab(1))])
        self.notebook.bind('<<NotebookTabChanged>>',self.check_tab_changed,add='+')

    def check_tab_changed(self,event=None):
        if self.checking() and self.frame is not None:
            self.guarded(lambda:self.process_frame(background=self.frame_live))

    def clear_check(self,message):
        if hasattr(self,'check_status'):
            self.check_status.set(message);self.check_center.set('—');self.check_angle.set('—')
            self.check_world.set('현재 측정 없음')

    def matching_intrinsics(self):
        value=self.station.get('intrinsics')
        if value and self.frame is not None and value.get('size')==list(self.frame.shape[1::-1]) and value.get('camera')==self.station['camera']:
            return value
        return None

    def candidates_for(self, profile):
        from jig_detection import inspection_candidates,profile_signature
        key=profile_signature(profile)
        token=self.candidate_frame_token()
        if getattr(self,'_candidate_token',None)!=token:
            self._candidate_cache={};self._candidate_token=token
        if key not in self._candidate_cache:
            self._candidate_cache[key]=inspection_candidates(self.frame,profile,self.station)
        return self._candidate_cache[key]

    def metric_pose_for(self, profile, *, candidates=None):
        """Share one metric projection between execution and overlay for a frame."""
        from jig_detection import detect_rectangle,profile_signature
        token=self.candidate_frame_token()
        if getattr(self,'_metric_token',None)!=token:
            self._metric_cache={};self._metric_token=token
        pending=self.reference_acquisition.pending
        enforce_heading=not bool(pending and pending.get('heading_only'))
        key=(profile_signature(profile),enforce_heading)
        if key not in self._metric_cache:
            self._metric_cache[key]=detect_rectangle(self.frame,profile,self.station,candidates=candidates,enforce_heading=enforce_heading)
        return self._metric_cache[key]

    def candidate_frame_token(self):
        from vision_teaching import calibration_signature
        return (self.frame_id,self.frame_at,id(self.frame),calibration_signature(self.station))

    def detection_overlay(self,overlay):
        import cv2
        import numpy as np
        from carrier_detection import image_pose
        if self.frame_live and time.monotonic()-self.frame_at>.6:
            self.clear_check('영상 갱신 중단');return overlay
        try:
            pending=self.reference_acquisition.pending
            draft=pending['target'] if pending else self.read_target()
            if draft['mode']!='rectangle':raise ValueError('잡기·놓기에서 지그를 따라가기를 선택하세요.')
            matches=self.candidates_for(draft)
            if draft['shape_model']=='rectangle':
                kept=[];ratio=draft['long_mm']/draft['short_mm']
                for candidate in matches:
                    sides=np.linalg.norm(np.roll(candidate.corners,-1,axis=0)-candidate.corners,axis=1)
                    pair=[(sides[0]+sides[2])/2,(sides[1]+sides[3])/2]
                    if abs(max(pair)/min(pair)-ratio)/ratio<.2:kept.append(candidate)
                matches=kept
            if not matches:
                message='도넛형 미검출 · 외곽과 내부 구멍의 가림을 확인하세요.' if draft['shape_model']=='ring' else '사각형 외곽 미검출'
                raise ValueError(message)
            if len(matches)>1:
                for candidate in matches:cv2.polylines(overlay,[candidate.corners.astype(int)],True,(0,165,255),2)
                self.clear_check(f'지그 후보 {len(matches)}개 · 검출 영역을 좁혀 하나만 선택하세요.')
                return overlay
            candidate=matches[0];corners=candidate.corners
            center,angle,direction=image_pose(corners,self.matching_intrinsics())
            quality=candidate.quality
            color=(0,220,80) if quality=='precise' else (0,165,255) if quality=='medium' else (160,160,160)
            cv2.polylines(overlay,[corners.astype(int)],True,color,3)
            cv2.drawMarker(overlay,tuple(center.astype(int)),color,cv2.MARKER_CROSS,24,3)
            cv2.arrowedLine(overlay,tuple(center.astype(int)),tuple((center+direction).astype(int)),color,3)
            cv2.putText(overlay,f"({center[0]:.1f}, {center[1]:.1f}) px  {angle:.1f} deg",tuple(center.astype(int)+[12,-12]),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
            source='실시간' if self.frame_live else '사진'
            status={'precise':'정밀','medium':'중간','low':'낮음 · 참고용'}[quality]
            self.check_status.set(f'{source} · {status}')
            self.check_center.set(f'{center[0]:.1f} · {center[1]:.1f}'+(' (추정)' if not candidate.precision_ok else ''))
            self.check_angle.set(f'{angle:.1f}°'+(' (추정)' if not candidate.precision_ok else ''))
            if quality=='low':
                self.check_world.set('실물 실행에 사용하지 않음 · '+candidate.reason)
            else:
                try:
                    _,value=self.metric_pose_for(draft,candidates=matches)
                    note=' · 등록한 지그 치수 사용'
                    if (self.station.get('extrinsics') or {}).get('source')=='cad_estimate':note+=' · 모델 추정'
                    if self.frame_live:
                        from vision_teaching import reference_key
                        note+='\n기준 '+self.tracker.status_text(reference_key(draft))
                    self.check_world.set(f"{value['x_mm']:.2f} · {value['y_mm']:.2f} mm · {value['yaw_deg']:.2f}°"+note)
                except ValueError as exc:self.check_world.set(str(exc))
        except ValueError as exc:self.clear_check(str(exc))
        return overlay

    def capture_folder(self, auto_calibrate=False):
        from vision_teaching import chessboard, calibrate_intrinsics, number
        import cv2
        self.idle();folder=Path(self.vars['photo_dir'].get()).expanduser()
        if not folder.is_dir():raise ValueError('보정 사진 폴더를 확인하세요.')
        files=sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
        board=self.board()
        cam={'source':self.vars['source'].get().strip(),'width':int(number(self.vars['width'].get(),320,4096)),
             'height':int(number(self.vars['height'].get(),240,2160))}
        if not cam['source']:raise ValueError('카메라 장치를 먼저 선택하세요.')
        if not files:raise ValueError('폴더에 사진이 없습니다.')
        def collect():
            samples=[];size=None;skipped=[];accepted=[]
            for path in files[:300]:
                frame=cv2.imread(str(path))
                if frame is None:skipped.append(path.name);continue
                shape=list(frame.shape[1::-1])
                if shape!=[cam['width'],cam['height']]:skipped.append(path.name);continue
                try:corners=chessboard(frame,board)
                except ValueError:skipped.append(path.name);continue
                samples.append(corners);size=shape;accepted.append(path.name)
            intrinsic=calibrate_intrinsics(samples,size,board,cam) if auto_calibrate else None
            if intrinsic is not None:
                intrinsic.update(source_files=accepted,source_directory=str(folder))
            return samples,size,skipped,intrinsic
        def done(value):
            self.samples,self.sample_size,skipped,intrinsic=value;self.sample_board=board
            self.sample_list.delete(*self.sample_list.get_children())
            for i in range(len(self.samples)):self.sample_list.insert('','end',values=(i+1,'계산 전'))
            self.sample_status.set(f'폴더 수집 {len(self.samples)}장 · 제외 {len(skipped)}장')
            if intrinsic is not None:
                if cam!=self.station['camera']:self.stop_camera()
                self.station.update(camera=cam,board=board,intrinsics=intrinsic,extrinsics=None)
                self.axes_confirm.set(False);self.save_station()
                for item,error in zip(self.sample_list.get_children(),intrinsic['errors_px']):self.sample_list.set(item,'error',f'{error:.3f}')
                self.sample_status.set(f"{len(self.samples)}장 보정 저장됨 · RMS {intrinsic['rms_px']:.3f}px · 제외 {len(skipped)}장")
                self.status.set(intrinsic.get('quality_note') or '렌즈 보정 저장 완료. 실제 mm 좌표를 사용하려면 로봇 좌표 연결을 설정하세요.')
            else:self.status.set('사진 수집 완료. 보정 계산을 누르세요.')
        self.run_job(collect,done)
        self.status.set(f'사진 {min(len(files),300)}장 분석'+(' · 렌즈 보정 계산 중…' if auto_calibrate else ' 중…'))

    def import_intrinsics(self):
        import json
        import numpy as np
        from app_choices import askopenfilename
        self.idle()
        path=askopenfilename(parent=self.window,title='내부 보정 JSON 선택',initialdir=Path(__file__).parent/'calibration',filetypes=[('보정 JSON','*.json')])
        if not path:return
        value=json.loads(Path(path).read_text())
        k=np.asarray(value['K'],float);d=np.asarray(value['D'],float)
        if k.shape!=(3,3) or not np.isfinite(k).all() or not np.isfinite(d).all() or value['camera']!=self.station['camera']:
            raise ValueError('카메라 장치·해상도가 보정 파일과 다릅니다. 카메라 설정을 맞추세요.')
        self.station['intrinsics']=value;self.station['extrinsics']=None;self.station['board']=value['board']
        for key in ('cols','rows','square_mm'):self.vars[key].set(str(value['board'][key]))
        self.save_station();self.sample_status.set(f"내부 보정 불러옴 · RMS {value['rms_px']:.3f}px")
        self.status.set(value.get('quality_note') or '내부 보정을 불러왔습니다. 로봇 좌표 보정은 다시 수행하세요.')

    def build_tool(self):
        import tkinter as tk
        from tkinter import ttk
        from vision_teaching import JOINTS
        from model_tcp import execution_summary
        tab = self.tabs[2]; tool = self.station['tool']
        ttk.Label(tab,text='집게 끝의 기준 위치 (TCP)',style='PanelHeading.TLabel').grid(row=0,column=0,columnspan=2,sticky='w')
        self.tool_reference=tk.StringVar(value=execution_summary(tool))
        ttk.Label(tab,textvariable=self.tool_reference,wraplength=460).grid(row=1,column=0,columnspan=2,sticky='ew',pady=6)
        edit=self.section(tab,2,'설치 변경 시에만 · TCP와 관절 영점 수정')
        for i,(key,values,axes) in enumerate((('xyz_mm',tool['xyz_mm'],('X mm','Y mm','Z mm')),('rpy_deg',tool['rpy_deg'],('좌우 기울기 °','앞뒤 기울기 °','방향 °')))):
            for j,(axis,v) in enumerate(zip(axes,values)):
                self.field(edit,2+i*3+j,'집게 끝 '+axis,'tool_'+key+str(j),v,increment=.5)
        ttk.Label(edit,text='추가 보정각 · 0°여도 저장한 모터 영점 변환은 유지됩니다',wraplength=460).grid(row=8,column=0,columnspan=2,sticky='w',pady=8)
        from teaching_ui import joint_label
        for i,(n,v) in enumerate(zip(JOINTS,tool['joint_offsets_deg'])):
            self.field(edit,9+i,joint_label(n)+' °','offset_'+str(i),v,increment=.1)
        self.button_row(edit,15,[('모델 끝점 불러오기',self.load_model_tip),('보정값 저장',self.save_tool),('선택한 자세의 집게 끝 확인',self.inspect_tcp)])
        preview=self.section(tab,3,'선택 · 가상 지그 위치로 미리보기')
        ttk.Label(preview,text='연습용 가상 지그 중심 · 실물 실행에 사용되지 않음',wraplength=460).grid(row=17,column=0,columnspan=2,sticky='ew')
        for i,(k,v) in enumerate((('ref_x',200),('ref_y',0),('ref_yaw',0))):
            self.field(preview,18+i,('지그 중심 X mm','지그 중심 Y mm','지그 +X 방향 °')[i],k,v,increment=1)
        self.button_row(preview,21,[('가상 지그 위치로 미리보기',lambda:self.preview_plan(offline=True))])
        ttk.Label(preview,text='지그 외곽을 읽지 못하면 실물 실행은 중단됩니다.\n위에서 대기·내리기·집게 열기/닫기·위로 빼기은 같은 목표를 공유합니다.\n3D는 자세·경로 확인용이며 충돌이나 실제 파지를 판정하지 않습니다.',wraplength=460,style='Muted.TLabel').grid(row=22,column=0,columnspan=2,sticky='ew',pady=12)


    def open_motor_calibration(self):
        if self.calibration_window is not None:
            self.calibration_window.window.lift();return
        from motor_calibration import MotorCalibrationWindow
        self.calibration_window=MotorCalibrationWindow(self)

    def idle(self):
        if self.app.state not in ('DISCONNECTED','READY','HOLD') or self.app.closing or self.app.preview_only:
            raise ValueError('동작을 중지한 뒤 현재 로봇의 에피소드에서 편집하세요.')
        if self.job is not None:
            raise ValueError('현재 계산이 끝난 뒤 다시 시도하세요.')
        if hasattr(self,'safe_execution') and self.safe_execution.pending is not None:
            raise ValueError('안전 자세 이동·지그 측정 중입니다. 중지 후 편집하세요.')

    def guarded(self, function):
        try:
            function()
        except Exception as exc:
            self.status.set(str(exc)); self.app.note(str(exc))

    def save_station(self):
        from teach_left_arm import atomic_json
        atomic_json(self.profile_path,self.station)
        self.reset_detection()
        self.clear_check('설정 저장됨 · 새 측정 대기')
        self.refresh_setup_status()

    def run_job(self, function, done, *, error_prefix=""):
        self.idle(); self.job=True
        self.app.vision_busy=True; self.app.refresh_buttons()
        self.status.set('계산 중입니다. 로봇에는 명령을 보내지 않습니다.')
        def work():
            try:
                self.completed.put((done,function(),None))
            except Exception as exc:
                self.completed.put((done,None,error_prefix+str(exc)))
        threading.Thread(target=work,name='vision-calculation',daemon=True).start()

    def start_camera(self):
        self.idle()
        from vision_teaching import number
        camera={'source':self.vars['source'].get().strip(),'width':int(number(self.vars['width'].get(),320,4096)),
                'height':int(number(self.vars['height'].get(),240,2160))}
        if not camera['source']:
            raise ValueError('상단 카메라 장치 번호나 경로를 입력하세요.')
        self.stop_camera()
        if camera!=self.station['camera']:
            self.station['camera']=camera
            self.station['intrinsics']=self.station['extrinsics']=None
            self.clear_samples()
        self.save_station()
        ctx=mp.get_context('spawn'); self.frames=ctx.Queue(maxsize=2); self.stopping=ctx.Event()
        self.process=ctx.Process(target=camera_worker,args=(camera['source'],camera['width'],camera['height'],self.frames,self.stopping),daemon=True)
        self.process.start(); self.frame=None; self.frame_live=False
        self.camera_status.set('상단 카메라 연결 중…')

    def stop_camera(self):
        self.cancel_execution('카메라 중지로 에피소드 준비를 취소했습니다.')
        self.clear_jig_reference('카메라를 중지하거나 다시 연결했습니다. 지그 기준을 다시 읽어 주세요.')
        if self.process:
            self.stopping.set(); self.process.join(.2)
            if self.process.is_alive():
                self.process.terminate(); self.process.join(.3)
            self.process=None
        if self.frames:
            self.frames.close(); self.frames=None
        self.frame_live=False; self.reset_detection();self.clear_check('영상 정지 · 현재 측정 없음')
        self.camera_status.set('카메라 연결 안 됨 · 이전 영상은 측정에 사용하지 않습니다.')

    def open_image(self):
        self.idle()
        from app_choices import askopenfilename
        import cv2
        p=askopenfilename(parent=self.window,title='보정 사진 열기',filetypes=[('영상','*.png *.jpg *.jpeg *.bmp')])
        if not p:return
        f=cv2.imread(p)
        if f is None:raise ValueError('사진을 읽을 수 없습니다.')
        self.stop_camera(); self.frame=f; self.frame_id+=1; self.frame_at=time.monotonic()
        self.inspect_outline(); self.camera_status.set('사진 검토 · 실시간 지그 실행에는 사용되지 않습니다.')

    def board(self):
        from vision_teaching import number
        return {'cols':int(number(self.vars['cols'].get(),3,20)),'rows':int(number(self.vars['rows'].get(),3,20)),
                'square_mm':number(self.vars['square_mm'].get(),1,100)}

    def current_frame(self):
        if self.frame is None or (self.frame_live and time.monotonic()-self.frame_at>.6):
            raise ValueError('새 카메라 영상이나 보정 사진을 먼저 여세요.')
        return self.frame.copy()

    def capture_board(self):
        from vision_teaching import chessboard
        self.idle(); f=self.current_frame(); b=self.board(); size=list(f.shape[1::-1])
        if self.samples and (self.sample_size!=size or self.sample_board!=b):
            raise ValueError('영상 크기나 체커 설정이 바뀌었습니다. 수집 사진을 초기화하세요.')
        def done(c):
            import numpy as np
            if any(np.mean(np.linalg.norm(c-other,axis=2))<8 for other in self.samples):
                raise ValueError('기존 사진과 위치가 너무 비슷합니다. 보드의 위치·기울기를 바꾸세요.')
            self.samples.append(c); self.sample_size=size; self.sample_board=b
            self.sample_list.insert('','end',values=(len(self.samples),'계산 전'))
            self.sample_status.set(f'수집 {len(self.samples)}장 · 위치와 기울기를 바꿔 촬영하세요.')
            self.status.set('체커보드 사진을 수집했습니다.')
        self.run_job(lambda:chessboard(f,b),done)

    def clear_samples(self):
        self.idle(); self.samples=[]; self.sample_size=None; self.sample_board=None
        self.sample_list.delete(*self.sample_list.get_children()); self.sample_status.set('수집 0장')

    def remove_sample(self):
        self.idle(); selection=self.sample_list.selection()
        if not selection:return
        index=self.sample_list.index(selection[0]); self.samples.pop(index)
        self.sample_list.delete(*self.sample_list.get_children())
        for i in range(len(self.samples)):self.sample_list.insert('','end',values=(i+1,'재계산 필요'))
        self.sample_status.set(f'수집 {len(self.samples)}장 · 내부 보정을 다시 계산하세요.')

    def calibrate(self):
        from vision_teaching import calibrate_intrinsics
        self.idle()
        if self.board()!=self.sample_board:raise ValueError('수집 당시의 체커 설정을 유지하세요.')
        samples=deepcopy(self.samples); size=self.sample_size; b=deepcopy(self.sample_board); cam=deepcopy(self.station['camera'])
        def done(value):
            self.station['board']=b; self.station['intrinsics']=value; self.station['extrinsics']=None
            self.axes_confirm.set(False)
            self.save_station()
            for item,error in zip(self.sample_list.get_children(),value['errors_px']):self.sample_list.set(item,'error',f'{error:.3f}')
            self.status.set(value.get('quality_note') or f"내부 보정 저장 완료 · RMS {value['rms_px']:.3f}px. 로봇 좌표 보정을 이어서 진행하세요.")
        self.run_job(lambda:calibrate_intrinsics(samples,size,b,cam),done)

    def show_board(self):
        from vision_teaching import chessboard
        import cv2
        self.idle(); f=self.current_frame(); b=self.board()
        def done(corners):
            c=corners.reshape(-1,2).astype(int)
            cv2.drawChessboardCorners(f,(b['cols'],b['rows']),corners,True)
            for index,label in ((0,'0'),(b['cols']-1,'+X'),((b['rows']-1)*b['cols'],'ROW')):
                cv2.putText(f,label,tuple(c[index]),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
            self.draw_frame(f); self.status.set('0번 코너 XYZ와 열(+X) 방향 θ를 입력하세요. ROW 화살표 쪽이 기준의 -Y인지 +Y인지 선택하세요.')
            self.board_overlay_until=time.monotonic()+8
        self.run_job(lambda:chessboard(f,b),done)

    def extrinsics(self):
        from vision_teaching import chessboard, calibrate_extrinsics, number
        self.idle()
        if not self.axes_confirm.get():raise ValueError('기준판의 0번 코너·+X·+Y를 실제 로봇 기준과 먼저 맞추세요.')
        f=self.current_frame(); s=deepcopy(self.station)
        if self.board()!=s['board']:raise ValueError('내부 보정 때의 체커 설정을 유지하세요.')
        origin=[number(self.vars['origin_'+a].get()) for a in 'XYZ']; yaw=number(self.vars['board_yaw'].get(),-360,360)
        row_sign=-1 if self.vars['board_row_sign'].get()=='-Y' else 1
        def done(value):
            self.station['extrinsics']=value; self.save_station()
            self.status.set(f"로봇 좌표 보정 저장 완료 · 재투영 오차 {value['error_px']:.3f}px. 별도 기준점의 mm 오차도 확인하세요.")
        self.run_job(lambda:calibrate_extrinsics(chessboard(f,s['board']),list(f.shape[1::-1]),s,origin,yaw,row_sign),done)

    def apply_model_camera(self):
        self.idle()
        from model_camera import model_extrinsics
        self.station['extrinsics']=model_extrinsics(self.station,self.vars['model_optical_offset'].get(),self.vars['model_image_roll'].get())
        self.clear_jig_reference('카메라 모델 보정이 바뀌었습니다. 새 측정이 필요합니다.')
        self.save_station();self.reset_detection();self.invalidate_paths();self.refresh_setup_status()
        self.status.set('모델 추정 좌표 보정을 적용했습니다. 실제 광학 중심·영상 축은 실측 미확인입니다. 목표 경로를 다시 계산하세요.')

    def read_target(self):
        from vision_teaching import validate_target
        t={k[2:]:v.get() for k,v in self.vars.items() if k.startswith('t_')}
        modes={'지그를 따라가기':'rectangle','이 자리에 고정':'fixed','직사각형 지그':'rectangle','로봇 고정':'fixed'}
        if t['mode'] not in modes:
            raise ValueError('이전 마커 목표는 읽기 전용입니다. 새 OpenCV 지그 목표를 등록하세요.')
        old=self.app.vision_plan.get('targets',{}).get(self.target_id,{})
        if old.get('mode')=='marker':
            raise ValueError('이전 마커 목표는 덮어쓰지 않습니다. 새 지그 목표를 등록하세요.')
        t['mode']=modes[t['mode']]
        t['operation']='pick' if t['operation']=='집기' else 'place'
        t['return_to_start']=t['return_to_start']=='집고 제자리에 놓기'
        t['appearance']='white_frame' if t['appearance']=='흰색 프레임' else 'contrast'
        t['shape_model']='ring' if t['shape_model'] in ('도넛형 (내부 구멍 있음)','3×2 하위 지그') else 'rectangle'
        t['detection_method']={'색상+윤곽 (기본)':'hybrid','윤곽 중심':'edge','흰색 중심':'white'}.get(t.get('detection_method'),'hybrid')
        for key in ('upper_strategy','grasp_side','jig_id','hole_layout'):
            if key in old:t[key]=deepcopy(old[key])
        t['geometry_verified']=bool(old.get('geometry_verified',False))
        return validate_target(t)

    def refresh_target_list(self):
        targets=self.app.vision_plan.get('targets',{})
        self.target_ids=list(targets)
        self.target_list['values']=[f"{t['name']} · {t['product']} · {t.get('jig_name','기존 마커' if t['mode']=='marker' else '고정')}" for t in targets.values()]
        if self.target_id in self.target_ids:self.target_list.current(self.target_ids.index(self.target_id))
        else:self.target_list.set('새 목표')
        if hasattr(self,'catalog'):self.catalog.refresh()
        self.refresh_roi_picker()
        names=[t['jig_name'] for t in self.station['jigs'].values() if t.get('mode')=='rectangle']
        self.detector_list['values']=['현재 에피소드',*sorted(set(names))]
        if self.detector_choice.get() not in self.detector_list['values']:self.detector_choice.set('현재 에피소드')

    def refresh_roi_picker(self):
        if not hasattr(self,'roi_jig_picker'):return
        from vision_teaching import reference_key
        targets=self.app.vision_plan.get('targets',{})
        unique={}
        for ident,t in targets.items():
            if t.get('mode')=='rectangle':unique.setdefault(reference_key(t),ident)
        self.roi_jig_ids=list(unique.values())
        self.roi_jig_picker['values']=[targets[i]['jig_name'] for i in self.roi_jig_ids]
        active=targets.get(self.target_id,{})
        ident=unique.get(reference_key(active)) if active.get('mode')=='rectangle' else None
        if ident in self.roi_jig_ids:self.roi_jig_picker.current(self.roi_jig_ids.index(ident))
        else:self.roi_jig_picker.set('티칭 화면에서 지그를 먼저 선택하세요')

    def select_roi_jig(self,_=None):
        try:
            self.idle();index=self.roi_jig_picker.current()
            if not 0<=index<len(self.roi_jig_ids):return
            self.image_roi.cancel();self.target_id=self.roi_jig_ids[index]
            self.refresh_target_list();self.load_target();self.image_roi.draw()
            from vision_teaching import reference_key
            self.app.vision_plan['teaching_jig_key']=reference_key(self.read_target())
            from jig_catalog import refresh_picker
            refresh_picker(self.app)
            self.roi_hint.set(self.read_target()['jig_name']+'의 감지 범위를 편집합니다.')
        except Exception as exc:self.status.set(str(exc));self.refresh_roi_picker()

    def apply_saved_detection_settings(self):
        """Camera search settings belong to this station, not an episode snapshot."""
        if self.app.preview_only:return False
        from jig_detection import validate_profile
        from vision_teaching import reference_key
        fields=('detection_method','expected_yaw_deg','yaw_tolerance_deg','roi_x_pct','roi_y_pct','roi_w_pct','roi_h_pct')
        updates=[]
        for target in self.app.vision_plan.get('targets',{}).values():
            if target.get('mode')!='rectangle':continue
            saved=self.station['jigs'].get(reference_key(target))
            if saved is None:continue
            values={k:saved[k] for k in fields if k in saved}
            if saved.get('preview_model'):
                from jig_stl import detection_height
                values['surface_z_mm']=detection_height(saved['preview_model'])
                from jig_stl import with_rim_metadata
                values['hole_layout']=with_rim_metadata(saved['preview_model']).get('hole_layout',[])
            if any(target.get(k)!=v for k,v in values.items()):
                validate_profile({**target,**values});updates.append((target,values))
        for target,values in updates:target.update(values)
        if updates:
            self.reset_detection();self.clear_jig_reference('저장된 감지 범위·도킹 방향을 적용했습니다. 실행 시 다시 측정합니다.')
        return bool(updates)

    def load_target(self):
        self.apply_saved_detection_settings()
        index=self.target_list.current()
        if index<0:return
        self.target_id=self.target_ids[index]; t=self.app.vision_plan['targets'][self.target_id]
        from jig_detection import PROFILE_DEFAULTS,validate_profile
        from vision_teaching import validate_target
        t=validate_target(t)
        if t.get('mode')=='rectangle':t={**t,**validate_profile(t)}
        for k,v in {**PROFILE_DEFAULTS,**t}.items():
            if 't_'+k in self.vars:
                value=({'rectangle':'지그를 따라가기','marker':'기존 마커 (전환 필요)','fixed':'이 자리에 고정'}[v] if k=='mode' else
                       '집기' if k=='operation' and v=='pick' else '놓기' if k=='operation' else
                       '집고 제자리에 놓기' if k=='return_to_start' and v else '잡기·놓기 4단계' if k=='return_to_start' else
                       '도넛형 (내부 구멍 있음)' if k=='shape_model' and v in ('carrier_6jigs','ring') else '직사각형' if k=='shape_model' else
                       {'hybrid':'색상+윤곽 (기본)','edge':'윤곽 중심','white':'흰색 중심'}[v] if k=='detection_method' else
                       '흰색 프레임' if k=='appearance' and v=='white_frame' else '명암 외곽' if k=='appearance' else str(v))
                self.vars['t_'+k].set(value)
        if t['mode']=='marker':self.status.set('이전 마커 파일은 읽기 전용입니다. 새 OpenCV 지그 목표를 등록하세요.')

    def import_simulation(self):
        self.idle()
        from simulation_bridge import PARTS,read_source,candidate
        from teach_left_arm import save_named_teaching
        kind=list(PARTS)[max(0,self.sim_part.current())]
        config=read_source();station=deepcopy(self.station)
        def done(docs):
            # Preserve the current draft through the existing persistence path.
            if self.app.steps and not self.app.preview_only:self.app.persist()
            paths={k:save_named_teaching(self.app.output,doc) for k,doc in docs.items()}
            self.app.refresh_library();self.app.load_path(paths[kind])
            if self.app.saved_path!=paths[kind]:
                raise ValueError('후보는 저장됐지만 현재 로봇 범위로 열지 못했습니다. 메인 화면의 오류를 확인하세요.')
            self.target_id='sim_A_'+kind;self.refresh_target_list();self.load_target()
            for key,value in zip(('ref_x','ref_y','ref_yaw'),[*config['carrier']['center_base_mm'][:2],config['carrier']['yaw_deg']]):self.vars[key].set(str(value))
            self.app.preview_selected_step()
            self.status.set('시작·종료 안전 자세를 포함한 9단계 후보 3개를 저장했습니다. 테두리 높이·집기 높이·안전 자세 경로를 실물에서 확인하세요.')
        self.run_job(lambda:{k:candidate(config,k,station) for k in PARTS},done)

    def add_safe_endpoints(self):
        self.idle()
        if not self.app.steps:raise ValueError('먼저 현재 자세를 스텝으로 저장하세요.')
        from simulation_bridge import read_source
        from safe_episode import model_safe_pose,with_safe_endpoints
        from teaching_settings import MAX_STEPS
        safe=self.app.backend.validate_target(model_safe_pose(read_source(),self.station['tool']))
        steps=with_safe_endpoints(self.app.steps,safe)
        if len(steps)>MAX_STEPS:raise ValueError('스텝 수 한도를 초과합니다.')
        self.app.steps=steps
        self.app.vision_plan['safe_pose_reference']={'source':'Test_teach safe_deg','model_candidate':True,'pose':deepcopy(safe)}
        self.app.refresh_steps(0);self.app.edited()
        self.status.set('처음과 끝에 같은 안전 자세를 넣었습니다. 안전 스텝의 관절값을 편집하면 양쪽이 함께 바뀝니다. 에피소드를 저장하세요.')

    def new_target(self):
        self.idle(); self.target_id=None; self.target_list.set('새 목표')
        self.vars['t_name'].set('새 목표');self.vars['t_mode'].set('지그를 따라가기')


    def save_target(self, measurement=None):
        self.idle()
        from vision_teaching import validate_plan, reference_key, calibration_signature
        from jig_detection import PROFILE_FIELDS
        t=self.read_target(); ident=self.target_id or uuid.uuid4().hex[:12]
        if measurement is not None:t['teaching_measurement']=deepcopy(measurement)
        old=self.app.vision_plan.get('targets',{}).get(ident,{})
        geometry_fields=[k for k in PROFILE_FIELDS if k!='geometry_verified']
        if old.get('mode')=='rectangle' and any(old.get(k)!=t.get(k) for k in geometry_fields):
            t['geometry_verified']=False
        plan=deepcopy(self.app.vision_plan)
        plan.setdefault('targets',{})[ident]=t
        # All slots share one explicitly named physical jig profile.
        if t['mode']=='rectangle':
            for other in plan['targets'].values():
                if other['mode']=='rectangle' and other['jig_name']==t['jig_name']:
                    other.update({k:deepcopy(t[k]) for k in PROFILE_FIELDS if k in t})
        plan['calibration_signature']=calibration_signature(self.station)
        plan=validate_plan(plan)
        self.app.vision_plan=plan; self.target_id=ident
        if t['mode']=='rectangle':
            key=reference_key(t)
            if not self.station['jigs'].get(key,{}).get('catalog_managed'):
                self.station['jigs'][key]=deepcopy(t)
        self.save_station()
        self.invalidate_paths(); self.app.edited(); self.refresh_target_list()
        self.status.set('공통 지그와 목표를 저장했습니다. 연결된 스텝의 경로를 다시 계산하세요.')


    def reset_detection_roi(self):
        self.idle()
        if hasattr(self,'image_roi'):self.image_roi.cancel()
        for k,value in [('roi_x_pct',0),('roi_y_pct',0),('roi_w_pct',100),('roi_h_pct',100)]:self.vars['t_'+k].set(str(value))
        self.save_detection_method()
        self.clear_jig_reference('감지 범위를 바꿨습니다. 지그 기준을 다시 읽으세요.')
        self.image_roi.draw();self.roi_hint.set('전체 영상에서 지그를 찾습니다.')

    def save_detection_method(self):
        self.idle()
        from vision_teaching import reference_key
        from jig_detection import PROFILE_FIELDS
        profile=self.read_target()
        if profile['mode']!='rectangle':raise ValueError('장비 설정에서 지그를 따라가는 목표를 먼저 등록하세요.')
        key=reference_key(profile)
        for target in self.app.vision_plan['targets'].values():
            if target.get('mode')=='rectangle' and reference_key(target)==key:
                for name in ('detection_method','roi_x_pct','roi_y_pct','roi_w_pct','roi_h_pct'):target[name]=profile[name]
        # ROI/method edits must not overwrite a newer catalog geometry with
        # the old geometry snapshot retained by a saved episode.
        from jig_detection import validate_profile
        stored=deepcopy(self.station['jigs'].get(key,{'mode':'rectangle',**validate_profile(profile)}))
        for name in ('detection_method','roi_x_pct','roi_y_pct','roi_w_pct','roi_h_pct'):stored[name]=profile[name]
        self.station['jigs'][key]=stored
        self.save_station()
        self.clear_jig_reference('감지 설정을 바꿨습니다. 실행 시 다시 측정합니다.')
        # ROI/method changes neither joint poses nor nominal preview trajectories.
        # Persist the working snapshot but keep the selected/saved episode usable.
        self.app.persist();self.app.refresh_buttons()
        self.status.set('감지 설정을 저장했습니다. 다른 에피소드에서도 이 지그의 마지막 범위를 사용합니다.')

    def reset_detection(self):
        self.tracker.clear();self.observations={}
        self.detected.delete(*self.detected.get_children())
        self._candidate_token=None;self._metric_token=None
        self.detection_generation+=1
        self.detection_displayed_at=0.

    def invalidate_paths(self):
        for step in self.app.steps:
            if step.get('motion'):step.pop('path',None)
        if self.app.preview:self.app.preview.invalidate_episode()

    def seed(self):
        if self.app.latest and time.monotonic()-self.app.latest['sample_time']<.6:
            return deepcopy(self.app.latest['follower'])
        index=self.app.selected()
        if index is not None:return deepcopy(self.app.steps[index]['pose'])
        if self.app.steps:return deepcopy(self.app.steps[-1]['pose'])
        from teach_left_arm import KEYS
        return dict.fromkeys(KEYS,0.)

    def add_steps(self):
        self.save_target()
        from vision_teaching import make_steps
        from teaching_settings import MAX_STEPS
        steps=make_steps(self.target_id,self.app.vision_plan['targets'][self.target_id],self.seed())
        if len(self.app.steps)+len(steps)>MAX_STEPS:raise ValueError('스텝 수 한도를 초과합니다.')
        self.app.steps.extend(steps)
        self.app.refresh_steps(len(self.app.steps)-len(steps)); self.app.edited()
        self.status.set(f'{len(steps)}단계를 추가했습니다. 저장한 지그 기준으로 경로를 계산하세요.')

    def observations_for(self, plan, offline=False, *, final=False):
        from vision_teaching import reference_key, number
        keys={reference_key(t) for t in plan['targets'].values() if t['mode'] in ('rectangle','marker')}
        if offline:
            return {k:{'x_mm':number(self.vars['ref_x'].get()),'y_mm':number(self.vars['ref_y'].get()),
                       'yaw_deg':number(self.vars['ref_yaw'].get(),-360,360)} for k in keys}
        if keys and (not self.frame_live or time.monotonic()-self.frame_at>.6):
            raise ValueError('현재 상단 카메라 영상이 필요합니다. 카메라 없이 확인하려면 집게 끝 위치 탭의 가상 지그 위치로 미리보기를 사용하세요.')
        try:snapshot=self.tracker.snapshot(keys,final=final)
        except ValueError:
            errors=getattr(self,'metric_errors',{})
            reason=next((errors[key] for key in keys if key in errors),None)
            if reason:raise ValueError(reason)
            raise
        from jig_detection import profile_signature
        for t in plan['targets'].values():
            if t['mode']=='rectangle' and snapshot[reference_key(t)].get('profile_signature')!=profile_signature(t):
                raise ValueError('검출 프로필과 목표의 지그 설정이 다릅니다. 해당 에피소드의 지그를 선택하세요.')
        return snapshot

    def capture_jig_reference(self):
        self.teaching_auto_pending=False
        self.idle()
        if self.manual.pending:raise ValueError('현재 팔 조정이 끝난 뒤 기준을 읽어 주세요.')
        target=self.read_target()
        if target['mode']!='rectangle':raise ValueError('잡기·놓기에서 위치 기준을 지그를 따라가기로 선택하세요.')
        self.reference_acquisition.begin({'targets':{'capture':target}},teaching=True)

    def schedule_teaching_reference(self, preserve=False):
        self.reference_acquisition.cancel('에피소드를 바꿨습니다.')
        if not preserve:self.teaching_reference.clear('새 에피소드의 지그를 읽습니다.')
        self.clear_jig_reference()
        self.teaching_auto_pending=any(t.get('mode')=='rectangle' for t in self.app.vision_plan['targets'].values()) and not self.app.preview_only
        if not self.teaching_auto_pending:return
        if not self.frame_live and not self.process and not self.app.backend.identity['demo']:
            try:self.start_camera()
            except Exception as exc:self.status.set('티칭 기준 대기 · '+str(exc))
        self.teaching_auto_pending=True
        self.status.set('에피소드 선택 · 카메라 연결 후 티칭 기준을 자동으로 읽습니다.')

    def poll_teaching_reference(self):
        if not self.teaching_auto_pending or not self.frame_live or self.app.state not in ('DISCONNECTED','READY','HOLD'):return
        if self.reference_acquisition.pending or self.safe_execution.pending or self.job:return
        self.teaching_auto_pending=False
        plan={'targets':{self.target_id:self.read_target()}}
        self.guarded(lambda:self.reference_acquisition.begin(plan,teaching=True,show=False))

    def end_teaching_reference(self):
        self.cancel_execution('작업을 마쳤습니다.')
        self.teaching_reference.clear('작업을 마쳤습니다.')
        self.clear_jig_reference('작업을 마쳤습니다.')
        self.update_jig_reference()

    def teaching_observations(self,plan):
        return self.teaching_reference.observations(plan,self.station)

    def clear_jig_reference(self, reason='작업 기준을 해제했습니다. 다음 도킹에서 다시 읽어 주세요.'):
        self.measurement_failure=''
        had_reference=self.jig_reference.value is not None
        self.jig_reference.clear(reason)
        if had_reference and hasattr(self,'manual'):self.manual.revision+=1
        if had_reference and self.app.state in ('PLAY','MOVE'):
            self.app.send('pause')
        if hasattr(self,'reference_status'):
            self.reference_status.set('기준 없음 · '+reason)
            self.reference_capture.configure(text='티칭 기준 다시 읽기')

    def update_jig_reference(self):
        for key,value in list(self.jig_reference.values.items()):
            try:self.jig_reference.check({'mode':'rectangle',**value['profile']},self.station)
            except ValueError:
                self.clear_jig_reference(self.jig_reference.reason)
                if self.app.state in ('PLAY','MOVE'):self.app.send('pause')
                break
            if self.frame_live and time.monotonic()-self.frame_at>.6:
                self.clear_jig_reference('카메라 영상이 끊겼습니다. 영상 복구 후 기준을 다시 읽어 주세요.');break
            if self.job or self.app.state in ('PLAY','MOVE'):
                try:current=self.tracker.snapshot([key])[key]
                except ValueError:current=None
                if current is not None and self.jig_reference.observe_key(key,current,position_tolerance=3.,angle_tolerance=5.):
                    self.clear_jig_reference(self.jig_reference.reason)
                    self.manual.revision+=1
                    if self.app.state in ('PLAY','MOVE'):self.app.send('pause')
                    break
        value=self.jig_reference.value
        if value is not None:
            p=value['observation']
            self.reference_status.set('저장된 기준 · 자동 확정 완료 %s\nX %.1f / Y %.1f mm / 방향 %.1f° · 가려져도 유지'%(value['captured_at'][11:19],p['x_mm'],p['y_mm'],p['yaw_deg']))
            self.reference_capture.configure(text='티칭 기준 다시 읽기')
        else:
            waiting=bool(self.safe_execution.pending or self.reference_acquisition.pending)
            failure=getattr(self,'measurement_failure','')
            message=('자동 측정 종료 · 다시 실행하세요.\n'+failure if failure else
                     '지그 위치 자동 읽기 중' if waiting else
                     '실행 대기 · 스텝/에피소드 실행 시 새 지그 위치를 자동으로 읽습니다.')
            if waiting and self.frame_live:
                try:
                    from vision_teaching import reference_key
                    draft=self.read_target()
                    if draft['mode']=='rectangle':message+='\n'+self.tracker.status_text(reference_key(draft))
                except ValueError:pass
            self.reference_status.set(message)
        try:
            target=self.read_target()
            anchor=self.teaching_reference.check(target,self.station)['observation']
            self.app.teaching_hint.set('티칭 기준 저장됨 · X %.1f / Y %.1f mm / %.1f°'%(anchor['x_mm'],anchor['y_mm'],anchor['yaw_deg']))
            self.reference_status.set('티칭 기준 유지 · X %.1f / Y %.1f mm / %.1f°\n'%(anchor['x_mm'],anchor['y_mm'],anchor['yaw_deg'])+self.reference_status.get())
        except ValueError:
            self.app.teaching_hint.set('티칭 기준 자동 측정 대기' if self.teaching_auto_pending or (self.reference_acquisition.pending and self.reference_acquisition.pending.get('teaching')) else '티칭 기준 없음 · 카메라에서 티칭 기준 다시 읽기')
        if self.execution_error:self.reference_status.set(self.execution_error+'\n\n'+self.reference_status.get())
        self.reference_capture.configure(text='티칭 기준 다시 읽기' if self.teaching_reference.value is not None else '티칭 기준 읽기')
        self.reference_capture.configure(state='normal' if self.app.state in ('DISCONNECTED','READY','HOLD') and not self.app.closing and not self.app.preview_only and not self.job and not self.manual.pending and not self.reference_acquisition.pending else 'disabled')

    def reference_observations(self, plan):
        """Camera-relative execution always uses an explicit docking reference."""
        self.update_jig_reference()
        return self.jig_reference.observations(plan,self.station)

    def preview_plan(self, offline=False):
        self.idle()
        from vision_teaching import calibration_signature, digest
        from planning_worker import compile_steps_isolated as compile_steps
        plan=deepcopy(self.app.vision_plan); plan['calibration_signature']=calibration_signature(self.station)
        plan['preview_tool']=deepcopy(self.station['tool'])
        from step_reference import used_plan
        active=used_plan(self.app.steps,plan)
        obs=self.observations_for(active,True) if offline else self.reference_observations(active)
        steps=deepcopy(self.app.steps); station=deepcopy(self.station); seed=self.seed()
        original=digest(self.app.steps)
        def done(value):
            if self.app.closing or digest(self.app.steps)!=original:raise ValueError('계산 중 스텝이 변경됐습니다. 다시 계산하세요.')
            self.app.steps=value; self.app.vision_plan=plan
            self.app.vision_plan['preview_kind']='offline' if offline else 'camera_snapshot'
            self.app.refresh_steps(0); self.app.edited(); self.app.preview_selected_step()
            self.status.set('경로를 3D에 반영했습니다. 에피소드를 저장하세요. 전체 실행은 새 측정으로 다시 계산하며, 안전 스텝이 있으면 도착 후 측정합니다.')
        self.run_job(lambda:compile_steps(steps,plan,station,obs,seed),done)

    def cancel_execution(self, reason):
        self.teaching_auto_pending=False
        self.execution_revision+=1
        self.safe_execution.cancel(reason)
        self.reference_acquisition.cancel(reason)

    def execute(self, index=None):
        self.execution_error=''
        self.teaching_auto_pending=False
        self.apply_saved_detection_settings()
        from safe_episode import validate_boundaries
        if index is None and validate_boundaries(self.app.steps):
            self.safe_execution.begin()
        else:
            from step_reference import used_plan
            selected=self.app.steps if index is None else [self.app.steps[index]]
            plan=used_plan(selected,self.app.vision_plan)
            rectangles=[t for t in plan['targets'].values() if t['mode']=='rectangle']
            if rectangles:
                self.reference_acquisition.begin(plan,lambda:self.execute_motion(index,prepared=True))
            else:self.execute_motion(index)

    def execute_motion(self, index=None, *, prepared=False):
        self.idle()
        from vision_teaching import calibration_signature, angle_delta, digest
        from planning_worker import compile_steps_isolated as compile_steps
        from teach_left_arm import validate_sequence
        from tkinter import messagebox
        import math
        if self.app.state != 'HOLD' or not self.app.latest:
            raise ValueError('로봇 연결·동기화 후 자세 유지 상태에서 실행하세요.')
        if time.monotonic()-self.app.latest['sample_time']>.6:
            raise ValueError('로봇 관절값 갱신을 기다리세요.')
        plan=deepcopy(self.app.vision_plan)
        steps=deepcopy((self.app.steps[1:] if prepared and self.app.steps and self.app.steps[0].get('safe_boundary') else self.app.steps) if index is None else [self.app.steps[index]])
        used={s['motion']['target'] for s in steps if s.get('motion')}
        plan['targets']={k:v for k,v in plan['targets'].items() if k in used}
        if index is None:
            rectangles=[t for t in plan['targets'].values() if t['mode']=='rectangle']
            if rectangles:
                # A whole episode always obtains a fresh temporal measurement.
                # The manual-teaching latch is never a fallback for this read.
                if not prepared:
                    measured=self.observations_for(plan)
                    from vision_teaching import reference_key
                    self.jig_reference.capture_many(rectangles,measured,self.station)
            obs=self.reference_observations(plan)
        else:
            # Individual steps may resume while the arm occludes the carrier.
            obs=self.reference_observations(plan)
        seed=deepcopy(self.app.latest['follower']); station=deepcopy(self.station)
        signature=calibration_signature(station); revision=digest([self.app.steps,self.app.vision_plan])
        execution_revision=self.execution_revision
        def done(compiled):
            if execution_revision!=self.execution_revision:raise ValueError('사용자 중지 또는 카메라 변경으로 실행을 취소했습니다.')
            if self.app.state!='HOLD' or self.app.session.stop.is_set():
                raise ValueError('계산 중 로봇 상태가 바뀌었습니다. 실행을 취소했습니다.')
            validate_sequence(self.app.backend,compiled,seed)
            if index is None and not (prepared and self.app.steps and self.app.steps[0].get('safe_boundary')) and not messagebox.askokcancel('에피소드 실행',
                    f'{self.app.teaching_title}\n지그 기준으로 {len(compiled)}개 스텝을 한 번 실행할까요?',parent=self.window):return
            # Revalidate the explicit reference ID, or obtain a new live image.
            # A teaching record on disk never counts as an active docking latch.
            fresh=self.reference_observations(plan)
            # The execution latch was just captured after the safe arrival.
            # Occlusion or a single missed contour must not require a second capture.
            if obs and (not self.frame_live or time.monotonic()-self.frame_at>.6):
                raise ValueError('카메라 영상 갱신이 중단됐습니다. 실행을 취소했습니다.')
            live_measurement={}
            for key in obs:
                assessment=self.tracker.assessment(key,final=True)
                if assessment['ready']:live_measurement[key]=assessment['value']
            for key,p in obs.items():
                q=fresh[key]
                if p.get('reference_id')!=q.get('reference_id'):
                    raise ValueError('실행 확인 중 지그 기준이 바뀌었습니다. 다시 실행하세요.')
                if math.hypot(p['x_mm']-q['x_mm'],p['y_mm']-q['y_mm'])>3 or abs(angle_delta(p['yaw_deg'],q['yaw_deg']))>5:
                    raise ValueError('확인 중 지그가 이동했습니다. 목표를 다시 계산해 실행하세요.')
                q=live_measurement.get(key)
                if q is None:continue
                if math.hypot(p['x_mm']-q['x_mm'],p['y_mm']-q['y_mm'])>3 or abs(angle_delta(p['yaw_deg'],q['yaw_deg']))>5:
                    raise ValueError('실행 확인 중 측정 위치·각도가 바뀌었습니다. 3초간 안정된 뒤 다시 실행하세요.')
            if calibration_signature(self.station)!=signature or digest([self.app.steps,self.app.vision_plan])!=revision:
                raise ValueError('계산 후 보정값 또는 스텝이 변경됐습니다. 다시 실행하세요.')
            if execution_revision!=self.execution_revision:raise ValueError('실행이 취소됐습니다.')
            if self.app.state!='HOLD' or self.app.session.stop.is_set():return
            self.app.send('play',{'steps':compiled,'title':self.app.teaching_title,'follower':seed,
                                  'vision_checked_at':time.monotonic(),
                                  'vision_evidence':{'calibration_signature':signature,'jigs':fresh}})
            self.status.set(('시험 모드 · 계산 오차 4mm/6° 허용 · 일반 티칭 도착 기준\n' if plan.get('execution_trial') else '')+('새 지그 측정으로 보정해 실행합니다. 동작 중 기준은 고정됩니다.' if index is None else '새 지그 측정으로 선택 스텝을 실행합니다.'))
        self.run_job(lambda:compile_steps(steps,plan,station,obs,seed,
                                          live=not self.app.backend.identity['demo'],single=index is not None),done,
                     error_prefix='실행 중단 · 지그 측정은 완료됨\n' if obs else '')

    def save_tool(self):
        self.idle()
        from vision_teaching import number
        from model_tcp import reference_matches,execution_summary
        old=self.station['tool']
        t={k:[number(self.vars['tool_'+k+str(i)].get(),-limit,limit) for i in range(3)]
           for k,limit in (('xyz_mm',200),('rpy_deg',180))}
        t['joint_offsets_deg']=[number(self.vars['offset_'+str(i)].get(),-180,180) for i in range(5)]
        changed=any(t[k]!=old[k] for k in t)
        reference=getattr(self,'pending_tool_reference',None) or old.get('reference')
        if reference_matches(t,reference):t['reference']=deepcopy(reference)
        for key in ('motor_limits_deg','motor_zero_reference'):
            if key in old:t[key]=deepcopy(old[key])
        t['verified']=bool(old.get('verified',False)) and not changed and getattr(self,'pending_tool_reference',None) is None
        self.station['tool']=t
        self.pending_tool_reference=None
        self.tool_reference.set(execution_summary(t))
        self.app.vision_plan['preview_tool']=deepcopy(t)
        self.save_station(); self.invalidate_paths(); self.app.edited()
        self.status.set('TCP·영점 값을 저장했습니다. 변경한 값으로 경로를 다시 계산합니다.')

    def load_model_tip(self):
        self.idle()
        from model_tcp import fixed_finger_tip
        reference=fixed_finger_tip()
        for i,value in enumerate(reference['xyz_mm']):self.vars['tool_xyz_mm'+str(i)].set(str(value))
        self.pending_tool_reference=reference
        self.status.set('왼쪽 고정 집게 끝단 중심을 불러왔습니다. 보정값 저장을 누르면 적용됩니다. 회전·관절 보정은 유지합니다.')

    def inspect_tcp(self):
        from vision_teaching import Kinematics
        tf=Kinematics(self.station['tool']).fk(self.seed())
        self.status.set('현재 자세로 계산한 집게 끝: X %.2f · Y %.2f · Z %.2f mm. 실제 측정값과 비교하세요.'%tuple(tf[:3,3]))

    def inspect_outline(self):
        self.idle();self.current_frame();self.select_tab(3)
        self._candidate_token=None
        self.process_frame()
        self.status.set('사진의 지그 구조와 외곽을 분석했습니다. 기준 저장·에피소드 실행에는 실시간 지그 위치 읽기가 필요합니다.')


    def draw_frame(self, frame=None):
        if self.app.navigation.current is not self.window:return
        from PIL import Image, ImageTk
        import cv2
        f=self.frame if frame is None else frame
        if f is None:return
        image=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
        w=max(200,self.canvas.winfo_width());h=max(160,self.canvas.winfo_height())
        image.thumbnail((w,h));self.photo=ImageTk.PhotoImage(image,master=self.canvas)
        self.canvas.delete('all');self.canvas.create_image(w//2,h//2,image=self.photo)
        if hasattr(self,'image_roi'):
            iw,ih=image.size
            self.image_roi.displayed((w//2-iw/2,h//2-ih/2,iw,ih))

    def detection_request(self):
        from jig_detection import profile_signature
        from vision_teaching import calibration_signature, digest
        if self.detector_choice.get()=='현재 에피소드':
            profiles=[t for t in self.app.vision_plan.get('targets',{}).values() if t['mode']=='rectangle']
        else:
            profiles=[t for t in self.station['jigs'].values() if t.get('mode')=='rectangle' and t['jig_name']==self.detector_choice.get()]
        pending=self.reference_acquisition.pending
        if pending:
            profiles=list(pending['plan']['targets'].values())
        elif self.checking():
            draft=self.read_target()
            if draft['mode']=='rectangle':profiles.append(draft)
        profiles={profile_signature(t):deepcopy(t) for t in profiles}
        signature=digest([calibration_signature(self.station),profiles,self.detector_choice.get(),self.checking(),bool(pending and pending.get('heading_only'))])
        return signature,profiles

    def process_frame(self, background=False):
        if background and self.frame_live:
            signature,profiles=self.detection_request()
            previous=getattr(self,'detection_request_signature',None)
            if previous is not None and previous!=signature:
                self.reset_detection();self.clear_check('설정 변경 · 새 측정 대기');self.draw_frame()
            self.detection_request_signature=signature
            if any(t.get('shape_model','ring') in ('carrier_6jigs','ring') for t in profiles.values()):
                token=(self.detection_generation,self.frame_id,self.frame_at,signature)
                if self.detection_busy or token==self.detection_submitted:return
                self.detection_busy=True;self.detection_submitted=token
                frame=self.frame.copy();station=deepcopy(self.station)
                completed=self.detection_completed
                def work():
                    from jig_detection import inspection_candidates
                    started=time.perf_counter()
                    try:
                        cache={key:inspection_candidates(frame,t,station) for key,t in profiles.items()}
                        completed.put((token,frame,cache,None,(time.perf_counter()-started)*1000))
                    except Exception as exc:completed.put((token,frame,{},str(exc)))
                # Finish this bounded image calculation before Python unloads OpenCV.
                # The worker retains only data and a queue, never Tk widgets.
                threading.Thread(target=work,name='jig-image-analysis',daemon=False).start()
                return
        self.process_frame_results()

    def finish_detection(self):
        try:token,frame,cache,error,*timing=self.detection_completed.get_nowait()
        except queue.Empty:return
        self.detection_busy=False
        generation,frame_id,frame_at,signature=token
        if not self.frame_live or generation!=self.detection_generation:return
        elapsed=timing[0] if timing else (time.monotonic()-frame_at)*1000
        self.detection_timing.set(f'검출 처리 {elapsed:.0f}ms · '+('지연됨: 검출 영역을 줄이거나 다른 방법을 선택하세요.' if time.monotonic()-frame_at>MAX_ANALYSIS_AGE else '최신 분석 사용'))
        if time.monotonic()-frame_at>MAX_ANALYSIS_AGE:return
        current,_=self.detection_request()
        if current!=signature:return
        if error:
            self.reset_detection();self.clear_check(error);return
        latest=self.frame,self.frame_id,self.frame_at
        try:
            self.frame,self.frame_id,self.frame_at=frame,frame_id,frame_at
            self._candidate_token=self.candidate_frame_token();self._candidate_cache=cache
            self.process_frame_results();self.detection_displayed_at=frame_at
        finally:self.frame,self.frame_id,self.frame_at=latest

    def process_frame_results(self):
        import cv2
        import numpy as np
        from vision_teaching import reference_key, target_pose
        from jig_detection import roi_pixels, DetectionAmbiguity
        frame=self.frame;overlay=frame.copy();targets=list(self.app.vision_plan.get('targets',{}).values())
        if self.detector_choice.get()=='현재 에피소드':
            known={reference_key(t):t for t in targets if t['mode']=='rectangle'}
        else:
            known={key:t for key,t in self.station['jigs'].items() if t.get('mode')=='rectangle' and t['jig_name']==self.detector_choice.get()}
        pending=self.reference_acquisition.pending
        if pending:
            targets=list(pending['plan']['targets'].values())
            known={reference_key(t):t for t in targets if t['mode']=='rectangle'}
        rows=[];self.observations={};self.metric_errors={};measured=set();found={};uncertain=set()
        for key,t in known.items():
            roi=roi_pixels(t,frame.shape[1::-1]);cv2.rectangle(overlay,roi[:2],roi[2:],(140,150,160),1)
            try:
                corners,value=self.metric_pose_for(t,candidates=self.candidates_for(t))
                found[key]=(corners,value,t)
            except DetectionAmbiguity as exc:
                uncertain.add(key);rows.append((t['jig_name'],t['product'],str(exc)))
            except ValueError as exc:
                self.metric_errors[key]=str(exc)
                rows.append((t['jig_name'],t['product'],str(exc)))
        # Do not let one physical rectangle masquerade as several selected products/jigs.
        ambiguous=set()
        for key,(c,_,_) in found.items():
            for other,(d,_,_) in found.items():
                if key>=other:continue
                intersection,_=cv2.intersectConvexConvex(c.astype(np.float32),d.astype(np.float32))
                if intersection>min(cv2.contourArea(c.astype(np.float32)),cv2.contourArea(d.astype(np.float32)))*.5:
                    ambiguous.update((key,other))
        for key,(corners,value,t) in found.items():
            if key in ambiguous:
                rows.append((t['jig_name'],t['product'],'여러 프로필이 같은 외곽과 겹칩니다. 제품·검출 영역을 구분하세요.'));continue
            self.observations[key]={**value,'product':t['product'],'at':self.frame_at}
            if self.frame_live and self.frame_at>=self.measurement_not_before:
                self.tracker.observe(key,value,self.frame_at,self.frame_id);measured.add(key)
            color=(0,220,80) if value.get('quality')=='precise' else (0,165,255)
            cv2.polylines(overlay,[corners.astype(int)],True,color,3)
            inverse=np.linalg.inv(self.station['extrinsics']['base_from_camera'])
            yaw=np.radians(value['yaw_deg'])
            center=np.array([value['x_mm'],value['y_mm'],t['surface_z_mm']])
            points=[center,center+np.array([np.cos(yaw),np.sin(yaw),0])*40]
            for slot in targets:
                if slot['mode']=='rectangle' and reference_key(slot)==key:
                    xyz,rotation=target_pose(slot,{key:value});points.extend((xyz,xyz+rotation[:,0]*20))
            uv,_=cv2.projectPoints(np.asarray(points),cv2.Rodrigues(inverse[:3,:3])[0],inverse[:3,3],np.array(self.station['intrinsics']['K']),np.array(self.station['intrinsics']['D']))
            for a,b in uv.reshape(-1,2,2).astype(int):
                cv2.drawMarker(overlay,tuple(a),(0,180,255),cv2.MARKER_CROSS,18,2)
                cv2.arrowedLine(overlay,tuple(a),tuple(b),(0,180,255),2)
            prefix=('정밀' if value.get('quality')=='precise' else '중간')+' '
            if (self.station.get('extrinsics') or {}).get('source')=='cad_estimate':prefix='모델·'+prefix
            rows.append((t['jig_name'],t['product'],prefix+f"{value['x_mm']:.1f} · {value['y_mm']:.1f} mm · {value['yaw_deg']:.1f}°"))
        self.detected.delete(*self.detected.get_children())
        for key in set(self.tracker.history)-measured:
            self.tracker.missing(key,self.frame_at,self.frame_id,hard=key in ambiguous or key in uncertain)
        for row in rows:self.detected.insert('','end',values=row)
        if not known:
            self.camera_status.set('직사각형 지그 목표를 저장하거나 검출할 프로필을 선택하세요.')
        if self.checking() and self.app.navigation.current is self.window:overlay=self.detection_overlay(overlay)
        if time.monotonic()>getattr(self,'board_overlay_until',0) or self.checking():self.draw_frame(overlay)

    def reset_detection_if_error(self, background=False):
        try:self.process_frame(background=background)
        except Exception as exc:
            self.reset_detection();self.clear_check(str(exc));self.status.set(str(exc))

    def poll(self):
        if self.closed:return
        self.manual.refresh()
        try:
            done,value,error=self.completed.get_nowait()
        except queue.Empty:pass
        else:
            self.job=None; self.app.vision_busy=False; self.app.refresh_buttons()
            if error:
                if error.startswith('실행 중단 · 지그 측정은 완료됨'):self.execution_error=error
                self.status.set(error);self.app.note(error)
            else:self.guarded(lambda:done(value))
        received=False
        if self.frames:
            while True:
                try:item=self.frames.get_nowait()
                except queue.Empty:break
                if item[0]=='error':
                    self.stop_camera()
                    message=item[1]+' · 카메라 연결 버튼으로 다시 시도하세요.'
                    self.status.set(message);self.camera_status.set(message);self.app.note(message)
                    break
                _,self.frame_at,self.frame_id,self.frame=item;self.frame_live=True;received=True
        if received:
            self.reset_detection_if_error(background=True)
            self.camera_status.set(f'상단 카메라 · {self.frame.shape[1]} × {self.frame.shape[0]} · 실시간')
        if self.frame_live and time.monotonic()-self.frame_at>.6:
            self.tracker.clear();self.observations={};self.detected.delete(*self.detected.get_children())
            self.camera_status.set('영상 갱신 중단 · 지그 실행 불가');self.clear_check('영상 갱신 중단')
        try:self.finish_detection()
        except Exception as exc:self.reset_detection();self.clear_check(str(exc))
        if self.frame_live and not self.detection_busy:
            self.reset_detection_if_error(background=True)
        if self.frame_live and self.detection_displayed_at and time.monotonic()-self.detection_displayed_at>MAX_ANALYSIS_AGE and self.detection_busy:
            self.clear_check('분석 중 · 새 측정 대기');self.draw_frame()
        if self.process and not self.process.is_alive():self.stop_camera()
        self.update_jig_reference()
        self.poll_teaching_reference()
        self.safe_execution.poll()
        self.reference_acquisition.poll()
        self.poll_job=self.window.after(80,self.poll)

    def close(self):
        if self.closed:return
        self.closed=True
        if self.calibration_window is not None:self.calibration_window.close()
        self.stop_camera()
        if getattr(self,'poll_job',None):self.window.after_cancel(self.poll_job)
        self.settings_window.destroy()
        self.window.destroy()
