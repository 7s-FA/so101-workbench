"""Choose a robot control mode before constructing hardware; no ports are opened."""
import json
from pathlib import Path


def read_selection(directory):
    try:
        data=json.loads((Path(directory)/'launch_mode.json').read_text())
        if data.get('mode') in ('follower_only','pair'):return data
    except (OSError,ValueError,TypeError):pass
    return {'mode':'follower_only','profile':str(Path(directory)/'follower.json')}


def validate_selection(mode,path):
    from teaching_profile import load_profile
    args,identity=load_profile(path,follower_only=mode=='follower_only')
    if mode=='pair' and identity.get('control_mode')=='follower_only':
        raise ValueError('리더·팔로워 함께 모드에는 두 팔을 등록한 프로필이 필요합니다.')
    from teach_left_arm import Hardware
    Hardware(Path(path),follower_only=mode=='follower_only')
    return mode=='follower_only',Path(path)


def register_pair(directory,leader_port,leader_json,follower_port,follower_json):
    """Import each real calibration unchanged into a new pair profile."""
    import tempfile,uuid
    from follower_setup import register_follower
    from teach_left_arm import atomic_json
    from teaching_profile import load_profile
    directory=Path(directory);name='pair_'+uuid.uuid4().hex[:10]
    data={'schema':1,'robot_type':'so101','pair_name':name,'side':'left',
          'leader_port':leader_port,'follower_port':follower_port}
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);test=tmp/'pair.json';atomic_json(test,data);load_profile(test)
        # Reuse the six-motor calibration validation for both arms.
        for role,source,port in [('leader',leader_json,leader_port),('follower',follower_json,follower_port)]:
            source=Path(source).expanduser().resolve(strict=True)
            normalized=tmp/(role+'_left.json');normalized.write_bytes(source.read_bytes())
            register_follower(tmp/(role+'.json'),port,normalized,name)
        # The destination is new, so existing arm profiles/calibrations are untouched.
        destination=directory/'calibration'/name;destination.mkdir(parents=True,exist_ok=False)
        for role in ('leader','follower'):
            (destination/(role+'_left.json')).write_bytes((tmp/(role+'_left.json')).read_bytes())
        profile=directory/(name+'.json');atomic_json(profile,data,overwrite=False)
    return profile


class ModePanel:
    def __init__(self,parent,directory,done,*,next_launch=False):
        import tkinter as tk
        from tkinter import ttk
        from app_choices import AppChoice,askopenfilename
        self.directory=Path(directory);self.done=done
        saved=read_selection(directory)
        box=ttk.Frame(parent,padding=20);box.pack(fill='both',expand=True);box.columnconfigure(1,weight=1)
        ttk.Label(box,text='로봇 실행 모드',style='PanelHeading.TLabel').grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,15))
        self.mode=tk.StringVar(master=parent,value='팔로워 단독' if saved['mode']=='follower_only' else '리더·팔로워 함께')
        choice=AppChoice(box,textvariable=self.mode,values=('팔로워 단독','리더·팔로워 함께'),state='readonly')
        choice.grid(row=1,column=0,columnspan=3,sticky='ew')
        self.profile=tk.StringVar(master=parent,value=saved['profile']);self.status=tk.StringVar(master=parent)
        ttk.Label(box,text='로봇 프로필').grid(row=2,column=0,sticky='w',pady=12)
        ttk.Entry(box,textvariable=self.profile).grid(row=2,column=1,sticky='ew',padx=8)
        def select():
            path=askopenfilename(parent=parent,initialdir=directory,filetypes=[('로봇 프로필','*.json')])
            if path:self.profile.set(path)
        ttk.Button(box,text='파일 선택',command=select).grid(row=2,column=2)
        def mode_changed(_=None):
            self.profile.set(str(self.directory/('follower.json' if self.mode.get()=='팔로워 단독' else 'robot.json')))
            self.status.set('함께 모드는 두 팔의 포트와 각각의 보정 JSON을 등록한 프로필을 선택하세요.' if self.mode.get()!='팔로워 단독' else '')
        choice.bind('<<ComboboxSelected>>',mode_changed)
        ttk.Label(box,text='팔로워 단독: 연결 시 토크 꺼짐 → 손으로 자세 맞추기·스텝 저장\n실물 실행 전 「현재 자세 유지 · 토크 켜기」를 누르세요.\n리더·팔로워 함께: 연결 확인 후 동기화 → 리더 따라가기',wraplength=690).grid(row=3,column=0,columnspan=3,sticky='w',pady=10)
        self.pair_frame=ttk.LabelFrame(box,text='새 리더·팔로워 프로필 등록',padding=10)
        self.pair_frame.grid(row=5,column=0,columnspan=3,sticky='ew',pady=8);self.pair_frame.columnconfigure(1,weight=1)
        self.inputs={};ports=sorted(str(p) for p in Path('/dev/serial/by-id').glob('*'))
        for row,(key,label) in enumerate([('leader_port','리더 USB'),('leader_json','리더 보정 JSON'),('follower_port','팔로워 USB'),('follower_json','팔로워 보정 JSON')]):
            var=tk.StringVar(master=parent);self.inputs[key]=var
            ttk.Label(self.pair_frame,text=label).grid(row=row,column=0,sticky='w',pady=4)
            field=AppChoice(self.pair_frame,textvariable=var,values=ports) if key.endswith('port') else ttk.Entry(self.pair_frame,textvariable=var)
            field.grid(row=row,column=1,sticky='ew',padx=8)
            if key.endswith('json'):
                def pick(v=var):
                    path=askopenfilename(parent=parent,initialdir=directory,filetypes=[('모터 보정 JSON','*.json')])
                    if path:v.set(path)
                ttk.Button(self.pair_frame,text='선택',command=pick).grid(row=row,column=2)
        ttk.Button(self.pair_frame,text='두 팔 등록',command=self.register).grid(row=4,column=1,sticky='e',pady=6)
        self.pair_frame.grid_remove()
        def show_pair():self.pair_frame.grid()
        ttk.Button(box,text='새 리더·팔로워 등록',command=show_pair).grid(row=4,column=0,columnspan=3,sticky='w')
        ttk.Label(box,textvariable=self.status,wraplength=690).grid(row=6,column=0,columnspan=3,sticky='ew',pady=12)
        ttk.Button(box,text='다음 실행 모드 저장' if next_launch else '이 모드로 앱 열기',command=self.save,style='Primary.TButton').grid(row=7,column=0,columnspan=3,sticky='ew')
        if next_launch:ttk.Label(box,text='현재 연결은 변경하지 않습니다. 앱을 종료한 뒤 opencv-teach로 다시 실행하세요.',wraplength=690).grid(row=8,column=0,columnspan=3,sticky='w',pady=8)
    def register(self):
        try:
            path=register_pair(self.directory,**{k:v.get().strip() for k,v in self.inputs.items()})
            self.mode.set('리더·팔로워 함께');self.profile.set(str(path));self.status.set('두 팔 등록 완료 · 원본 보정 JSON은 유지했습니다.')
        except Exception as exc:self.status.set(str(exc))
    def save(self):
        try:
            mode='follower_only' if self.mode.get()=='팔로워 단독' else 'pair'
            selected=validate_selection(mode,self.profile.get().strip())
            from teach_left_arm import atomic_json
            atomic_json(self.directory/'launch_mode.json',{'mode':mode,'profile':str(selected[1].resolve())})
            self.done(selected)
        except Exception as exc:self.status.set(str(exc))


def choose_mode(directory):
    import tkinter as tk
    from teaching_ui import configure_ui
    root=tk.Tk();configure_ui(root);root.title('OpenCV 티칭 · 실행 모드');root.geometry('780x710')
    result=[]
    def done(value):result.append(value);root.destroy()
    ModePanel(root,directory,done);root.mainloop()
    return result[0] if result else None


def open_mode_settings(app):
    from app_navigation import AppPage
    from teach_left_arm import ROOT
    page=AppPage(app);page.title('실행 모드 설정')
    def done(_):app.note('실행 모드를 저장했습니다. 앱 재시작 후 적용됩니다.');page.destroy()
    ModePanel(page,ROOT,done,next_launch=True)
