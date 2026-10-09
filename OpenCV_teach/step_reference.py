"""Step-level fixed/follow choices. Changes are edits, never motor commands."""
from copy import deepcopy
import uuid


def used_plan(steps, plan):
    result=deepcopy(plan)
    used={s['motion']['target'] for s in steps if s.get('motion')}
    result['targets']={key:value for key,value in plan.get('targets',{}).items() if key in used}
    return result


def needs_measurement(steps, plan):
    return any(t.get('mode')=='rectangle' for t in used_plan(steps,plan)['targets'].values())


def reference_mode(step, plan):
    if step.get('safe_boundary') or step.get('gripper_only'):return '고정 위치'
    motion=step.get('motion')
    return '지그 따라가기' if motion and plan['targets'].get(motion['target'],{}).get('mode')=='rectangle' else '고정 위치'


def change_reference(steps, plan, indices, mode, station, *, profile=None, observation=None, gripper_only=False):
    """Keep original joints when disabling, and preserve relative targets to restore."""
    from teach_left_arm import pose
    from vision_teaching import validate_plan,calibration_signature,reference_key
    from manual_teaching import measured_target
    result=deepcopy(steps);updated=deepcopy(plan)
    if mode not in ('고정 위치','지그 따라가기'):raise ValueError('위치 기준을 선택하세요.')
    for index in indices:
        s=result[index]
        if s.get('safe_boundary') and (mode!='고정 위치' or gripper_only):
            raise ValueError('안전 자세는 고정 위치로만 사용합니다.')
        if s.get('safe_boundary'):continue
        s.setdefault('fixed_pose',pose(s['pose']))
        if mode=='고정 위치' or gripper_only:
            if s.get('motion'):s['follow_motion']=s.pop('motion')
            s.pop('path',None);s['pose']=deepcopy(s['fixed_pose'])
            if gripper_only:s['gripper_only']=True
            else:s.pop('gripper_only',None)
            continue
        s.pop('gripper_only',None)
        original=s.get('motion') or s.get('follow_motion')
        old_target=updated['targets'].get(original['target'],{}) if original else {}
        if original and old_target.get('mode')=='rectangle' and (profile is None or reference_key(old_target)==reference_key(profile)):
            s['motion']=deepcopy(original);s.pop('path',None);continue
        if profile is None or profile.get('mode')!='rectangle' or observation is None:
            raise ValueError('장비 설정에서 지그를 등록하고 카메라 창의 지그 기준 저장을 먼저 누르세요.')
        target=measured_target(profile,s['fixed_pose'],station['tool'],observation)
        target['name']=s['name'][:60]
        target['return_to_start']=False
        key='step_'+uuid.uuid4().hex[:12]
        updated['targets'][key]=target
        s['motion']={'target':key,'phase':'waypoint'}
        s['follow_motion']=deepcopy(s['motion'])
        s['teaching_reference']={k:observation[k] for k in ('x_mm','y_mm','yaw_deg')}
        s.pop('path',None)
    updated['calibration_signature']=calibration_signature(station)
    return result,validate_plan(updated)


class StepReferenceEditor:
    def __init__(self, app, index):
        import tkinter as tk
        from tkinter import ttk
        from app_choices import AppChoice
        self.app,self.index=app,index
        self.step=app.steps[index]
        from app_navigation import AppPage
        self.window=AppPage(app,section="episode",dock_preview=True);self.window.title('스텝 설정');self.window.geometry('520x460')
        self.window.minsize(500,450)
        self.window.transient(app.root);self.window.protocol('WM_DELETE_WINDOW',self.close)
        box=ttk.Frame(self.window,padding=18);box.pack(fill='both',expand=True);box.columnconfigure(1,weight=1)
        ttk.Label(box,text=self.step['name'],style='EditorStep.TLabel',wraplength=450).grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,14))
        self.mode=tk.StringVar(master=self.window,value=reference_mode(self.step,app.vision_plan))
        ttk.Label(box,text='위치 기준').grid(row=2,column=0,sticky='w')
        self.choice=AppChoice(box,textvariable=self.mode,values=('고정 위치','지그 따라가기'),state='readonly')
        self.choice.grid(row=2,column=1,sticky='ew',pady=6)
        ttk.Label(box,text='고정 위치: 티칭한 관절값으로 이동\n지그 따라가기: 저장한 지그 기준으로 위치·각도 보정',wraplength=450).grid(row=3,column=0,columnspan=2,sticky='w',pady=12)
        original=self.step.get('motion') or self.step.get('follow_motion')
        from jig_catalog import app_profiles
        from vision_teaching import reference_key
        self.profiles=app_profiles(app)
        for target in app.vision_plan['targets'].values():
            if target.get('mode')=='rectangle':self.profiles[reference_key(target)]=target
        self.jig_keys=list(self.profiles)
        self.jig_choice=AppChoice(box,values=[self.profiles[k]['jig_name'] for k in self.jig_keys],state='readonly')
        self.jig_choice.grid(row=1,column=1,sticky='ew',pady=6)
        ttk.Label(box,text='따라갈 지그').grid(row=1,column=0,sticky='w')
        bound=app.vision_plan['targets'].get(original['target'],{}) if original else {}
        selected=reference_key(bound) if bound.get('mode')=='rectangle' else app.vision_plan.get('teaching_jig_key')
        if selected in self.jig_keys:self.jig_choice.current(self.jig_keys.index(selected))
        def toggle(_=None):
            self.jig_choice.configure(state='readonly' if self.mode.get()=='지그 따라가기' and not self.step.get('safe_boundary') and not self.step.get('gripper_only') else 'disabled')
        self.choice.bind('<<ComboboxSelected>>',toggle);toggle()
        self.group=tk.BooleanVar(master=self.window,value=bool(original and original['phase']!='waypoint'))
        self.group_box=ttk.Checkbutton(box,text='같은 집기·놓기 동작의 단계에 함께 적용',variable=self.group)
        self.group_box.grid(row=4,column=0,columnspan=2,sticky='w')
        ttk.Button(box,text='카메라에서 지그 기준 읽기',command=self.open_camera).grid(row=5,column=0,columnspan=2,sticky='w',pady=8)
        self.message=tk.StringVar(master=self.window,value='설정만 저장합니다. 로봇은 움직이지 않습니다.')
        ttk.Label(box,textvariable=self.message,wraplength=450).grid(row=6,column=0,columnspan=2,sticky='ew',pady=12)
        actions=ttk.Frame(self.window.footer);actions.pack(fill='x')
        ttk.Button(actions,text='적용',command=self.apply,style='Primary.TButton').pack(side='right')
        ttk.Button(actions,text='취소',command=self.close).pack(side='right',padx=8)
        if self.step.get('safe_boundary'):
            self.choice.configure(state='disabled');self.group_box.configure(state='disabled');self.group.set(False)
            self.message.set('안전 자세는 고정 위치입니다. 관절값 편집에서 자세를 바꾸세요.')
        if self.step.get('gripper_only'):
            self.choice.configure(state='disabled');self.group_box.configure(state='disabled');self.group.set(False)
            self.message.set('기존 집게 동작입니다. 관절값 편집에서 벌림을 조정하세요.')

    def selected_profile(self):
        index=self.jig_choice.current()
        if not 0<=index<len(self.jig_keys):raise ValueError('이 스텝이 따라갈 지그를 선택하세요.')
        return self.profiles[self.jig_keys[index]]

    def open_camera(self):
        try:
            from jig_catalog import select_plan
            view=self.app.open_vision(show=False)
            if view is None:return
            profile=self.selected_profile()
            plan,ident,_=select_plan(self.app.steps,self.app.vision_plan,profile,view.station)
            self.app.vision_plan=plan;view.target_id=ident;view.refresh_target_list();view.load_target()
            self.app.edited();view.schedule_teaching_reference(preserve=True);view.select_tab(3)
        except Exception as exc:self.message.set(str(exc))

    def apply(self):
        from teach_left_arm import validate_program,atomic_json
        try:
            if self.app.state not in ('HOLD','READY','DISCONNECTED') or self.app.vision_busy or self.app.closing:
                raise ValueError('동작을 멈춘 뒤 스텝을 설정하세요.')
            if self.index>=len(self.app.steps) or self.app.steps[self.index] is not self.step:
                raise ValueError('선택 스텝이 변경됐습니다. 창을 다시 여세요.')
            view=self.app.open_vision(show=False)
            if view is None:raise ValueError('장비 설정을 열 수 없습니다.')
            indices=[self.index];original=self.step.get('motion') or self.step.get('follow_motion')
            if self.group.get() and original:
                indices=[i for i,s in enumerate(self.app.steps) if (s.get('motion') or s.get('follow_motion') or {}).get('target')==original['target']]
            profile=observation=None
            if self.mode.get()=='지그 따라가기':
                from vision_teaching import reference_key
                profile=self.selected_profile()
                old=self.app.vision_plan['targets'].get(original['target'],{}) if original else {}
                if old.get('mode')!='rectangle' or reference_key(old)!=reference_key(profile):
                    observation=view.teaching_observations({'targets':{'selected':profile}})[reference_key(profile)]
            steps,plan=change_reference(self.app.steps,self.app.vision_plan,indices,self.mode.get(),view.station,
                                        profile=profile,observation=observation,gripper_only=bool(self.step.get('gripper_only')))
            selected_index=self.index
            from safe_episode import validate_boundaries,model_safe_pose,with_safe_endpoints
            if needs_measurement(steps,plan) and not validate_boundaries(steps):
                from simulation_bridge import read_source
                safe=self.app.backend.validate_target(model_safe_pose(read_source(),view.station['tool']))
                steps=with_safe_endpoints(steps,safe);selected_index+=1
                plan['safe_pose_reference']={'source':'Test_teach safe_deg','model_candidate':True,'pose':deepcopy(safe)}
            document={**self.app.document(),'schema':2,'steps':steps,'vision_plan':plan}
            validate_program(document,self.app.episode_identity)
            atomic_json(self.app.program_path,document)
            self.app.steps,self.app.vision_plan=steps,plan
            self.app.refresh_steps(selected_index);self.app.edited();view.refresh_target_list()
            self.app.note(f'{len(indices)}개 스텝: {self.mode.get()} 적용. 에피소드를 저장하세요.')
            self.close()
        except Exception as exc:self.message.set(str(exc))

    def close(self):
        self.app.reference_editor=None
        self.window.destroy();self.app.refresh_buttons()


def editable_servos(step):
    """Computed arm joints are visible but cannot be overwritten as raw goals."""
    from teach_left_arm import KEYS
    return {'gripper.pos'} if step.get('motion') or step.get('gripper_only') else set(KEYS)


def edit_calculated_gripper(steps,plan,index,value):
    result=deepcopy(steps);updated=deepcopy(plan)
    item=result[index];binding=item['motion'];target=updated['targets'][binding['target']]
    def field(phase):
        after=phase in ('grip','retreat','return')
        return ('closed' if after else 'open') if target['operation']=='pick' else ('open' if after else 'closed')
    if binding['phase']=='waypoint':
        item['pose']['gripper.pos']=value
        item.setdefault('fixed_pose',deepcopy(item['pose']))['gripper.pos']=value
        item.pop('path',None)
    else:
        target[field(binding['phase'])]=value
        for step in result:
            motion=step.get('motion')
            if not motion or motion['target']!=binding['target']:continue
            step.pop('path',None)
            if motion['phase']=='waypoint':continue
            step['pose']['gripper.pos']=target[field(motion['phase'])]
            if 'fixed_pose' in step:step['fixed_pose']['gripper.pos']=step['pose']['gripper.pos']
    return result,updated


class InlineStepSettings(StepReferenceEditor):
    """Inspector replacing only the 3D region; episode selection stays available."""
    def __init__(self,app,parent):
        import tkinter as tk
        from tkinter import ttk
        from app_choices import AppChoice
        self.app=app;self.index=None;self.step=None;self.token=None;self.profiles={};self.jig_keys=[]
        self.window=box=ttk.Frame(parent,padding=(8,8),style='PanelBody.TFrame');box.columnconfigure(0,weight=1)
        self.title=tk.StringVar(master=box,value='스텝을 선택하세요')
        ttk.Label(box,text='스텝 설정',style='PanelHeading.TLabel').grid(row=0,column=0,sticky='w',pady=(0,6))
        ttk.Label(box,textvariable=self.title,wraplength=430,width=1,style='PreviewStatus.TLabel').grid(row=1,column=0,sticky='ew',pady=(0,10))
        ttk.Label(box,text='위치 기준').grid(row=2,column=0,sticky='w')
        self.mode=tk.StringVar(master=box,value='고정 위치')
        self.choice=AppChoice(box,textvariable=self.mode,values=('고정 위치','지그 따라가기'),state='readonly',width=1)
        self.choice.grid(row=3,column=0,sticky='ew',pady=(3,8));self.choice.bind('<<ComboboxSelected>>',lambda _:self.refresh_controls())
        ttk.Label(box,text='따라갈 지그').grid(row=4,column=0,sticky='w')
        self.jig_choice=AppChoice(box,state='readonly',width=1);self.jig_choice.grid(row=5,column=0,sticky='ew',pady=(3,8))
        self.group=tk.BooleanVar(master=box,value=False)
        self.group_box=ttk.Checkbutton(box,text='연결된 단계 함께 적용',variable=self.group)
        self.group_box.grid(row=6,column=0,sticky='w',pady=5)
        self.apply_button=ttk.Button(box,text='스텝 설정 적용',command=self.apply,style='Compact.Primary.TButton')
        self.apply_button.grid(row=7,column=0,sticky='ew',pady=(4,8))
        self.camera_button=ttk.Button(box,text='지그 기준 읽기',command=self.open_camera,style='Compact.TButton')
        self.camera_button.grid(row=8,column=0,sticky='ew')
        self.message=tk.StringVar(master=box,value='')
        ttk.Label(box,textvariable=self.message,wraplength=430,width=1,style='Muted.TLabel').grid(row=9,column=0,sticky='ew',pady=8)
        self.refresh()

    def refresh(self):
        from vision_teaching import reference_key
        from jig_catalog import app_profiles
        index=self.app.selected()
        step=self.app.steps[index] if index is not None and index<len(self.app.steps) else None
        targets=self.app.vision_plan.get('targets',{})
        token=(id(step),repr(step.get('motion') or step.get('follow_motion')) if step else None,
               tuple((k,t.get('jig_name'),t.get('jig_id')) for k,t in targets.items()))
        if token!=self.token:
            self.token=token;self.index=index;self.step=step
            self.profiles=app_profiles(self.app)
            for target in targets.values():
                if target.get('mode')=='rectangle':self.profiles[reference_key(target)]=target
            self.jig_keys=list(self.profiles)
            self.jig_choice['values']=[self.profiles[k]['jig_name'] for k in self.jig_keys]
            original=(step.get('motion') or step.get('follow_motion')) if step else None
            bound=targets.get(original['target'],{}) if original else {}
            selected=reference_key(bound) if bound.get('mode')=='rectangle' else self.app.vision_plan.get('teaching_jig_key')
            if selected in self.jig_keys:self.jig_choice.current(self.jig_keys.index(selected))
            else:self.jig_choice.set('지그를 선택하세요')
            self.mode.set(reference_mode(step,self.app.vision_plan) if step else '고정 위치')
            self.group.set(bool(original and original['phase']!='waypoint'))
            self.message.set('안전 자세는 고정 위치입니다.' if step and step.get('safe_boundary') else '적용은 설정만 저장합니다.\n실제 이동은 스텝 실행을 누르세요.' if step else '')
        self.title.set(step['name'] if step else '스텝을 선택하세요')
        self.refresh_controls()

    def refresh_controls(self):
        idle=(self.app.state in ('DISCONNECTED','READY','HOLD') and not self.app.vision_busy and not self.app.closing
              and not self.app.preview_only and not self.app.finish_pending and self.app.pose_editor is None
              and self.app.reference_editor is None and self.app.name_editor is None)
        editable=bool(self.step and idle and not self.step.get('safe_boundary') and not self.step.get('gripper_only'))
        self.choice.configure(state='readonly' if editable else 'disabled')
        follows=editable and self.mode.get()=='지그 따라가기'
        self.jig_choice.configure(state='readonly' if follows else 'disabled')
        self.group_box.configure(state='normal' if editable else 'disabled')
        self.apply_button.configure(state='normal' if editable else 'disabled')
        self.camera_button.configure(state='normal' if follows else 'disabled')

    def apply(self):
        if self.step is None:return
        super().apply()

    def open_camera(self):
        if self.step is None:return
        super().open_camera()

    def close(self):
        self.token=None;self.refresh()
        self.message.set('스텝 설정을 적용했습니다.\n에피소드 저장으로 파일에 반영하세요.')
