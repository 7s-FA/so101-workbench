"""Follower-only incremental adjustment and measured-pose teaching.

Geometry helpers are hardware-free. The panel sends bounded requests through
the existing Session; only that control loop reads or writes motors.
"""
from copy import deepcopy
import math
import time
import uuid

JOG_RATE = 5.0  # degrees / second; gripper normalized units / second
MAX_JOINT_CHANGE = 5.0
MAX_SAMPLE_AGE = .6
JOG_TIMEOUT_SECONDS = 2.5
JOG_REFERENCE_SECONDS = 10.


def checked_jog(request, current, now, validate):
    from teach_left_arm import pose, difference
    start, goal = pose(request['start']), pose(request['goal'])
    age = now-float(request['sample_time'])
    if not math.isfinite(age) or not 0 <= age <= MAX_SAMPLE_AGE:
        raise ValueError('측정값이 오래됐습니다. 새 관절값을 기다린 뒤 다시 누르세요.')
    if difference(start, current) > .5:
        raise ValueError('팔 위치가 바뀌었습니다. 멈춘 뒤 다시 누르세요.')
    if difference(start, goal) > MAX_JOINT_CHANGE+1e-8:
        raise ValueError('한 번에 움직일 관절 변화가 큽니다. 간격을 줄이거나 관절별로 조정하세요.')
    return validate(goal)


def jog_goal(current, tool, axis, amount, reference=None, now=None, hold_target=None):
    from scipy.spatial.transform import Rotation
    from vision_teaching import Kinematics, number
    from teach_left_arm import pose
    current = pose(current)
    if axis in current:
        limit = MAX_JOINT_CHANGE
        amount = number(amount, -limit, limit)
        now = time.monotonic() if now is None else now
        held = pose(hold_target) if hold_target is not None else current
        coarse = abs(amount) == MAX_JOINT_CHANGE
        base = current[axis] if coarse else held[axis]
        if (not coarse and reference and reference.get('axis') == axis and amount*reference.get('direction', 0) > 0
                and 0 <= now-reference['at'] <= JOG_REFERENCE_SECONDS
                and abs(current[axis]-reference['measured']) <= .5):
            base = float(reference['requested'])
        goal = {**held, axis: base+amount}
        return pose(goal)
    if axis not in ('X','Y','Z','turn'):
        raise ValueError('이동 방향을 확인하세요.')
    amount = number(amount, -2, 2)
    kin = Kinematics(tool);tf = kin.fk(current);xyz = tf[:3,3].copy();rotation=tf[:3,:3]
    if axis == 'turn':rotation=Rotation.from_euler('z',amount,degrees=True).as_matrix()@rotation
    else:xyz['XYZ'.index(axis)]+=amount
    goal=kin.ik(xyz,rotation,current,current['gripper.pos'])
    if max(abs(goal[k]-current[k]) for k in goal)>MAX_JOINT_CHANGE:
        raise ValueError('이 방향은 관절을 크게 바꿔야 합니다. 관절별 조정으로 먼저 자세를 맞추세요.')
    return goal


def jog_hold_pose(request, goal, current, project):
    axis = request.get('axis')
    # Keep the previous holding command on untouched joints; reading back a
    # load-related error must not shift their commanded pose on every click.
    hold = {**goal, axis: current[axis]} if axis in current else current.copy()
    return project(hold)


def jog_outcome(request, goal, current, arrived, now):
    """Report requested vs measured change without claiming a deadband move arrived."""
    axis=request.get('axis')
    reference=None
    if axis in current and request.get('increment'):
        reference={'axis':axis,'requested':goal[axis],'measured':current[axis],
                   'direction':1 if request['increment']>0 else -1,'at':now}
    message='미세 조정 처리 완료 · 자세 유지' if arrived else '미세 조정 목표에 미도달 · 현재 자세 유지, 다시 조정할 수 있습니다.'
    if axis in current:
        unit='단위' if axis=='gripper.pos' else '°'
        message+=f" · 요청 변화 {goal[axis]-request['start'][axis]:+.2f}{unit} / 실제 변화 {current[axis]-request['start'][axis]:+.2f}{unit}"
        if request.get('unit') == 'ticks':
            from motor_positions import TICK_DEGREES
            asked=round((goal[axis]-request['start'][axis])/TICK_DEGREES)
            moved=round((current[axis]-request['start'][axis])/TICK_DEGREES)
            message=('모터 조정 완료' if arrived else '목표 미도달 · 다시 조정 가능')+f' · 요청 {asked:+d} / 실제 {moved:+d} 틱'
    return {'token':request.get('token'),'arrived':bool(arrived),'goal':goal.copy(),
            'pose':current.copy(),'message':message,'reference':reference}


def jog_arrived(request, goal, current):
    from motor_positions import TICK_DEGREES
    return all(abs(goal[k]-current[k]) <=
               (TICK_DEGREES*.5 if request.get('unit') == 'ticks' and k == request.get('axis')
                else .5 if k == 'gripper.pos' else .35) for k in goal)


def measured_target(target, measured_pose, tool, observation=None):
    """Preserve the complete measured TCP orientation, including wrist roll."""
    from vision_teaching import Kinematics, validate_target
    from scipy.spatial.transform import Rotation
    t=deepcopy(target);tf=Kinematics(tool).fk(measured_pose)
    xyz=tf[:3,3].copy();rotation=tf[:3,:3]
    if t['mode']=='rectangle':
        if observation is None:raise ValueError('카메라가 읽은 지그 위치가 필요합니다.')
        inverse=Rotation.from_euler('z',-observation['yaw_deg'],degrees=True).as_matrix()
        xyz=inverse@(xyz-[observation['x_mm'],observation['y_mm'],0])
        rotation=inverse@rotation
    elif t['mode']!='fixed':raise ValueError('지그 기준 또는 이 자리에 고정을 선택하세요.')
    roll,pitch,yaw=Rotation.from_matrix(rotation).as_euler('xyz',degrees=True)
    t.update(x_mm=float(xyz[0]),y_mm=float(xyz[1]),z_mm=float(xyz[2]),
             roll_deg=float(roll),pitch_deg=float(pitch+180),yaw_deg=float(yaw))
    return validate_target(t)


class ManualTeachingPanel:
    def __init__(self, view, parent):
        import tkinter as tk
        from tkinter import ttk
        from app_choices import AppChoice
        from teaching_ui import joint_label
        self.view,self.app=view,view.app
        self.pending=None;self.buttons=[];self.readouts={};self.sliders={};self.context=None;self.revision=0
        self.status=view.status
        self.position=tk.StringVar(value='집게 끝 위치: 연결 후 표시')
        self.target_label=tk.StringVar(value='먼저 잡기·놓기에서 목표를 선택하세요.')
        title='실물을 보며 위치 맞추기'+(' · 데모, 실물 연결 없음' if self.app.backend.identity['demo'] else '')
        ttk.Label(parent,text=title,style='PanelHeading.TLabel').grid(row=0,column=0,columnspan=4,sticky='w',pady=6)
        ttk.Label(parent,text='① ±로 위치 맞추기 → ② 위치 저장\n지그를 따라가는 위치를 저장할 때는 지그 기준을 먼저 읽으세요.',wraplength=440).grid(row=1,column=0,columnspan=4,sticky='ew',pady=4)
        actions=ttk.Frame(parent);actions.grid(row=2,column=0,columnspan=4,sticky='ew',pady=5)
        self.connect_button=ttk.Button(actions,text='팔로워 연결',command=self.app.connect)
        self.connect_button.pack(side='left')
        ttk.Button(actions,text='움직임 멈추기 · 힘 유지',command=self.pause).pack(side='left',padx=5)
        ttk.Label(parent,textvariable=self.position,wraplength=440).grid(row=3,column=0,columnspan=4,sticky='ew',pady=6)
        from motor_positions import TICK_STEPS
        self.joint_step=tk.StringVar(value='5');self.xyz_step=tk.StringVar(value='1');self.grip_step=tk.StringVar(value='2')
        ttk.Label(parent,text='관절 한 번 이동 (모터 틱)').grid(row=5,column=0,sticky='w')
        AppChoice(parent,textvariable=self.joint_step,values=TICK_STEPS,state='readonly',width=8).grid(row=5,column=1,columnspan=3,sticky='ew')
        from teach_left_arm import KEYS
        for row,key in enumerate(KEYS,6):
            ttk.Label(parent,text='집게 벌림' if key=='gripper.pos' else joint_label(key)).grid(row=row,column=0,sticky='w',pady=3)
            self.button(parent,row,1,'− 닫기' if key=='gripper.pos' else '−',lambda k=key:self.jog(k,-1))
            self.readouts[key]=tk.StringVar(value='—')
            cell=ttk.Frame(parent);cell.grid(row=row,column=2,sticky='ew')
            ttk.Label(cell,textvariable=self.readouts[key],width=20,anchor='center').pack(fill='x')
            from motor_slider import MotorSlider
            slider=MotorSlider(cell,5 if key=='gripper.pos' else 50,'벌림' if key=='gripper.pos' else '틱',
                lambda d,k=key:self.view.guarded(lambda:self.jog(k,1,amount=d,from_measured=True)),
                lambda d,k=key:self.slider_description(k,d))
            slider.frame.pack(fill='x');self.sliders[key]=slider
            self.button(parent,row,3,'+ 열기' if key=='gripper.pos' else '+',lambda k=key:self.jog(k,1))
        ttk.Label(parent,text='집게 한 번 이동 (0~100)').grid(row=12,column=0,sticky='w')
        AppChoice(parent,textvariable=self.grip_step,values=('1','2','5'),state='readonly',width=8).grid(row=12,column=1,columnspan=3,sticky='ew')
        tip=self.view.section(parent,13,'집게 끝을 직선으로 조금씩 이동')
        tip.master.grid_configure(columnspan=4)
        ttk.Label(tip,text='X·Y는 로봇 바닥의 축 기준, Z는 위·아래입니다.\n화면의 좌우와 다를 수 있습니다. 도달하지 못하면 관절별로 조정하세요.',wraplength=430).grid(row=0,column=0,columnspan=4,sticky='ew',pady=4)
        ttk.Label(tip,text='한 번 이동 (mm / °)').grid(row=1,column=0,sticky='w')
        AppChoice(tip,textvariable=self.xyz_step,values=('0.5','1','2'),state='readonly',width=8).grid(row=1,column=1,columnspan=3,sticky='ew')
        for row,(axis,label) in enumerate((('X','X 방향'),('Y','Y 방향'),('Z','높이'),('turn','집게 방향')),2):
            ttk.Label(tip,text=label).grid(row=row,column=0,sticky='w',pady=4)
            self.button(tip,row,1,'− 아래' if axis=='Z' else '−',lambda a=axis:self.jog(a,-1))
            slider=MotorSlider(tip,2,'°' if axis=='turn' else 'mm',
                lambda d,a=axis:self.view.guarded(lambda:self.jog(a,1,amount=d)),step=.5)
            slider.frame.grid(row=row,column=2,sticky='ew');self.sliders[axis]=slider
            self.button(tip,row,3,'+ 위' if axis=='Z' else '+',lambda a=axis:self.jog(a,1))
        footer=ttk.Frame(view.pages[4],padding=(7,4));footer.grid(row=1,column=0,columnspan=2,sticky='ew')
        footer.columnconfigure(0,weight=1);footer.columnconfigure(1,weight=1)
        ttk.Label(footer,textvariable=self.target_label,wraplength=230).grid(row=0,column=0,sticky='ew',pady=3)
        self.button(footer,1,0,'맞춘 위치를 목표에 반영',lambda:self.measure('target'))
        self.button(footer,1,1,'현재 자세 스텝 저장',lambda:self.measure('step'))
        grip=ttk.Frame(footer);grip.grid(row=2,column=0,columnspan=2,sticky='ew',pady=2)
        self.button(grip,0,0,'현재 벌림을 열림값으로',lambda:self.measure('open'))
        self.button(grip,0,1,'현재 벌림을 닫힘값으로',lambda:self.measure('closed'))
        ttk.Checkbutton(footer,text='저장 시 지그 보정',variable=self.app.capture_follow).grid(row=0,column=1,sticky='e')
        ttk.Label(parent,text='저장한 지그 기준은 팔에 가려져도 유지됩니다.\n재도킹하거나 지그를 움직이면 기준을 다시 읽어 주세요.\n카메라 보정 전에도 관절 조정·자세 스텝 저장은 가능합니다.',wraplength=440,style='Muted.TLabel').grid(row=14,column=0,columnspan=4,sticky='ew',pady=8)
        self.view.button_row(parent,15,[('잡기·놓기 목표 선택',lambda:view.select_tab(1)),('카메라 켜기',view.start_camera)]).grid_configure(columnspan=4)
        parent.columnconfigure(0,weight=1);parent.columnconfigure(2,weight=1)

    def button(self,parent,row,col,text,action,span=1):
        from tkinter import ttk
        b=ttk.Button(parent,text=text,command=lambda:self.view.guarded(action),style='Compact.TButton')
        b.grid(row=row,column=col,columnspan=span,sticky='ew',padx=3,pady=3);self.buttons.append(b);return b

    def current(self):
        self.view.idle()
        if self.pending:raise ValueError('이전 조정이 처리될 때까지 기다리세요.')
        sample=self.app.latest
        if self.app.state!='HOLD' or not sample or self.app.session.stop.is_set():
            raise ValueError('팔로워를 연결하고 자세 유지 상태에서 조정하세요.')
        if not 0<=time.monotonic()-sample['sample_time']<=MAX_SAMPLE_AGE:
            raise ValueError('새 관절값을 기다리세요.')
        if self.app.pose_editor is not None:raise ValueError('관절 편집창을 닫고 조정하세요.')
        return deepcopy(sample)

    def send(self,command,payload):
        self.pending={'token':payload['token'],'at':time.monotonic(),'revision':self.revision}
        self.app.send(command,payload);self.refresh()

    def slider_description(self,axis,delta):
        sample=self.app.latest
        if not sample:return f'{delta:+d}'
        if axis=='gripper.pos':return f"목표 벌림 {sample['follower'][axis]+delta:.1f} / 100"
        from motor_positions import to_ticks
        raw=to_ticks(self.app.backend,axis,sample['follower'][axis])
        return f'목표 {raw+delta} 틱 ({delta:+d})'

    def jog(self,axis,sign,*,amount=None,from_measured=False):
        from vision_teaching import digest
        sample=self.current();tool=deepcopy(self.view.station['tool'])
        revision=self.revision
        delta=float((self.grip_step if axis=='gripper.pos' else self.joint_step if axis.endswith('.pos') else self.xyz_step).get())*sign if amount is None else amount
        raw_joint=axis.endswith('.pos') and axis!='gripper.pos'
        held=deepcopy(sample.get('hold_target') or sample['follower'])
        reference=sample.get('jog_reference')
        if from_measured:
            held[axis]=sample['follower'][axis];reference=None
        def done(goal):
            if revision!=self.revision or self.view.closed:return
            fresh=self.current()
            if digest(tool)!=digest(self.view.station['tool']):raise ValueError('집게 끝 설정이 바뀌었습니다. 다시 조정하세요.')
            request={'token':uuid.uuid4().hex,'start':sample['follower'],'goal':goal,'sample_time':sample['sample_time']}
            if axis in sample['follower']:request.update(axis=axis,increment=delta)
            if raw_joint:request['unit']='ticks'
            if from_measured:request['from_measured']=True
            checked_jog(request,fresh['follower'],time.monotonic(),self.app.backend.validate_target)
            self.send('jog',request);self.status.set('작은 이동을 요청했습니다. 실물을 확인하세요.')
        from motor_positions import tick_goal
        self.view.run_job(lambda:tick_goal(self.app.backend,sample['follower'],axis,delta,reference,hold_target=held) if raw_joint
                          else jog_goal(sample['follower'],tool,axis,delta,reference,hold_target=held),done)

    def jog_completed(self, result):
        self.status.set(result['message'])

    def pause(self):
        self.revision+=1
        self.app.send('pause');self.status.set('이동을 멈추고 현재 자세를 유지합니다.')

    def measure(self,purpose):
        from vision_teaching import digest,calibration_signature
        sample=self.current()
        target=self.view.read_target() if purpose!='step' else None
        capture_context=self.app.new_capture_context() if purpose=='step' else None
        self.context={'capture_context':capture_context,'purpose':purpose,'target':target,'target_id':self.view.target_id,
                      'target_digest':digest(target),'station':calibration_signature(self.view.station)}
        if purpose=='target' and target['mode']=='rectangle':
            self.view.update_jig_reference()
            self.context['jig_reference']=self.view.teaching_reference.check(target,self.view.station)
        self.send('measure_target',{'token':uuid.uuid4().hex,'start':sample['follower'],'sample_time':sample['sample_time']})
        self.status.set('멈춘 실물 관절값을 읽어 저장합니다.')

    def completed(self,result):
        from vision_teaching import digest,calibration_signature,reference_key
        if not self.pending or result.get('token')!=self.pending['token']:return
        pending=self.pending;self.pending=None
        if pending['revision']!=self.revision:return
        if not result.get('ok'):
            self.status.set(result['error']);self.view.status.set(result['error']);return
        if result['kind']=='jog':return
        try:
            pose=result['pose'];context=self.context
            if not 0<=time.monotonic()-result['sample_time']<=MAX_SAMPLE_AGE:raise ValueError('저장할 관절값이 오래됐습니다. 다시 누르세요.')
            if self.app.state!='HOLD' or self.app.session.stop.is_set():raise ValueError('팔이 멈춘 뒤 다시 저장하세요.')
            if context['purpose']=='step':
                from teaching_settings import MAX_STEPS
                from teach_left_arm import next_step_name
                if len(self.app.steps)>=MAX_STEPS:raise ValueError('스텝 수 한도를 초과했습니다.')
                step={'name':next_step_name(self.app.steps),'pose':pose,'captured_at':result['captured_at']}
                self.app.store_captured_step(step,context['capture_context'])
                self.status.set('실제로 읽은 현재 자세를 스텝으로 저장했습니다. 메인 화면에서 에피소드 저장을 누르세요.')
                return
            if context['station']!=calibration_signature(self.view.station) or context['target_id']!=self.view.target_id or context['target_digest']!=digest(self.view.read_target()):
                raise ValueError('목표나 보정값이 바뀌었습니다. 다시 저장하세요.')
            t=context['target'];purpose=context['purpose']
            if purpose=='target':
                observation=None
                if t['mode']=='rectangle':
                    self.view.update_jig_reference()
                    active=self.view.teaching_reference.check(t,self.view.station)
                    if active['id']!=context['jig_reference']['id']:
                        raise ValueError('저장하는 동안 지그 기준이 바뀌었습니다. 다시 저장하세요.')
                    observation=active['observation']
                changed=measured_target(t,pose,self.view.station['tool'],observation)
                keys=('x_mm','y_mm','z_mm','yaw_deg','pitch_deg','roll_deg')
            else:
                changed={**t,purpose:pose['gripper.pos']};keys=(purpose,)
            from vision_teaching import validate_target
            changed=validate_target(changed)
            for key in keys:self.view.vars['t_'+key].set(str(changed[key]))
            self.view.save_target(measurement={'kind':'follower_measured','purpose':purpose,
                'pose':deepcopy(pose),'captured_at':result['captured_at'],
                'calibration_signature':context['station'],'tcp_position_source':'kinematics_from_measured_joints',
                'reference':observation if purpose=='target' else None,
                'jig_reference':deepcopy(context.get('jig_reference'))})
            self.status.set('맞춘 값을 목표에 저장했습니다. 집기·놓기 화면에서 경로를 다시 계산하세요.')
        except Exception as exc:self.status.set(str(exc));self.view.status.set(str(exc))

    def refresh(self):
        sample=self.app.latest
        fresh=bool(sample and 0<=time.monotonic()-sample['sample_time']<=MAX_SAMPLE_AGE)
        # A missing reply is not permission to execute; the session also rejects
        # expired requests. Clear UI waiting only after the queue TTL has passed.
        if self.pending and time.monotonic()-self.pending['at']>2:
            self.pending=None;self.status.set('조정 응답이 없습니다. 연결 상태와 현재 위치를 확인하세요.')
        enabled=fresh and self.app.state=='HOLD' and not self.pending and not self.view.job and not self.app.preview_only and not self.app.closing and self.app.pose_editor is None
        for button in self.buttons:button.configure(state='normal' if enabled else 'disabled')
        for slider in self.sliders.values():slider.enable(enabled)
        self.connect_button.configure(state='normal' if self.app.state=='DISCONNECTED' and not self.app.closing and not self.app.preview_only else 'disabled')
        if fresh:
            from vision_teaching import Kinematics
            q=sample['follower'];xyz=Kinematics(self.view.station['tool']).fk(q)[:3,3]
            self.position.set('집게 끝 (계산값)  X %.1f · Y %.1f · 높이 %.1f mm'%tuple(xyz))
            from motor_positions import to_ticks
            for key,var in self.readouts.items():
                raw=sample.get('motor_raw',{}).get('follower',{}).get(key)
                if key!='gripper.pos' and raw is None:raw=to_ticks(self.app.backend,key,q[key])
                var.set(f"{q[key]:.2f} / 100" if key=='gripper.pos' else f"{raw} 틱 · {q[key]:.2f}°")
        else:
            self.position.set('집게 끝 위치: 새 관절값을 기다립니다.' if self.app.state!='DISCONNECTED' else '집게 끝 위치: 팔로워 연결 후 표시')
            for var in self.readouts.values():var.set('—')
        name=self.view.vars['t_name'].get();self.target_label.set('저장할 목표: '+name+' · '+self.view.vars['t_operation'].get())
