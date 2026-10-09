"""Independent jig profiles and explicit episode selection; no robot commands."""
from copy import deepcopy
from pathlib import Path
import json
import uuid


def profiles(station):
    from jig_detection import validate_profile
    result={}
    for key,value in station.get('jigs',{}).items():
        if value.get('mode')!='rectangle':continue
        try:
            result[key]={'mode':'rectangle',**validate_profile(value)}
            if value.get('preview_model') is not None:
                from jig_stl import validate_model
                result[key]['preview_model']=validate_model(value['preview_model'])
        except (ValueError,KeyError,TypeError):continue
    return result


def register(station,profile,key=None):
    from jig_detection import validate_profile
    from vision_teaching import reference_key
    value={'mode':'rectangle',**validate_profile(profile),'catalog_managed':True}
    if profile.get('preview_model') is not None:
        from jig_stl import with_rim_metadata,detection_height
        value['preview_model']=with_rim_metadata(profile['preview_model'])
        value['surface_z_mm']=detection_height(value['preview_model'])
        if value['shape_model']=='ring':value['hole_layout']=deepcopy(value['preview_model'].get('hole_layout',[]))
        validate_profile(value)
    available=profiles(station)
    if key is not None and key not in available:raise ValueError('선택한 지그가 없습니다. 목록을 다시 확인하세요.')
    if any(k!=key and p['jig_name'].casefold()==value['jig_name'].casefold() for k,p in available.items()):
        raise ValueError('같은 이름의 지그가 있습니다. 구별할 이름을 입력하세요.')
    if key is not None and value['jig_name']!=available[key]['jig_name']:
        raise ValueError('기존 지그 이름은 유지하세요. 새 이름은 복사하여 등록할 수 있습니다.')
    if key is None:
        value['jig_id']='jig_'+uuid.uuid4().hex[:16];key=reference_key(value)
    elif 'jig_id' in available[key]:value['jig_id']=available[key]['jig_id']
    if key in available and any(value.get(k)!=available[key].get(k) for k in ('long_mm','short_mm','surface_z_mm','shape_model')):
        value['geometry_verified']=False
    updated=deepcopy(station);updated.setdefault('jigs',{})[key]=value
    return updated,key


def episode_keys(steps,plan):
    from vision_teaching import reference_key
    keys=set()
    for step in steps:
        binding=step.get('motion') or step.get('follow_motion')
        target=plan.get('targets',{}).get(binding.get('target')) if binding else None
        if target and target.get('mode')=='rectangle':keys.add(reference_key(target))
    return keys


def teaching_target(profile):
    from vision_teaching import validate_target
    return validate_target({**profile,'name':'직접 티칭 기준','mode':'rectangle','operation':'pick',
        'x_mm':0.,'y_mm':0.,'z_mm':30.,'yaw_deg':0.,'roll_deg':180.,'pitch_deg':180.,
        'hover_mm':20.,'retreat_mm':20.,'open':80.,'closed':20.,'speed_mm_s':10.,'wait_s':.4})


def select_plan(steps,plan,profile,station):
    from vision_teaching import reference_key,calibration_signature
    key=reference_key(profile)
    result=deepcopy(plan)
    result.setdefault('targets',{})
    matches=[k for k,t in result['targets'].items() if t.get('mode')=='rectangle' and reference_key(t)==key]
    added=not matches
    ident=matches[0] if matches else 'jig_'+uuid.uuid4().hex[:12]
    if added:result['targets'][ident]=teaching_target(profile)
    result['teaching_jig_key']=key
    result['calibration_signature']=calibration_signature(station)
    return result,ident,added


def app_profiles(app):
    from vision_teaching import digest
    view=getattr(app,'vision',None)
    if view:return profiles(view.station)
    path=app.output/'vision'/(digest(app.backend.identity)[:16]+'.json')
    try:
        station=json.loads(path.read_text())
        return profiles(station) if station.get('identity')==app.backend.identity else {}
    except (OSError,ValueError,TypeError):return {}


def refresh_picker(app):
    if not hasattr(app,'jig_picker'):return
    from vision_teaching import reference_key
    available=app_profiles(app);app.jig_picker_keys=sorted(available,key=lambda k:available[k]['jig_name'])
    labels=[available[k]['jig_name'] for k in app.jig_picker_keys]
    app.jig_picker['values']=labels
    keys=episode_keys(app.steps,app.vision_plan)
    selected=app.vision_plan.get('teaching_jig_key') or (next(iter(keys)) if len(keys)==1 else None)
    if selected in app.jig_picker_keys:app.jig_picker.current(app.jig_picker_keys.index(selected))
    elif keys:
        names=sorted({t.get('jig_name','') for t in app.vision_plan.get('targets',{}).values() if t.get('mode')=='rectangle'})
        app.jig_picker.set(' / '.join(names)+' (에피소드에 저장됨)')
    else:app.jig_picker.set('지그를 선택하세요' if labels else '먼저 지그를 등록하세요')


def choose(app,key):
    view=app.open_vision(show=False)
    if view is None:return
    try:
        view.idle()
        if app.preview_only:raise ValueError('다른 로봇의 에피소드에서는 지그를 변경할 수 없습니다.')
        if getattr(app,'pose_editor',None) or getattr(app,'reference_editor',None):raise ValueError('스텝 편집을 마친 뒤 지그를 선택하세요.')
        profile=profiles(view.station).get(key)
        if profile is None:raise ValueError('등록된 지그를 다시 선택하세요.')
        plan,ident,changed=select_plan(app.steps,app.vision_plan,profile,view.station)
        previous=app.vision_plan.get('teaching_jig_key')
        app.vision_plan=plan;view.target_id=ident;view.refresh_target_list();view.load_target()
        if previous!=key or changed:app.edited()
        if hasattr(view,'image_roi'):view.image_roi.cancel()
        try:view.teaching_reference.check(view.read_target(),view.station)
        except ValueError:view.schedule_teaching_reference(preserve=True)
        app.note('새 스텝의 지그: '+profile['jig_name']+' · 기존 스텝의 지그는 유지합니다.')
        refresh_picker(app)
    except Exception as exc:app.note(str(exc));refresh_picker(app)


def install_picker(app,parent):
    from tkinter import ttk
    from app_choices import AppChoice
    row=ttk.Frame(parent,style='PanelBody.TFrame');row.grid(row=3,column=0,sticky='ew',pady=(5,0));row.columnconfigure(1,weight=1)
    ttk.Label(row,text='새 스텝의 지그',style='PanelMuted.TLabel').grid(row=0,column=0,padx=(0,10))
    app.jig_picker=AppChoice(row,state='readonly');app.jig_picker.grid(row=0,column=1,sticky='ew',padx=(0,10))
    def select(_=None):
        index=app.jig_picker.current()
        if 0<=index<len(app.jig_picker_keys):choose(app,app.jig_picker_keys[index])
    app.jig_picker.bind('<<ComboboxSelected>>',select)
    def manage():
        view=app.open_vision(show=False)
        if view:view.select_tab(1)
    ttk.Button(row,text='지그 등록·관리',command=manage,style='Compact.TButton').grid(row=0,column=2)
    refresh_picker(app)


class CatalogPanel:
    def __init__(self,view,parent):
        import tkinter as tk
        from tkinter import ttk
        from app_choices import AppChoice
        self.view=view;self.app=view.app;self.key=None;self.keys=[];self.vars={};self.fields={}
        self.frame=ttk.Frame(parent);self.frame.grid(row=0,column=0,columnspan=2,sticky='nsew');self.frame.columnconfigure(1,weight=1)
        ttk.Label(self.frame,text='지그 등록·관리',style='PanelHeading.TLabel').grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,8))
        self.list=AppChoice(self.frame,state='readonly');self.list.grid(row=1,column=0,columnspan=2,sticky='ew');self.list.bind('<<ComboboxSelected>>',self.load)
        bar=ttk.Frame(self.frame);bar.grid(row=2,column=0,columnspan=2,sticky='ew',pady=8)
        ttk.Button(bar,text='새 지그',command=self.new).pack(side='left',padx=(0,6))
        ttk.Button(bar,text='선택 지그 복사',command=self.copy).pack(side='left')
        fields=[('지그 이름','jig_name',None),('외곽·내부 구조','shape_model',('직사각형','도넛형 (내부 구멍 있음)')),
                ('긴 변 (mm)','long_mm',None),('짧은 변 (mm)','short_mm',None),('검출면 높이 Z (mm)','surface_z_mm',None),
                ('색상·외곽','appearance',('흰색 프레임','명암 외곽')),
                ('검출 방법','detection_method',('색상+윤곽 (기본)','윤곽 중심','흰색 중심')),
                ('도킹 기준 방향 (°)','expected_yaw_deg',None),('방향 허용 범위 (±°)','yaw_tolerance_deg',None),
                ('외곽 크기 허용 오차 (%)','size_tolerance_pct',None)]
        for row,(label,name,choices) in enumerate(fields,4):
            if name=='jig_name':row=3
            ttk.Label(self.frame,text=label).grid(row=row,column=0,sticky='w',padx=(0,12),pady=3)
            var=tk.StringVar(master=self.frame);self.vars[name]=var
            widget=AppChoice(self.frame,textvariable=var,values=choices,state='readonly') if choices else ttk.Entry(self.frame,textvariable=var)
            widget.grid(row=row,column=1,sticky='ew',pady=3)
            self.fields[name]=widget
        self.heading_button=ttk.Button(self.frame,text='카메라 각도 가져오기',command=self.capture_heading)
        self.heading_button.grid(row=14,column=0,columnspan=2,sticky='w',pady=5)
        self.status=tk.StringVar(master=self.frame)
        ttk.Label(self.frame,text='직사각형은 외곽, 도넛형은 외곽과 내부 구멍을 확인합니다.\n도넛형에 STL을 등록하면 구멍 배치를 검출에 참고합니다.\n감지 범위는 이 지그를 선택한 뒤 카메라 영상에서 끌어 지정하세요.',wraplength=650).grid(row=15,column=0,columnspan=2,sticky='ew',pady=10)
        ttk.Label(self.frame,textvariable=self.status,wraplength=650,style='Muted.TLabel').grid(row=16,column=0,columnspan=2,sticky='ew',pady=6)
        actions=ttk.Frame(view.pages[1],padding=(10,6));actions.grid(row=1,column=0,columnspan=2,sticky='ew')
        self.save_button=ttk.Button(actions,text='지그 저장',command=self.save,style='Primary.TButton')
        self.save_button.pack(side='left',padx=(0,8))
        ttk.Button(actions,text='이 지그로 티칭',command=self.use).pack(side='left')
        from jig_stl_panel import JigSTLPanel
        self.stl=JigSTLPanel(self,self.frame)
        self.refresh();self.new()
    def refresh(self):
        available=profiles(self.view.station);self.keys=sorted(available,key=lambda k:available[k]['jig_name'])
        self.list['values']=[available[k]['jig_name'] for k in self.keys]
        if self.key in self.keys:self.list.current(self.keys.index(self.key))
        else:self.list.set('새 지그')
        refresh_picker(self.app)
    def fill(self,p):
        shapes={'rectangle':'직사각형','ring':'도넛형 (내부 구멍 있음)'}
        methods={'hybrid':'색상+윤곽 (기본)','edge':'윤곽 중심','white':'흰색 중심'}
        for key,var in self.vars.items():
            value=p[key]
            if key=='shape_model':value=shapes[value]
            if key=='detection_method':value=methods[value]
            if key=='appearance':value='흰색 프레임' if value=='white_frame' else '명암 외곽'
            var.set(str(value))
        if hasattr(self,'stl'):self.stl.load(p.get('preview_model'))
    def new(self):
        from jig_detection import PROFILE_DEFAULTS
        self.key=None;self.original={'mode':'rectangle',**PROFILE_DEFAULTS,'product':'공통 지그','shape_model':'rectangle','jig_name':'새 지그'}
        self.fill(self.original);self.list.set('새 지그');self.status.set('이름과 실물 치수를 입력하세요. 기존 지그·스텝은 변경하지 않습니다.')
    def copy(self):
        self.key=None;self.vars['jig_name'].set(self.vars['jig_name'].get()+' 복사');self.list.set('새 지그');self.status.set('복사본을 새 이름으로 저장하세요.')
    def load(self,_=None):
        index=self.list.current()
        if not 0<=index<len(self.keys):return
        self.key=self.keys[index];self.original=deepcopy(profiles(self.view.station)[self.key]);self.fill(self.original)
        self.status.set('검출 설정을 편집할 수 있습니다. 기존 에피소드의 치수·티칭 기준은 자동 변경하지 않습니다.')
    def read(self):
        p={**self.original,**{key:var.get() for key,var in self.vars.items()}}
        p['shape_model']={'직사각형':'rectangle','도넛형 (내부 구멍 있음)':'ring'}[p['shape_model']]
        p['appearance']='white_frame' if p['appearance']=='흰색 프레임' else 'contrast'
        p['detection_method']={'색상+윤곽 (기본)':'hybrid','윤곽 중심':'edge','흰색 중심':'white'}[p['detection_method']]
        model=self.stl.read()
        if model is not None:
            from jig_stl import detection_height
            p['preview_model']=model;p['surface_z_mm']=detection_height(model)
            p['hole_layout']=deepcopy(model.get('hole_layout',[]))
        else:
            p.pop('preview_model',None)
            if self.original.get('preview_model'):p.pop('hole_layout',None)
        return p
    def capture_heading(self):
        """Acquire the catalog draft without modifying episode/reference data."""
        from vision_teaching import reference_key,digest
        from jig_detection import validate_profile
        view=self.view
        try:
            view.idle()
            if view.reference_acquisition.pending:raise ValueError('현재 지그 측정이 끝난 뒤 다시 누르세요.')
            target=teaching_target(validate_profile(self.read()))
            from vision_teaching import pixels_on_plane
            size=list(view.frame.shape[1::-1]) if view.frame is not None else [view.station['camera']['width'],view.station['camera']['height']]
            try:pixels_on_plane([[size[0]/2,size[1]/2]],target['surface_z_mm'],view.station,size)
            except ValueError as exc:
                message='각도를 적용하지 못했습니다 · '+str(exc)
                self.status.set(message);view.status.set(message);self.app.note(message)
                view.select_tab(0);return
            before=digest(self.read());selected=self.key;choice=view.detector_choice.get()
            auto_pending=view.teaching_auto_pending
            def restore():
                view.detector_choice.set(choice);view.teaching_auto_pending=auto_pending
                view.reset_detection();view.select_tab(1)
            def done(observations):
                restore()
                if self.key!=selected or digest(self.read())!=before:
                    self.status.set('지그 설정이 바뀌어 측정 각도를 적용하지 않았습니다.');return
                yaw=observations[reference_key(target)]['yaw_deg']
                self.vars['expected_yaw_deg'].set(f'{yaw:.2f}')
                message=f'도킹 기준 방향 {yaw:.2f}° 입력됨 · 아직 저장 전입니다. 지그 저장을 누르세요.'
                self.status.set(message);view.status.set(message)
                self.fields['expected_yaw_deg'].focus_set()
            def cancelled(reason):
                restore();self.status.set(reason+' · 기존 도킹 기준 방향을 유지합니다.')
            view.teaching_auto_pending=False
            try:
                view.reference_acquisition.begin({'targets':{'heading':target}},done,heading_only=True,on_cancel=cancelled)
            except Exception:
                view.teaching_auto_pending=auto_pending;raise
        except Exception as exc:self.status.set(str(exc))

    def save(self):
        try:
            self.view.idle()
            if self.app.preview_only:raise ValueError('현재 로봇의 에피소드에서 지그를 등록하세요.')
            from teach_left_arm import atomic_json
            station,key=register(self.view.station,self.read(),self.key)
            atomic_json(self.view.profile_path,station);self.view.station=station;self.key=key
            self.original=deepcopy(station['jigs'][key]);self.refresh();self.view.refresh_target_list()
            changed=self.view.apply_saved_detection_settings()
            if changed:self.view.load_target()
            self.status.set('저장됨 · '+self.original['jig_name']+f' · 도킹 기준 방향 {self.original["expected_yaw_deg"]:.2f}°'+(' · 현재 에피소드 검출에 반영했습니다.' if changed else ''))
            return True
        except Exception as exc:self.status.set(str(exc));return False
    def use(self):
        if not self.save():return
        choose(self.app,self.key);self.app.navigation.home()
