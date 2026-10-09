"""STL registration only. Jig display requires a confirmed camera reference."""

class JigSTLPanel:
    def __init__(self,catalog,parent):
        import tkinter as tk
        from tkinter import ttk
        from app_choices import AppChoice
        self.catalog=catalog;self.view=catalog.view;self.app=catalog.app
        self.model=None;self.generation=0
        box=ttk.LabelFrame(parent,text='STL 형상',padding=8)
        box.grid(row=4,column=0,columnspan=2,sticky='ew',pady=(2,8));box.columnconfigure(1,weight=1)
        self.frame=box;self.unit=tk.StringVar(master=box,value='mm');self.label=tk.StringVar(master=box,value='STL 없음')
        bar=ttk.Frame(box);bar.grid(row=0,column=0,columnspan=3,sticky='ew')
        ttk.Button(bar,text='STL 파일 선택',command=self.pick).pack(side='left')
        AppChoice(bar,textvariable=self.unit,values=('mm','cm','m'),state='readonly',width=4).pack(side='left',padx=5)
        ttk.Button(bar,text='STL 해제',command=lambda:self.load(None)).pack(side='left')
        ttk.Label(box,textvariable=self.label,wraplength=540).grid(row=1,column=0,columnspan=3,sticky='ew',pady=5)
        ttk.Button(box,text='STL 외곽 크기 적용',command=self.use_dimensions).grid(row=2,column=0,columnspan=3,sticky='w',pady=5)
        self.status=tk.StringVar(master=box,value='지그 위치·각도를 감지한 뒤에만 시뮬에 표시합니다.')
        ttk.Label(box,textvariable=self.status,wraplength=540,style='Muted.TLabel').grid(row=3,column=0,columnspan=3,sticky='ew')
        ttk.Label(box,text='검출면 높이는 STL 바깥 테두리의 윗면에서 자동 계산합니다.\n파일 등록만으로는 지그를 시뮬에 배치하지 않습니다.',wraplength=540,style='Muted.TLabel').grid(row=4,column=0,columnspan=3,sticky='ew',pady=(4,0))

    def load(self,model):
        from jig_stl import with_rim_metadata,detection_height
        self.generation+=1
        self.model=with_rim_metadata(model) if model is not None else None
        field=self.catalog.fields.get('surface_z_mm')
        if field:field.configure(state='readonly' if self.model else 'normal')
        if self.model:
            d=self.model['dimensions_mm'];self.label.set(self.model['source_name']+' · %.1f × %.1f × %.1f mm'%tuple(d))
            self.unit.set(self.model.get('unit','mm'))
            self.catalog.vars['surface_z_mm'].set(f'{detection_height(self.model):g}')
            self.status.set('STL 외곽 윗면 %.1fmm + 설치 바닥 Z %.1fmm = 검출면 Z %.1fmm'%(self.model['rim_height_mm'],self.model['bottom_z_mm'],detection_height(self.model)))
        else:
            self.label.set('STL 없음');self.status.set('지그 위치·각도를 감지한 뒤에만 시뮬에 표시합니다.')

    def read(self):
        from jig_stl import with_rim_metadata
        return with_rim_metadata(self.model) if self.model is not None else None

    def pick(self):
        from app_choices import askopenfilename
        from jig_stl import import_stl
        try:self.view.idle()
        except Exception as exc:self.status.set(str(exc));return
        path=askopenfilename(parent=self.frame,initialdir=str(__import__('pathlib').Path.home()/'Downloads'),filetypes=[('STL 모델','*.stl')])
        if not path:return
        generation=self.generation;unit=self.unit.get()
        def done(model):
            if generation!=self.generation:
                self.status.set('선택한 지그가 바뀌어 STL 가져오기를 적용하지 않았습니다.');return
            self.load(model)
            self.view.status.set('STL 가져오기 완료 · 외곽 검출면 높이를 계산했습니다. 지그 저장을 누르세요.')
        self.view.run_job(lambda:import_stl(path,unit),done,error_prefix='STL 가져오기 실패: ')

    def use_dimensions(self):
        try:
            self.view.idle();model=self.read()
            if model is None:raise ValueError('STL을 먼저 선택하세요.')
            d=model['dimensions_mm']
            from jig_detection import validate_profile
            validate_profile({**self.catalog.read(),'long_mm':d[0],'short_mm':d[1]})
            self.catalog.vars['long_mm'].set(f'{d[0]:g}');self.catalog.vars['short_mm'].set(f'{d[1]:g}')
            self.status.set('STL의 긴 변·짧은 변을 입력했습니다. 외곽 검출면 높이는 자동 계산됩니다. 지그 저장을 누르세요.')
        except Exception as exc:self.status.set(str(exc))

