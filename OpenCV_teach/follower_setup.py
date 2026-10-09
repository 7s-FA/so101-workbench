"""Register a real follower and import its existing calibration; never opens USB."""
from app_choices import AppChoice

import json
from pathlib import Path
import re


def register_follower(profile, port, calibration_file, name='opencv_follower_right'):
    from teaching_profile import load_profile
    from teach_left_arm import JOINTS, atomic_json
    import tempfile
    profile=Path(profile).resolve()
    source=Path(calibration_file).expanduser().resolve(strict=True)
    data=json.loads(source.read_text())
    if not isinstance(data,dict) or set(data)!=set(JOINTS):
        raise ValueError('SO-101의 몸체 5관절과 그리퍼 보정 JSON을 선택하세요.')
    for i,joint in enumerate(JOINTS,1):
        entry=data[joint]
        if any(type(entry.get(k)) is not int for k in ('id','drive_mode','homing_offset','range_min','range_max')):
            raise ValueError(f'{joint}: 정수 보정값이 필요합니다.')
        if entry['id']!=i or not 0<=entry['range_min']<entry['range_max']<=4095 or entry['drive_mode'] not in (0,1):
            raise ValueError(f'{joint}: 모터 ID·범위·방향을 확인하세요.')
        if not -2047<=entry['homing_offset']<=2047:
            raise ValueError(f'{joint}: 영점 보정 범위가 올바르지 않습니다.')
    ident=source.stem
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',ident):
        raise ValueError('보정 JSON 파일명은 영문·숫자·밑줄·하이픈을 사용하세요.')
    document=dict(schema=1,robot_type='so101',mode='follower_only',pair_name=name,
                  side='right' if ident.endswith('_right') else 'left',follower_id=ident,follower_port=port)
    with tempfile.TemporaryDirectory() as tmp:
        check=Path(tmp)/'profile.json';check.write_text(json.dumps(document));load_profile(check)
    destination=profile.parent/'calibration'/name/(ident+'.json')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Preserve the provided calibration bytes, not a fabricated/renormalized copy.
    if source!=destination:destination.write_bytes(source.read_bytes())
    atomic_json(profile,document)
    return profile


def choose_follower_profile(profile, error=''):
    import tkinter as tk
    from tkinter import ttk
    from app_choices import askopenfilename
    from teaching_ui import configure_ui
    root=tk.Tk();configure_ui(root);root.title('OpenCV 티칭 · 실물 팔로워 설정');root.geometry('760x430')
    # Keep this entry screen independent from the motion controller.
    frame=ttk.Frame(root,padding=20);frame.pack(fill='both',expand=True);frame.columnconfigure(1,weight=1)
    ttk.Label(frame,text='팔로워 단독 모드 · 실물 장치 등록',font=('',15,'bold')).grid(row=0,column=0,columnspan=2,sticky='w',pady=(0,18))
    ttk.Label(frame,text='리더암 없이 사용할 SO-101 팔로워 포트와 기존 보정 JSON을 선택하세요.\n등록만으로 모터를 연결하거나 움직이지 않습니다.',wraplength=700).grid(row=1,column=0,columnspan=2,sticky='w',pady=(0,16))
    ports=sorted(str(p) for p in Path('/dev/serial/by-id').glob('*'))
    port=tk.StringVar(value=ports[0] if len(ports)==1 else '')
    source=tk.StringVar();name=tk.StringVar(value='opencv_follower_right');status=tk.StringVar(value=error)
    for row,(label,var) in enumerate((('팔로워 USB 포트',port),('보정 JSON 파일',source),('로봇 프로필 이름',name)),2):
        ttk.Label(frame,text=label).grid(row=row,column=0,sticky='w',padx=(0,12),pady=6)
        entry=AppChoice(frame,textvariable=var,values=ports) if row==2 else ttk.Entry(frame,textvariable=var)
        entry.grid(row=row,column=1,sticky='ew',pady=6)
    ttk.Button(frame,text='보정 JSON 선택',command=lambda:source.set(askopenfilename(parent=root,filetypes=[('SO-101 보정','*.json')]) or source.get())).grid(row=5,column=1,sticky='e',pady=8)
    result=[]
    def save():
        try:
            selected=register_follower(profile,port.get().strip(),source.get().strip(),name.get().strip())
            from teach_left_arm import Hardware
            Hardware(selected,follower_only=True)
        except Exception as exc:status.set(str(exc));return
        result.append(selected);root.destroy()
    ttk.Button(frame,text='등록하고 실물 GUI 열기',command=save).grid(row=6,column=0,columnspan=2,sticky='ew',pady=8)
    ttk.Label(frame,textvariable=status,wraplength=700).grid(row=7,column=0,columnspan=2,sticky='w',pady=8)
    root.mainloop()
    return result[0] if result else None
