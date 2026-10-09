"""Shared Korean presentation for the teaching and diagnostic windows.

Serialized joint names, states, register names and controller messages stay unchanged.
"""

import re

JOINT_LABELS = {
    "shoulder_pan": "어깨 회전", "shoulder_lift": "어깨 들기", "elbow_flex": "팔꿈치",
    "wrist_flex": "손목 굽힘", "wrist_roll": "손목 회전", "gripper": "그리퍼",
}
STATE_LABELS = {
    "DISCONNECTED": "연결 안 됨", "CONNECTING": "연결 확인 중", "READY": "연결됨 · 자세 맞춤 필요",
    "SYNCING": "리더와 동기화 중", "HOLD": "자세 유지", "TEACH": "티칭 중 · 리더 따라가기",
    "MOVE": "스텝 실행 중", "PLAY": "에피소드 실행 중", "STOPPING": "정지 처리 중",
    "demo": "데모", "DEMO": "데모", "read_only": "읽기 전용", "readonly": "읽기 전용", "teaching": "티칭",
}
# Shared industrial HMI palette. Keep presentation tokens in this module.
BG, INK, MUTED, BORDER = "#dce1e4", "#202c34", "#4e606b", "#96a3ab"
PANEL, SURFACE, STRIPE = "#edf0f2", "#f9fafb", "#eaf0f3"
CHROME, CHROME_TEXT, BAND = "#26333c", "#f5f8fa", "#c7d0d6"
ACCENT, DANGER = "#00677d", "#b32632"


def joint_label(key):
    name = key.removesuffix(".pos")
    return JOINT_LABELS.get(name, name)


def unit_label(key):
    return "0–100" if key == "gripper.pos" else "°"


def state_label(state):
    return STATE_LABELS.get(state, state)


def step_correction(step, plan):
    """Describe the execution reference, never infer it from the editable name."""
    motion = step.get("motion")
    if step.get("safe_boundary"):
        if motion:
            return "확인 필요", "안전 자세에 지그 목표가 연결돼 있습니다. 스텝 구성을 확인하세요."
        return "미적용", "안전 자세 · 로봇 기준 고정\n지그가 이동·회전해도 저장된 관절값으로 이동합니다."
    if step.get("gripper_only"):
        return "집게만", "집게 전용 · 팔 위치 유지\n현재 팔 위치를 유지하고 집게 벌림만 조절합니다."
    if not motion:
        return "미적용", "일반 관절 스텝 · 로봇 기준 고정\n지그 위치·각도와 무관하게 저장된 관절값으로 이동합니다."
    target = plan.get("targets", {}).get(motion.get("target"), {})
    if target.get("mode") == "rectangle":
        return "적용 · "+target.get("jig_name","지그"), "지그를 따라가기 · "+target.get("jig_name","지그")+" · 위치·각도 보정\n매번 실행 시 자동 측정합니다(최대 3초). 티칭 기준은 별도로 유지합니다."
    if target.get("mode") == "fixed":
        return "미적용", "이 자리에 고정 · 로봇 기준 목표\n지그 위치·각도와 무관하게 저장된 목표 위치·방향으로 계산합니다."
    return "확인 필요", "목표의 위치 기준을 확인하세요. 기존 마커 목표는 지그 목표로 전환해야 합니다."


def configure_ui(root):
    import tkinter.font as font
    from tkinter import ttk
    master = root._root()
    root.configure(background=BG)
    if getattr(master, "_teaching_ui_ready", False):
        return
    master._teaching_ui_ready = True
    families = set(font.families(root))
    family = next((name for name in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic")
                   if name in families), font.nametofont("TkDefaultFont").actual("family"))
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        font.nametofont(name).configure(family=family, size=10)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=(family, 10), background=BG, foreground=INK)
    style.configure("Chrome.TFrame", background=CHROME)
    style.configure("Panel.TFrame", background=PANEL, bordercolor=BORDER, borderwidth=1, relief="solid")
    style.configure("PanelBody.TFrame", background=PANEL)
    style.configure("Band.TFrame", background=BAND)
    style.configure("Title.TLabel", font=(family, 16, "bold"), background=CHROME, foreground=CHROME_TEXT)
    style.configure("Model.TLabel", font=(family, 12, "bold"), background=ACCENT, foreground=CHROME_TEXT, padding=(12, 8))
    style.configure("Chrome.TLabel", background=CHROME, foreground="#c8d5dd")
    style.configure("Mode.TLabel", background="#d5e7ec", foreground="#134653", padding=(10, 5), font=(family, 10, "bold"))
    style.configure("Demo.TLabel", background="#f4dc9a", foreground="#624900", padding=(10, 5), font=(family, 10, "bold"))
    style.configure("Section.TLabel", font=(family, 10, "bold"), background=BAND)
    style.configure("Band.TLabel", background=BAND, foreground=MUTED)
    style.configure("Panel.TLabel", background=PANEL)
    style.configure("PanelHeading.TLabel", background=PANEL, font=(family, 11, "bold"))
    style.configure("EditorStep.TLabel", font=(family, 11, "bold"))
    style.configure("Muted.TLabel", foreground=MUTED)
    style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED)
    style.configure("Status.TLabel", font=(family, 11, "bold"), foreground=ACCENT)
    style.configure("PreviewStatus.TLabel", font=(family, 10, "bold"), foreground=ACCENT, background=PANEL)
    style.configure("TLabelframe", bordercolor=BORDER, borderwidth=1)
    style.configure("TLabelframe.Label", font=(family, 10, "bold"))
    style.configure("TButton", padding=(12, 8), bordercolor="#718895", borderwidth=2,
                    focusthickness=2, focuscolor=ACCENT, background="#ffffff", relief="raised",
                    font=(family,10,'bold'))
    style.map("TButton", background=[("disabled", "#e2e7ea"), ("pressed", "#bfd3dc"), ("active", "#e6f2f6")],
              foreground=[("disabled", "#788994")], bordercolor=[("focus", ACCENT),('active',ACCENT)],
              relief=[('pressed','sunken'),('!pressed','raised')])
    style.configure('Nav.TButton',padding=(16,9),background='#ffffff',font=(family,10,'bold'))
    style.configure('Active.Nav.TButton',background=ACCENT,foreground='#ffffff',bordercolor='#73d2e2',relief='sunken')
    style.map('Active.Nav.TButton',background=[('pressed','#004d60'),('active','#005a6e'),('!disabled',ACCENT)],
              foreground=[('!disabled','#ffffff')],bordercolor=[('!disabled','#73d2e2')],relief=[('!disabled','sunken')])
    style.configure("Compact.TButton", padding=(7, 3))
    style.configure("Compact.Primary.TButton", padding=(7, 3))
    style.configure("Episode.TEntry", padding=3)
    style.configure("Operation.TButton", padding=(10, 8), font=(family, 10, "bold"))
    for name, color, active in (("Primary.TButton", ACCENT, "#004d60"),
                                ("Stop.TButton", DANGER, "#8b1925")):
        style.configure(name, background=color, foreground="#ffffff", bordercolor=color, focuscolor="#ffffff")
        style.map(name, background=[("disabled", "#d0d9de"), ("pressed", active), ("active", active)],
                  foreground=[("disabled", "#586e7a"), ("!disabled", "#ffffff")],
                  bordercolor=[("disabled", BORDER), ("focus", INK)])
    style.configure("Run.Primary.TButton", padding=(12, 8), font=(family, 10, "bold"))
    style.configure("Operation.Stop.TButton", padding=(14, 8), font=(family, 10, "bold"))
    style.configure('Joint.Treeview.Heading',padding=(4,2))
    style.configure("Treeview", rowheight=32, background=SURFACE, fieldbackground=SURFACE,
                    bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, foreground=INK)
    style.configure("Treeview.Heading", padding=(5, 7), background="#d4dde3",
                    foreground=INK, font=(family, 10, "bold"), relief="flat", bordercolor=BORDER)
    style.map("Treeview", background=[("selected", "#c1dfe7")], foreground=[("selected", "#123e4a")])
    style.map("Treeview.Heading", background=[("active", "#c7d5dc")])
    style.configure("TEntry", padding=6, fieldbackground=SURFACE, bordercolor=BORDER)
    style.configure("EpisodeName.TEntry", padding=(2, 0), borderwidth=0, relief="flat",
                    bordercolor=BAND, lightcolor=BAND, darkcolor=BAND)
    style.map("EpisodeName.TEntry", fieldbackground=[("readonly", BAND)],
              foreground=[("readonly", INK)], bordercolor=[("focus", ACCENT)])
    style.configure("TSpinbox", padding=6, fieldbackground=SURFACE, bordercolor=BORDER, arrowsize=18)
    style.configure("Editor.TSpinbox",padding=(5,3))
    style.configure("TCombobox", padding=5, bordercolor=BORDER)
    style.map("TCombobox", fieldbackground=[("readonly", SURFACE)], selectbackground=[("readonly", SURFACE)],
              selectforeground=[("readonly", INK)])
    style.configure("TNotebook", background=BG, bordercolor=BORDER)
    style.configure("TNotebook.Tab", padding=(12, 7), background=BAND)
    style.map("TNotebook.Tab", background=[("selected", SURFACE)], foreground=[("selected", ACCENT)])
    root.tk.call("package", "require", "msgcat")
    root.tk.call("::msgcat::mclocale", "ko")
    for source, target in {"&OK": "확인", "OK": "확인", "&Cancel": "취소", "Cancel": "취소",
                           "&Yes": "예", "&No": "아니요", "Yes": "예", "No": "아니요",
                           "&Open": "열기", "Open": "열기", "&Save": "저장", "Save": "저장",
                           "Save As": "다른 이름으로 저장", "File name:": "파일 이름:",
                           "Files of type:": "파일 형식:", "Look in:": "위치:",
                           "All Files": "모든 파일", "Directory:": "폴더:", "&Directory:": "폴더:",
                           "File &name:": "파일 이름:", "File &names:": "파일 이름:",
                           "Files of &type:": "파일 형식:", "&Selection:": "선택:",
                           "Show &Hidden Files and Directories": "숨김 파일과 폴더 표시",
                           "Show &Hidden Directories": "숨김 폴더 표시",
                           'File "%1$s"  does not exist.': '파일 "%1$s"이(가) 없습니다.',
                           'Directory "%1$s" does not exist.': '폴더 "%1$s"이(가) 없습니다.',
                           'Invalid file name "%1$s".': '파일 이름 "%1$s"이(가) 올바르지 않습니다.'}.items():
        root.tk.call("::msgcat::mcset", "ko", source, target)


def ask_teaching_title(parent, initialvalue=""):
    import tkinter as tk
    from tkinter import ttk
    top=parent.winfo_toplevel()
    panel=ttk.Frame(top,padding=22,relief='solid',borderwidth=2)
    top._title_panel=panel
    host=getattr(top,'_title_editor_host',top)
    panel.place(in_=host,relx=.02,rely=.08,relwidth=.96,height=200);panel.lift()
    ttk.Label(panel,text='에피소드 이름',style='PanelHeading.TLabel').pack(anchor='w',pady=(0,12))
    value=tk.StringVar(master=top,value=initialvalue);done=tk.BooleanVar(master=top,value=False);result=[]
    entry=ttk.Entry(panel,textvariable=value);entry.pack(fill='x');entry.selection_range(0,'end')
    buttons=ttk.Frame(panel);buttons.pack(fill='x',pady=16)
    def finish(save):
        if save:result.append(value.get())
        done.set(True)
    ttk.Button(buttons,text='취소',command=lambda:finish(False)).pack(side='right',padx=(8,0))
    ttk.Button(buttons,text='저장',command=lambda:finish(True),style='Primary.TButton').pack(side='right')
    entry.bind('<Return>',lambda _:finish(True));entry.bind('<Escape>',lambda _:finish(False))
    panel.bind('<Destroy>',lambda e:done.set(True) if e.widget is panel else None)
    try:
        panel.grab_set();entry.focus_set();top.wait_variable(done)
    finally:
        try:panel.grab_release();panel.destroy()
        except tk.TclError:pass
        top._title_panel=None
    return result[0] if result else None


# Controller diagnostics remain intact in session files and stdout. Translate at the UI boundary.
MESSAGES = {
    "Connection cancelled": "연결을 취소했습니다.",
    "Already connected": "이미 연결되어 있습니다.",
    "Synchronizing to approved leader pose. Keep the leader still.": "승인한 리더 자세로 이동합니다. 리더를 움직이지 마세요.",
    "Sync complete; ready for Start teaching": "자세 맞춤 완료. 리더 따라가기로 티칭을 시작하세요.",
    "Sync target limited to follower calibration range.": "자세 맞춤 목표를 팔로워의 보정 범위로 제한했습니다.",
    "Leader moved during sync. Holding current follower pose; approve sync again.": "자세 맞춤 중 리더가 움직여 현재 자세를 유지합니다. 리더와 동기화를 다시 실행하세요.",
    "Playback complete": "에피소드의 모든 스텝 실행을 마쳤습니다.", "Step reached": "선택한 스텝 실행을 마쳤습니다.",
    "Position display is stale; wait for a fresh sample.": "관절값이 오래되었습니다. 새 측정값을 기다려 주세요.",
    "Wait for the follower to settle before saving.": "팔로워가 안정된 후 스텝을 추가하세요.",
    "Command queue full": "명령이 밀려 있습니다. 잠시 후 다시 시도하세요.",
    "Command queue full; teaching has not stopped. Try again or use STOP.": "명령이 밀려 따라가기를 멈추지 못했습니다. 다시 시도하거나 정지 버튼을 누르세요.",
    "Teaching paused. No steps to save; use Add step first.": "티칭을 멈췄습니다. 스텝 추가를 누른 후 에피소드를 저장하세요.",
    "DEMO disconnected; no hardware accessed": "데모 연결을 종료했습니다. 실물 장치는 사용하지 않았습니다.",
    "Torque OFF verified / ports closed": "토크 해제를 확인하고 포트를 닫았습니다.",
    "Ports closed after read-only checks; motor torque was not changed": "읽기 전용 확인 후 포트를 닫았습니다. 모터 토크는 변경하지 않았습니다.",
    "torque OFF not confirmed": "토크 해제를 확인하지 못했습니다. 모터 전원을 차단하세요.",
    "GUI heartbeat lost; stopping": "화면 응답이 끊겨 로봇을 정지합니다.",
    "Torque OFF attempted; collecting post-stop motor evidence (about 5s).": "토크 해제를 시도했습니다. 정지 후 모터 상태를 수집합니다(약 5초).",
    "Wrong arm, calibration, schema, or demo/live program": "로봇·보정값·파일 형식 또는 데모/실물 모드가 일치하지 않습니다.",
    "Wrong position units": "관절값의 단위가 올바르지 않습니다.",
    "Expected exactly six joint positions": "6개 관절의 값이 모두 필요합니다.",
    "Each step needs a name": "모든 스텝에 이름이 필요합니다.", "Step name too long": "스텝 이름이 너무 깁니다.",
    "Hold or stop teaching/playback before editing a step.": "자세를 유지하거나 정지한 후 스텝을 편집하세요.",
    "This step is no longer in the teaching. Select a step again.": "이 스텝이 목록에 없습니다. 스텝을 다시 선택하세요.",
    "Poses changed since preview. Keep both arms still and approve a new preview.": "확인 후 자세가 바뀌었습니다. 두 팔을 멈춘 상태에서 다시 확인하세요.",
    "Position changed since playback preview; approve again": "실행 확인 후 자세가 바뀌었습니다. 다시 확인하세요.",
    "This tool requires the verified LeRobot 0.6.1 installation": "검증된 LeRobot 0.6.1 환경이 필요합니다.",
    "Disk full": "저장 공간이 부족합니다. 공간을 확보한 후 다시 적용하세요.",
    "No alarm observed": "알람 없음", "stopped": "정지됨",
    "Unexpected model": "예상한 모터 모델이 아닙니다.",
    "Temperature difference (warning only)": "두 온도값의 차이가 큽니다(경고).",
    "Torque ON; monitor cannot switch it OFF": "토크 켜짐 · 읽기 전용 진단으로 해제할 수 없습니다.",
    "Voltage outside device limits": "전압이 장치 허용 범위를 벗어났습니다.",
}


def message_ko(message):
    text = str(message)
    if not text:
        return ""
    if text in MESSAGES:
        return MESSAGES[text]
    exception = re.fullmatch(r"(\w+(?:Error|Exception)): (.+)", text, re.DOTALL)
    if exception:
        return f"{exception[1]}: {message_ko(exception[2])}"
    for prefix, label in (("Saved: ", "초안 저장: "), ("Teaching saved: ", "에피소드 저장 완료: "),
                           ("SAVE FAILED: ", "저장 실패: "), ("Open failed: ", "파일 열기 실패: "),
                           ("Playback blocked: ", "실행할 수 없습니다: "), ("CUT MOTOR POWER: ", "모터 전원을 차단하세요: "),
                           ("Cannot save this pose: ", "이 자세를 저장할 수 없습니다: "),
                           ("Finish synchronization before saving: ", "자세 맞춤을 마친 후 저장하세요: "),
                           ("Log save failed: ", "로그 저장 실패: "), ("Telemetry close failed: ", "측정 기록 종료 실패: "),
                           ("Dashboard log save failed: ", "진단 로그 저장 실패: "),
                           ("Fault capture failed: ", "오류 상태 기록 실패: "),
                           ("Read/packet error: ", "읽기·패킷 오류: "),
                           ("Status alarm: ", "상태 알람: ")):
        if text.startswith(prefix):
            return label + message_ko(text[len(prefix):])
    match = re.fullmatch(r"Outside follower calibration range: (\w+\.pos)=([^,]+), allowed \[([^,]+), ([^]]+)\]", text)
    if match:
        key, value, low, high = match.groups()
        return f"{joint_label(key)}: 보정 범위를 벗어났습니다. {low}~{high} {unit_label(key)} 안에서 입력하세요(현재 {value})."
    match = re.fullmatch(r"Out of range: (\w+\.pos)=(.+)", text)
    if match:
        key, value = match.groups()
        low, high = (0, 100) if key == "gripper.pos" else (-180, 180)
        unit = "" if key == "gripper.pos" else "°"
        return f"{joint_label(key)}: {low}~{high}{unit} 안에서 입력하세요(현재 {value})."
    match = re.fullmatch(r"Invalid position: (.+)", text)
    if match:
        return f"{joint_label(match[1])}: 올바른 숫자를 입력하세요."
    match = re.fullmatch(r"Temperature >= (.+) raw", text)
    if match:
        return f"온도가 {match[1]} raw 이상입니다."
    match = re.fullmatch(r"Torque mismatch: expected (.+), got (.+)", text)
    if match:
        return f"토크 상태 불일치: 예상 {match[1]}, 측정 {match[2]}"
    match = re.fullmatch(r"Invalid steps \(maximum (\d+)\)", text)
    if match:
        return f"스텝 목록이 올바르지 않습니다(최대 {match[1]}개)."
    match = re.fullmatch(r"Select a teaching with 1-(\d+) steps", text)
    if match:
        return f"1~{match[1]}개 스텝이 있는 에피소드를 선택하세요."
    match = re.fullmatch(r"Calibration mismatch: (.+); no calibration was written", text)
    if match:
        return f"{match[1]} 보정값이 일치하지 않습니다. 보정값은 변경하지 않았습니다."
    match = re.fullmatch(r"(.+): no measurable forward progress for ([\d.]+)s; no automatic retry", text)
    if match:
        joints = ", ".join(joint_label(key.strip()) for key in match[1].split(","))
        return f"{joints}: {match[2]}초 동안 이동 진행이 감지되지 않아 정지했습니다. 자동 재시도하지 않습니다."
    match = re.fullmatch(r"Playing (.+) \((\d+) / (\d+)\)", text)
    if match:
        return f"{match[1]} 이동 중 ({match[2]}/{match[3]})"
    match = re.fullmatch(r"Maximum (\d+) steps\.", text)
    if match:
        return f"최대 {match[1]}개 스텝까지 저장할 수 있습니다."
    match = re.fullmatch(r"Use a 1-(\d+) character teaching title\.", text)
    if match:
        return f"에피소드 이름은 1~{match[1]}자로 입력하세요."
    match = re.fullmatch(r"Use a single-line step name of 1-(\d+) characters\.", text)
    if match:
        return f"스텝 이름은 줄바꿈 없이 1~{match[1]}자로 입력하세요."
    match = re.fullmatch(r"Leader/follower gap >([\d.]+)\. Use Sync to leader before teaching\.", text)
    if match:
        return f"리더와 팔로워의 차이가 {match[1]}을 넘습니다. 리더와 동기화를 누른 후 리더 따라가기를 켜세요."
    match = re.fullmatch(r"(?:step(\d+): wrist move|Wrist (move|sync)) >([\d.]+) degrees blocked(?:.*)", text)
    if match:
        return f"손목 회전 차이가 {match[3]}°를 넘어 이동이 차단되었습니다. 중간 스텝을 추가하거나 전원을 끈 상태에서 자세와 배선을 확인하세요."
    match = re.fullmatch(r"(SYNCING|MOVE|PLAY) timeout \((.+) s\)", text)
    if match:
        return f"{state_label(match[1])}: 제한 시간 {match[2]}초를 초과했습니다."
    if "TEMPERATURE" in text:
        text = text.replace("TEMPERATURE FAULT", "온도 이상").replace("TEMPERATURE", "온도")
        text = text.replace("DEMO", "데모").replace("follower_left raw", "팔로워 원시값")
        text = text.replace("(partial; unread=--)", "(일부 측정 · 미수신=--)")
        for name, label in JOINT_LABELS.items():
            text = text.replace(name + "=", label + "=")
        return text
    # Unknown device/OS details are retained verbatim for diagnosis.
    if re.search(r"[가-힣]", text) or text.startswith("/"):
        return text
    return "장치·파일 상태: " + text
