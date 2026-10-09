"""Read-only motor health window; may share the teaching worker's existing bus."""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import queue
import signal
import threading
import tkinter as tk
from tkinter import ttk

from teaching_ui import configure_ui, joint_label, state_label, message_ko, SURFACE, INK, BORDER, ACCENT, MUTED

from motor_diagnostics import Collector, SETTINGS, demo_frame, make_bus, readonly_worker
from temperature_guard import pending_label

from teaching_settings import (TEMPERATURE_LIMIT, TEMPERATURE_DURATION_S, TEMPERATURE_MAX_GAP_S, TRACKING_ERROR, TRACKING_SENT_ERROR, STALL_SECONDS,
                               GUI_HEARTBEAT_TIMEOUT, SYNC_TIMEOUT, MOVE_TIMEOUT,DASHBOARD_POLL_MS, DASHBOARD_HISTORY_SIZE, DASHBOARD_EVENT_SIZE,
                               DASHBOARD_GEOMETRY, DASHBOARD_STALE_SECONDS)

ROOT = Path(__file__).resolve().parent


def fmt(value):
    return "--" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


class Dashboard:
    def __init__(self, root, collector=None, *, stop_motion=None, demo=False, managed=False):
        self.root, self.collector = root, collector
        self.embedded = collector is not None or managed
        self.poll_job = None
        self.stop_motion = stop_motion
        self.demo = demo
        self.thread = None
        self.stop_event = threading.Event()
        self.closing = False
        self.frame = None
        self.history = deque(maxlen=DASHBOARD_HISTORY_SIZE)
        self.selected = 6
        self.demo_index = 0
        self.last_event = None
        self.events = deque(maxlen=DASHBOARD_EVENT_SIZE)
        self.root.title("SO-101 | 상태 모니터링" + (" | 데모" if demo else ""))
        self.root.geometry(DASHBOARD_GEOMETRY)
        self.root.minsize(960, 660)
        configure_ui(root)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)
        top = ttk.Frame(root, style="Chrome.TFrame", padding=(14, 12))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        mode = "데모" if demo else "동작 상태 측정" if self.embedded else "읽기 전용"
        ttk.Label(top, text="상태 모니터링", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(top, text=mode, style="Demo.TLabel" if demo else "Mode.TLabel").grid(row=0, column=1, padx=12)
        self.connect_button = ttk.Button(top, text="진단 연결", command=self.connect)
        self.stop_button = ttk.Button(top, text="정지 / 토크 해제" if self.embedded else "모니터링 중지",
                                      command=stop_motion if self.embedded else self.stop_monitor, style="Stop.TButton" if self.embedded else "TButton")
        if not self.embedded and not demo:
            self.connect_button.grid(row=0, column=2, padx=4)
        if not demo:
            self.stop_button.grid(row=0, column=3, padx=4)
        self.status = tk.StringVar(value="측정 대기 · 메인 화면에서 팔로워를 연결해 주세요." if self.embedded else "연결 안 됨")
        self.status_label = tk.Label(root, textvariable=self.status, anchor="w", bg="#e9edf1",
                                    fg="#263238", padx=12, pady=8, wraplength=1100)
        self.status_label.grid(row=1, column=0, sticky="ew")
        table_frame = ttk.Frame(root, padding=(12, 10, 12, 4))
        table_frame.grid(row=2, column=0, sticky="ew")
        table_frame.columnconfigure(0, weight=1)
        columns = ("id", "joint", "temp", "single", "delta", "current", "load", "voltage", "torque", "status", "packet", "condition")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", height=6, selectmode="browse")
        titles = ("ID", "관절", "블록 온도", "단일 온도", "온도차", "전류", "부하", "전압", "토크", "상태", "패킷", "진단 결과")
        for col, title in zip(columns, titles):
            width = 140 if col == "joint" else 220 if col == "condition" else 68 if col != "id" else 36
            self.table.heading(col, text=title)
            self.table.column(col, width=width, minwidth=width, stretch=col in ("joint", "condition"))
        self.table.tag_configure("error", background="#ffe4e2", foreground="#951d25")
        self.table.tag_configure("warning", background="#fff1d0", foreground="#704300")
        self.table.tag_configure("ok", background=SURFACE, foreground=INK)
        self.table.grid(row=0, column=0, sticky="ew")
        scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(xscrollcommand=scroll.set)
        scroll.grid(row=1, column=0, sticky="ew")
        self.table.bind("<<TreeviewSelect>>", self.select)
        lower = ttk.Panedwindow(root, orient="horizontal")
        lower.grid(row=3, column=0, sticky="nsew", padx=12, pady=8)
        left = ttk.Frame(lower)
        right = ttk.Frame(lower)
        lower.add(left, weight=3)
        lower.add(right, weight=2)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(2, weight=1)
        controls = ttk.Frame(left)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.heading = tk.StringVar(value="ID6 그리퍼")
        ttk.Label(controls, textvariable=self.heading, font="TkHeadingFont").pack(side="left")
        self.metric = tk.StringVar(value="온도")
        pick = ttk.Combobox(controls, textvariable=self.metric, state="readonly", width=16,
                            values=("온도", "전압", "전류", "부하", "위치 오차"))
        pick.pack(side="right", padx=8)
        pick.bind("<<ComboboxSelected>>", lambda _: self.draw())
        self.position = tk.StringVar(value="목표 —    현재 —    오차 —")
        self.position_label = ttk.Label(left, textvariable=self.position, font="TkDefaultFont", wraplength=650)
        self.position_label.grid(row=1, column=0, sticky="w", pady=4)
        left.bind("<Configure>", lambda e: self.position_label.configure(wraplength=max(150, e.width - 16)))
        self.canvas = tk.Canvas(left, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER, height=230)
        self.canvas.grid(row=2, column=0, sticky="nsew", padx=(0, 10))
        self.canvas.bind("<Configure>", lambda _: self.draw())
        notebook = ttk.Notebook(right)
        notebook.pack(fill="both", expand=True)
        self.details = {}
        for title, label in (("Limits", "제한값"), ("Events", "이벤트"), ("Raw packet", "원시 패킷")):
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=label)
            text = tk.Text(tab, wrap="word", width=36, height=12, font="TkDefaultFont", background=SURFACE, foreground=INK, relief="flat", padx=8, pady=8)
            bar = ttk.Scrollbar(tab, orient="vertical", command=text.yview)
            text.configure(yscrollcommand=bar.set)
            bar.pack(side="right", fill="y")
            text.pack(fill="both", expand=True)
            text.configure(state="disabled")
            self.details[title] = text
        self.footer = tk.StringVar(value="온도·전류·부하·전압·장치 제한값은 원시값(raw)으로 표시합니다.")
        self.footer_label = ttk.Label(root, textvariable=self.footer, style="Chrome.TLabel", wraplength=1100, padding=(12, 8))
        self.footer_label.grid(row=4, column=0, sticky="ew")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Configure>", self.resize, add="+")
        self.poll()

    def resize(self, event):
        if event.widget == self.root:
            self.status_label.configure(wraplength=max(200, event.width - 30))
            self.footer_label.configure(wraplength=max(200, event.width - 30))

    def connect(self):
        if self.thread and self.thread.is_alive():
            return
        try:
            self.frame = None
            self.history.clear()
            self.table.delete(*self.table.get_children())
            for title in self.details:
                self.put_text(title, "")
            self.collector = Collector(make_bus(), ROOT / "teaching/motor-diagnostics")
            self.stop_event.clear()
            self.thread = threading.Thread(target=readonly_worker, args=(self.collector, self.stop_event))
            self.thread.start()
            self.connect_button.configure(state="disabled")
            self.status.set("진단 연결 중 · 읽기 전용")
        except Exception as exc:
            self.status.set(f"연결 실패: {message_ko(exc)}")
            self.status_label.configure(bg="#ffe4e2", fg="#951d25")

    def stop_monitor(self):
        self.stop_event.set()
        self.status.set("모니터링을 중지합니다. 모터 토크는 변경하지 않습니다.")

    def close(self):
        if self.embedded:
            self.root.withdraw()
            return
        self.closing = True
        self.stop_monitor()

    def dispose(self):
        """The owning application is closing; cancel only this window's timer."""
        if self.closing:
            return
        if not self.embedded:
            self.close()
            return
        self.closing = True
        if self.poll_job is not None:
            self.root.after_cancel(self.poll_job)
            self.poll_job = None
        if self.root.winfo_exists():
            self.root.destroy()

    def select(self, _=None):
        selected = self.table.selection()
        if selected:
            self.selected = int(selected[0])
            self.detail()

    def put_text(self, title, text):
        widget = self.details[title]
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.configure(state="disabled")

    def accept(self, frame):
        previous_time = self.frame.get("time") if self.frame else None
        self.frame = frame
        if frame.get("connected") and frame.get("time") != previous_time:
            self.history.append(frame)
        for row in frame.get("rows", []):
            r = row["raw"]
            first, second = r.get("Present_Temperature"), row.get("temperature_single")
            delta = first - second if first is not None and second is not None else None
            problems = row.get("issues", [])
            severity = "error" if any(p[0] == "error" for p in problems) else "warning" if problems else "ok"
            values = (row["id"], joint_label(row["name"]), fmt(first), fmt(second), fmt(delta),
                      fmt(r.get("Present_Current")), fmt(row.get("load_signed")), fmt(r.get("Present_Voltage")),
                      fmt(r.get("Torque_Enable")), fmt(r.get("Status")),
                      fmt(row.get("packets", {}).get("block_packet_error_raw")),
                      message_ko(problems[0][1]) if problems else "알람 없음")
            key = str(row["id"])
            if self.table.exists(key):
                self.table.item(key, values=values, tags=(severity,))
            else:
                self.table.insert("", "end", iid=key, values=values, tags=(severity,))
        if not self.table.selection() and self.table.exists(str(self.selected)):
            self.table.selection_set(str(self.selected))
        findings = [f"ID{r['id']} {p[1]}" for r in frame.get("rows", []) for p in r.get("issues", [])]
        event = frame.get("outcome") if not frame.get("connected") else "; ".join(findings)
        if event and event != self.last_event:
            self.events.append(f"{frame.get('event_time', frame.get('time', ''))}\n{message_ko(event)}\n")
        self.last_event = event
        self.detail()

    def detail(self):
        if not self.frame:
            return
        row = next((r for r in self.frame.get("rows", []) if r["id"] == self.selected), None)
        if row is None:
            return
        self.heading.set(f"ID{row['id']} {joint_label(row['name'])}")
        self.position.set(f"목표 {fmt(row.get('goal'))}   현재 {fmt(row.get('position'))}   "
                          f"오차 {fmt(row.get('error'))} [{row.get('unit', 'raw')}]")
        limits = ["동작 정지 조건 (참고)", f"구동 중 온도: {TEMPERATURE_LIMIT} raw 이상 연속 {TEMPERATURE_DURATION_S:g}초",
                  f"고온 감시 공백: {TEMPERATURE_MAX_GAP_S:g}초 초과 시 즉시 중단",
                  "시작 전 고온·통신 오류·모터 알람: 즉시 차단", "두 온도값의 차이: 경고만 표시",
                  f"추종: 요청 오차 > {TRACKING_ERROR}, 전송 오차 > {TRACKING_SENT_ERROR}, {STALL_SECONDS}초 동안 진행 없음",
                  f"화면 응답 제한: {GUI_HEARTBEAT_TIMEOUT}초", f"동기화 제한: {SYNC_TIMEOUT}초", f"스텝 이동 제한: {MOVE_TIMEOUT}초", "",
                  "장치 설정 (원시값·레지스터 이름)"]
        limits += [f"{key}: {fmt(row.get('settings', {}).get(key))}" for key in SETTINGS]
        limits += [f"Torque_Limit: {fmt(row['raw'].get('Torque_Limit'))}", "",
                   "부하·전류의 원시값은 장치 보호 설정값과 직접 비교할 수 없습니다."]
        failure = self.frame.get("health_failure", {})
        if failure:
            limits += ["", "마지막 상태 검사", message_ko(failure.get("error", "")),
                       str(failure.get("last_check", ""))]
        context = self.frame.get("context", {})
        tracking = context.get("tracking", {}).get(row["name"] + ".pos", {})
        if tracking:
            limits += ["", "추종 상태", *[f"{k}: {fmt(v)}" for k, v in tracking.items()]]
        self.put_text("Limits", "\n".join(limits))
        self.put_text("Events", "\n".join(reversed(self.events)))
        import json
        self.put_text("Raw packet", json.dumps({"telemetry": row.get("packets", {}),
            "final_health_check": failure.get("last_check")}, indent=2))
        self.draw()

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if width < 80 or height < 80:
            return
        data = [next((r for r in frame.get("rows", []) if r["id"] == self.selected), {}) for frame in self.history]
        metric = self.metric.get()
        if metric == "온도":
            traces = [("블록 읽기", "#087e8b", [r.get("raw", {}).get("Present_Temperature") for r in data]),
                      ("단일 읽기", "#be5816", [r.get("temperature_single") for r in data])]
        else:
            values = [r.get("raw", {}).get("Present_Current") if metric == "전류" else
                      r.get("raw", {}).get("Present_Voltage") if metric == "전압" else
                      r.get("load_signed") if metric == "부하" else r.get("error") for r in data]
            traces = [(metric, "#087e8b", values)]
        values = [v for _, _, vs in traces for v in vs if isinstance(v, (int, float))]
        low, high = min([0, *values]), max([TEMPERATURE_LIMIT + 5 if metric == "온도" else 1, *values])
        high += (high - low) * .1
        x0, x1, y0, y1 = 46, width - 16, 34, height - 28
        y = lambda v: y1 - (v - low) / (high - low) * (y1 - y0)
        for i in range(5):
            value = low + (high - low) * i / 4
            canvas.create_line(x0, y(value), x1, y(value), fill="#e5e9ed")
            canvas.create_text(x0 - 7, y(value), text=f"{value:.0f}", anchor="e", fill="#52606d")
        if metric == "온도":
            canvas.create_line(x0, y(TEMPERATURE_LIMIT), x1, y(TEMPERATURE_LIMIT), fill="#c52b38", dash=(5, 4))
            canvas.create_text(x1, y(TEMPERATURE_LIMIT) - 8, text=f"고온 기준 {TEMPERATURE_LIMIT:g} raw", anchor="e", fill="#c52b38")
        for trace_index, (name, color, vs) in enumerate(traces):
            canvas.create_text(x0 + trace_index * 100, 15, text=name, anchor="w", fill=color)
            previous = None
            for i, v in enumerate(vs):
                if v is None:
                    previous = None
                    continue
                point = (x0 + i / max(1, len(vs) - 1) * (x1 - x0), y(v))
                if previous:
                    canvas.create_line(*previous, *point, fill=color, width=2)
                previous = point
        canvas.create_text(x0, height - 10, text=f"측정 {len(data)}회", anchor="w", fill="#52606d")

    def poll(self):
        self.poll_job = None
        if self.closing and not (self.thread and self.thread.is_alive()):
            self.root.destroy()
            return
        if self.demo and not self.closing:
            if self.demo_index % 5 == 0:
                self.accept(demo_frame(self.demo_index // 5))
            self.demo_index += 1
        if self.collector:
            while True:
                try:
                    self.accept(self.collector.events.get_nowait())
                except queue.Empty:
                    break
        if self.thread and not self.thread.is_alive() and not self.closing:
            self.connect_button.configure(state="normal")
        if self.frame:
            frame = self.frame
            if frame.get("context", {}).get("loop_dt_s"):
                state_fps = 1 / frame["context"]["loop_dt_s"]
            else:
                state_fps = None
            age = (datetime.now().astimezone() - datetime.fromisoformat(frame["time"])).total_seconds() if frame.get("time") else 0
            errors = sum(any(p[0] == "error" for p in r.get("issues", [])) for r in frame.get("rows", []))
            state = frame.get("context", {}).get("state", frame.get("mode", ""))
            pending = [v for row in frame.get('rows', []) for v in row.get('temperature_pending', [])]
            if not frame.get("connected"):
                suffix = " · 토크 해제 확인" if frame.get("torque_off_verified") else ""
                self.status.set("연결 종료 · " + message_ko(frame.get("outcome", "")) + suffix)
                color = "#ffe4e2" if frame.get("cleanup_errors") or "Error" in frame.get("outcome", "") else "#e9edf1"
            elif age > DASHBOARD_STALE_SECONDS:
                self.status.set(f"측정 지연 · {age:.1f}초 동안 새 데이터 없음 · {state_label(state)}")
                color = "#fff1d0"
            else:
                self.status.set(f"{state_label(state)} · 정지 조건 해당 모터 {errors}개 · "
                                f"{age:.1f}초 전 측정 · 주기 {frame.get('cycle_ms', 0):.1f} ms" +
                                (f" · 제어 {state_fps:.1f} Hz" if state_fps else ""))
                if pending:
                    self.status.set(self.status.get()+' · '+pending_label(pending))
                color = "#ffe4e2" if errors else "#fff1d0" if pending else "#e4f2ec"
            self.status_label.configure(bg=color)
            cleanup = " · 토크 해제 미확인: 모터 전원을 차단하세요" if frame.get("cleanup_errors") else ""
            self.footer.set("원시 측정값(raw) · " + frame.get("log_path", "데모") + cleanup)
        self.poll_job = self.root.after(DASHBOARD_POLL_MS, self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Synthetic data, no serial access")
    args = parser.parse_args()
    root = tk.Tk()
    dashboard = Dashboard(root, demo=args.demo)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: dashboard.close())
    root.mainloop()
    if dashboard.thread:
        dashboard.thread.join()


if __name__ == "__main__":
    main()
