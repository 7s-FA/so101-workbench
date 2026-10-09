"""Desktop entry point: one app instance, persistent startup errors, il runtime."""
import fcntl
import os
from pathlib import Path
import subprocess
import tkinter as tk
from tkinter import messagebox


def notice(title, message, *, error=False):
    root = tk.Tk(className='OpenCV-teach')
    root.withdraw()
    (messagebox.showerror if error else messagebox.showinfo)(title, message, parent=root)
    root.destroy()


def main():
    app = Path(__file__).resolve().parent
    state = Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) / 'opencv-teach'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'app.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            notice('OpenCV 티칭', '이미 실행 중입니다. 작업 표시줄에서 OpenCV 티칭 창을 선택하세요.')
            return 0
        log_path = state / 'app.log'
        with log_path.open('w') as log:
            result = subprocess.run(['bash', str(app / 'run.sh')], cwd=app,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            notice('OpenCV 티칭 실행 실패', f'앱을 시작하지 못했습니다.\n실행 기록: {log_path}\n터미널에서 opencv-teach --check-only로 설정을 확인하세요.', error=True)
        return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
