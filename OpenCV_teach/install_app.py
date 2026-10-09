"""Register the existing il-backed GUI in the current user's application menu."""
import os
from pathlib import Path
import shutil
import subprocess
import sys


def quote(value):
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$') + '"'


def main():
    app = Path(__file__).resolve().parent
    applications = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'applications'
    applications.mkdir(parents=True, exist_ok=True)
    entry = applications / 'opencv-teach.desktop'
    entry.write_text('\n'.join([
        '[Desktop Entry]', 'Type=Application', 'Name=OpenCV 티칭',
        'Comment=팔로워 스텝 저장 · 상단 카메라 지그 감지',
        f'Exec={quote(sys.executable)} {quote(app / "desktop_launch.py")}',
        f'Path={app}', f'Icon={app / "assets/opencv-teach.svg"}',
        'Terminal=false', 'Categories=Science;Engineering;',
        'StartupWMClass=Opencv-teach', 'Keywords=opencv;robot;teaching;로봇;티칭;', ''
    ]), encoding='utf-8')
    if shutil.which('update-desktop-database'):
        subprocess.run(['update-desktop-database', str(applications)], check=True)
    print(f'앱 메뉴에 OpenCV 티칭 등록: {entry}')


if __name__ == '__main__':
    main()
