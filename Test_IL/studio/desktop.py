"""Address-bar-free local window; hardware remains owned by the existing server."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import signal
import urllib.request


def browser_command(url, profile):
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('화면을 열 수 없습니다. 웹 실행은 --web 옵션을 사용하세요.')
    browser = next((p for name in ('google-chrome', 'chromium', 'chromium-browser')
                    if (p := shutil.which(name))), None)
    if not browser:
        raise RuntimeError('Chrome 또는 Chromium이 필요합니다. 웹 실행은 --web 옵션을 사용하세요.')
    return [browser, f'--app={url}', f'--user-data-dir={Path(profile).resolve()}',
            '--no-first-run', '--no-default-browser-check', '--disable-background-mode',
            '--disable-session-crashed-bubble', '--window-size=1600,1000',
            '--class=so101-imitation-learning']


def server_ready(url, instance):
    try:
        with urllib.request.urlopen(url + '/api/health', timeout=.5) as response:
            return json.load(response).get('instance') == instance
    except (OSError, ValueError):
        return False


class DesktopWindow:
    def __init__(self, url, profile, log_path, instance):
        self.command = browser_command(url, profile)
        self.url, self.instance = url, instance
        self.log_path = Path(log_path)
        self.process = None
        self.log = None

    async def watch(self, on_exit):
        try:
            # Open only after this server (not an unrelated occupant) is listening.
            for _ in range(200):
                if await asyncio.to_thread(server_ready, self.url, self.instance):
                    break
                await asyncio.sleep(.1)
            else:
                raise RuntimeError('프로그램 시작 시간이 초과되었습니다.')
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = self.log_path.open('w')
            self.process = await asyncio.create_subprocess_exec(
                *self.command, stdout=self.log, stderr=self.log)
            code = await self.process.wait()
            on_exit('프로그램 창 닫힘' if code == 0 else f'프로그램 창 종료 ({code})')
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f'DESKTOP: {error}', flush=True)
            on_exit(str(error))

    async def close(self):
        if self.process and self.process.returncode is None:
            try:
                self.process.send_signal(signal.SIGTERM)
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            except ProcessLookupError:
                pass
        if self.log:
            self.log.close()
