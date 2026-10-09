import asyncio
import json
from unittest.mock import patch

from studio.desktop import DesktopWindow, browser_command, server_ready
import pytest


def test_desktop_uses_separate_profile_and_keeps_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.setattr('studio.desktop.shutil.which', lambda _: '/usr/bin/google-chrome')
    cmd = browser_command('http://127.0.0.1:8779', tmp_path / 'profile with spaces')
    assert '--app=http://127.0.0.1:8779' in cmd
    assert f'--user-data-dir={tmp_path}/profile with spaces' in cmd
    assert '--disable-background-mode' in cmd
    assert '--class=so101-imitation-learning' in cmd
    assert '--no-sandbox' not in cmd
    assert not any('remote-debugging' in arg for arg in cmd)


def test_desktop_requires_display_or_explicit_web_mode(monkeypatch):
    monkeypatch.delenv('DISPLAY', raising=False)
    monkeypatch.delenv('WAYLAND_DISPLAY', raising=False)
    with pytest.raises(RuntimeError, match='--web'):
        browser_command('http://127.0.0.1:8768', '/tmp/profile')


def test_ready_rejects_another_server():
    import io
    with patch('studio.desktop.urllib.request.urlopen', return_value=io.BytesIO(json.dumps({'instance': 'other'}).encode())):
        assert not server_ready('http://127.0.0.1:8768', 'ours')


def test_window_close_requests_owner_shutdown(tmp_path, monkeypatch):
    events = []
    class Process:
        returncode = 0
        async def wait(self): return 0
    async def spawn(*args, **kwargs): return Process()
    monkeypatch.setattr('studio.desktop.browser_command', lambda *args: ['browser'])
    monkeypatch.setattr('studio.desktop.server_ready', lambda *args: True)
    monkeypatch.setattr('studio.desktop.asyncio.create_subprocess_exec', spawn)
    async def run():
        window = DesktopWindow('http://127.0.0.1:8779', tmp_path, tmp_path/'log', 'ours')
        await window.watch(events.append)
        await window.close()
    asyncio.run(run())
    assert events == ['프로그램 창 닫힘']


def test_browser_failure_does_not_leave_server_waiting(tmp_path, monkeypatch):
    events = []
    async def spawn(*args, **kwargs): raise OSError('launch failed')
    monkeypatch.setattr('studio.desktop.browser_command', lambda *args: ['browser'])
    monkeypatch.setattr('studio.desktop.server_ready', lambda *args: True)
    monkeypatch.setattr('studio.desktop.asyncio.create_subprocess_exec', spawn)
    async def run():
        window = DesktopWindow('http://127.0.0.1:8779', tmp_path, tmp_path/'log', 'ours')
        await window.watch(events.append)
        await window.close()
    asyncio.run(run())
    assert events == ['launch failed']
