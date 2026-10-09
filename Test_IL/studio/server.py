import argparse
import asyncio
import json
import os
import signal
import secrets
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from .config import ROOT, TEACH, ISAAC, IL_PYTHON, load, CAMERAS, camera_devices
from .engine import Engine


def create_app(settings=None, data_root=None, *, launch_sim=True, sim_gui=False, test_mode=False, desktop=False, renderer='mujoco'):
    settings = settings or load()
    if sim_gui:renderer='isaac'
    if renderer not in ('mujoco','isaac'):raise ValueError('Unknown renderer')
    sim_image_path=ROOT/'outputs'/('sim.jpg' if renderer=='mujoco' else 'isaac.jpg')
    engine = Engine(settings,data_root)
    children = []
    logs = []
    instance = secrets.token_hex(16)
    window = None
    if desktop:
        from .desktop import DesktopWindow
        window = DesktopWindow(f"http://127.0.0.1:{settings['port']}",
                               ROOT/'outputs/desktop-profile', ROOT/'outputs/desktop.log', instance)

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        desktop_task = None
        if window:
            def window_closed(reason):
                from teach_io.lifecycle import notify
                notify('exit', reason=reason)
                os.kill(os.getpid(), signal.SIGINT)
            desktop_task = asyncio.create_task(window.watch(window_closed))
        if launch_sim:
            for p in (sim_image_path,ROOT/'outputs/sim_status.json'):
                p.unlink(missing_ok=True)
            stream = (ROOT/'outputs/sim.log').open('w')
            logs.append(stream)
            command = [str(IL_PYTHON if renderer=='mujoco' else ISAAC),str(ROOT/'scripts'/('pose_view.py' if renderer=='mujoco' else 'sim_view.py')),'--server',f"http://127.0.0.1:{settings['port']}"]
            if renderer=='mujoco':command+=['--fps',str(settings.get('preview_fps',15))]
            elif not sim_gui:command.append('--headless')
            # Delay HTTP polling until uvicorn has bound its listening socket.
            async def start_child():
                await asyncio.sleep(.5)
                failure=None
                try:
                    child = subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
                    children.append(child)
                    code = await asyncio.to_thread(child.wait)
                except Exception as error:
                    failure=str(error);code=-1
                if code == 0 and sim_gui:
                    from teach_io.lifecycle import notify
                    notify('exit',reason='Isaac window closed')
                    os.kill(os.getpid(),signal.SIGINT)
                else:
                    # The display has no authority to command or stop the robot.
                    from .store import atomic_json
                    path=ROOT/'outputs/sim_status.json'
                    previous=json.loads(path.read_text()) if path.exists() else {}
                    atomic_json(path,dict(previous,state='error',time=time.time(),renderer=renderer,
                        error=failure or previous.get('error') or '자세 화면이 종료되었습니다. outputs/sim.log를 확인하세요.'))
            task = asyncio.create_task(start_child())
        try:
            yield
        finally:
            if desktop_task:
                desktop_task.cancel()
                await asyncio.gather(desktop_task, return_exceptions=True)
            if launch_sim:
                task.cancel()
                await asyncio.gather(task,return_exceptions=True)
            for child in children:
                if child.poll() is None:child.send_signal(signal.SIGINT)
            try:
                await asyncio.to_thread(engine.close)
            finally:
                for child in children:
                    try:await asyncio.to_thread(child.wait,3)
                    except subprocess.TimeoutExpired:
                        child.kill();await asyncio.to_thread(child.wait)
                for stream in logs:stream.close()
                if window:
                    await window.close()

    app = FastAPI(lifespan=lifespan)
    app.state.engine = engine

    @app.get('/api/health')
    def health():
        return dict(app='so101-imitation-learning', instance=instance)

    @app.middleware('http')
    async def local_requests(request, call_next):
        host = request.headers.get('host','').split(':')[0]
        if host not in ('127.0.0.1','localhost','testserver'):
            return Response('Local access only',status_code=403)
        if request.method not in ('GET','HEAD'):
            origin = request.headers.get('origin')
            if origin and origin not in (f"http://127.0.0.1:{settings['port']}",f"http://localhost:{settings['port']}"):
                return Response('Cross-origin commands refused',status_code=403)
            if request.headers.get('x-studio-command') != '1':
                return Response('Missing command header',status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response

    @app.get('/')
    def index():return FileResponse(ROOT/'web/index.html')

    @app.get('/api/status')
    def status():
        value = engine.public.copy()
        path = ROOT/'outputs/sim_status.json'
        sim = json.loads(path.read_text()) if launch_sim and path.exists() else dict(state='disabled' if not launch_sim else 'loading')
        if sim.get('state')=='ready' and time.time()-sim.get('time',0)>2:sim['state']='stale'
        value.update(sim=sim,test_mode=test_mode)
        return value

    @app.get('/api/sim-state')
    def sim_state():
        value=engine.sim_state()
        return {key:value[key] for key in ('display_angles','display_source','display_robot','timestamp')}

    @app.post('/api/heartbeat')
    def heartbeat():engine.ping();return dict(ok=True)

    @app.get('/api/devices')
    def devices():
        return dict(ports=[str(p) for p in sorted(Path('/dev/serial/by-id').glob('usb-1a86_*'))],
            cameras=camera_devices(),defaults=load(),robot=__import__('studio.robot',fromlist=['profile']).profile())

    @app.get('/api/learning')
    def learning():return engine.learning.status()

    @app.get('/api/calibration')
    def calibration():
        from .calibration import catalog,GUIDE
        from .robot import profile
        return dict(files=catalog(),robot=profile(),guide=GUIDE,
                    ports=[str(p) for p in sorted(Path('/dev/serial/by-id').glob('usb-1a86_*'))])

    @app.get('/api/calibration/reference')
    def calibration_reference():
        from .robot import OCV
        return FileResponse(OCV/'assets/so101/calibration_middle.jpg')

    @app.get('/api/training/{job_id}/log')
    def training_log(job_id:str):
        if job_id not in {j['id'] for j in engine.learning.jobs()}:raise HTTPException(404)
        path=engine.learning.root/job_id/'train.log'
        if not path.exists():return dict(text='로그 준비 중')
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-12000))
            return dict(text=stream.read().decode('utf-8',errors='replace'))

    @app.get('/api/episodes')
    def episodes():return engine.store.list()

    @app.get('/api/episodes/{number}')
    def episode(number:int):
        try:
            meta,samples = engine.store.episode(number)
            return dict(metadata=meta,samples=samples)
        except (ValueError,OSError) as e:raise HTTPException(404,str(e))

    @app.post('/api/command/{op}')
    def command(op:str, payload:dict):
        if test_mode and op in ('connect','teleop','replay','model_run','register_leader','calibration_start','calibration_select','calibration_apply','calibration_command'):
            raise HTTPException(409,'Hardware commands are disabled in test mode')
        try:return engine.call(op,**payload)
        except Exception as e:raise HTTPException(409,str(e))

    @app.get('/api/image/sim')
    def sim_image():
        path = sim_image_path
        if not launch_sim or not path.exists():return Response(status_code=204)
        status_path=ROOT/'outputs/sim_status.json'
        sim=json.loads(status_path.read_text()) if status_path.exists() else {}
        if sim.get('state')!='ready' or time.time()-sim.get('time',0)>2:return Response(status_code=204)
        return Response(path.read_bytes(),media_type='image/jpeg')

    @app.get('/api/image/{name}')
    def camera_image(name:str):
        if name=='table-depth':
            if engine.mode=='preview' and engine.samples:
                depth=engine.samples[engine.play_index]['cameras']['table'].get('depth')
                if not depth:return Response(status_code=204)
                try:
                    from .depth import preview_file
                    path=engine.store.path(engine.selected)/'table_depth'/Path(depth['file']).name
                    return Response(preview_file(path,depth['scale_m_per_unit']),media_type='image/jpeg')
                except (ValueError,OSError):return Response(status_code=204)
            try:
                depth=engine.cameras['table'].snapshot()['depth']
                return Response(depth['preview_jpeg'],media_type='image/jpeg')
            except (KeyError,RuntimeError):return Response(status_code=204)
        if name not in CAMERAS:raise HTTPException(404)
        if engine.mode=='preview' and engine.samples:
            sample = engine.samples[engine.play_index]
            # Never accept a filename supplied by a request or embedded episode path.
            filename = Path(sample['cameras'][name]['file']).name
            try:
                path = engine.store.path(engine.selected)/name/filename
                return Response(path.read_bytes(),media_type='image/png' if path.suffix=='.png' else 'image/jpeg')
            except (ValueError,FileNotFoundError):
                # Selection can be deleted by the control owner during a preview fetch.
                return Response(status_code=204)
        try:return Response(engine.cameras[name].snapshot()['jpeg'],media_type='image/jpeg')
        except (KeyError,RuntimeError):return Response(status_code=204)

    app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int)
    parser.add_argument('--no-sim',action='store_true')
    parser.add_argument('--sim-gui',action='store_true')
    parser.add_argument('--renderer',choices=('mujoco','isaac'),default='mujoco')
    parser.add_argument('--test-mode',action='store_true')
    parser.add_argument('--data-dir',type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--desktop', action='store_true', help='주소창 없는 독립 창')
    mode.add_argument('--web', action='store_true', help='웹 화면만 실행')
    args = parser.parse_args()
    settings = load()
    if args.port:settings['port']=args.port
    app = create_app(settings,args.data_dir,launch_sim=not args.no_sim,sim_gui=args.sim_gui,test_mode=args.test_mode,desktop=args.desktop,renderer=args.renderer)
    print(f"SO-101 Imitation Learning: http://127.0.0.1:{settings['port']} (motor enable requires the web controls)",flush=True)
    uvicorn.run(app,host='127.0.0.1',port=settings['port'],log_level='warning',timeout_graceful_shutdown=4)


if __name__=='__main__':main()
