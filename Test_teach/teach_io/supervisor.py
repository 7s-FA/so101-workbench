"""Own one Isaac process group and bound shutdown independently of Kit/GIL.

Only an intentional exit releases the registered motor port. Unexpected crashes
retain the existing hold behavior. No process-name searches or unrelated kills.
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IO_PYTHON=ROOT.parent/'Sim_real/.venv-io/bin/python'


def release_port(port):
    try:
        result=subprocess.run([str(IO_PYTHON),str(ROOT/'scripts/hardware.py'),
            'release','--port',port],input='SUPPORTED\n',text=True,
            capture_output=True,timeout=5,start_new_session=True)
        if result.returncode:raise RuntimeError(result.stderr or result.stdout)
        print('EXIT: all six motor torques OFF (launcher verified)',flush=True)
        return True
    except Exception as error:
        print(f'EXIT TORQUE OFF NOT CONFIRMED: {error}\n'
              'Support the arm and run ./Test_teach/torque_off.sh.',flush=True)
        return False


def supervise(command, *, grace=4., release=release_port):
    read_fd,write_fd=os.pipe2(os.O_NONBLOCK|os.O_CLOEXEC)
    env=dict(os.environ,TEST_TEACH_LIFECYCLE_FD=str(write_fd))
    process=None;old_handlers={};buffer=b''
    state=dict(requested=False,intentional=False,deadline=None,port=None,released=False,signal=None)

    def request(reason, intentional=True):
        state['intentional']|=intentional
        if not state['requested']:
            state.update(requested=True,deadline=time.monotonic()+grace)
            print(f'EXIT: {reason}; stopping Test_teach...',flush=True)

    def on_signal(number, _frame):
        already=state['requested']
        request(signal.Signals(number).name)
        state['signal']=number
        if already:state['deadline']=time.monotonic()
        elif process is not None and process.poll() is None:
            try:process.send_signal(signal.SIGINT)
            except ProcessLookupError:pass

    def drain():
        nonlocal buffer
        while True:
            try:data=os.read(read_fd,4096)
            except BlockingIOError:break
            if not data:break
            buffer+=data
            while b'\n' in buffer:
                line,buffer=buffer.split(b'\n',1)
                try:message=json.loads(line)
                except ValueError:continue
                event=message.get('event')
                if event=='motor_port':state.update(port=message['port'],released=False)
                elif event=='released' and message.get('port')==state['port']:state['released']=True
                elif event=='exit':request(message.get('reason','window close'))
                elif event=='cleanup':request('cleanup',intentional=message.get('intentional',False))

    def stop_group():
        # New session => every member belongs to this launch, including orphaned
        # camera/USB children. SIGKILL avoids Kit's recursive SIGTERM shutdown.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            print('EXIT: OS has not reaped Isaac yet; a driver may be blocked.',flush=True)

    try:
        for number in (signal.SIGINT,signal.SIGTERM):
            old_handlers[number]=signal.signal(number,on_signal)
        process=subprocess.Popen(command,env=env,pass_fds=(write_fd,),start_new_session=True)
        os.close(write_fd);write_fd=None
        if state['requested']:process.send_signal(signal.SIGINT)
        forced=False
        while True:
            drain()
            if process.poll() is not None:break
            if state['deadline'] is not None and time.monotonic()>=state['deadline']:
                forced=True
                print('EXIT: cleanup deadline reached; removing this launch only.',flush=True)
                stop_group();break
            time.sleep(.05)
        drain()
        # Also remove children if Kit fast-exited before its Python finally block.
        stop_group()
        release_failed=False
        if state['intentional'] and state['port'] and not state['released']:
            release_failed=not release(state['port'])
        if release_failed:return 2
        if state['signal']:return 128+state['signal']
        if forced:return 0 if state['intentional'] else 1
        return process.returncode if process.returncode is not None else 1
    finally:
        if process is not None and process.poll() is None:stop_group()
        os.close(read_fd)
        if write_fd is not None:os.close(write_fd)
        for number,handler in old_handlers.items():signal.signal(number,handler)
