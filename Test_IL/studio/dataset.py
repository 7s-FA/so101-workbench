"""Background LeRobot writer. Raw recordings remain intact if encoding fails."""
import fcntl
import json
import subprocess
import threading
from .config import ROOT, IL_PYTHON

class DatasetWriter:
    def __init__(self, root):
        self.root = root
        self.thread = None
        self.process = None
        self.lock = threading.Lock()
        self.stopping = False
        self.again = False
        self.status = {'state':'idle','message':'학습 파일은 기록을 저장하면 함께 만듭니다.'}
        if not self.exports_current():
            self.status = {'state':'stale','message':'저장된 기록 중 학습 파일에 반영되지 않은 내용이 있습니다. 다시 준비합니다.'}
        elif any((self.root/'tasks').glob('*/lerobot/studio_manifest.json')):
            self.status = {'state':'ready','message':'LeRobot 학습 파일이 저장되어 있습니다.','path':str(self.root/'tasks')}

    def exports_current(self):
        # A previous process can exit during encoding while leaving a valid but
        # older published generation. Its existence alone does not mean ready.
        with (self.root/'.source.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH)
            try:
                for folder in (self.root/'tasks').glob('*'):
                    if not folder.is_dir():continue
                    records=[json.loads(p.read_text()) for p in folder.glob('episodes/episode_*/metadata.json')]
                    path=folder/'lerobot/studio_manifest.json'
                    if not path.exists():
                        if records:return False
                        continue
                    manifest=json.loads(path.read_text())
                    if manifest.get('task_name')!=folder.name:return False
                    saved={m['id']:(m.get('result'),m.get('included'),m.get('fingerprint')) for m in records}
                    exported={m['studio_id']:(m.get('result'),m.get('included'),m.get('source_fingerprint'))
                              for m in manifest['episodes']}
                    if saved!=exported:return False
            except (OSError,ValueError,KeyError,TypeError):
                return False
        return True

    def request(self):
        with self.lock:
            if self.stopping:return
            if self.thread and self.thread.is_alive():
                self.again = True
                return
            self.status = {'state':'writing','message':'LeRobot 학습 파일을 만드는 중입니다. 기록은 계속 삭제할 수 있습니다.'}
            self.thread = threading.Thread(target=self.run,daemon=True)
            self.thread.start()

    def run(self):
        while True:
            try:
                with self.lock:
                    if self.stopping:return
                    self.process = subprocess.Popen([str(IL_PYTHON),str(ROOT/'scripts/write_dataset.py'),
                        '--data-root',str(self.root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    process = self.process
                stdout,stderr = process.communicate()
                if process.returncode:
                    raise RuntimeError((stderr or stdout)[-1200:])
                exported=json.loads(stdout.strip().splitlines()[-1]) if stdout.strip() else {}
                if exported.get('stale'):
                    with self.lock:self.again=True
                empty=exported.get('episodes')==0
                self.status = {'state':'writing' if exported.get('stale') else 'ready','message':'삭제·추가된 기록을 반영하고 있습니다. 기록은 계속 삭제할 수 있습니다.' if exported.get('stale') else '저장한 기록이 없어 현재 학습 파일을 비웠습니다.' if empty else 'LeRobot 학습 파일 저장 완료',
                               'path':None if empty else str(self.root/'tasks'), 'cleanup':exported.get('cleanup')}
            except Exception as error:
                self.status = {'state':'error','message':'학습 파일을 만들지 못했습니다. 원본 기록은 보관되어 있습니다.','detail':str(error)}
            with self.lock:
                self.process = None
                if self.stopping or not self.again:return
                self.again = False
                self.status = {'state':'writing','message':'추가된 기록의 LeRobot 학습 파일을 만드는 중입니다.'}

    def close(self):
        with self.lock:
            self.stopping = True
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:process.kill()
        if self.thread:self.thread.join(timeout=3)
