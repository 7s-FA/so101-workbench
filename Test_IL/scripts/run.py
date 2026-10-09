#!/usr/bin/env python3
"""Use the existing bounded shutdown supervisor, including motor-release fallback."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'Test_teach')]
from teach_io.supervisor import supervise
import subprocess

def release_so101(port):
    result=subprocess.run(['/path/to/venv/bin/python','-m','studio.so101_worker','--release-only','--port',port],cwd=ROOT,timeout=10)
    return result.returncode==0
if __name__=='__main__':
    args = sys.argv[1:]
    if '--web' not in args and '--desktop' not in args:
        args = ['--desktop', *args]
    sys.exit(supervise(['/path/to/venv/bin/python','-m','studio.server',*args],grace=8.,release=release_so101))
