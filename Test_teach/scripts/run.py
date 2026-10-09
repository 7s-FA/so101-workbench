#!/usr/bin/env python3
"""Launcher outside Isaac: stays responsive when a renderer call hangs."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from teach_io.supervisor import supervise

if __name__=='__main__':
    arguments=sys.argv[1:]
    legacy='--legacy-omx' in arguments
    if legacy:arguments.remove('--legacy-omx')
    sys.exit(supervise(['/path/to/isaac/env_isaaclab/bin/python',
        str(ROOT/'scripts'/('sim.py' if legacy else 'so101_sim.py')),*arguments]))
