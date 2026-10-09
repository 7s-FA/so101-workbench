#!/usr/bin/env python3
"""A standalone read-only monitor for already-running camera/telemetry sources."""
import argparse
import signal
import threading
from arm_sim.monitor import ensure_monitor

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8767)
a=p.parse_args();server=ensure_monitor(a.port)
print(f'통합 모니터 http://127.0.0.1:{a.port}/ · 종료 Ctrl+C',flush=True)
if server is not None:
    stop=threading.Event()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stop.set())
    try:stop.wait()
    finally:server.close()
