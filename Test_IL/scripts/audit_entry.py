"""Validate the current SO-101 entry coordinates; the scene is a display only."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from studio.hardware import audit_entry
if __name__=='__main__':
    request=json.load(sys.stdin)
    print(json.dumps(audit_entry(request['start'],request['goal'])))
