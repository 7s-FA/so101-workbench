"""Restore pinned upstream dependencies; never replace an existing checkout."""
import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for project in ('Sim_real','Demo'):
    for item in json.loads((root/'vendor-sources.json').read_text()):
        target=root/project/'vendor'/item['name']
        if target.exists():
            print('Keeping existing:',target)
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',item['url'],str(target)],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',item['revision']],check=True)
