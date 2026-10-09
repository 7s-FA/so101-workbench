"""Offline test data only; metadata forbids any physical replay."""
import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from studio.store import Store
from studio.config import fingerprint,ROOT
import json

def make_fixture(root):
    store=Store(root)
    store.begin(dict(fingerprint=fingerprint(),synthetic=True,source='integration_test'))
    home=np.radians(json.loads((ROOT/'config/scene.json').read_text())['start_angles_deg'])
    for i in range(91):
        t=i/30
        q=home.copy();q[0]+=.15*np.sin(t)
        frames={}
        for name,color in [('gripper',(48,83,87)),('table',(72,76,84))]:
            image=Image.new('RGB',(640,480),color)
            draw=ImageDraw.Draw(image);draw.text((30,30),f'TEST FIXTURE / {name} / frame {i:03d}',fill='white',font_size=26)
            draw.rectangle((100+i*2,170,160+i*2,230),fill=(100,190,199))
            buffer=io.BytesIO();image.save(buffer,format='JPEG')
            frames[name]=dict(jpeg=buffer.getvalue(),acquired=100+t,sequence=i,shape=[480,640,3])
        store.append(dict(t=t,state=q.tolist(),action=q.tolist(),command=q.tolist()),frames)
    return store.save('Success')

if __name__=='__main__':print(json.dumps(make_fixture(sys.argv[1])))
