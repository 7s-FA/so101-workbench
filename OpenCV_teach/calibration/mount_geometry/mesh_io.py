import numpy as np
from pathlib import Path

def read(p):
 data=Path(p).read_bytes();n=int.from_bytes(data[80:84],'little')
 assert 84+50*n==len(data)
 a=np.frombuffer(data,dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]),offset=84,count=n)
 return a['vertices'].astype(float),a['normal'].astype(float)
