"""Experimental universal nest for one housing, cap OR 30 mm ball at a time.

The ring supports the ball below the square parts' support plane. The four
corner stops leave the centre of each side open for the fingers. This is a
geometric proposal, not a validated production fixture.
"""
import math
import numpy as np

OUTER_SIZE=.068
SQUARE_SEAT=.008
BALL_BASE=.006+math.sqrt(.015**2-.010**2)-.015


def boxes():
    result=[('Base',(0.,0.,.001),(.068,.068,.002))]
    for sx in (-1,1):
        for sy in (-1,1):
            name=f'{sx}_{sy}'.replace('-','n')
            result.extend([
                ('Pad'+name,(sx*.023,sy*.023,.005),(.009,.009,.006)),
                ('StopX'+name,(sx*.0275,sy*.023,.009),(.003,.009,.006)),
                ('StopY'+name,(sx*.023,sy*.0275,.009),(.009,.003,.006)),
            ])
    return result


def cradle_triangles(segments=48):
    triangles=[]
    for i in range(segments):
        a,b=2*math.pi*i/segments,2*math.pi*(i+1)/segments
        v=np.array([[r*math.cos(t),r*math.sin(t),z] for z in (.002,.006) for r in (.010,.014) for t in (a,b)])
        for face in [(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]:
            triangles.extend([v[list((face[0],face[1],face[2]))],v[list((face[0],face[2],face[3]))]])
    return np.asarray(triangles)
