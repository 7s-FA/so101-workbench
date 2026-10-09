"""Read-only STS3215 telemetry aggregation; never synthesizes missing voltage."""
import collections
import math
import time

NAMES=['베이스','어깨','팔꿈치','손목 굽힘','손목 회전','그리퍼']

class VoltageState:
    def __init__(self, clock=time.monotonic):
        self.clock=clock;self.start=clock();self.events=collections.deque(maxlen=30)
        self.motors={i:{'id':i,'name':name,'voltage':None,'minimum':None,'maximum':None,
            'lower':None,'upper':None,'last_valid':None,'temperature':None,'load_raw':None,'error':0,
            'error_samples':0,'fault_events':0,'comm_errors':0,'samples':0,
            'history':collections.deque(maxlen=8000),'fault_active':False} for i,name in enumerate(NAMES,1)}

    def limits(self,i,lower,upper):
        self.motors[i]['lower']=lower;self.motors[i]['upper']=upper

    def update(self,i,data,comm,error):
        m=self.motors[i];t=self.clock()-self.start
        valid=comm==0 and len(data)==15
        if not valid:
            m['voltage']=None;m['temperature']=None;m['load_raw']=None;m['comm_errors']+=1
            m['history'].append([round(t,3),None])
            return
        v=round(data[6]*.1,1);status=data[9] | error
        # STS Present_Load at 60:61, sign-magnitude bit 10 (vendor LeRobot table).
        # Keep native units: this is not a calibrated force or torque measurement.
        load=data[4] | data[5]<<8
        m['load_raw']=-(load&1023) if load&1024 else load&1023
        m['voltage']=v;m['temperature']=data[7];m['error']=status;m['last_valid']=t;m['samples']+=1
        m['minimum']=v if m['minimum'] is None else min(v,m['minimum'])
        m['maximum']=v if m['maximum'] is None else max(v,m['maximum'])
        outside=(m['lower'] is not None and v<m['lower']) or (m['upper'] is not None and v>m['upper'])
        fault=bool(status or outside)
        if fault:
            m['error_samples']+=1
            if not m['fault_active']:
                m['fault_events']+=1
                self.events.appendleft({'t':round(t,3),'id':i,'voltage':v,'error':status})
        m['fault_active']=fault
        m['history'].append([round(t,3),v])

    def snapshot(self):
        t=self.clock()-self.start;motors=[]
        for m in self.motors.values():
            out={k:v for k,v in m.items() if k not in ('history','last_valid','fault_active')}
            out['age']=None if m['last_valid'] is None else t-m['last_valid']
            out['stale']=out['age'] is None or out['age']>.5 or m['voltage'] is None
            if out['stale']:
                for key in ('voltage','temperature','load_raw'):out[key]=None
            # 100 ms min/max envelopes preserve single-sample dips in the display.
            bins={}
            for stamp,v in m['history']:
                if stamp<t-120:continue
                key=int(stamp*10)
                if key not in bins:bins[key]=[key/10,v,v]
                elif v is not None:
                    b=bins[key];b[1]=v if b[1] is None else min(b[1],v);b[2]=v if b[2] is None else max(b[2],v)
            out['history']=list(bins.values());motors.append(out)
        return {'elapsed':t,'motors':motors,'events':list(self.events)}
