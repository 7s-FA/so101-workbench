"""Native Isaac UI for measured motor -> simulated joint comparison."""
import json
import math
import time
from pathlib import Path
import omni.ui as ui
from .observation import PoseObservation
from .observe_client import Observer

TEXT_STYLE={'Label':{'font_size':18},'Button.Label':{'font_size':18}}


class ObservePanel:
    def __init__(self,port,mapping,limits,apply_pose,begin,end,camera,output):
        self.port=port;self.mapping=Path(mapping);self.limits=limits
        self.apply_pose=apply_pose;self.begin=begin;self.end=end;self.camera=camera
        self.output=Path(output);self.reader=None;self.decoder=None;self.last=None
        self.frames=0;self.error=None;self.saved_at=0.;self.rows=[]
        self.window=ui.Window('OMX pose monitor',width=900,height=630,visible=False)
        self.window.frame.style=TEXT_STYLE
        with self.window.frame:
            self.scroll=ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF)
            with self.scroll, ui.VStack(spacing=6,height=0,style=TEXT_STYLE):
                ui.Label('Real OMX -> Simulated OMX',height=32,style={'font_size':23})
                ui.Label('READ ONLY  |  No motor commands  |  Physics trial paused',height=44,word_wrap=True)
                self.status=ui.Label('Waiting',word_wrap=True,height=48)
                with ui.HStack(height=38,spacing=8):
                    ui.Button('Connect / Restart',clicked_fn=self.start)
                    ui.Button('Stop observation',clicked_fn=self.stop)
                widths=[ui.Percent(v) for v in (22,12,16,16,15,19)]
                with ui.HStack(height=30):
                    for title,width in zip(['Joint','Ticks','Real deg','Sim deg','Error deg','Status'],widths):
                        ui.Label(title,width=width,style={'font_size':15})
                for name in ['11 - Base','12 - Shoulder','13 - Elbow','14 - Wrist pitch','15 - Wrist roll','16 - Gripper']:
                    with ui.HStack(height=30):
                        labels=[]
                        for value,width in zip([name,'—','—','—','—','Waiting'],widths):
                            labels.append(ui.Label(value,width=width,style={'font_size':15}))
                        self.rows.append(labels)
                self.timing=ui.Label('Waiting for a new sample',height=28)
                with ui.CollapsableFrame('Reading details',collapsed=True,height=0):
                    with ui.VStack(spacing=6,height=0):
                        ui.Label('Error compares motor and simulated angles; it does not verify physical calibration.',word_wrap=True,height=46)
                        ui.Label('URDF LIMIT: the virtual joint stops at its model limit. Switch to Simulation only to reset the trial.',word_wrap=True,height=46)

    @property
    def active(self):return self.reader is not None

    def start(self):
        self.window.visible=True
        if self.active:return True
        self.error=None
        if not self.begin():
            self.status.text='Cannot observe while real motor control is enabled. Launch with start.sh.'
            return False
        try:
            cfg=json.loads(self.mapping.read_text());self.decoder=PoseObservation(cfg)
            self.reader=Observer(self.port,self.mapping)
            assurance=('Calibration file' if cfg.get('verified') else
                       'Single-pose link correction (provisional)' if cfg.get('link_body_calibration') else
                       'Pose mapping (unverified)')
            self.status.text='Connected | Torque OFF | '+assurance
            self.frames=0;self.started=time.monotonic();self.saved_at=0
            return True
        except Exception as e:
            self.stop();self.error=str(e);self.status.text='Connection failed: '+str(e);return False

    def stop(self):
        if self.reader:
            self.reader.close();self.reader=None
        self.end()
        self.status.text='Stopped | Last virtual pose retained | Reset before the trial'

    def poll(self):
        if not self.active:return
        if not self.window.visible:self.stop();return
        try:
            reply=self.reader.update()
            if not reply:return
            if self.last and self.last['received']==reply['received']:return
            decoded=self.decoder.decode([m['ticks'] for m in reply['sample']],self.limits)
            actual=self.apply_pose(decoded['display_angles_rad'])
            errors=[a-b for a,b in zip(actual,decoded['motor_angles_rad'])]
            self.last=dict(**decoded,sim_angles_rad=list(actual),error_rad=errors,
                received=reply['received'],read_ms=reply['read_ms'],torque=reply['torque'],
                captured_at=time.time(),mapping_file=str(self.mapping),mode='read_only_pose_observation')
            self.frames+=1
            for i,row in enumerate(self.rows):
                row[1].text=str(decoded['ticks'][i])
                row[2].text=f'{math.degrees(decoded["motor_angles_rad"][i]):.2f}'
                row[3].text=f'{math.degrees(actual[i]):.2f}'
                row[4].text=f'{math.degrees(errors[i]):+.2f}'
                row[5].text='URDF LIMIT' if decoded['limited'][i] else 'Following'
                row[5].set_style({'color':0xff69b9ff if decoded['limited'][i] else 0xffa7d58b})
            age=(time.monotonic()-reply['received'])*1000
            rate=self.frames/max(.001,time.monotonic()-self.started)
            self.timing.text=f'Received {rate:.1f} Hz  |  USB read {reply["read_ms"]:.1f} ms  |  Sample age {age:.0f} ms'
            if time.monotonic()-self.saved_at>.5:
                self.output.parent.mkdir(parents=True,exist_ok=True)
                self.output.write_text(json.dumps(self.last,ensure_ascii=False,indent=2)+'\n');self.saved_at=time.monotonic()
        except Exception as e:
            self.stop();self.error=str(e);self.status.text='Connection stopped: '+str(e)
            self.timing.text='Live updates stopped. Values below are the last received sample.'

    def shutdown(self):
        if self.reader:self.reader.close();self.reader=None
