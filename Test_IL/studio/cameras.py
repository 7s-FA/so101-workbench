"""One owner per camera, timestamped JPEG samples shared by preview and recorder."""
import threading
import time
import json
import subprocess
from pathlib import Path
import cv2

MAX_FRAME_AGE = .2
LOG_PATH = Path(__file__).resolve().parents[1]/'outputs/diagnostics/cameras.jsonl'

def learning_image(frame):
    """Lossless camera pixels for recording/inference; JPEG is preview only.

    Legacy/test frames without raw pixels retain their original JPEG bytes.
    Encoding runs in the disk/inference worker, never the motor control loop.
    """
    image=frame.get('bgr')
    if image is None:return '.jpg',frame['jpeg']
    if str(image.dtype)!='uint8' or len(image.shape)!=3 or image.shape[2]!=3:
        raise ValueError('학습 영상은 uint8 컬러 영상이어야 합니다.')
    ok,encoded=cv2.imencode('.png',image,[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok:raise RuntimeError('학습 영상 PNG 변환 실패')
    return '.png',encoded.tobytes()


class CameraUnavailable(RuntimeError):
    """Missing observations, not a fault in the motor transport."""


def camera_status(camera, now=None):
    if camera is None:
        return dict(connected=False,state='disconnected',error='Not connected',age=None)
    now = time.monotonic() if now is None else now
    frame = camera.frame
    age = now-frame['acquired'] if frame else None
    live = not camera.error and age is not None and age <= MAX_FRAME_AGE
    return dict(connected=live,state='live' if live else 'recovering',
                error=camera.error or (None if live else 'Camera frame is stale'),age=age,
                device=camera.device,failures=getattr(camera,'failures',0),
                reconnects=getattr(camera,'reconnects',0),
                actual=getattr(camera,'actual',{}).copy(),last_error=getattr(camera,'last_error',None))


class Camera:
    def __init__(self, device, fps=30):
        self.device = device
        self.fps = fps
        self.frame = None
        self.error = 'Connecting'
        self.failures = 0
        self.reconnects = 0
        self.last_error = None
        self.actual = {}
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def log(self, event, **detail):
        try:
            LOG_PATH.parent.mkdir(parents=True,exist_ok=True)
            with LOG_PATH.open('a') as stream:
                stream.write(json.dumps(dict(time=time.time(),device=self.device,event=event,**detail))+'\n')
        except OSError:pass  # Diagnostics must not stop capture.

    def run(self):
        sequence = 0
        attempts = 0
        while not self.stop_event.is_set():
            cap = None
            first = True
            rate_started = None
            rate_count = 0
            try:
                # Resolve the stable USB identity again after unplug/replug.
                cap = cv2.VideoCapture(str(Path(self.device).resolve()), cv2.CAP_V4L2)
                if not cap.isOpened():
                    raise RuntimeError('Cannot open camera; check USB and other programs')
                for prop,value in ((cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG')),
                                   (cv2.CAP_PROP_FRAME_WIDTH,640),(cv2.CAP_PROP_FRAME_HEIGHT,480),
                                   (cv2.CAP_PROP_FPS,self.fps),(cv2.CAP_PROP_BUFFERSIZE,1)):
                    cap.set(prop,value)
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                self.actual = dict(width=cap.get(cv2.CAP_PROP_FRAME_WIDTH),height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                   fps=cap.get(cv2.CAP_PROP_FPS),format=''.join(chr((fourcc>>8*i)&255) for i in range(4)))
                # Keep automatic exposure, but prevent it silently lowering FPS.
                # Unsupported UVC controls are diagnostic warnings, not failures.
                try:
                    fixed = subprocess.run(['v4l2-ctl','--device',str(Path(self.device).resolve()),
                        '--set-ctrl=exposure_dynamic_framerate=0','--get-ctrl=exposure_dynamic_framerate'],
                        capture_output=True,text=True,timeout=2)
                    self.actual['fixed_frame_rate'] = fixed.returncode==0 and 'exposure_dynamic_framerate: 0' in fixed.stdout
                    if not self.actual['fixed_frame_rate']:self.log('fixed_fps_unavailable',detail=(fixed.stderr or fixed.stdout)[-300:])
                except (OSError,subprocess.TimeoutExpired) as error:
                    self.actual['fixed_frame_rate'] = False
                    self.log('fixed_fps_unavailable',detail=str(error))
                while not self.stop_event.is_set():
                    tick = time.monotonic()
                    ok, image = cap.read()
                    acquired = time.monotonic()
                    if self.stop_event.is_set():break
                    if not ok or image is None:
                        raise RuntimeError('Camera frame read failed')
                    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not ok:
                        raise RuntimeError('JPEG encoding failed')
                    sequence += 1
                    self.frame = dict(jpeg=encoded.tobytes(),bgr=image.copy(),acquired=acquired,sequence=sequence,shape=list(image.shape))
                    self.error = None
                    if rate_started is None:rate_started=acquired
                    else:
                        rate_count += 1
                        if acquired-rate_started>=2:
                            self.actual['measured_fps']=round(rate_count/(acquired-rate_started),2)
                            rate_started=acquired;rate_count=0
                    if first:
                        if attempts:self.reconnects += 1
                        self.log('recovered' if attempts else 'stream_started',actual=self.actual,sequence=sequence)
                        first = False
                    if acquired-tick > MAX_FRAME_AGE:self.log('slow_read',seconds=acquired-tick,sequence=sequence)
            except Exception as error:
                self.error = str(error)
                self.last_error = dict(message=self.error,time=time.time())
                self.failures += 1
                self.log('capture_error',message=self.error,device_present=Path(self.device).exists(),sequence=sequence)
            finally:
                if cap is not None:cap.release()
            attempts += 1
            # Reopen only the camera. Never resume motion/recording automatically.
            if self.stop_event.wait(.5 if attempts==1 else 2.):break

    def snapshot(self):
        f = self.frame
        # A single bad read may recover before the last real frame expires.
        # Never invent a timestamp or serve stale imagery as a new observation.
        if f is None or time.monotonic()-f['acquired'] > MAX_FRAME_AGE:
            raise CameraUnavailable(self.error or 'Camera frame is stale')
        return f

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)


def realsense_devices():
    """Enumerate SDK identities, never choose an IR/depth V4L node as RGB."""
    try:
        import pyrealsense2 as rs
        return ['realsense:'+d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()
                if 'D435' in d.get_info(rs.camera_info.name)]
    except (ImportError,RuntimeError):return []


class RealSenseCamera(Camera):
    """Synchronized D435 RGB/Z16; ACT receives only the color stream."""
    def run(self):
        import numpy as np
        sequence=0;attempts=0
        while not self.stop_event.is_set():
            pipeline=None;started=False
            try:
                import pyrealsense2 as rs
                pipeline=rs.pipeline();config=rs.config()
                serial=self.device.removeprefix('realsense:');config.enable_device(serial)
                config.enable_stream(rs.stream.color,640,480,rs.format.bgr8,int(self.fps))
                config.enable_stream(rs.stream.depth,640,480,rs.format.z16,int(self.fps))
                profile=pipeline.start(config);started=True
                device=profile.get_device();model=device.get_info(rs.camera_info.name)
                if 'D435' not in model:raise RuntimeError('선택한 장치는 RealSense D435가 아닙니다.')
                depth_scale=float(device.first_depth_sensor().get_depth_scale())
                def intrinsics(stream):
                    value=profile.get_stream(stream).as_video_stream_profile().get_intrinsics()
                    return dict(width=value.width,height=value.height,fx=value.fx,fy=value.fy,
                                ppx=value.ppx,ppy=value.ppy,model=str(value.model),coeffs=list(value.coeffs))
                extrinsic=profile.get_stream(rs.stream.depth).get_extrinsics_to(profile.get_stream(rs.stream.color))
                self.actual=dict(width=640,height=480,fps=self.fps,format='RGB + Z16',model=model,serial=serial,
                    depth_training=False,depth_recording=True,depth_scale_m_per_unit=depth_scale,
                    depth_aligned_to='color',depth_filters='none',
                    intrinsics=dict(color=intrinsics(rs.stream.color),depth=intrinsics(rs.stream.depth)),
                    depth_to_color=dict(rotation=list(extrinsic.rotation),translation_m=list(extrinsic.translation)))
                align=rs.align(rs.stream.color)
                from .depth import encode
                first=True
                while not self.stop_event.is_set():
                    frames=pipeline.wait_for_frames(timeout_ms=500);acquired=time.monotonic()
                    frames=align.process(frames);frame=frames.get_color_frame();depth_frame=frames.get_depth_frame()
                    if not frame or not depth_frame:raise RuntimeError('D435 컬러·깊이 영상 수신 실패')
                    image=np.asanyarray(frame.get_data());raw_depth=np.asanyarray(depth_frame.get_data())
                    if raw_depth.shape!=image.shape[:2]:raise RuntimeError('D435 깊이·컬러 정렬 크기가 다릅니다.')
                    depth=encode(raw_depth,depth_scale)
                    depth.update(sensor_timestamp_ms=depth_frame.get_timestamp(),frame_number=depth_frame.get_frame_number(),
                                 timestamp_domain=str(depth_frame.get_frame_timestamp_domain()))
                    ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])
                    if not ok:raise RuntimeError('D435 JPEG 변환 실패')
                    sequence+=1
                    self.frame=dict(jpeg=encoded.tobytes(),bgr=image.copy(),acquired=acquired,sequence=sequence,shape=list(image.shape),
                        sensor_timestamp_ms=frame.get_timestamp(),frame_number=frame.get_frame_number(),
                        timestamp_domain=str(frame.get_frame_timestamp_domain()),depth=depth)
                    self.error=None
                    if first:
                        if attempts:self.reconnects+=1
                        self.log('recovered' if attempts else 'stream_started',actual=self.actual);first=False
            except Exception as error:
                self.error=str(error);self.failures+=1;self.last_error=dict(message=self.error,time=time.time())
                self.log('capture_error',message=self.error,sequence=sequence)
            finally:
                if started:
                    try:pipeline.stop()
                    except RuntimeError:pass
            attempts+=1
            if self.stop_event.wait(.5 if attempts==1 else 2):break


def open_camera(device,fps=30):
    return (RealSenseCamera if device.startswith('realsense:') else Camera)(device,fps)
