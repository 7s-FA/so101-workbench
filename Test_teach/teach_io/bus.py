"""OMX-F-only SDK adapter. No EEPROM, ID, homing or mode writes."""
import fcntl
import math
import termios
from .mapping import IDS, MODELS
from .motion_timing import motor_profile

POWER_PROFILE = 'omx_f_manufacturer_v1'
# ROBOTIS open_manipulator_description/ros2_control/omx_f.ros2_control.xacro
# RAM-only adaptation: keep the installed modes, EEPROM limits and our timing.
POSITION_GAINS = {80: 1000, 82: 0, 84: 1000}  # D, I, P
GRIPPER_CURRENT_MA = 600


class Bus:
    def __init__(self, device):
        import dynamixel_sdk as sdk
        self.sdk = sdk
        self.port = sdk.PortHandler(device)
        self.packet = sdk.PacketHandler(2.0)
        self.owned = False
        if not self.port.openPort():
            raise RuntimeError(f'Cannot open {device}')
        try:
            fcntl.flock(self.port.ser.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.ioctl(self.port.ser.fileno(), termios.TIOCEXCL)
            if not self.port.setBaudRate(1000000):
                raise RuntimeError('Cannot set 1 Mbps')
        except Exception:
            self.port.closePort()
            raise

    def read(self, motor, address, size):
        value, comm, error = getattr(self.packet, f'read{size}ByteTxRx')(self.port, motor, address)
        if comm or error:
            raise RuntimeError(f'ID {motor} read {address}: {comm}/{error}')
        return value

    def write(self, motor, address, size, value):
        if address < 64:
            raise ValueError('EEPROM writes are prohibited')
        comm, error = getattr(self.packet, f'write{size}ByteTxRx')(self.port, motor, address, int(value))
        if comm or error:
            raise RuntimeError(f'ID {motor} write {address}: {comm}/{error}')

    def inspect(self):
        result = []
        for motor, expected in zip(IDS, MODELS):
            model = self.read(motor, 0, 2)
            if model != expected:
                raise RuntimeError(f'ID {motor}: expected model {expected}, got {model}')
            result.append(dict(id=motor, model=model, mode=self.read(motor,11,1),
                drive=self.read(motor,10,1), offset=self.read(motor,20,4),
                torque=self.read(motor,64,1), watchdog=self.read(motor,98,1),
                min_ticks=self.read(motor,52,4), max_ticks=self.read(motor,48,4)))
        return result

    def sample(self, detailed=False, packed=False, *, temperature_guard=None, active=False):
        if temperature_guard:temperature_guard.check_timeouts()
        result = []
        for motor in IDS:
            if packed:
                # One contiguous response for PWM/current/velocity/position/
                # trajectory/voltage/temperature, retaining status error bits.
                data,comm,status=self.packet.readTxRx(self.port,motor,124,23)
                if comm or status or len(data)!=23:
                    raise RuntimeError(f'ID {motor} feedback block: {comm}/{status}, bytes={len(data)}')
                def read(m, address, size):
                    if 124<=address and address+size<=147:
                        return int.from_bytes(bytes(data[address-124:address-124+size]),'little')
                    return self.read(m,address,size)
            else:
                read=self.read
            # Direct TxRx retains motor status error bits, unlike SDK GroupSyncRead.
            raw = read(motor,132,4)
            raw = raw-(1<<32) if raw & (1<<31) else raw
            voltage = read(motor,144,2)*.1
            temp = read(motor,146,1)
            error = read(motor,70,1)
            effort = read(motor,126,2)
            effort = effort-65536 if effort & 32768 else effort
            lo, hi = (6.5,14.) if motor <= 13 else (3.7,6.)
            if not lo <= voltage <= hi or error or (temperature_guard is None and temp >= 55):
                raise RuntimeError(f'ID {motor}: voltage={voltage:.1f}V temperature={temp} error={error}')
            if temperature_guard:
                temperature_guard.observe(motor,'packed' if packed else 'single',temp,active=active)
            result.append(dict(id=motor,ticks=raw,voltage=voltage,temperature=temp,
                load_percent=effort*.1 if motor<=13 else None,
                current_ma=effort if motor>=14 else None))
            if detailed:
                def signed(value,bits):return value-(1<<bits) if value&(1<<(bits-1)) else value
                result[-1].update(velocity_rad_s=signed(read(motor,128,4),32)*.229*math.tau/60,
                    trajectory_ticks=signed(read(motor,140,4),32),
                    pwm_percent=signed(read(motor,124,2),16)*.113)
        return result

    def release(self):
        """Explicit user-requested torque OFF; try every joint and verify readback."""
        failures=[]
        for motor in IDS:
            for address in (64,98):
                try:self.write(motor,address,1,0)
                except Exception as error:failures.append(str(error))
        for motor in IDS:
            try:
                if self.read(motor,64,1) or self.read(motor,98,1):failures.append(f'ID {motor}: release not confirmed')
            except Exception as error:failures.append(str(error))
        if failures:raise RuntimeError('; '.join(failures))
        self.owned=False

    def positions(self, ticks):
        group = self.sdk.GroupSyncWrite(self.port,self.packet,116,4)
        try:
            for motor, value in zip(IDS,ticks):
                data = list((int(value)&0xffffffff).to_bytes(4,'little'))
                if not group.addParam(motor,data):
                    raise RuntimeError('Cannot construct goal packet')
            result = group.txPacket()
            if result:
                raise RuntimeError(f'Position transmission failed: {result}')
        finally:
            group.clearParam()

    def enable(self, info, current_ticks, reference=False):
        for i, entry in enumerate(info):
            expected = (3,4) if i<5 else (5,)
            if entry['mode'] not in expected:
                raise RuntimeError('Use position/extended position mode for arm and current-position mode for gripper; mode is not changed automatically')
            if entry['torque'] or entry['watchdog']:
                raise RuntimeError('Start with torque OFF and no watchdog fault; another controller may own the arm')
        # Reject an incompatible EEPROM velocity limit before any motor write.
        for i,entry in enumerate(info):
            if not reference and not entry['drive'] & 4 and motor_profile(i)[1]>self.read(entry['id'],44,4):
                raise RuntimeError(f"ID {entry['id']}: requested profile exceeds stored Velocity Limit; EEPROM is unchanged")
        if any(not 0 <= t <= 4095 for t in current_ticks):
            raise RuntimeError('Multi-turn/wrapped startup is not supported; recalibrate in one revolution')
        # Read every output cap before any write. Never enlarge an EEPROM cap.
        settings=[]
        for motor in IDS:
            pwm_limit=self.read(motor,36,2)
            if not 0 < pwm_limit <= 885:
                raise RuntimeError(f'ID {motor}: invalid PWM Limit {pwm_limit}')
            registers={**POSITION_GAINS,100:pwm_limit}
            if reference:
                # The workshop only overrides elbow PID; keep other installed
                # gains and power levels, never raise the EEPROM caps.
                registers={a:self.read(motor,a,2) for a in (80,82,84,100)}
                registers[100]=min(registers[100],pwm_limit)
                if motor==13:registers.update({84:1500,82:0,80:600})
            if motor==16:
                current_limit=self.read(motor,38,2)
                if not 0 < current_limit <= 1750:
                    raise RuntimeError(f'ID 16: invalid Current Limit {current_limit}')
                registers[102]=min(self.read(motor,102,2) if reference else GRIPPER_CURRENT_MA,current_limit)
            settings.append((motor,registers))
        self.owned = True
        # Hold measured pose BEFORE torque on; avoid a jump to the old goal.
        self.positions(current_ticks)
        # Set and verify the complete output profile while torque is still OFF.
        # A failed readback prevents enabling any joint.
        for motor,registers in settings:
            for address,value in registers.items():
                self.write(motor,address,2,value)
                actual=self.read(motor,address,2)
                if actual!=value:
                    raise RuntimeError(f'ID {motor}: output setting {address} readback {actual} != {value}; torque not enabled')
        self.power_profile=dict(name='lerobot_reference' if reference else POWER_PROFILE,motors=[dict(id=m,
            position_p=r[84],position_i=r[82],position_d=r[80],goal_pwm=r[100],
            gripper_current_ma=r.get(102)) for m,r in settings])
        for entry in info:
            motor = entry['id']
            self.write(motor,98,1,15)  # 300 ms independent bus watchdog
            if reference:
                # Restore RAM defaults: the reference program does not install
                # our slow demo velocity/acceleration profiles.
                for address in (108,112):
                    self.write(motor,address,4,0)
                    if self.read(motor,address,4)!=0:
                        raise RuntimeError(f'ID {motor}: reference profile readback failed; torque not enabled')
            elif entry['drive'] & 4:    # time-based profile, milliseconds
                self.write(motor,108,4,100)
                self.write(motor,112,4,200)
            else:                    # velocity-based profile, motor units
                acceleration,velocity=motor_profile(IDS.index(motor))
                self.write(motor,108,4,acceleration)
                self.write(motor,112,4,velocity)
        for motor in IDS:
            self.write(motor,64,1,1)
        after=[x['ticks'] for x in self.sample()]
        if max(abs(a-b) for a,b in zip(after,current_ticks))>32:
            raise RuntimeError('Encoder reference changed at torque enable; stopped')

    def hold(self):
        if self.owned:
            # Keep support torque. Do not drop the arm/object on application exit.
            try:
                ticks=[]
                for motor in IDS:
                    t=self.read(motor,132,4);ticks.append(t-(1<<32) if t&(1<<31) else t)
                self.positions(ticks)
            except Exception:
                pass  # Watchdog remains armed; no reconnect/replay.

    def close(self):
        self.hold()
        try:
            fcntl.ioctl(self.port.ser.fileno(),termios.TIOCNXCL)
        finally:
            self.port.closePort()
