"""Release-to-apply delta slider: dragging is a preview, never a command."""
class MotorSlider:
    def __init__(self, parent, limit, unit, apply, describe=None, step=1):
        import tkinter as tk
        from tkinter import ttk
        self.apply=apply;self.describe=describe;self.unit=unit;self.limit=limit;self.step=step;self.enabled=False
        self.frame=ttk.Frame(parent);self.value=tk.DoubleVar(master=parent,value=0.);self.label=tk.StringVar(master=parent,value='')
        self.scale=ttk.Scale(self.frame,from_=-limit,to=limit,variable=self.value,command=self.preview)
        self.scale.pack(fill='x');ttk.Label(self.frame,textvariable=self.label,anchor='center',style='Muted.TLabel').pack(fill='x')
        self.scale.bind('<ButtonRelease-1>',self.release)
        for key,delta in [('Left',-1),('Down',-1),('Right',1),('Up',1)]:
            self.scale.bind('<KeyPress-'+key+'>',lambda e,d=delta:self.key(d))
            self.scale.bind('<KeyRelease-'+key+'>',self.release)
        self.preview()

    def key(self, delta):
        if self.enabled:self.value.set(max(-self.limit,min(self.limit,self.snapped()+delta*self.step)));self.preview()
        return 'break'

    def snapped(self):return round(self.value.get()/self.step)*self.step

    def preview(self, *_):
        delta=self.snapped()
        text=f'{delta:+g} {self.unit}' if delta else '밀고 놓으면 조정'
        if delta and self.describe:text=self.describe(delta)
        self.label.set(text)

    def release(self, _event=None):
        delta=self.snapped();self.value.set(0);self.preview()
        if self.enabled and delta:self.apply(delta)
        return 'break'

    def enable(self, enabled):
        if self.enabled!=bool(enabled):
            self.enabled=bool(enabled)
            self.scale.configure(state='normal' if self.enabled else 'disabled')
            if not self.enabled:self.value.set(0);self.preview()


class ValueSlider:
    """A slider bound to an existing numeric editor, without hardware dispatch."""
    def __init__(self, parent, variable, low, high, step=1):
        import tkinter as tk
        from tkinter import ttk
        self.variable=variable;self.low=low;self.high=high;self.step=step;self.lock=False
        self.value=tk.DoubleVar(master=parent,value=low)
        self.scale=ttk.Scale(parent,from_=low,to=high,variable=self.value,command=self.changed)
        self.trace=variable.trace_add('write',self.sync)
        self.scale.bind('<Destroy>',self.destroy);self.sync()

    def changed(self, value):
        if self.lock:return
        amount=max(self.low,min(self.high,round(float(value)/self.step)*self.step))
        self.variable.set(f'{amount:g}')

    def sync(self, *_):
        import math
        try:value=float(self.variable.get())
        except (ValueError,TypeError):return
        if not math.isfinite(value):return
        self.lock=True
        try:self.value.set(max(self.low,min(self.high,value)))
        finally:self.lock=False

    def destroy(self, _event):
        self.variable.trace_remove('write',self.trace)


class MotorValue:
    """Map the encoder editor to the existing degree-valued teaching model."""
    def __init__(self, backend, key, position):
        import tkinter as tk
        self.backend,self.key,self.position=backend,key,position
        self.raw=tk.StringVar(master=position._root);self.lock=False
        self.sync_position()
        self.raw_trace=self.raw.trace_add('write',self.sync_raw)
        self.position_trace=self.position.trace_add('write',self.sync_position)

    def sync_position(self,*_):
        from motor_positions import to_ticks
        if self.lock:return
        self.lock=True
        try:self.raw.set(str(to_ticks(self.backend,self.key,float(self.position.get()))))
        except (ValueError,OverflowError):self.raw.set(self.position.get())
        finally:self.lock=False

    def sync_raw(self,*_):
        from motor_positions import from_ticks
        import math
        if self.lock:return
        self.lock=True
        try:
            raw=float(self.raw.get())
            if not math.isfinite(raw) or raw!=int(raw) or not 0<=raw<=4095:raise ValueError
            self.position.set(repr(from_ticks(self.backend,self.key,int(raw))))
        except ValueError:self.position.set('nan')
        finally:self.lock=False

    def close(self,*_):
        self.raw.trace_remove('write',self.raw_trace)
        self.position.trace_remove('write',self.position_trace)
