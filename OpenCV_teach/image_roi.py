"""Mouse-drawn detection area in image coordinates, independent of letterboxing."""
class ImageROI:
    fields=('roi_x_pct','roi_y_pct','roi_w_pct','roi_h_pct')
    def __init__(self,view):
        self.view=view;self.canvas=view.canvas;self.bounds=None;self.active=False;self.start=None;self.end=None
        for event,action in [('<ButtonPress-1>',self.press),('<B1-Motion>',self.motion),('<ButtonRelease-1>',self.release)]:
            self.canvas.bind(event,action,add='+')

    def toggle(self):
        if self.active:self.cancel();return
        self.view.idle();self.view.current_frame()
        if not self.bounds:raise ValueError('카메라 영상을 먼저 표시하세요.')
        self.active=True;self.target=self.view.target_id
        self.canvas.configure(cursor='crosshair');self.view.roi_button.configure(text='범위 지정 취소',style='Primary.TButton')
        self.view.roi_hint.set('영상에서 마우스를 누른 채 끌고 놓으세요. 놓으면 선택한 지그의 범위를 저장합니다.')

    def cancel(self):
        self.active=False;self.start=None;self.end=None
        self.canvas.configure(cursor='');self.view.roi_button.configure(text='감지 범위 직접 지정',style='TButton')
        self.draw()

    def displayed(self,bounds):
        if self.start is not None and self.bounds!=bounds:
            self.start=self.end=None
            self.view.roi_hint.set('영상 크기가 바뀌었습니다. 범위를 다시 끌어 주세요.')
        self.bounds=bounds;self.draw()

    def point(self,event,clip=True):
        if not self.bounds:return None
        x,y,w,h=self.bounds
        if not clip and not(x<=event.x<=x+w and y<=event.y<=y+h):return None
        return max(x,min(x+w,event.x)),max(y,min(y+h,event.y))

    def press(self,event):
        if not self.active:return
        self.start=self.point(event,clip=False);self.end=self.start;self.draw();return 'break'

    def motion(self,event):
        if self.active and self.start is not None:
            self.end=self.point(event);self.draw();return 'break'

    def release(self,event):
        if not self.active or self.start is None:return
        self.end=self.point(event)
        x0,y0=self.start;x1,y1=self.end
        if abs(x1-x0)<8 or abs(y1-y0)<8:
            self.start=self.end=None;self.draw();self.view.roi_hint.set('범위가 너무 작습니다. 지그 전체가 들어가도록 다시 끌어 주세요.');return 'break'
        x,y,w,h=self.bounds
        left=round(max(0,min(x0,x1)-x)/w*100,6);top=round(max(0,min(y0,y1)-y)/h*100,6)
        values=(left,top,min(100-left,round(abs(x1-x0)/w*100,6)),min(100-top,round(abs(y1-y0)/h*100,6)))
        before={k:self.view.vars['t_'+k].get() for k in self.fields}
        try:
            self.view.idle();self.view.current_frame()
            if self.target!=self.view.target_id:raise ValueError('선택한 지그가 바뀌었습니다. 다시 지정하세요.')
            for k,v in zip(self.fields,values):self.view.vars['t_'+k].set(repr(v))
            from jig_detection import roi_pixels
            roi_pixels(self.view.read_target(),self.view.frame.shape[1::-1])
            self.view.save_detection_method()
            self.view.clear_jig_reference('감지 범위를 바꿨습니다. 지그 기준을 다시 읽으세요.')
        except Exception as exc:
            for k,v in before.items():self.view.vars['t_'+k].set(v)
            self.view.roi_hint.set(str(exc));self.start=self.end=None;self.draw();return 'break'
        self.cancel();self.view.roi_hint.set('파란 테두리 안에서만 선택한 지그를 찾습니다. 다음 실행 전에 지그 기준을 다시 읽으세요.')
        self.view.status.set('마우스로 지정한 감지 범위를 저장했습니다.')
        return 'break'

    def draw(self):
        self.canvas.delete('detection-area')
        if not self.bounds:return
        x,y,w,h=self.bounds
        if self.start and self.end:
            x0,y0=self.start;x1,y1=self.end
        else:
            try:a,b,c,d=(float(self.view.vars['t_'+k].get()) for k in self.fields)
            except (KeyError,ValueError):return
            if (a,b,c,d)==(0,0,100,100) and not self.active:return
            x0,y0,x1,y1=x+w*a/100,y+h*b/100,x+w*(a+c)/100,y+h*(b+d)/100
        self.canvas.create_rectangle(x0,y0,x1,y1,outline='#00c8ff',width=3,dash=(7,4),tags='detection-area')
        lx,ly=min(x0,x1),min(y0,y1)
        self.canvas.create_rectangle(lx,ly,lx+85,ly+25,fill='#005a70',outline='',tags='detection-area')
        self.canvas.create_text(lx+5,ly+4,text='감지 범위',anchor='nw',fill='white',font=('sans',11,'bold'),tags='detection-area')
