"""Single-window task pages with shared robot controls and back navigation."""
import tkinter as tk
from tkinter import ttk


class Navigation:
    def __init__(self, app):
        self.app=app;self.current=None;self.history=[];self.pages=[]
        self.host=ttk.Frame(app.root)
        self.host.columnconfigure(0,weight=1);self.host.rowconfigure(1,weight=1)
        bar=ttk.Frame(self.host,padding=(14,8));bar.grid(row=0,column=0,sticky='ew')
        ttk.Button(bar,text='← 뒤로',command=self.back).pack(side='left')
        ttk.Button(bar,text='에피소드로',command=self.home).pack(side='left',padx=8)
        self.heading=tk.StringVar(master=app.root);ttk.Label(bar,textvariable=self.heading,style='EditorStep.TLabel',wraplength=280).pack(side='left',padx=12,fill='x',expand=True)
        self.body=ttk.Frame(self.host);self.body.grid(row=1,column=0,sticky='nsew')
        self.body.rowconfigure(0,weight=1);self.body.columnconfigure(0,weight=1)
        self.activate('episode')

    def activate(self,section):
        self.active_section=section
        for key,label in [('episode','에피소드'),('manual','실물 조정'),('vision','카메라'),('settings','장비 설정')]:
            button=self.app.buttons.get(key)
            if button is not None and button.winfo_exists():
                button.configure(style='Active.Nav.TButton' if key==section else 'Nav.TButton',
                                 text=('● ' if key==section else '')+label)

    def position(self,page):
        self.host.grid_forget();self.host.place_forget()
        if page.dock_preview and self.app.preview is not None:
            self.host.place(in_=self.app.left_workspace,x=0,y=0,relwidth=1,relheight=1)
        else:self.host.grid(row=2,column=0,rowspan=3,sticky='nsew')
        self.host.tkraise();self.activate(page.section)

    def show(self,page):
        if self.current is page:
            self.activate(page.section);return
        if self.current:
            self.current.outer.grid_remove()
        self.history.append(self.current)
        self.current=page;self.position(page)
        self.host.tkraise();page.outer.grid(row=0,column=0,sticky='nsew');page.outer.tkraise()
        self.heading.set(page.caption);page.after_idle(page.fit)

    def hide(self,page):
        page.outer.grid_remove()
        if self.current is not page:
            self.history=[p for p in self.history if p is not page];return
        self.current=None
        while self.history:
            prior=self.history.pop()
            if prior is None:break
            if prior.winfo_exists():
                self.current=prior;self.position(prior);prior.outer.grid(row=0,column=0,sticky='nsew')
                self.heading.set(prior.caption);prior.after_idle(prior.fit);return
        self.host.grid_remove();self.host.place_forget();self.activate('episode')

    def back(self):
        page=self.current
        if page:
            if page.close_action:page.close_action()
            else:self.hide(page)

    def home(self):
        self.history=[]
        self.back()

    def navigate(self, action):
        page=self.current
        if page and not page.persistent:
            self.back()
            if self.current is page:return  # asynchronous calibration cleanup
        action()


class PageContainer(ttk.Frame):
    """Mark parent-led destruction so AppPage cannot destroy its parent twice."""
    def destroy(self):
        if getattr(self,'destroying',False):return
        self.destroying=True
        super().destroy()


class AppPage(tk.Frame):
    """Content frame exposing only the old windows' small layout API.

    Serial/control code is unchanged. Oversized forms scroll inside the shared
    workspace instead of resizing the desktop or opening another OS window.
    """
    def __init__(self,app,*,show=True,persistent=False,fluid=False,section='settings',dock_preview=False):
        self.nav=app.navigation;self.caption='';self.close_action=None;self.destroying=False
        self.persistent=persistent;self.section=section;self.dock_preview=dock_preview
        self.fluid=fluid
        self.outer=PageContainer(self.nav.body);self.outer.columnconfigure(0,weight=1);self.outer.rowconfigure(0,weight=1)
        self.viewport=tk.Canvas(self.outer,highlightthickness=0,background='#eaf0f3')
        self.viewport.grid(row=0,column=0,sticky='nsew')
        self.scroll=ttk.Scrollbar(self.outer,orient='vertical',command=self.viewport.yview)
        self.scroll.grid(row=0,column=1,sticky='ns');self.viewport.configure(yscrollcommand=self.scroll.set)
        self.footer=ttk.Frame(self.outer,padding=(12,6));self.footer.grid(row=1,column=0,columnspan=2,sticky='ew')
        super().__init__(self.viewport,background='#eaf0f3')
        self.item=self.viewport.create_window(0,0,window=self,anchor='nw')
        self.viewport.bind('<Configure>',self.fit);self.bind('<Configure>',self.fit)
        self.nav.pages.append(self)
        if show:self.nav.show(self)

    def fit(self,_event=None):
        if self.destroying or not self.winfo_exists():return
        width=max(1,self.viewport.winfo_width())
        height=self.viewport.winfo_height() if self.fluid else max(self.winfo_reqheight(),self.viewport.winfo_height())
        self.viewport.itemconfigure(self.item,width=width,height=height)
        self.viewport.configure(scrollregion=(0,0,width,height))
        if height<=self.viewport.winfo_height()+1:self.scroll.grid_remove()
        else:self.scroll.grid()

    def title(self,value=None):
        if value is None:return self.caption
        self.caption=value
        if self.nav.current is self:self.nav.heading.set(value)
    def geometry(self,*_):return self.winfo_geometry()
    def state(self):return 'normal' if self.nav.current is self else 'withdrawn'
    def minsize(self,*_):pass
    def resizable(self,*_):pass
    def transient(self,*_):pass
    def protocol(self,name,command=None):
        if name=='WM_DELETE_WINDOW':self.close_action=command
    def deiconify(self):self.nav.show(self)
    def lift(self,*_):self.nav.show(self)
    def withdraw(self):self.nav.hide(self)
    def destroy(self):
        if self.destroying:return
        self.destroying=True;self.nav.hide(self)
        if self in self.nav.pages:self.nav.pages.remove(self)
        super().destroy()
        if not getattr(self.outer,'destroying',False):self.outer.destroy()
