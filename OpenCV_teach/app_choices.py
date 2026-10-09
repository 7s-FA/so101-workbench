"""In-window selection lists. No native dropdown or file-manager popup."""
from pathlib import Path
import fnmatch
import tkinter as tk
from tkinter import ttk

from teaching_ui import SURFACE, INK, BORDER, ACCENT


class AppChoice(ttk.Frame):
    """Small Combobox-compatible adapter using a list placed inside the app window."""
    def __init__(self, master, *, values=(), textvariable=None, state='readonly', width=18, **kw):
        super().__init__(master, **kw)
        self.variable=textvariable if textvariable is not None else tk.StringVar(self)
        self._values=tuple(str(v) for v in values);self._state=state;self.panel=None
        self.button=ttk.Button(self,textvariable=self.variable,width=width,command=self.open)
        self.button.pack(side='left',fill='x',expand=True)
        self.arrow=ttk.Button(self,text='▾',width=2,command=self.open)
        self.arrow.pack(side='right')
        self.configure(state=state)
        for key in ('<Down>','<Up>','<Return>','<space>'):
            self.button.bind(key,lambda e:(self.open(),'break')[1])
        self.bind('<Destroy>',lambda e:self.close() if e.widget is self else None,add='+')

    def configure(self, cnf=None, **kw):
        if isinstance(cnf,dict):kw={**cnf,**kw}
        if 'values' in kw:
            self._values=tuple(str(v) for v in kw.pop('values'))
            self.close()
        if 'state' in kw:
            self._state=kw.pop('state')
            if hasattr(self,'button'):
                for button in (self.button,self.arrow):button.configure(state='disabled' if self._state=='disabled' else 'normal')
            if self._state=='disabled':self.close()
        return super().configure(**kw) if kw else None

    config=configure

    def cget(self,key):
        if key=='state':return self._state
        if key=='values':return self._values
        return super().cget(key)

    def __getitem__(self,key):return self.cget(key)
    def __setitem__(self,key,value):self.configure(**{key:value})
    def get(self):return self.variable.get()
    def set(self,value):self.variable.set(value)

    def current(self,index=None):
        if index is not None:
            self.variable.set(self._values[index]);return index
        try:return self._values.index(self.variable.get())
        except ValueError:return -1

    def open(self):
        if self._state=='disabled':return
        if self.panel is not None:self.close();return
        top=self.winfo_toplevel();old=getattr(top,'_active_choice',None)
        if old:old.close()
        top._active_choice=self
        self.top=top;top.update_idletasks()
        self.panel=tk.Frame(top,background=SURFACE,highlightbackground=BORDER,highlightthickness=1)
        self.listbox=tk.Listbox(self.panel,exportselection=False,activestyle='none',background=SURFACE,
            foreground=INK,selectbackground=ACCENT,selectforeground='white',borderwidth=0,
            highlightthickness=0,height=min(max(len(self._values),1),8),font='TkDefaultFont')
        self.listbox.pack(side='left',fill='both',expand=True,padx=3,pady=3)
        scroll=ttk.Scrollbar(self.panel,command=self.listbox.yview);scroll.pack(side='right',fill='y')
        self.listbox.configure(yscrollcommand=scroll.set)
        for value in self._values:self.listbox.insert('end',value)
        if not self._values:self.listbox.insert('end','선택할 항목이 없습니다')
        index=max(0,self.current());self.listbox.selection_set(index);self.listbox.activate(index);self.listbox.see(index)
        width=min(max(self.winfo_width(),220),max(220,top.winfo_width()-16))
        height=min(260,max(36,len(self._values)*28+8))
        x=min(max(8,self.winfo_rootx()-top.winfo_rootx()),max(8,top.winfo_width()-width-8))
        below=self.winfo_rooty()-top.winfo_rooty()+self.winfo_height()
        y=below if below+height<top.winfo_height()-8 else max(8,below-self.winfo_height()-height)
        self.panel.place(x=x,y=y,width=width,height=height);self.panel.lift()
        self.listbox.bind('<ButtonRelease-1>',self.choose)
        self.listbox.bind('<Return>',self.choose)
        self.listbox.bind('<Escape>',lambda e:self.close())
        self.listbox.bind('<Tab>',lambda e:(self.close(),self.button.focus_set(),'break')[-1])
        self.outside_binding=top.bind('<ButtonPress-1>',self.outside,add='+')
        self.panel.grab_set();self.listbox.focus_set()

    def outside(self,event):
        if self.panel is None:return
        widget=self.winfo_containing(event.x_root,event.y_root)
        if widget is None or (widget is not self.panel and not str(widget).startswith(str(self.panel)+'.')):
            self.close()

    def choose(self,event=None):
        if not self._values:return 'break'
        selection=self.listbox.curselection()
        if selection:
            self.variable.set(self._values[selection[0]])
            self.close();self.button.focus_set();self.event_generate('<<ComboboxSelected>>')
        return 'break'

    def close(self):
        panel=getattr(self,'panel',None)
        if panel is None:return
        self.panel=None
        try:
            self.top.unbind('<ButtonPress-1>',self.outside_binding)
            if panel.grab_current() is panel:panel.grab_release()
            panel.destroy()
            if getattr(self.top,'_active_choice',None) is self:self.top._active_choice=None
        except tk.TclError:pass




def askopenfilename(*, parent, title='파일 선택', initialdir=None, filetypes=()):
    """Synchronous result with Tk events alive, rendered as an in-app panel."""
    top=parent.winfo_toplevel();top.update_idletasks()
    panel=ttk.Frame(top,padding=14,relief='solid',borderwidth=1)
    panel.place(relx=.03,rely=.04,relwidth=.94,relheight=.92);panel.lift()
    panel.columnconfigure(0,weight=1);panel.rowconfigure(3,weight=1)
    ttk.Label(panel,text=title,style='PanelHeading.TLabel').grid(row=0,column=0,sticky='w',pady=(0,8))
    pathvar=tk.StringVar(master=top,value=str(Path(initialdir or Path.home()/'Downloads').expanduser()))
    status=tk.StringVar(master=top);done=tk.BooleanVar(master=top,value=False);result=[];paths={}
    navigation=ttk.Frame(panel);navigation.grid(row=1,column=0,sticky='ew',pady=4);navigation.columnconfigure(1,weight=1)
    current=[Path(pathvar.get())]
    entry=ttk.Entry(navigation,textvariable=pathvar);entry.grid(row=0,column=1,sticky='ew',padx=6)
    ttk.Label(panel,text='폴더를 두 번 눌러 열고 파일을 선택하세요.').grid(row=2,column=0,sticky='w',pady=4)
    tree=ttk.Treeview(panel,columns=('name','kind'),show='headings',selectmode='browse')
    tree.heading('name',text='이름');tree.heading('kind',text='종류');tree.column('kind',width=80,stretch=False)
    tree.grid(row=3,column=0,sticky='nsew');bar=ttk.Scrollbar(panel,command=tree.yview);bar.grid(row=3,column=1,sticky='ns');tree.configure(yscrollcommand=bar.set)
    patterns=[pattern for _,pats in filetypes for pattern in (pats if isinstance(pats,(tuple,list)) else pats.split())]
    def refresh(directory=None):
        try:
            directory=Path(directory or pathvar.get()).expanduser().resolve(strict=True)
            files=sorted(directory.iterdir(),key=lambda p:(not p.is_dir(),p.name.casefold()))
        except (OSError,ValueError) as exc:status.set(str(exc));return
        current[0]=directory;pathvar.set(str(directory));paths.clear();tree.delete(*tree.get_children())
        for path in files:
            if path.name.startswith('.') or (not path.is_dir() and patterns and not any(fnmatch.fnmatch(path.name.lower(),p.lower()) for p in patterns)):continue
            iid=tree.insert('','end',values=(path.name,'폴더' if path.is_dir() else '파일'));paths[iid]=path
        status.set(f'{len(paths)}개 항목')
    def finish():
        selected=tree.selection()
        if not selected:return
        path=paths[selected[0]]
        if path.is_dir():refresh(path);return
        if not path.is_file():status.set('파일이 없습니다. 목록을 새로고침하세요.');return
        result.append(str(path));done.set(True)
    ttk.Button(navigation,text='상위 폴더',command=lambda:refresh(current[0].parent)).grid(row=0,column=0)
    ttk.Button(navigation,text='이동',command=refresh).grid(row=0,column=2)
    entry.bind('<Return>',lambda e:refresh())
    tree.bind('<Double-1>',lambda e:finish());tree.bind('<Return>',lambda e:finish())
    ttk.Label(panel,textvariable=status).grid(row=4,column=0,sticky='w',pady=6)
    actions=ttk.Frame(panel);actions.grid(row=5,column=0,sticky='e')
    ttk.Button(actions,text='취소',command=lambda:done.set(True)).pack(side='left',padx=6)
    ttk.Button(actions,text='선택한 파일 열기',command=finish).pack(side='left')
    binding=top.bind('<Escape>',lambda e:done.set(True),add='+')
    panel.bind('<Destroy>',lambda e:done.set(True) if e.widget is panel else None)
    previous=top.grab_current()
    try:
        refresh();panel.grab_set();tree.focus_set();top.wait_variable(done)
    finally:
        try:
            top.unbind('<Escape>',binding);panel.grab_release();panel.destroy()
            if previous and previous.winfo_exists():previous.grab_set()
        except tk.TclError:pass
    return result[0] if result else ''
