"""Default native docks: viewport left, modes below, controls right."""
import omni.ui as ui


class TeachLayout:
    def __init__(self, modes, panels):
        self.modes=modes
        self.panels=panels
        self.initialized=False
        self.visible={}
        self.pending=set()
        self.focus_modes_next=False

    def update(self):
        content=ui.Workspace.get_window('Content')
        prop=ui.Workspace.get_window('Property')
        if not content or not prop:return
        if self.focus_modes_next:
            self.modes.focus()
            self.focus_modes_next=False
        if not self.initialized:
            self.modes.dock_in(content,ui.DockPosition.SAME)
            self.focus_modes_next=True
            ui.Workspace.set_dock_id_height(content.dock_id,max(230,ui.Workspace.get_main_window_height()*.23))
            ui.Workspace.set_dock_id_width(prop.dock_id,ui.Workspace.get_main_window_width()*.30)
            stage=ui.Workspace.get_window('Stage')
            if stage:ui.Workspace.set_dock_id_height(stage.dock_id,ui.Workspace.get_main_window_height()*.23)
            settings=ui.Workspace.get_window('Simulation Settings')
            if settings:settings.visible=False
            self.initialized=True
        for panel in self.panels:
            # Defer one frame so Kit has registered a newly shown window.
            if panel in self.pending:
                if panel.visible:
                    panel.dock_in(prop,ui.DockPosition.SAME)
                    panel.focus()
                self.pending.remove(panel)
            if panel.visible and not self.visible.get(panel,False):self.pending.add(panel)
            self.visible[panel]=panel.visible

    def snapshot(self):
        result={}
        for title in ('Viewport','Content','Property','Stage',self.modes.title,*[p.title for p in self.panels]):
            w=ui.Workspace.get_window(title)
            if w:result[title]={'docked':w.docked,'dock_id':w.dock_id,'visible':w.visible,'width':w.width,'height':w.height,'selected':w.is_selected_in_dock()}
        return result
