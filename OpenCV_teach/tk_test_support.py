"""Cancel Tk callbacks through their owning widget before destroying test roots."""
def cancel_jobs(root):
    def children(widget):
        for child in widget.winfo_children():
            yield child
            yield from children(child)
    owners=[root,*children(root)]
    for job in root.tk.splitlist(root.tk.call('after','info')):
        script=root.tk.call('after','info',job)[0]
        owner=next((w for w in owners if script in (w._tclCommands or [])),root)
        owner.after_cancel(job)
