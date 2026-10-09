"""Small, nonblocking notifications to the launcher; never accesses hardware."""
import json
import os


def notify(event, **values):
    descriptor=os.environ.get('TEST_TEACH_LIFECYCLE_FD')
    if descriptor is None:return
    try:
        os.write(int(descriptor),(json.dumps(dict(event=event,**values))+'\n').encode())
    except (OSError,ValueError):
        # A vanished supervisor must not interrupt motor cleanup.
        pass
