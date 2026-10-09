"""Remove unreferenced generated datasets, never raw episodes or model files."""
from contextlib import contextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import re
import shutil


@contextmanager
def training_reference_guard(root):
    """Protect dataset selection until its training job reference is on disk."""
    with (Path(root)/'.lerobot-retention.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _references(root, versions):
    kept = set()

    def protect(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Invalid training dataset reference')
        target = Path(value)
        if not target.is_absolute():
            raise ValueError('Relative training dataset reference')
        target = target.resolve()
        if target.is_relative_to(versions):
            relative = target.relative_to(versions)
            if not relative.parts:
                raise ValueError('Dataset points to the versions directory itself')
            if not (versions/relative.parts[0]).is_dir():
                raise ValueError('Referenced dataset is missing: '+str(target))
            kept.add(relative.parts[0])

    # Appending the filename preserves dangling links; glob('*/lerobot') can
    # silently omit them, which would make an uncertain reference look unused.
    for link in [root/'lerobot', *(folder/'lerobot' for folder in sorted((root/'tasks').glob('*')))]:
        if link.is_symlink():
            protect(str(link.resolve()))

    # All job states are retained, including completed/interrupted runs. They
    # can still be used for same-data resume even after episodes are edited.
    for path in sorted((root/'training').glob('*/job.json')):
        job = json.loads(path.read_text())
        protect(job['dataset'])
        for config_path in sorted((path.parent/'output/checkpoints').glob('*/pretrained_model/train_config.json')):
            config = json.loads(config_path.read_text())
            if not isinstance(config, dict):
                raise ValueError('Invalid checkpoint dataset configuration')
            dataset = config.get('dataset')
            if dataset is not None:
                protect(dataset['root'])
    return kept


def _disk_bytes(path):
    total = path.lstat().st_blocks*512
    for base, dirs, files in os.walk(path, followlinks=False):
        for name in dirs+files:
            total += (Path(base)/name).lstat().st_blocks*512
    return total


def prune_dataset_versions(root, *, dry_run=False):
    """Best-effort cleanup after successful export; busy/uncertain means keep.

    Lock order: writer -> retention -> source. Training takes retention before
    the existing nonblocking writer admission check, so it cannot deadlock with
    cleanup or lose its dataset between snapshot() and job.json publication.
    """
    root = Path(root).resolve()
    versions = root/'.lerobot_versions'
    report = dict(status='dry_run' if dry_run else 'cleaned', kept=[], candidates=[],
                  removed=[], candidate_bytes=0, removed_bytes=0, errors=[])
    if not versions.exists():
        return report
    try:
        if versions.is_symlink() or not versions.is_dir():
            raise ValueError('Generated dataset directory must not be a link')
        with (root/'.lerobot-write.lock').open('a') as writer:
            try:
                fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                report['status'] = 'busy'
                return report
            with training_reference_guard(root), (root/'.source.lock').open('a') as source:
                fcntl.flock(source, fcntl.LOCK_EX)
                kept = _references(root, versions)
                report['kept'] = sorted(kept)
                candidates = [p for p in sorted(versions.iterdir())
                              if re.fullmatch(r'[0-9a-f]{32}', p.name)
                              and not p.is_symlink() and p.is_dir() and p.name not in kept]
                # Complete reference/size inspection before the first deletion.
                sizes = {p.name: _disk_bytes(p) for p in candidates}
                report['candidates'] = list(sizes)
                report['candidate_bytes'] = sum(sizes.values())
                if not dry_run:
                    for path in candidates:
                        try:
                            shutil.rmtree(path)
                            report['removed'].append(path.name)
                            report['removed_bytes'] += sizes[path.name]
                        except OSError as error:
                            report['errors'].append(str(error))
                    if report['errors']:
                        report['status'] = 'partial'
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report['status'] = 'skipped'
        report['errors'].append(str(error))
    if report['errors']:
        logging.warning('Dataset cleanup %s: %s', report['status'], '; '.join(report['errors']))
    return report
