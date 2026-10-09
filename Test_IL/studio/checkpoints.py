"""Keep completed checkpoints within one training run."""
import json
import logging
import shutil
from pathlib import Path


def prune_previous_checkpoints(current):
    """Call only after save and publication; never traverse another run or links."""
    current = Path(current)
    if current.is_symlink() or not current.name.isdecimal():
        raise ValueError('Invalid checkpoint directory')
    required = [
        'studio_ready.json', 'pretrained_model/model.safetensors',
        'pretrained_model/train_config.json', 'pretrained_model/policy_preprocessor.json',
        'pretrained_model/policy_postprocessor.json', 'training_state/training_step.json',
        'training_state/optimizer_state.safetensors', 'training_state/optimizer_param_groups.json',
        'training_state/rng_state.safetensors',
    ]
    if not all((current / name).is_file() for name in required):
        raise ValueError('New checkpoint is incomplete; previous models preserved')
    step = int(current.name)
    for name in ('studio_ready.json', 'training_state/training_step.json'):
        if json.loads((current / name).read_text())['step'] != step:
            raise ValueError('Checkpoint step mismatch; previous models preserved')
    removed = []
    for old in current.parent.iterdir():
        if old.is_symlink() or not old.is_dir() or not old.name.isdecimal():
            continue
        if int(old.name) >= step or not (old / 'studio_ready.json').is_file():
            continue
        try:
            # Hide from the model selector before deleting large files.
            (old / 'studio_ready.json').unlink()
            shutil.rmtree(old)
            removed.append(old.name)
        except OSError:
            logging.exception('Could not remove old checkpoint %s; latest model is preserved', old)
    return removed
