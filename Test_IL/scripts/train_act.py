#!/usr/bin/env python3
"""Use the installed workshop training loop, optimizer, processors and checkpoints."""
import json
import os
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ['TOKENIZERS_PARALLELISM']='false'
from studio.store import atomic_json
from studio.checkpoints import prune_previous_checkpoints


class TrainingStopped(Exception):
    """Raised only after a complete checkpoint has been published."""


def run(directory, smoke=False):
    import torch
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.scripts import lerobot_train as training
    from lerobot.utils.utils import init_logging
    directory=Path(directory);path=directory/'job.json';job=json.loads(path.read_text());p=job['params']
    # Keep USB/camera owner responsive during simultaneous collection.
    os.nice(10)
    resumed=job.get('resume_checkpoint')
    if p['device']=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA를 사용할 수 없습니다. CPU를 선택하세요.')
    torch.set_num_threads(2)
    policy=ACTConfig(device=p['device'],push_to_hub=False,chunk_size=p['chunk_size'],n_action_steps=p['n_action_steps'],
        optimizer_lr=p['learning_rate'],optimizer_lr_backbone=p['learning_rate'],
        pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1' if p['pretrained_backbone'] else None)
    if smoke:
        if not job['synthetic']:raise ValueError('Tiny smoke model requires synthetic fixtures')
        policy.dim_model=32;policy.dim_feedforward=64;policy.n_encoder_layers=1;policy.n_decoder_layers=1;policy.n_vae_encoder_layers=1;policy.latent_dim=8
    cfg=TrainPipelineConfig(dataset=DatasetConfig(job['repo_id'],root=job['dataset'],episodes=job['episodes'],video_backend='pyav'),
        policy=policy,output_dir=directory/'output',job_name=job['name'],steps=p['steps'],batch_size=p['batch_size'],
        num_workers=p['num_workers'],seed=p['seed'],save_freq=p['save_freq'],log_freq=min(20,p['steps']),eval_freq=0)
    if resumed:
        cfg=TrainPipelineConfig.from_pretrained(Path(resumed)/'train_config.json',local_files_only=True)
        cfg.resume=True;cfg.output_dir=directory/'output';cfg.job_name=job['name']
        cfg.dataset=DatasetConfig(job['repo_id'],root=job['dataset'],episodes=job['episodes'],video_backend='pyav')
        cfg.steps=p['steps'];cfg.save_freq=p['save_freq'];cfg.eval_freq=0
        cfg.policy.device=p['device'];cfg.policy.push_to_hub=False;cfg.policy.pretrained_backbone_weights=None
        cfg.wandb.enable=False
        # Installed LeRobot validates resume via this CLI argument even when
        # called programmatically; policy/optimizer/RNG loading stays native.
        sys.argv=[sys.argv[0],f'--config_path={Path(resumed)/"train_config.json"}']
    original_update=training.update_policy;step=job.get('start_step',0);began=time.monotonic()
    def update(*args,**kwargs):
        nonlocal step
        value=original_update(*args,**kwargs);step+=1
        if (directory/'stop_requested').exists():
            cfg.save_freq=1  # Native loop checkpoints this completed update.
        if step==job.get('start_step',0)+1 or step%5==0 or step==p['steps'] or (directory/'stop_requested').exists():
            atomic_json(directory/'progress.json',dict(step=step,total=p['steps'],loss=float(value[0].loss.avg),elapsed=time.monotonic()-began))
        return value
    training.update_policy=update
    original_save=training.save_checkpoint
    def save(*args,**kwargs):
        # The native loop has already decided to save; persist the normal
        # cadence rather than the temporary save-on-stop value.
        cfg.save_freq=p['save_freq']
        result=original_save(*args,**kwargs)
        atomic_json(Path(kwargs['checkpoint_dir'])/'studio_ready.json',dict(step=kwargs['step']))
        # Update the native alias before pruning, including save-on-stop.
        training.update_last_checkpoint(Path(kwargs['checkpoint_dir']))
        if p.get('keep_latest_only',False):
            prune_previous_checkpoints(Path(kwargs['checkpoint_dir']))
        if (directory/'stop_requested').exists():
            raise TrainingStopped()
        return result
    training.save_checkpoint=save
    init_logging()
    job.update(state='running',message='ACT 학습 중');atomic_json(path,job)
    try:training.train(cfg)
    except TrainingStopped:
        job=json.loads(path.read_text())
        job.update(state='cancelled',message=f'{step:,}회 학습 상태를 저장하고 중단했습니다.',ended_at=time.time(),saved_step=step)
        atomic_json(path,job)
        return
    job=json.loads(path.read_text())
    if job['state']!='cancelled':job.update(state='completed',message='ACT 학습 완료',ended_at=time.time());atomic_json(path,job)

if __name__=='__main__':
    directory=Path(sys.argv[1]);smoke='--smoke' in sys.argv;sys.argv=sys.argv[:1]
    try:run(directory,smoke)
    except Exception as error:
        path=directory/'job.json';job=json.loads(path.read_text());job.update(state='error',message=str(error));atomic_json(path,job)
        raise
