#!/usr/bin/env python3
"""Official ACT policy and its saved pre/post processors. Never opens hardware."""
import base64
import io
import json
import os
from pathlib import Path
import sys
import time
from contextlib import redirect_stdout
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_DATASETS_OFFLINE']='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from studio.execution import options
protocol=sys.stdout

def send(value):protocol.write(json.dumps(value,allow_nan=False)+'\n');protocol.flush()

def main():
    with redirect_stdout(sys.stderr):
        import numpy as np
        import torch
        from PIL import Image
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.control_utils import predict_action, prepare_observation_for_inference
        torch.set_num_threads(2)
        checkpoint,device=sys.argv[1:3]
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA를 사용할 수 없습니다.')
        config=PreTrainedConfig.from_pretrained(checkpoint)
        runtime=options(json.loads(sys.argv[3]) if len(sys.argv)>3 else {},dict(n_action_steps=config.n_action_steps,chunk_size=config.chunk_size))
        config.device=device;config.pretrained_backbone_weights=None
        config.n_action_steps=runtime['action_steps']
        config.temporal_ensemble_coeff=runtime['ensemble_coeff'] if runtime['mode']=='ensemble' else None
        policy=ACTPolicy.from_pretrained(checkpoint,config=config,local_files_only=True,strict=True)
        policy.config.device=device;policy.to(device);policy.eval();policy.reset()
        pre,post=make_pre_post_processors(policy.config,pretrained_path=checkpoint,
            preprocessor_overrides={'device_processor':{'device':device}},
            postprocessor_overrides={'device_processor':{'device':'cpu'}})
        pre.reset();post.reset()
        def infer(obs):
            computed=runtime['mode']!='original' or len(policy._action_queue)==0
            if device=='cuda':torch.cuda.synchronize()
            began=time.monotonic()
            if runtime['mode']=='prefetch':
                with torch.inference_mode():
                    batch=pre(prepare_observation_for_inference(dict(obs),torch.device(device),None,None))
                    result=post(policy.predict_action_chunk(batch)).detach().cpu().numpy()[0]
                if result.ndim!=2 or result.shape[1]!=6 or not np.isfinite(result).all():raise ValueError('잘못된 동작 묶음입니다.')
                payload=dict(actions=result.tolist())
            else:
                result=predict_action(obs,policy,torch.device(device),pre,post,policy.config.use_amp)
                action=result.detach().cpu().numpy().reshape(-1)
                if action.shape!=(6,) or not np.isfinite(action).all():raise ValueError('모델이 올바른 6개 관절 목표를 내지 않았습니다.')
                payload=dict(action=action.tolist())
            elapsed=time.monotonic()-began
            return dict(**payload,computed_chunk=computed,compute_ms=elapsed*1000 if computed else None,
                        remaining=len(policy._action_queue) if runtime['mode']=='original' else 0)
        # Warm the real inference path before motors can be enabled.
        obs={'observation.state':np.zeros(6,dtype=np.float32)}
        for key,feature in policy.config.image_features.items():
            _,height,width=feature.shape;obs[key]=np.zeros((height,width,3),dtype=np.uint8)
        warm=infer(obs)
        policy.reset();pre.reset();post.reset()
        send(dict(kind='ready',warm_inference_ms=warm['compute_ms'],options=runtime))
        for line in sys.stdin:
            item=json.loads(line);started=time.monotonic()
            obs={'observation.state':np.asarray(item['state'],dtype=np.float32)}
            for src,dst in [('gripper','wrist'),('table','top')]:
                with Image.open(io.BytesIO(base64.b64decode(item['cameras'][src]))) as img:
                    obs['observation.images.'+dst]=np.array(img.convert('RGB'),copy=True)
            result=infer(obs)
            send(dict(**result,acquired=item['acquired'],observation_at=item.get('observation_at',item['acquired']),elapsed=time.monotonic()-started))


if __name__=='__main__':
    try:main()
    except Exception as error:
        send(dict(error=str(error)))
        raise
