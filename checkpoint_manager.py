"""
checkpoint_manager.py
完整的checkpoint保存和恢复系统
"""

import os
import torch
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class CheckpointManager:
    """管理模型的checkpoint保存和加载"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.checkpoint_dir = os.path.join(model_path, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, 
                       gaussians,
                       iteration: int,
                       optimizer_state: Optional[dict] = None,
                       scene_info: Optional[dict] = None,
                       config: Optional[dict] = None,
                       save_full: bool = True):
        """
        保存完整的训练checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:07d}.pth")
        
        # 收集所有需要保存的状态
        checkpoint = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'model_type': gaussians.__class__.__name__,
        }
        
        # 保存模型参数
        model_state = {
            'active_sh_degree': gaussians.active_sh_degree,
            'max_sh_degree': gaussians.max_sh_degree,
            '_xyz': gaussians._xyz.detach().cpu(),
            '_features_dc': gaussians._features_dc.detach().cpu(),
            '_features_rest': gaussians._features_rest.detach().cpu(),
            '_opacity': gaussians._opacity.detach().cpu(),
            '_scaling': gaussians._scaling.detach().cpu(),
            '_rotation': gaussians._rotation.detach().cpu(),
            'spatial_lr_scale': gaussians.spatial_lr_scale,
        }
        
        # 添加模型特定的参数
        if hasattr(gaussians, '_shape'):
            model_state['_shape'] = gaussians._shape.detach().cpu()
            model_state['shape_strngth'] = gaussians.shape_strngth if hasattr(gaussians, 'shape_strngth') else 1.0
            model_state['prune_shape_threshold'] = gaussians.prune_shape_threshold if hasattr(gaussians, 'prune_shape_threshold') else 0.2
        
        if hasattr(gaussians, '_wave'):
            model_state['_wave'] = gaussians._wave.detach().cpu()
            model_state['use_splitting'] = gaussians.use_splitting if hasattr(gaussians, 'use_splitting') else False
            model_state['_max_splits'] = gaussians._max_splits if hasattr(gaussians, '_max_splits') else 10
            model_state['_split_factor'] = gaussians._split_factor if hasattr(gaussians, '_split_factor') else 1.6
        
        checkpoint['model_state'] = model_state
        
        # 保存完整状态
        if save_full:
            full_state = {
                'max_radii2D': gaussians.max_radii2D.detach().cpu() if hasattr(gaussians, 'max_radii2D') else None,
                'xyz_gradient_accum': gaussians.xyz_gradient_accum.detach().cpu() if hasattr(gaussians, 'xyz_gradient_accum') else None,
                'denom': gaussians.denom.detach().cpu() if hasattr(gaussians, 'denom') else None,
                'percent_dense': gaussians.percent_dense if hasattr(gaussians, 'percent_dense') else 0.01,
            }
            checkpoint['full_state'] = full_state
        
        # 保存优化器状态
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        elif hasattr(gaussians, 'optimizer') and gaussians.optimizer is not None:
            checkpoint['optimizer_state'] = gaussians.optimizer.state_dict()
        
        # 保存场景信息
        if scene_info is not None:
            checkpoint['scene_info'] = scene_info
        
        # 保存配置
        if config is not None:
            checkpoint['config'] = config
        
        # 保存checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # 保存元信息
        meta_path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:07d}_meta.json")
        meta_info = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'num_gaussians': gaussians._xyz.shape[0],
            'model_type': gaussians.__class__.__name__,
            'checkpoint_file': os.path.basename(checkpoint_path),
        }
        
        if hasattr(gaussians, '_wave'):
            wave_norms = torch.norm(gaussians._wave, dim=1)
            meta_info['wave_stats'] = {
                'active_waves': int((wave_norms > 0.01).sum()),
                'mean_norm': float(wave_norms.mean()),
                'max_norm': float(wave_norms.max()),
            }
        
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)
        
        print(f"[Checkpoint] Saved checkpoint at iteration {iteration}")
        print(f"  File: {checkpoint_path}")
        print(f"  Gaussians: {gaussians._xyz.shape[0]}")
        if 'wave_stats' in meta_info:
            print(f"  Active waves: {meta_info['wave_stats']['active_waves']}")
        
        return checkpoint_path
    
    def load_checkpoint(self, 
                       gaussians,
                       checkpoint_path: Optional[str] = None,
                       iteration: Optional[int] = None,
                       load_optimizer: bool = True,
                       strict: bool = False) -> Dict[str, Any]:
        """
        加载checkpoint
        """
        # 确定要加载的checkpoint路径
        if checkpoint_path is None:
            if iteration is not None:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{iteration:07d}.pth")
            else:
                # 加载最新的checkpoint
                checkpoints = list(Path(self.checkpoint_dir).glob("checkpoint_*.pth"))
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
                checkpoint_path = str(max(checkpoints, key=lambda x: x.stat().st_mtime))
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"[Checkpoint] Loading checkpoint from {checkpoint_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 恢复模型状态
        model_state = checkpoint['model_state']
        
        # 基本参数
        gaussians.active_sh_degree = model_state['active_sh_degree']
        gaussians.max_sh_degree = model_state['max_sh_degree']
        gaussians._xyz = torch.nn.Parameter(model_state['_xyz'].cuda().requires_grad_(True))
        gaussians._features_dc = torch.nn.Parameter(model_state['_features_dc'].cuda().requires_grad_(True))
        gaussians._features_rest = torch.nn.Parameter(model_state['_features_rest'].cuda().requires_grad_(True))
        gaussians._opacity = torch.nn.Parameter(model_state['_opacity'].cuda().requires_grad_(True))
        gaussians._scaling = torch.nn.Parameter(model_state['_scaling'].cuda().requires_grad_(True))
        gaussians._rotation = torch.nn.Parameter(model_state['_rotation'].cuda().requires_grad_(True))
        gaussians.spatial_lr_scale = model_state['spatial_lr_scale']
        
        # 恢复模型特定参数
        if '_shape' in model_state and hasattr(gaussians, '_shape'):
            gaussians._shape = torch.nn.Parameter(model_state['_shape'].cuda().requires_grad_(True))
            if 'shape_strngth' in model_state:
                gaussians.shape_strngth = model_state['shape_strngth']
            if 'prune_shape_threshold' in model_state:
                gaussians.prune_shape_threshold = model_state['prune_shape_threshold']
        
        if '_wave' in model_state and hasattr(gaussians, '_wave'):
            gaussians._wave = torch.nn.Parameter(model_state['_wave'].cuda().requires_grad_(True))
            if 'use_splitting' in model_state:
                gaussians.use_splitting = model_state['use_splitting']
            if '_max_splits' in model_state:
                gaussians._max_splits = model_state['_max_splits']
            if '_split_factor' in model_state:
                gaussians._split_factor = model_state['_split_factor']
        
        # 恢复完整状态
        if 'full_state' in checkpoint:
            full_state = checkpoint['full_state']
            if full_state['max_radii2D'] is not None:
                gaussians.max_radii2D = full_state['max_radii2D'].cuda()
            if full_state['xyz_gradient_accum'] is not None:
                gaussians.xyz_gradient_accum = full_state['xyz_gradient_accum'].cuda()
            if full_state['denom'] is not None:
                gaussians.denom = full_state['denom'].cuda()
            if 'percent_dense' in full_state:
                gaussians.percent_dense = full_state['percent_dense']
        
        # 恢复优化器状态
        optimizer_state = None
        if load_optimizer and 'optimizer_state' in checkpoint:
            optimizer_state = checkpoint['optimizer_state']
            if hasattr(gaussians, 'optimizer') and gaussians.optimizer is not None:
                try:
                    gaussians.optimizer.load_state_dict(optimizer_state)
                    print("  Optimizer state loaded")
                except Exception as e:
                    print(f"  Warning: Failed to load optimizer state: {e}")
        
        # 返回加载信息
        load_info = {
            'iteration': checkpoint['iteration'],
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'num_gaussians': gaussians._xyz.shape[0],
            'model_type': checkpoint.get('model_type', 'unknown'),
            'optimizer_loaded': optimizer_state is not None,
            'config': checkpoint.get('config', None),
            'scene_info': checkpoint.get('scene_info', None),
        }
        
        print(f"  Loaded iteration: {load_info['iteration']}")
        print(f"  Gaussians: {load_info['num_gaussians']}")
        
        if hasattr(gaussians, '_wave'):
            wave_norms = torch.norm(gaussians._wave, dim=1)
            active_waves = (wave_norms > 0.01).sum()
            print(f"  Active waves: {active_waves}/{len(wave_norms)}")
        
        return load_info
    
    def list_checkpoints(self) -> list:
        """列出所有可用的checkpoints"""
        checkpoints = []
        
        for meta_file in Path(self.checkpoint_dir).glob("checkpoint_*_meta.json"):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            checkpoints.append(meta)
        
        # 按迭代次数排序
        checkpoints.sort(key=lambda x: x['iteration'])
        
        return checkpoints
    
    def cleanup_old_checkpoints(self, keep_latest: int = 5, keep_iterations: list = None):
        """清理旧的checkpoints，保留最新的几个"""
        if keep_iterations is None:
            keep_iterations = []
        
        checkpoints = list(Path(self.checkpoint_dir).glob("checkpoint_*.pth"))
        if len(checkpoints) <= keep_latest:
            return
            
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # 确定要删除的checkpoints
        to_delete = []
        for ckpt in checkpoints[:-keep_latest]:
            # 提取迭代次数
            try:
                iteration = int(ckpt.stem.split('_')[1])
                if iteration not in keep_iterations:
                    to_delete.append(ckpt)
            except:
                continue
        
        # 删除旧的checkpoints
        for ckpt in to_delete:
            ckpt.unlink()
            # 同时删除对应的meta文件
            meta_file = ckpt.parent / f"{ckpt.stem}_meta.json"
            if meta_file.exists():
                meta_file.unlink()
            print(f"  Deleted old checkpoint: {ckpt.name}")
        
        if to_delete:
            print(f"[Checkpoint] Cleaned up {len(to_delete)} old checkpoints")