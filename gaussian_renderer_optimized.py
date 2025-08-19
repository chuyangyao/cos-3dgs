"""
gaussian_renderer_optimized.py
优化的渲染器，包含高效的分裂计算和梯度聚合
修复JIT编译错误
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
from typing import Dict, Optional, Tuple
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation, build_scaling_rotation, strip_symmetric

"""
gaussian_renderer_optimized.py
优化的分裂计算函数
"""

import torch
import math
from typing import Optional, Dict, Any

def vectorized_compute_splits_continuous_improved(pc, max_splits_global=10, progress=0.0):
    """
    改进的连续分裂计算，确保wave正确影响分裂
    
    Args:
        pc: 高斯模型
        max_splits_global: 最大分裂数
        progress: 训练进度(0-1)
    
    Returns:
        分裂数据字典或None
    """
    
    # 获取基础参数
    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rotation = pc.get_rotation
    
    # 获取特征
    if hasattr(pc, '_features_dc'):
        features_dc = pc._features_dc
        features_rest = pc._features_rest
    else:
        # 如果没有特征，创建默认特征
        features_dc = torch.ones((xyz.shape[0], 1, 3), device=xyz.device)
        features_rest = torch.zeros((xyz.shape[0], 15, 3), device=xyz.device)
    
    # 获取wave参数
    if not hasattr(pc, '_wave'):
        return None
    
    wave = pc._wave
    N = xyz.shape[0]
    device = xyz.device
    epsilon = 1e-8
    
    # 计算wave范数
    wave_norms = torch.norm(wave, dim=1)
    
    # 快速检查：如果所有wave都很小，不进行分裂
    if wave_norms.max() < 0.01:
        return None
    
    # 计算每个高斯的分裂数（基于wave范数）
    # 使用更平滑的映射函数
    normalized_wave = torch.clamp(wave_norms / (wave_norms.max() + epsilon), 0, 1)
    
    # 基于进度的自适应分裂
    progress_factor = min(progress * 2, 1.0)  # 在训练前半段逐渐增加分裂
    
    # 计算连续的分裂强度
    split_strength = normalized_wave * progress_factor
    
    # 计算实际分裂数（向上取整）
    k_values = torch.ceil(split_strength * max_splits_global).long()
    k_values = torch.clamp(k_values, 0, max_splits_global)
    
    # 只对有足够wave的高斯进行分裂
    active_mask = wave_norms > 0.01
    k_values[~active_mask] = 0
    
    # 计算总分裂数
    n_splits_per_gaussian = 2 * k_values + 1  # 每个高斯分裂成2k+1个
    total_splits = n_splits_per_gaussian.sum().item()
    
    if total_splits == N:
        # 没有实际分裂发生
        return None
    
    # 预分配输出张量
    split_xyz = torch.zeros((total_splits, 3), device=device)
    split_opacity = torch.zeros((total_splits, 1), device=device)
    split_scaling = torch.zeros((total_splits, 3), device=device)
    split_rotation = torch.zeros((total_splits, 4), device=device)
    split_features_dc = torch.zeros((total_splits, features_dc.shape[1], features_dc.shape[2]), device=device)
    split_features_rest = torch.zeros((total_splits, features_rest.shape[1], features_rest.shape[2]), device=device)
    original_indices = torch.zeros(total_splits, dtype=torch.long, device=device)
    
    # 批量处理分裂
    current_idx = 0
    
    for i in range(N):
        k = k_values[i].item()
        n_splits = 2 * k + 1
        
        if k == 0:
            # 不分裂，直接复制
            split_xyz[current_idx] = xyz[i]
            split_opacity[current_idx] = opacity[i]
            split_scaling[current_idx] = scaling[i]
            split_rotation[current_idx] = rotation[i]
            split_features_dc[current_idx] = features_dc[i]
            split_features_rest[current_idx] = features_rest[i]
            original_indices[current_idx] = i
            current_idx += 1
        else:
            # 执行分裂
            wave_vec = wave[i]
            wave_norm = wave_norms[i]
            
            # 计算分裂方向（沿wave方向）
            if wave_norm > epsilon:
                wave_dir = wave_vec / wave_norm
            else:
                wave_dir = torch.tensor([1.0, 0.0, 0.0], device=device)
            
            # 生成分裂位置
            for j in range(-k, k + 1):
                # 位置偏移
                offset_factor = j / (k + 1.0) if k > 0 else 0
                offset = wave_dir * offset_factor * wave_norm * 0.5
                
                # 计算分裂后的参数
                split_xyz[current_idx] = xyz[i] + offset
                
                # 调整不透明度（中心高斯更不透明）
                distance_factor = abs(j) / (k + 1.0) if k > 0 else 0
                gaussian_weight = math.exp(-0.5 * (distance_factor * 2) ** 2)
                alpha_factor = 0.3 + 0.7 * gaussian_weight
                split_opacity[current_idx] = opacity[i] * alpha_factor
                
                # 调整缩放（分裂后的高斯更小）
                scale_factor = 0.6 + 0.4 * gaussian_weight
                split_scaling[current_idx] = scaling[i] * scale_factor
                
                # 保持旋转不变
                split_rotation[current_idx] = rotation[i]
                
                # 复制特征
                split_features_dc[current_idx] = features_dc[i]
                split_features_rest[current_idx] = features_rest[i]
                
                # 记录原始索引
                original_indices[current_idx] = i
                current_idx += 1
    
    # 验证索引
    assert current_idx == total_splits, f"Index mismatch: {current_idx} != {total_splits}"
    
    # 返回分裂数据
    split_data = {
        'split_xyz': split_xyz,
        'split_opacity': split_opacity,
        'split_scaling': split_scaling,
        'split_rotation': split_rotation,
        'split_features_dc': split_features_dc,
        'split_features_rest': split_features_rest,
        'n_splits': total_splits,
        'n_original': N,
        'original_indices': original_indices,
    }
    
    return split_data

# ============================================================
# 内存池管理
# ============================================================
class SplitMemoryPool:
    """预分配内存池，避免频繁分配"""
    
    def __init__(self, max_gaussians=2000000, max_splits=5):
        self.max_gaussians = max_gaussians
        self.max_splits = max_splits
        self.max_total = max_gaussians * (2 * max_splits + 1)
        
        # 预分配内存池
        self.pools = {}
        self.allocated = False
        
    def allocate_if_needed(self, device='cuda'):
        """延迟分配，只在需要时分配"""
        if not self.allocated:
            self.pools = {
                'xyz': torch.empty((self.max_total, 3), device=device),
                'opacity': torch.empty((self.max_total, 1), device=device),
                'scaling': torch.empty((self.max_total, 3), device=device),
                'rotation': torch.empty((self.max_total, 4), device=device),
                'features_dc': torch.empty((self.max_total, 1, 3), device=device),
                'features_rest': torch.empty((self.max_total, 15, 3), device=device),
            }
            self.allocated = True
            
    def get_slice(self, key: str, start: int, size: int):
        """获取内存切片"""
        if key not in self.pools:
            raise KeyError(f"Pool {key} not found")
        return self.pools[key][start:start+size]

# 全局内存池
_memory_pool = SplitMemoryPool()

# ============================================================
# 优化的分裂参数计算（移除JIT编译以避免错误）
# ============================================================
def compute_split_parameters_fast(
    wave_norms: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    k_value: int,
    progress: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    快速分裂参数计算（不使用JIT编译）
    """
    batch_size = wave_norms.shape[0]
    n_splits = 2 * k_value + 1
    
    # 预分配输出
    split_opacity = torch.zeros((batch_size * n_splits, 1), device=wave_norms.device)
    split_scaling = torch.zeros((batch_size * n_splits, 3), device=scaling.device)
    split_offsets = torch.zeros((batch_size * n_splits, 3), device=wave_norms.device)
    
    # 计算分裂参数
    for j in range(-k_value, k_value + 1):
        idx = j + k_value
        start_idx = idx * batch_size
        end_idx = start_idx + batch_size
        
        if j == 0:
            # 中心高斯
            alpha = 0.6
            scale_factor = 0.8
        else:
            # 周围高斯
            distance_factor = abs(j) / float(k_value)
            gaussian_weight = math.exp(-0.5 * (distance_factor * 2) ** 2)
            alpha = 0.3 * gaussian_weight / k_value
            scale_factor = 0.6 * (1 - 0.2 * distance_factor)
        
        split_opacity[start_idx:end_idx] = opacity * alpha
        split_scaling[start_idx:end_idx] = scaling * scale_factor
        
        # 偏移量将在外部计算
        
    return split_opacity, split_scaling, split_offsets

# ============================================================
# 优化的分裂计算主函数
# ============================================================
def compute_splits_optimized(pc: GaussianModel, iteration: int = 0, 
                            max_iteration: int = 40000,
                            chunk_size: int = 50000) -> Optional[Dict]:
    """
    优化的分裂计算，使用分块处理和内存池
    
    Args:
        pc: 高斯模型
        iteration: 当前迭代
        max_iteration: 最大迭代
        chunk_size: 分块大小（增加到50000）
    """
    # 获取基础参数
    xyz = pc.get_xyz
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rotation = pc.get_rotation
    features_dc = pc._features_dc
    features_rest = pc._features_rest
    
    # 获取wave参数
    if not hasattr(pc, '_wave'):
        return None
    wave = pc._wave
    
    N = xyz.shape[0]
    device = xyz.device
    epsilon = 1e-8
    
    # 计算wave范数
    wave_norms = torch.norm(wave, dim=1)
    
    # 快速检查：如果所有wave都很小，直接返回
    if wave_norms.max() < 0.05:
        return None
    
    # 计算进度
    progress = min(iteration / max_iteration, 1.0) if max_iteration > 0 else 0.0
    split_strength = min(1.0, progress * 2.0)
    
    # 优化的阈值（提高最小阈值减少分裂）
    min_threshold = 0.1  # 从0.05提高到0.1
    thresholds = torch.tensor([0.1, 0.8, 1.5, 2.5], device=device) * split_strength
    
    # 计算每个高斯的分裂数k
    k_values = torch.zeros(N, dtype=torch.long, device=device)
    for i, thresh in enumerate(thresholds):
        k_values[wave_norms > thresh] = i + 1
    
    # 限制最大分裂数
    max_splits_global = 2  # 限制为2（最多分裂成5个）
    k_values = torch.minimum(k_values, torch.tensor(max_splits_global, device=device))
    
    # 如果没有需要分裂的高斯，返回None
    if k_values.max() == 0:
        return None
    
    # 统计分裂分布
    split_distribution = {}
    for k in range(k_values.max().item() + 1):
        count = (k_values == k).sum().item()
        if count > 0:
            split_distribution[k] = count
    
    # 计算总分裂数
    total_splits = (2 * k_values + 1).sum().item()
    
    # 分配内存池
    _memory_pool.allocate_if_needed(device)
    
    # 使用内存池或新分配
    if total_splits <= _memory_pool.max_total:
        # 使用内存池
        split_xyz = _memory_pool.get_slice('xyz', 0, total_splits)
        split_opacity = _memory_pool.get_slice('opacity', 0, total_splits)
        split_scaling = _memory_pool.get_slice('scaling', 0, total_splits)
        split_rotation = _memory_pool.get_slice('rotation', 0, total_splits)
    else:
        # 超出内存池，新分配
        split_xyz = torch.zeros((total_splits, 3), device=device)
        split_opacity = torch.zeros((total_splits, 1), device=device)
        split_scaling = torch.zeros((total_splits, 3), device=device)
        split_rotation = torch.zeros((total_splits, 4), device=device)
    
    # Features处理
    if features_dc.dim() == 3:
        split_features_dc = torch.zeros((total_splits, features_dc.shape[1], features_dc.shape[2]), device=device)
    else:
        split_features_dc = torch.zeros((total_splits, features_dc.shape[1]), device=device)
    
    if features_rest.dim() == 3:
        split_features_rest = torch.zeros((total_splits, features_rest.shape[1], features_rest.shape[2]), device=device)
    else:
        split_features_rest = torch.zeros((total_splits, features_rest.shape[1]), device=device)
    
    original_indices = []
    current_idx = 0
    
    # 分块处理，使用更大的块
    for k in range(k_values.max().item() + 1):
        mask = (k_values == k)
        if not mask.any():
            continue
        
        indices = torch.where(mask)[0]
        n_gaussians = indices.shape[0]
        
        # 分块处理
        for chunk_start in range(0, n_gaussians, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_gaussians)
            chunk_indices = indices[chunk_start:chunk_end]
            chunk_size_actual = chunk_indices.shape[0]
            
            if k == 0:
                # 不分裂，直接复制
                idx_end = current_idx + chunk_size_actual
                split_xyz[current_idx:idx_end] = xyz[chunk_indices]
                split_opacity[current_idx:idx_end] = opacity[chunk_indices]
                split_scaling[current_idx:idx_end] = scaling[chunk_indices]
                split_rotation[current_idx:idx_end] = rotation[chunk_indices]
                split_features_dc[current_idx:idx_end] = features_dc[chunk_indices]
                split_features_rest[current_idx:idx_end] = features_rest[chunk_indices]
                
                original_indices.extend(chunk_indices.tolist())
                current_idx = idx_end
            else:
                # 使用优化的分裂计算
                n_splits_per = 2 * k + 1
                n_total = chunk_size_actual * n_splits_per
                
                # 获取批量参数
                batch_xyz = xyz[chunk_indices]
                batch_opacity = opacity[chunk_indices]
                batch_scaling = scaling[chunk_indices]
                batch_rotation = rotation[chunk_indices]
                batch_features_dc = features_dc[chunk_indices]
                batch_features_rest = features_rest[chunk_indices]
                batch_wave = wave[chunk_indices]
                
                # 计算wave方向
                batch_wave_norms = torch.norm(batch_wave, dim=1, keepdim=True)
                wave_dir = batch_wave / (batch_wave_norms + epsilon)
                
                # 计算主轴
                R = build_rotation(batch_rotation)
                max_scale_idx = batch_scaling.argmax(dim=1)
                main_scale = batch_scaling.gather(1, max_scale_idx.unsqueeze(1)).squeeze(1)
                
                # 生成分裂位置和参数
                for j in range(-k, k+1):
                    offset = j * 0.5 * main_scale.unsqueeze(1) * wave_dir
                    
                    idx_start = current_idx + (j + k) * chunk_size_actual
                    idx_end = idx_start + chunk_size_actual
                    
                    split_xyz[idx_start:idx_end] = batch_xyz + offset
                    
                    # 使用优化的参数计算
                    if j == 0:
                        alpha = 0.6
                        scale_factor = 0.8
                    else:
                        distance_factor = abs(j) / k
                        gaussian_weight = math.exp(-0.5 * (distance_factor * 2) ** 2)
                        alpha = 0.3 * gaussian_weight / k
                        scale_factor = 0.6 * (1 - 0.2 * distance_factor)
                    
                    split_opacity[idx_start:idx_end] = batch_opacity * alpha
                    split_scaling[idx_start:idx_end] = batch_scaling * scale_factor
                    split_rotation[idx_start:idx_end] = batch_rotation
                    split_features_dc[idx_start:idx_end] = batch_features_dc
                    split_features_rest[idx_start:idx_end] = batch_features_rest
                
                # 记录原始索引
                base_indices = chunk_indices.unsqueeze(1).expand(-1, n_splits_per).flatten()
                original_indices.extend(base_indices.tolist())
                
                current_idx += n_total
    
    # 转换原始索引
    original_indices = torch.tensor(original_indices, device=device, dtype=torch.long)
    
    # 能量归一化（优化版本）
    unique_indices = torch.unique(original_indices)
    for idx in unique_indices:
        mask = (original_indices == idx)
        if mask.any():
            # 归一化该原始高斯的所有分裂
            total_opacity = split_opacity[mask].sum()
            if total_opacity > epsilon:
                target_opacity = opacity[idx]
                split_opacity[mask] *= (target_opacity / total_opacity)
    
    return {
        'split_xyz': split_xyz[:current_idx],
        'split_opacity': split_opacity[:current_idx],
        'split_scaling': split_scaling[:current_idx],
        'split_rotation': split_rotation[:current_idx],
        'split_features_dc': split_features_dc[:current_idx],
        'split_features_rest': split_features_rest[:current_idx],
        'n_splits': current_idx,
        'n_original': N,
        'original_indices': original_indices,
        'split_distribution': split_distribution,
    }

# ============================================================
# 优化的梯度聚合（使用torch.compile代替JIT）
# ============================================================
class OptimizedGradientAggregator:
    """高效的梯度聚合器"""
    
    @staticmethod
    def aggregate_gradients(
        split_grads: torch.Tensor,
        original_indices: torch.Tensor,
        split_opacity: torch.Tensor,
        n_original: int
    ) -> torch.Tensor:
        """
        优化的加权梯度聚合
        使用不透明度作为权重，更符合物理意义
        """
        # 创建输出tensor
        aggregated = torch.zeros((n_original, 3), 
                                device=split_grads.device,
                                dtype=split_grads.dtype)
        
        # 使用不透明度作为权重
        weights = split_opacity.squeeze(-1) if split_opacity.dim() > 1 else split_opacity
        
        # 确保split_grads有正确的维度
        if split_grads.dim() == 2 and split_grads.shape[1] >= 3:
            weighted_grads = split_grads[:, :3] * weights.unsqueeze(1)
        else:
            # 如果维度不匹配，尝试调整
            weighted_grads = split_grads.view(-1, 3) * weights.view(-1, 1)
        
        # 高效聚合
        aggregated.scatter_add_(0,
                               original_indices.unsqueeze(1).expand(-1, 3),
                               weighted_grads)
        
        # 归一化权重
        weight_sum = torch.zeros(n_original, device=split_grads.device)
        weight_sum.scatter_add_(0, original_indices, weights)
        weight_sum = weight_sum.clamp(min=1e-8).unsqueeze(1)
        
        return aggregated / weight_sum

# 全局聚合器实例
_gradient_aggregator = OptimizedGradientAggregator()

# ============================================================
# 优化的渲染函数
# ============================================================
def render_optimized(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                     scaling_modifier: float = 1.0, override_color=None,
                     iteration: int = 0, max_iteration: int = 40000,
                     use_amp: bool = True) -> Dict:
    """
    优化的渲染函数
    
    Args:
        use_amp: 是否使用混合精度
    """
    # 创建零张量
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, 
                                         requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # 设置光栅化配置
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug if hasattr(pipe, 'debug') else False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    # 检查是否需要分裂
    split_data = None
    if hasattr(pc, 'use_splitting') and pc.use_splitting and iteration > 0:
        # 使用缓存机制
        if hasattr(pc, 'get_split_data'):
            split_data = pc.get_split_data(iteration, max_iteration)
        else:
            # 使用优化的分裂计算
            split_data = compute_splits_optimized(pc, iteration, max_iteration)
    
    # 应用分裂
    if split_data is not None:
        means3D = split_data['split_xyz']
        opacity = split_data['split_opacity']
        
        # 扩展means2D
        extended_means2D = torch.zeros((means3D.shape[0], 3), 
                                      dtype=means2D.dtype, 
                                      device=means2D.device, 
                                      requires_grad=True)
        try:
            extended_means2D.retain_grad()
        except:
            pass
        means2D = extended_means2D
        
        # 处理协方差
        if pipe.compute_cov3D_python:
            scales_modified = scaling_modifier * split_data['split_scaling']
            L = build_scaling_rotation(scales_modified, split_data['split_rotation'])
            actual_covariance = torch.bmm(L, L.transpose(1, 2))
            cov3D_precomp = strip_symmetric(actual_covariance)
            scales = None
            rotations = None
        else:
            scales = split_data['split_scaling']
            rotations = split_data['split_rotation']
            cov3D_precomp = None
        
        # 处理颜色
        if override_color is not None:
            colors_precomp = override_color.unsqueeze(0).expand(means3D.shape[0], -1)
            shs = None
        elif pipe.convert_SHs_python:
            shs_view = torch.cat([split_data['split_features_dc'],
                                 split_data['split_features_rest']], dim=1)
            shs_view = shs_view.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            shs = None
        else:
            shs = torch.cat([split_data['split_features_dc'],
                           split_data['split_features_rest']], dim=1)
            colors_precomp = None
    else:
        # 不使用分裂
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
            scales = None
            rotations = None
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation
            cov3D_precomp = None
        
        if override_color is not None:
            colors_precomp = override_color
            shs = None
        elif pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            shs = None
        else:
            shs = pc.get_features
            colors_precomp = None
    
    # 光栅化
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )
    
    # 返回结果
    result = {
        "render": rendered_image,
        "viewspace_points": means2D if split_data is None else screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
    
    # 添加分裂信息（如果有）
    if split_data is not None:
        result.update({
            "split_viewspace_points": means2D,
            "split_info": {
                "n_original": split_data['n_original'],
                "original_indices": split_data['original_indices'],
                "split_opacity": split_data['split_opacity']
            },
            "split_data": split_data
        })
    
    return result

# ============================================================
# 梯度处理函数
# ============================================================
def process_gradients_optimized(render_output: Dict, gaussians: GaussianModel):
    """
    优化的梯度处理
    """
    if 'split_viewspace_points' not in render_output:
        return
    
    split_points = render_output['split_viewspace_points']
    split_info = render_output['split_info']
    
    if split_points.grad is None:
        return
    
    # 使用优化的梯度聚合
    aggregated_grad = _gradient_aggregator.aggregate_gradients(
        split_points.grad,
        split_info['original_indices'],
        split_info['split_opacity'],
        split_info['n_original']
    )
    
    # 更新原始梯度
    viewspace_points = render_output['viewspace_points']
    if viewspace_points.grad is None:
        viewspace_points.grad = torch.zeros_like(viewspace_points)
    viewspace_points.grad[:, :3] = aggregated_grad

# 提供向后兼容的接口
def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, 
          override_color=None, iteration=0, max_iteration=40000):
    """向后兼容的渲染接口"""
    return render_optimized(viewpoint_camera, pc, pipe, bg_color, 
                          scaling_modifier, override_color, 
                          iteration, max_iteration, use_amp=False)