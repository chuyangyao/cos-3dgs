"""
gaussian_renderer/__init__.py
修复后的渲染器，添加调试验证和改进的分裂计算
"""

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from gaussian_renderer_optimized import vectorized_compute_splits_continuous_improved

# 导入调试和配置管理
from debug_validator import debug_validator
from config_manager import config_manager

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, 
          scaling_modifier=1.0, override_color=None, iteration=0, max_iteration=40000):
    """
    Render the scene with improved continuous splitting.
    
    Background tensor (bg_color) must be on GPU!
    """
    
    # 创建零张量
    screenspace_points = torch.zeros_like(pc.get_xyz, memory_format=torch.contiguous_format, 
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
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取原始高斯参数
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    
    # 颜色处理
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            shs = None
        else:
            shs = pc.get_features
            colors_precomp = None
    else:
        colors_precomp = override_color
        shs = None

    cov3D_precomp = None
    
    # 记录原始高斯数量
    n_original = means3D.shape[0]
    
    # 获取配置
    config = config_manager.config
    
    # 根据迭代决定是否使用分裂
    use_splitting = config.should_use_splitting(iteration) and hasattr(pc, 'use_splitting') and pc.use_splitting
    
    # 定期验证（只在训练时）
    if iteration > 0 and iteration % 500 == 0:
        validation_result = debug_validator.validate_splitting(pc, iteration, verbose=(iteration % 2000 == 0))
    
    # 调试输出
    if iteration > 25000 and iteration % 100 == 0:
        print(f"\n[DEBUG Render] Iter {iteration}:")
        print(f"  use_splitting: {use_splitting}")
        if hasattr(pc, '_wave'):
            wave_norms = torch.norm(pc._wave, dim=1)
            active_waves = (wave_norms > 0.01).sum()
            print(f"  Wave: mean={wave_norms.mean():.4f}, max={wave_norms.max():.4f}, active={active_waves}/{len(wave_norms)}")
    
    split_data = None
    
    # 分裂计算
    if use_splitting and iteration > 0:
        # 使用改进的分裂计算
        progress = min(iteration / max_iteration, 1.0)
        max_splits = pc._max_splits if hasattr(pc, '_max_splits') else 10
        
        # 使用缓存机制（如果有）
        if hasattr(pc, 'get_split_data'):
            split_data = pc.get_split_data(iteration, max_iteration)
        else:
            split_data = vectorized_compute_splits_continuous_improved(
                pc, max_splits_global=max_splits, progress=progress
            )
        
        if split_data is not None:
            # 只在关键迭代打印信息
            if iteration % 500 == 0:
                print(f"[Iter {iteration}] Original: {n_original}, Split: {split_data['split_xyz'].shape[0]}")
            
            # 使用分裂的高斯替换原始高斯
            means3D = split_data['split_xyz']
            opacity = split_data['split_opacity']
            scales = split_data['split_scaling']
            rotations = split_data['split_rotation']
            
            # 扩展means2D
            extended_means2D = torch.zeros(
                (means3D.shape[0], 3), 
                device=means2D.device, 
                dtype=means2D.dtype,
                requires_grad=means2D.requires_grad
            )
            screenspace_points = extended_means2D
            means2D = extended_means2D
            
            # 处理颜色
            if colors_precomp is not None:
                # 扩展预计算颜色
                original_indices = split_data['original_indices']
                colors_precomp = colors_precomp[original_indices]
            elif shs is not None:
                # 使用分裂的SH系数
                shs = torch.cat([split_data['split_features_dc'], 
                               split_data['split_features_rest']], dim=1)
    
    # 警告：检查颜色是否为None
    if shs is None and colors_precomp is None:
        print(f"[WARNING] Both shs and colors_precomp are None at iteration {iteration}, using default features")
        # 创建默认颜色
        colors_precomp = torch.ones((means3D.shape[0], 3), device="cuda") * 0.5
    
    # 渲染
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

    # 返回渲染结果
    return {
        "render": rendered_image,
        "viewspace_points": means2D if split_data is None else screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "split_viewspace_points": means2D if split_data is not None else None,
        "split_info": {
            "n_original": n_original,
            "original_indices": split_data['original_indices'] if split_data else None
        } if split_data is not None else None,
        "split_data": split_data
    }