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
from .splits import compute_splits_precise

# 导入调试和配置管理
from debug_validator import debug_validator
from config_manager import config_manager as CFG_MGR

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
        # 预分裂渲染用预评估颜色；为兼容动态/评估，仍传 active_sh_degree 但shs=None
        sh_degree=pc.active_sh_degree if hasattr(pc, 'active_sh_degree') else 0,
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
    
    # 颜色处理（统一在Python端将SH评估为颜色，保证梯度与形状一致）
    shs = None
    if override_color is not None:
        colors_precomp = override_color
    else:
        expected_sh = (pc.max_sh_degree + 1) ** 2
        # 初始不分裂时的特征: get_features 返回 [N, SH, 3]，需转为 [N, 3, SH]
        shs_view = pc.get_features.transpose(1, 2).contiguous()
        # 裁剪/填充到 expected_sh（沿最后一维）
        cur_sh = shs_view.shape[2]
        if cur_sh < expected_sh:
            pad = expected_sh - cur_sh
            shs_view = torch.cat([shs_view, torch.zeros((shs_view.shape[0], 3, pad), device=shs_view.device, dtype=shs_view.dtype)], dim=2)
        elif cur_sh > expected_sh:
            shs_view = shs_view[:, :, :expected_sh]
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

    cov3D_precomp = None
    
    # 记录原始高斯数量
    n_original = means3D.shape[0]
    
    # 获取配置
    config = CFG_MGR.config
    
    # 根据迭代决定是否使用分裂（方法从管理器调用）
    use_splitting = CFG_MGR.should_use_splitting(iteration) and hasattr(pc, 'use_splitting') and pc.use_splitting
    
    # 定期验证（只在训练时）
    if iteration > 0 and iteration % 500 == 0:
        validation_result = debug_validator.validate_splitting(pc, iteration, verbose=(iteration % 2000 == 0))
    
    # 调试输出
    if iteration > 25000 and iteration % 100 == 0:
        print(f"\n[DEBUG Render] Iter {iteration}:")
        print(f"  use_splitting: {use_splitting}")
        if hasattr(pc, '_wave'):
            wave_norms = torch.norm(pc._wave, dim=1)
            thr = getattr(CFG_MGR.config, 'wave_threshold', 1e-4)
            active_waves = (wave_norms > thr).sum()
            print(f"  Wave: mean={wave_norms.mean():.4f}, max={wave_norms.max():.4f}, active={active_waves}/{len(wave_norms)} (thr={thr})")
    
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
            # 优先使用精确且高效的实现，失败再回退
            split_data = compute_splits_precise(pc, iteration=iteration, max_iteration=max_iteration, max_k=max_splits)
            if split_data is None:
                split_data = vectorized_compute_splits_continuous_improved(
                    pc, max_splits_global=max_splits, progress=progress
                )
        
        if split_data is not None:
            # 只在关键迭代打印信息
            if iteration % 500 == 0:
                print(f"[Iter {iteration}] Original: {n_original}, Split: {split_data['split_xyz'].shape[0]}")
            
            # 使用分裂的高斯参数用于光栅化
            # 安全阈值：若分裂结果过大（例如仅中心复制导致的 N->N），则直接放弃此次分裂
            try:
                max_ratio = getattr(CFG_MGR.config, 'split_max_points_ratio', 1.10)
            except Exception:
                max_ratio = 1.10
            if split_data['split_xyz'].shape[0] > int(max_ratio * n_original):
                split_data = None
            
        if split_data is not None:
            split_means3D = split_data['split_xyz']
            opacity = split_data['split_opacity']
            scales = split_data['split_scaling']
            rotations = split_data['split_rotation']
            
            # 分裂版屏幕空间点，仅用于光栅化与split梯度；保留原screenspace_points供上游聚合梯度
            split_means2D = torch.zeros(
                (split_means3D.shape[0], 3), 
                device=means2D.device, 
                dtype=means2D.dtype,
                requires_grad=True
            )
            means3D = split_means3D
            
            # 处理颜色：分裂后用分裂的SH在Python端评估
            split_shs_view = torch.cat([
                split_data['split_features_dc'], 
                split_data['split_features_rest']
            ], dim=1).transpose(1, 2).contiguous()
            # 裁剪/填充到 expected_sh
            cur_sh = split_shs_view.shape[2]
            expected_sh = (pc.max_sh_degree + 1) ** 2
            if cur_sh < expected_sh:
                pad = expected_sh - cur_sh
                split_shs_view = torch.cat([split_shs_view, torch.zeros((split_shs_view.shape[0], 3, pad), device=split_shs_view.device, dtype=split_shs_view.dtype)], dim=2)
            elif cur_sh > expected_sh:
                split_shs_view = split_shs_view[:, :, :expected_sh]
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, split_shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            shs = None

    # 此时始终使用 colors_precomp，shs=None，避免后端反传形状不匹配
    
    # 警告：检查颜色是否为None
    if shs is None and colors_precomp is None:
        print(f"[WARNING] Both shs and colors_precomp are None at iteration {iteration}, using default features")
        # 创建默认颜色
        colors_precomp = torch.ones((means3D.shape[0], 3), device="cuda") * 0.5
    
    # 渲染
    means2D_input = split_means2D if ('split_data' in locals() and split_data is not None) else means2D
    # 强制保证输入张量的设备/类型/形状（非原地）
    def to_f32_cuda(t: torch.Tensor) -> torch.Tensor:
        return t.to(device="cuda", dtype=torch.float32).contiguous()
    means3D = to_f32_cuda(means3D)
    means2D_input = to_f32_cuda(means2D_input)
    scales = to_f32_cuda(scales)
    rotations = to_f32_cuda(rotations)
    opacity = to_f32_cuda(opacity)
    if opacity.dim() == 1:
        opacity = opacity.view(-1, 1)
    # 纠正异常不透明度（非原地）
    if (opacity.max() > 1.5) or (opacity.min() < -0.5):
        opacity = torch.sigmoid(opacity)
    # 颜色
    colors_precomp = to_f32_cuda(colors_precomp)
    # 防 NaN/Inf（非原地）
    def sanitize(t: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    means3D = sanitize(means3D)
    means2D_input = sanitize(means2D_input)
    scales = sanitize(scales)
    rotations = sanitize(rotations)
    opacity = sanitize(opacity)
    colors_precomp = sanitize(colors_precomp)
    # 诊断打印（可临时启用）
    if pipe.debug:
        try:
            cmin, cmax, cmean = float(colors_precomp.min()), float(colors_precomp.max()), float(colors_precomp.mean())
            omin, omax, omean = float(opacity.min()), float(opacity.max()), float(opacity.mean())
            print(f"[Render Stats] colors_precomp min/max/mean: {cmin:.4f}/{cmax:.4f}/{cmean:.4f}")
            print(f"[Render Stats] opacity min/max/mean: {omin:.4f}/{omax:.4f}/{omean:.4f}")
        except Exception:
            pass

    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D_input,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp
    )

    # 返回渲染结果
    # 如果发生分裂，聚合visibility与radii到原始点数（向量化）
    if 'split_data' in locals() and split_data is not None:
        original_indices = split_data['original_indices']
        n_original = split_data['n_original']
        # radii 形状 [n_splits]
        try:
            original_radii = torch.zeros((n_original,), device=radii.device)
            if hasattr(original_radii, 'scatter_reduce_'):
                original_radii.scatter_reduce_(0, original_indices, radii, reduce='amax', include_self=True)
            elif hasattr(original_radii, 'index_reduce_'):
                original_radii.index_reduce_(0, original_indices, radii, reduce='amax', include_self=True)
            else:
                # 排序后用unique_consecutive聚合
                sort_idx = torch.argsort(original_indices)
                sorted_idx = original_indices[sort_idx]
                sorted_r = radii[sort_idx]
                uniq, counts = torch.unique_consecutive(sorted_idx, return_counts=True)
                max_vals = torch.zeros_like(uniq, dtype=sorted_r.dtype)
                start = 0
                for j, c in enumerate(counts):
                    seg = sorted_r[start:start+int(c.item())]
                    max_vals[j] = seg.max()
                    start += int(c.item())
                original_radii = torch.zeros((n_original,), device=radii.device)
                original_radii[uniq] = max_vals
            out_visibility = original_radii > 0
            out_radii = original_radii
        except Exception:
            # 退化：仅聚合可见性（any），半径直接置零
            out_visibility = torch.zeros((n_original,), dtype=torch.bool, device=radii.device)
            out_visibility[original_indices[radii > 0]] = True
            out_radii = torch.zeros((n_original,), device=radii.device)
    else:
        out_visibility = radii > 0
        out_radii = radii

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": out_visibility,
        "radii": out_radii,
        "split_viewspace_points": split_means2D if ('split_data' in locals() and split_data is not None) else None,
        "split_info": {
            "n_original": n_original,
            "original_indices": split_data['original_indices'] if split_data else None,
            "split_opacity": split_data['split_opacity'] if split_data else None
        } if split_data is not None else None,
        "split_data": split_data
    }