# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.laplacian_model import LaplacianModel
from scene.new_model import SplitLaplacianModel, SplitGaussianModel # Ensure SplitGaussianModel is also imported if needed for isinstance checks
from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation # Import necessary function

def compute_dp_dwave_jacobian(wave, rot_matrix, k, eps=1e-5):
    """Computes the Jacobian d(offset_k) / dwave.

    Args:
        wave: The wave vector (3,).
        rot_matrix: The rotation matrix R (3, 3).
        k: The split index (integer).
        eps: Small value to avoid division by zero. 增大到1e-5提高稳定性。

    Returns:
        Jacobian matrix J_offset_wave (3, 3).
    """
    w = wave
    norm_w_sq = torch.dot(w, w) + eps # ||w||^2
    norm_w_pow4 = norm_w_sq * norm_w_sq

    # Calculate R * (||w||^2 * I - 2 * w * w^T)
    w_outer_w = torch.outer(w, w) # w * w^T (3, 3)
    term_in_brackets = norm_w_sq * torch.eye(3, device=w.device) - 2 * w_outer_w # (3, 3)
    rotated_term = torch.matmul(rot_matrix, term_in_brackets) # R * [...] (3, 3)

    jacobian = (2 * math.pi * k / norm_w_pow4) * rotated_term
    return jacobian

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. Modified to compute and return wave Jacobians, 
    and to handle SplitLaplacianModel by incorporating shape.
    
    Background tensor (bg_color) must be on GPU!
    """
    # 修复梯度设置问题
    screenspace_points = torch.zeros(pc.get_xyz.shape[0], 3, dtype=pc.get_xyz.dtype, device="cuda", requires_grad=True)
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        print(f"Warning: Could not retain grad on screenspace_points: {e}")

    # Set up rasterization configuration
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

    means3D = pc.get_xyz
    opacity = pc.get_opacity
    n_original = means3D.shape[0]

    has_shape = isinstance(pc, SplitLaplacianModel)

    split_jacobians_info = {
        "original_indices": [], 
        "split_indices_pos": [],
        "split_indices_neg": [],
        "jacobians_pos": [], 
        "jacobians_neg": []
    }
    
    gaussian_stats = {
        'n_original': n_original,
        'n_split': 0,
        'split_ratio': 1.0,
        'effective_render_count': n_original,
        'active_waves': 0, 
        'active_waves_001': 0,
        'active_waves_05': 0,
        'max_splits_per_gaussian': 0,
        'all_splitting_gaussians': 0, 
        'avg_splits_all': 0.0
    }

    # 修复分裂条件检查：添加调试信息
    can_split = hasattr(pc, 'use_splitting') and pc.use_splitting and \
                  hasattr(pc, '_wave') and pc._wave is not None and pc._wave.numel() > 0 and \
                  hasattr(pc, '_max_splits') and pc._max_splits > 0
                  
    # 注释掉调试输出
    # if hasattr(pc, 'use_splitting'):
    #     print(f"[调试] use_splitting: {pc.use_splitting}, _max_splits: {getattr(pc, '_max_splits', 'N/A')}")
    #     if hasattr(pc, '_wave') and pc._wave is not None:
    #         wave_norm_debug = torch.norm(pc._wave, dim=1)
    #         active_count_debug = (wave_norm_debug > 0.001).sum().item()
    #         print(f"[调试] Wave向量总数: {pc._wave.shape[0]}, 活跃数量(>0.001): {active_count_debug}")
    
    # print(f"[调试] 分裂条件can_split: {can_split}")
                  
    render_means2D_for_rasterizer = screenspace_points
    actual_total_render = n_original
    split_points_added = 0

    if can_split:
        wave_vectors = pc.get_wave
        wave_norm = torch.norm(wave_vectors, dim=1)
        
        gaussian_stats['active_waves'] = (wave_norm > 0.01).sum().item()
        gaussian_stats['active_waves_001'] = (wave_norm > 0.001).sum().item()
        gaussian_stats['active_waves_05'] = (wave_norm > 0.05).sum().item()
        
        # print(f"[调试] 统计 - 活跃波向量(>0.01): {gaussian_stats['active_waves']}, (>0.001): {gaussian_stats['active_waves_001']}")
        
        max_k_splits = pc._max_splits 
        max_total_alloc = n_original * (1 + 2 * max_k_splits)
            
        extended_means3D = torch.zeros((max_total_alloc, 3), dtype=means3D.dtype, device=means3D.device)
        extended_opacity = torch.zeros((max_total_alloc, 1), dtype=opacity.dtype, device=opacity.device)
        extended_means3D[:n_original] = means3D
        extended_opacity[:n_original] = opacity
        
        current_scales_no_shape = pc.get_scaling 
        current_rotations = pc.get_rotation
        
        extended_scales_no_shape = None
        extended_rotations_aug = None
        extended_cov3D = None
        extended_shs = None
        extended_shapes = None

        if not pipe.compute_cov3D_python:
            extended_scales_no_shape = torch.zeros((max_total_alloc, 3), dtype=current_scales_no_shape.dtype, device=current_scales_no_shape.device)
            extended_rotations_aug = torch.zeros((max_total_alloc, 4), dtype=current_rotations.dtype, device=current_rotations.device)
            extended_scales_no_shape[:n_original] = current_scales_no_shape
            extended_rotations_aug[:n_original] = current_rotations
        else:
             cov3D_precomp_orig = pc.get_covariance(scaling_modifier)
             extended_cov3D = torch.zeros((max_total_alloc, 6), dtype=cov3D_precomp_orig.dtype, device=cov3D_precomp_orig.device)
             extended_cov3D[:n_original] = cov3D_precomp_orig
        
        if has_shape:
            current_shapes = pc.get_shape
            extended_shapes = torch.zeros((max_total_alloc, current_shapes.shape[1]), dtype=current_shapes.dtype, device=current_shapes.device)
            extended_shapes[:n_original] = current_shapes

        if override_color is None:
            if not pipe.convert_SHs_python:
                shs_orig = pc.get_features
                extended_shs = torch.zeros((max_total_alloc, shs_orig.shape[1], shs_orig.shape[2]), dtype=shs_orig.dtype, device=shs_orig.device)
                extended_shs[:n_original] = shs_orig
        
        decay_rate = getattr(pipe, 'split_opacity_decay_rate', 0.20) 
        camera_pos = viewpoint_camera.camera_center
        view_importance_threshold = getattr(pipe, 'split_view_importance_threshold', 0.4)
        splits_per_original_gaussian = torch.zeros(n_original, dtype=torch.int, device="cuda")

        # 计数器用于调试
        total_candidates = 0
        skipped_by_norm = 0
        skipped_by_offset = 0
        actual_splits = 0

        for idx in range(n_original):
            w_i = wave_vectors[idx]
            norm_w_i = wave_norm[idx]
            
            # 大幅放宽wave阈值条件
            if norm_w_i < 0.001:  # 从1e-6放宽到0.001
                skipped_by_norm += 1
                continue
            
            total_candidates += 1
            
            rot_matrix = build_rotation(pc._rotation[idx].unsqueeze(0))[0]
            local_decay_rate = decay_rate
            point_to_camera = camera_pos - means3D[idx]
            point_to_camera_norm = torch.norm(point_to_camera)
            if point_to_camera_norm > 1e-6:
                point_to_camera_dir = point_to_camera / point_to_camera_norm
                wave_dir_norm = w_i / norm_w_i
                view_importance = torch.abs(torch.dot(wave_dir_norm, point_to_camera_dir))
                if view_importance > view_importance_threshold:
                    local_decay_rate = decay_rate * (1.0 - (view_importance - view_importance_threshold) * 0.7)

            max_original_scale_component = torch.max(current_scales_no_shape[idx]).item()
            current_gaussian_splits_count_for_debug = 0
            consecutive_skips = 0
            max_consecutive_skips = 2  # 减少从3到2

            for k in range(1, min(max_k_splits + 1, 4)):  # 限制最大分裂次数为3，避免过度分裂
                # === 鲁棒的偏移计算机制 ===
                
                # 1. 多重约束的基础偏移计算
                wave_magnitude = norm_w_i.item()
                
                # 约束1：基于wave强度的自适应偏移
                wave_based_offset = wave_magnitude * 0.05  # 降低基础系数
                
                # 约束2：基于高斯球尺度的相对偏移
                scale_based_offset = max_original_scale_component * 0.1  # 相对于高斯球尺度
                
                # 约束3：基于分裂次数的递减偏移
                split_decay_factor = 1.0 / (1.0 + 0.5 * (k - 1))  # 分裂次数越多，偏移越小
                
                # 约束4：绝对偏移上限
                absolute_limit = 0.02  # 绝对上限
                
                # 选择最小的偏移作为基础偏移
                candidate_offsets = [
                    wave_based_offset,
                    scale_based_offset,
                    absolute_limit
                ]
                base_offset = min(candidate_offsets) * split_decay_factor
                
                # 2. 应用分裂索引的影响
                adaptive_offset = base_offset * min(k, 3)  # 限制k的影响，最多3倍
                
                # 3. 动态尺度限制（更严格的约束）
                # 基于当前高斯球的实际尺度动态调整限制
                dynamic_scale_limit = max_original_scale_component * (2.0 + k * 0.5)  # 随分裂次数适度增长
                
                # 4. 最终安全检查
                adaptive_offset = min(adaptive_offset, dynamic_scale_limit, 0.05)  # 多重上限保护
                
                if adaptive_offset > dynamic_scale_limit:
                    consecutive_skips += 1
                    skipped_by_offset += 1
                    if consecutive_skips >= max_consecutive_skips: 
                        break
                    continue
                
                consecutive_skips = 0
                current_gaussian_splits_count_for_debug += 2
                actual_splits += 1
                pos_idx = n_original + split_points_added
                neg_idx = n_original + split_points_added + 1

                if pos_idx >= max_total_alloc or neg_idx >= max_total_alloc: 
                    break 

                pos_offset_dir = w_i / norm_w_i 
                pos_offset_vec = adaptive_offset * pos_offset_dir
                neg_offset_vec = -pos_offset_vec

                jacobian_k = compute_dp_dwave_jacobian(w_i.detach(), rot_matrix.detach(), k)
                jacobian_pos = jacobian_k
                jacobian_neg = -jacobian_k

                split_jacobians_info["original_indices"].append(idx)
                split_jacobians_info["split_indices_pos"].append(pos_idx)
                split_jacobians_info["split_indices_neg"].append(neg_idx)
                split_jacobians_info["jacobians_pos"].append(jacobian_pos)
                split_jacobians_info["jacobians_neg"].append(jacobian_neg)

                extended_means3D[pos_idx] = means3D[idx] + torch.matmul(rot_matrix, pos_offset_vec.unsqueeze(-1)).squeeze(-1)
                extended_means3D[neg_idx] = means3D[idx] + torch.matmul(rot_matrix, neg_offset_vec.unsqueeze(-1)).squeeze(-1)
                
                # 修复opacity计算
                opacity_factor = torch.exp(torch.tensor(-k * local_decay_rate, device=opacity.device, dtype=torch.float32))
                extended_opacity[pos_idx] = opacity[idx] * opacity_factor
                extended_opacity[neg_idx] = opacity[idx] * opacity_factor

                if pipe.compute_cov3D_python:
                    base_cov = extended_cov3D[idx].unsqueeze(0) 
                    wave_dir_cov = pos_offset_dir.unsqueeze(0)
                    min_orig_scale_val = torch.min(current_scales_no_shape[idx]).item()
                    scale_factor = pc.compute_split_scale_factor(norm_w_i.item(), k, min_orig_scale_val) 
                    transformed_cov = pc.compute_directional_covariance(base_cov, wave_dir_cov, scale_factor)
                    extended_cov3D[pos_idx] = transformed_cov[0]
                    extended_cov3D[neg_idx] = transformed_cov[0]
                elif extended_scales_no_shape is not None:
                    base_scale_no_shape = current_scales_no_shape[idx]
                    base_rot_val = current_rotations[idx]
                    min_orig_scale_val = torch.min(base_scale_no_shape).item()
                    split_factor_val = pc.compute_split_scale_factor(norm_w_i.item(), k, min_orig_scale_val) 
                    perp_factor = 1.0 / (math.sqrt(split_factor_val) + 1e-8)
                    scaling_dir_factors = torch.ones(3, device=w_i.device)
                    wave_abs_global = torch.abs(pos_offset_dir)
                    max_dim_global = torch.argmax(wave_abs_global)
                    scaling_dir_factors[max_dim_global] = split_factor_val
                    other_dims_global = [d for d in range(3) if d != max_dim_global.item()]
                    scaling_dir_factors[other_dims_global] = perp_factor
                    new_scale_no_shape = base_scale_no_shape * scaling_dir_factors
                    extended_scales_no_shape[pos_idx] = new_scale_no_shape
                    extended_rotations_aug[pos_idx] = base_rot_val
                    extended_scales_no_shape[neg_idx] = new_scale_no_shape
                    extended_rotations_aug[neg_idx] = base_rot_val

                if has_shape and extended_shapes is not None:
                    extended_shapes[pos_idx] = current_shapes[idx]
                    extended_shapes[neg_idx] = current_shapes[idx]
                
                if override_color is None and not pipe.convert_SHs_python and extended_shs is not None:
                    extended_shs[pos_idx] = shs_orig[idx]
                    extended_shs[neg_idx] = shs_orig[idx]

                split_points_added += 2
            
            splits_per_original_gaussian[idx] = current_gaussian_splits_count_for_debug
            if current_gaussian_splits_count_for_debug > 0:
                gaussian_stats['all_splitting_gaussians'] += 1

        # 调试输出分裂统计
        print(f"[调试] 分裂统计 - 候选点: {total_candidates}, 被norm跳过: {skipped_by_norm}, 被offset跳过: {skipped_by_offset}, 实际分裂: {actual_splits}, 新增点: {split_points_added}")

        actual_total_render = n_original + split_points_added
        means3D = extended_means3D[:actual_total_render]
        opacity = extended_opacity[:actual_total_render]

        # 修复梯度张量创建
        if split_points_added > 0:
            split_screenspace_points_for_cat = torch.zeros((split_points_added, 3), dtype=screenspace_points.dtype, requires_grad=True, device="cuda")
            render_means2D_for_rasterizer = torch.cat([screenspace_points, split_screenspace_points_for_cat], dim=0)
            try:
                render_means2D_for_rasterizer.retain_grad()
            except Exception as e:
                print(f"Warning: Could not retain grad on extended screenspace_points: {e}")
        else:
            render_means2D_for_rasterizer = screenspace_points

        # Finalize other tensors for rendering based on actual_total_render
        final_scales_for_raster = None
        final_rotations_for_raster = None
        final_cov3D_for_raster = None
        final_shs_for_raster = None
        final_colors_for_raster = None

        if pipe.compute_cov3D_python:
            final_cov3D_for_raster = extended_cov3D[:actual_total_render]
        elif extended_scales_no_shape is not None:
            final_rotations_for_raster = extended_rotations_aug[:actual_total_render]
            scales_component = extended_scales_no_shape[:actual_total_render]
            if has_shape and extended_shapes is not None:
                shape_component = extended_shapes[:actual_total_render]
                final_scales_for_raster = scales_component * shape_component
            else:
                final_scales_for_raster = scales_component
        else: 
             final_scales_for_raster = pc.get_scaling[:actual_total_render] # Should be sliced if means3D is sliced
             if has_shape: final_scales_for_raster = final_scales_for_raster * pc.get_shape[:actual_total_render]
             final_rotations_for_raster = pc.get_rotation[:actual_total_render]

        if override_color is None:
            if pipe.convert_SHs_python:
                temp_shs_features = pc.get_features
                expanded_shs_for_eval = torch.zeros((actual_total_render, temp_shs_features.shape[1], temp_shs_features.shape[2]), 
                                                    dtype=temp_shs_features.dtype, device=temp_shs_features.device)
                expanded_shs_for_eval[:n_original] = temp_shs_features
                for i in range(len(split_jacobians_info["original_indices"])):
                    orig_idx_sh = split_jacobians_info["original_indices"][i]
                    pos_split_idx_sh = split_jacobians_info["split_indices_pos"][i]
                    neg_split_idx_sh = split_jacobians_info["split_indices_neg"][i]
                    if pos_split_idx_sh < actual_total_render:
                       expanded_shs_for_eval[pos_split_idx_sh] = temp_shs_features[orig_idx_sh]
                    if neg_split_idx_sh < actual_total_render:
                       expanded_shs_for_eval[neg_split_idx_sh] = temp_shs_features[orig_idx_sh]
                
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(actual_total_render, 1))
                dir_pp_normalized = dir_pp/(dir_pp.norm(dim=1, keepdim=True) + 1e-8)
                shs_view_expanded = expanded_shs_for_eval.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view_expanded, dir_pp_normalized)
                final_colors_for_raster = torch.clamp_min(sh2rgb + 0.5, 0.0)
            elif extended_shs is not None:
                final_shs_for_raster = extended_shs[:actual_total_render]
        else: 
            final_colors_for_raster = torch.zeros((actual_total_render, 3), dtype=override_color.dtype, device=override_color.device)
            if override_color.shape[0] == n_original:
                final_colors_for_raster[:n_original] = override_color
                for i in range(len(split_jacobians_info["original_indices"])):
                    orig_idx_c = split_jacobians_info["original_indices"][i]
                    pos_split_idx_c = split_jacobians_info["split_indices_pos"][i]
                    neg_split_idx_c = split_jacobians_info["split_indices_neg"][i]
                    if pos_split_idx_c < actual_total_render:
                        final_colors_for_raster[pos_split_idx_c] = override_color[orig_idx_c]
                    if neg_split_idx_c < actual_total_render:
                        final_colors_for_raster[neg_split_idx_c] = override_color[orig_idx_c]
            else: 
                num_repeats = (actual_total_render // override_color.shape[0]) + 1
                final_colors_for_raster = override_color.repeat(num_repeats, 1)[:actual_total_render]
        
        gaussian_stats['n_split'] = split_points_added
        gaussian_stats['effective_render_count'] = actual_total_render
        if n_original > 0:
            gaussian_stats['split_ratio'] = actual_total_render / n_original
        if gaussian_stats['all_splitting_gaussians'] > 0:
            active_splitting_gaussians_norms = splits_per_original_gaussian[splits_per_original_gaussian > 0].float()
            if active_splitting_gaussians_norms.numel() > 0: # Ensure there are elements before calling mean
                 gaussian_stats['avg_splits_all'] = active_splitting_gaussians_norms.mean().item()
            else:
                 gaussian_stats['avg_splits_all'] = 0.0 # Or handle as appropriate
        max_splits_val = splits_per_original_gaussian.max().item()
        gaussian_stats['max_splits_per_gaussian'] = max_splits_val if max_splits_val > 0 else 0

    else: # Not splitting, prepare tensors for rasterizer directly from pc
        means3D = pc.get_xyz # Use original means3D directly
        opacity = pc.get_opacity # Use original opacity
        # render_means2D_for_rasterizer is already screenspace_points
        
        final_scales_for_raster = pc.get_scaling
        if has_shape:
            final_scales_for_raster = final_scales_for_raster * pc.get_shape
        final_rotations_for_raster = pc.get_rotation
        
        final_cov3D_for_raster = None
        if pipe.compute_cov3D_python:
             final_cov3D_for_raster = pc.get_covariance(scaling_modifier)

        final_shs_for_raster = None
        final_colors_for_raster = None
        if override_color is None:
            if pipe.convert_SHs_python:
                dir_pp = (means3D - viewpoint_camera.camera_center.repeat(n_original, 1))
                dir_pp_normalized = dir_pp/(dir_pp.norm(dim=1, keepdim=True) + 1e-8)
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                final_colors_for_raster = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                final_shs_for_raster = pc.get_features
        else:
            final_colors_for_raster = override_color
        
        gaussian_stats['effective_render_count'] = n_original

    rendered_image, radii_rendered = rasterizer(
        means3D = means3D, # This is either original or extended means3D
        means2D = render_means2D_for_rasterizer, # This is either original or extended screenspace points
        shs = final_shs_for_raster, 
        colors_precomp = final_colors_for_raster,
        opacities = opacity, # This is either original or extended opacity
        scales = final_scales_for_raster, 
        rotations = final_rotations_for_raster, 
        cov3D_precomp = final_cov3D_for_raster)

    visibility_filter_final = torch.zeros(n_original, dtype=torch.bool, device="cuda")
    original_radii_final = torch.zeros(n_original, device="cuda")
    visibility_filter_rendered_mask = radii_rendered > 0 # Visibility of all points passed to rasterizer

    if can_split and split_points_added > 0 :
        # Visibility of original points that were part of the render batch
        visibility_filter_final[:n_original] = visibility_filter_rendered_mask[:n_original]
        original_radii_final[:n_original] = radii_rendered[:n_original]
        # Aggregate from children
        for i in range(len(split_jacobians_info["original_indices"])):
            orig_idx = split_jacobians_info["original_indices"][i]
            # pos_split_idx and neg_split_idx are indices in the *extended* batch passed to rasterizer
            pos_split_idx_in_raster_batch = split_jacobians_info["split_indices_pos"][i]
            neg_split_idx_in_raster_batch = split_jacobians_info["split_indices_neg"][i]

            # Check if these children were actually rendered (i.e., within actual_total_render)
            # and if they were visible according to radii_rendered
            if pos_split_idx_in_raster_batch < actual_total_render and visibility_filter_rendered_mask[pos_split_idx_in_raster_batch]:
                visibility_filter_final[orig_idx] = True
                original_radii_final[orig_idx] = torch.max(original_radii_final[orig_idx], radii_rendered[pos_split_idx_in_raster_batch])
            
            if neg_split_idx_in_raster_batch < actual_total_render and visibility_filter_rendered_mask[neg_split_idx_in_raster_batch]:
                visibility_filter_final[orig_idx] = True
                original_radii_final[orig_idx] = torch.max(original_radii_final[orig_idx], radii_rendered[neg_split_idx_in_raster_batch])
    else: # No splitting happened or not enabled
        visibility_filter_final = visibility_filter_rendered_mask[:n_original] 
        original_radii_final = radii_rendered[:n_original]

    # CRITICAL: The viewspace_points returned for gradient calculation MUST be the original screenspace_points
    # that corresponds to pc.get_xyz and has its gradient retained.
    returned_viewspace_points = screenspace_points 

    return {"render": rendered_image,
            "viewspace_points": returned_viewspace_points, # THIS IS THE KEY CHANGE
            "visibility_filter" : visibility_filter_final, 
            "radii": original_radii_final, 
            "split_jacobians_info": split_jacobians_info if (can_split and split_points_added > 0) else None, 
            "gaussian_stats": gaussian_stats
            }

# ... (render_laplacian, render_new)
# It seems render_new was an older attempt. The main `render` function is now modified.
# I will comment out the old render_new to avoid confusion if it's not used elsewhere.
# If it IS used, it would also need similar updates or be replaced by the main `render`.
# For now, assuming the main `render` function is the one to be used by the training script.

def render_laplacian(viewpoint_camera, pc : LaplacianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
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

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling * pc.get_shape
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

# Commenting out the old render_new as the main `render` function is now the primary one.
# def render_new(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
#    """
#    使用分裂拉普拉斯方法渲染场景。
#    结合了高斯分裂和拉普拉斯形状特性。
#    (This function would need similar Jacobian and shape handling as the main render function)
#    """
#    # ... implementation ...
#    pass

def render_fast_split(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None, 
                     split_budget=None, high_freq_regions=None, smart_split=True):
    """
    快速分裂渲染，支持智能分裂策略
    
    Args:
        viewpoint_camera: 视点相机
        pc: 点云模型
        pipe: 渲染管线
        bg_color: 背景颜色
        scaling_modifier: 缩放修正因子
        override_color: 覆盖颜色
        split_budget: 分裂预算限制
        high_freq_regions: 高频区域掩码
        smart_split: 是否启用智能分裂
    """
    # 在推理时创建简单的屏幕空间点，不需要梯度
    requires_grad = torch.is_grad_enabled()  # 检查是否在训练模式
    screenspace_points = torch.zeros(pc.get_xyz.shape[0], 3, dtype=pc.get_xyz.dtype, device="cuda", requires_grad=requires_grad)
    
    if requires_grad:
        try:
            screenspace_points.retain_grad()
        except Exception as e:
            print(f"Warning: Could not retain grad on screenspace_points: {e}")

    # Set up rasterization configuration
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

    # 准备基础高斯球数据
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    n_original = means3D.shape[0]

    # 检查是否可以分裂
    can_split = hasattr(pc, 'use_splitting') and pc.use_splitting and \
                hasattr(pc, '_wave') and pc._wave is not None and pc._wave.numel() > 0 and \
                hasattr(pc, '_max_splits') and pc._max_splits > 0

    # 初始化统计信息
    gaussian_stats = {
        "n_original": n_original,
        "n_split": 0,
        "split_ratio": 1.0,
        "active_waves": 0,
        "active_waves_001": 0,
        "active_waves_05": 0,
        "effective_render_count": n_original
    }

    # 添加调试信息（在推理时也显示）
    if can_split:
        wave = pc.get_wave
        wave_norm = torch.norm(wave, dim=1)
        
        # 更新统计信息
        gaussian_stats['active_waves'] = (wave_norm > 0.01).sum().item()
        gaussian_stats['active_waves_001'] = (wave_norm > 0.001).sum().item()
        gaussian_stats['active_waves_05'] = (wave_norm > 0.05).sum().item()
        
        # print(f"[调试] 统计 - 活跃波向量(>0.01): {gaussian_stats['active_waves']}, (>0.001): {gaussian_stats['active_waves_001']}")
        
        # 使用更宽松的分裂阈值进行渲染
        split_mask = wave_norm > 0.0005  # 进一步放宽阈值，从0.001到0.0005
        
        if split_mask.any():
            split_indices = torch.where(split_mask)[0]
            n_splits = len(split_indices)
            
            # print(f"[快速渲染] 潜在分裂点: {n_splits}")
            
            # 在推理时使用更宽松的限制
            if not requires_grad:  # 推理模式
                max_splits_per_batch = min(n_splits, 50000)  # 推理时允许更多分裂
            else:  # 训练模式
                # 使用智能分裂预算（如果提供）
                if split_budget is not None and smart_split:
                    max_splits_per_batch = min(n_splits, split_budget)
                    # print(f"[智能分裂] 使用分裂预算: {split_budget}")
                else:
                    # 原有的显存基础限制
                    allocated_gb = torch.cuda.memory_allocated() / 1024**3
                    if allocated_gb > 14.0:
                        max_splits_per_batch = min(n_splits, 10000)
                    elif allocated_gb > 10.0:
                        max_splits_per_batch = min(n_splits, 20000)
                    else:
                        max_splits_per_batch = min(n_splits, 40000)
            
            if n_splits > max_splits_per_batch:
                # 智能选择分裂点：优先选择wave较大的点
                wave_norms_subset = wave_norm[split_mask]
                _, top_indices = torch.topk(wave_norms_subset, max_splits_per_batch)
                split_indices = split_indices[top_indices]
                n_splits = max_splits_per_batch
                # print(f"[快速渲染] 限制分裂点数量为: {n_splits}")
            
            if n_splits > 0:
                # 获取scaling信息用于分裂计算
                current_scales = pc.get_scaling
                
                # 更宽松的分裂偏移计算
                valid_split_indices = []
                split_offsets = []
                
                for i, idx in enumerate(split_indices):
                    w_i = wave[idx]
                    norm_w_i = wave_norm[idx]
                    max_scale = torch.max(current_scales[idx]).item()
                    min_scale = torch.min(current_scales[idx]).item()
                    
                    # === 快速渲染的鲁棒偏移计算 ===
                    
                    # 1. 多重约束的偏移计算
                    wave_magnitude = norm_w_i.item()
                    
                    # 约束1：基于wave强度的保守偏移
                    wave_offset = wave_magnitude * 0.02  # 进一步降低系数
                    
                    # 约束2：基于最小尺度的相对偏移（更保守）
                    scale_offset = min_scale * 0.15  # 基于最小尺度而非最大尺度
                    
                    # 约束3：绝对最小偏移保证
                    min_offset = 0.0002  # 降低最小偏移
                    
                    # 约束4：绝对最大偏移限制
                    max_offset = 0.01  # 严格的上限
                    
                    # 选择合适的偏移
                    base_offset = max(min(wave_offset, scale_offset, max_offset), min_offset)
                    
                    # 2. 动态尺度限制（基于高斯球的实际形状）
                    # 考虑高斯球的各向异性
                    scale_ratio = max_scale / (min_scale + 1e-8)
                    anisotropy_factor = 1.0 / (1.0 + 0.1 * torch.log(torch.tensor(scale_ratio + 1.0)))
                    
                    # 调整偏移以适应各向异性
                    adjusted_offset = base_offset * anisotropy_factor.item()
                    
                    # 3. 最终的尺度限制检查
                    conservative_scale_limit = min_scale * 3.0  # 基于最小尺度的保守限制
                    
                    if adjusted_offset <= conservative_scale_limit:
                        split_offsets.append(adjusted_offset)  # 使用调整后的偏移
                        valid_split_indices.append(idx)
                
                # print(f"[快速渲染] 通过条件检查的分裂点: {len(valid_split_indices)}")
                
                if valid_split_indices:
                    valid_split_indices = torch.tensor(valid_split_indices, device=means3D.device)
                    n_valid_splits = len(valid_split_indices)
                    
                    # 计算分裂方向
                    split_directions = wave[valid_split_indices] / wave_norm[valid_split_indices].unsqueeze(1)
                    
                    # 创建分裂点
                    split_pos_offsets = []
                    split_neg_offsets = []
                    
                    for i, offset in enumerate(split_offsets):
                        direction = split_directions[i]
                        split_pos_offsets.append(direction * offset)
                        split_neg_offsets.append(-direction * offset)
                    
                    if split_pos_offsets:
                        split_pos_offsets = torch.stack(split_pos_offsets)
                        split_neg_offsets = torch.stack(split_neg_offsets)
                        
                        # 创建分裂点位置
                        split_pos = means3D[valid_split_indices] + split_pos_offsets
                        split_neg = means3D[valid_split_indices] + split_neg_offsets
                        extended_means3D = torch.cat([means3D, split_pos, split_neg], dim=0)
                        
                        # 扩展其他属性
                        decay_factor = 0.8  # 分裂点透明度衰减
                        extended_opacity = torch.cat([
                            opacity, 
                            opacity[valid_split_indices] * decay_factor,
                            opacity[valid_split_indices] * decay_factor
                        ], dim=0)
                        
                        # 扩展屏幕空间点 - 修复梯度错误
                        split_points_added = 2 * len(valid_split_indices)
                        
                        # 为分裂点创建正确的2D屏幕空间占位符
                        # 这些将在光栅化过程中被正确投影
                        split_screenspace_placeholder = torch.zeros(
                            split_points_added, 3, 
                            dtype=screenspace_points.dtype, 
                            device=screenspace_points.device,
                            requires_grad=requires_grad
                        )
                        
                        # 使用torch.cat创建扩展的屏幕空间点，避免原地操作
                        extended_screenspace = torch.cat([
                            screenspace_points, 
                            split_screenspace_placeholder
                        ], dim=0)
                        
                        if requires_grad:
                            try:
                                extended_screenspace.retain_grad()
                            except Exception as e:
                                print(f"Warning: Could not retain grad on extended screenspace: {e}")
                        
                        means3D = extended_means3D
                        opacity = extended_opacity
                        means2D = extended_screenspace
                        
                        # 更新统计信息
                        gaussian_stats["n_split"] = split_points_added
                        gaussian_stats["split_ratio"] = extended_means3D.shape[0] / n_original
                        gaussian_stats["effective_render_count"] = extended_means3D.shape[0]
                        
                        # print(f"[快速渲染] ✅ 成功分裂 {len(valid_split_indices)} 个高斯球，新增 {split_points_added} 个分裂点")
                    else:
                        # print(f"[快速渲染] ❌ 没有创建有效的分裂偏移")
                        pass
                else:
                    # print(f"[快速渲染] ❌ 没有通过条件检查的分裂点")
                    pass
            else:
                # print(f"[快速渲染] ❌ 分裂点数量为0")
                pass
        else:
            # print(f"[快速渲染] ❌ 没有满足阈值的wave向量")
            pass
    else:
        # print(f"[快速渲染] ❌ 分裂条件不满足 - use_splitting: {getattr(pc, 'use_splitting', 'N/A')}, _max_splits: {getattr(pc, '_max_splits', 'N/A')}")
        pass

    # 处理scaling和rotation (保持原来的逻辑)
    if isinstance(pc, SplitLaplacianModel):
        # 处理SplitLaplacianModel的shape参数
        base_scaling = pc.get_scaling
        if hasattr(pc, '_shape') and pc._shape.numel() > 0:
            shape_factor = pc.shape_activation(pc._shape)
            effective_scaling = base_scaling * shape_factor.unsqueeze(1)
        else:
            effective_scaling = base_scaling
        
        # 如果有分裂点，扩展scaling
        if means3D.shape[0] > n_original:
            n_splits = (means3D.shape[0] - n_original) // 2
            split_indices = valid_split_indices[:n_splits]
            extended_scaling = torch.cat([
                effective_scaling,
                effective_scaling[split_indices],
                effective_scaling[split_indices]
            ], dim=0)
            extended_rotation = torch.cat([
                pc._rotation,
                pc._rotation[split_indices],
                pc._rotation[split_indices]
            ], dim=0)
        else:
            extended_scaling = effective_scaling
            extended_rotation = pc._rotation
    else:
        # 标准高斯模型处理
        base_scaling = pc.get_scaling
        if means3D.shape[0] > n_original and 'valid_split_indices' in locals():
            n_splits = (means3D.shape[0] - n_original) // 2
            split_indices = valid_split_indices[:n_splits]
            extended_scaling = torch.cat([
                base_scaling,
                base_scaling[split_indices],
                base_scaling[split_indices]
            ], dim=0)
            extended_rotation = torch.cat([
                pc._rotation,
                pc._rotation[split_indices],
                pc._rotation[split_indices]
            ], dim=0)
        else:
            extended_scaling = base_scaling
            extended_rotation = pc._rotation

    # 处理SH特征
    if means3D.shape[0] > n_original and 'valid_split_indices' in locals():
        # 扩展SH特征
        n_splits = (means3D.shape[0] - n_original) // 2
        split_indices = valid_split_indices[:n_splits]
        extended_features_dc = torch.cat([
            pc._features_dc,
            pc._features_dc[split_indices],
            pc._features_dc[split_indices]
        ], dim=0)
        extended_features_rest = torch.cat([
            pc._features_rest,
            pc._features_rest[split_indices],
            pc._features_rest[split_indices]
        ], dim=0)
        shs = torch.cat([extended_features_dc, extended_features_rest], dim=1)
    else:
        shs = torch.cat([pc._features_dc, pc._features_rest], dim=1)

    # 渲染设置
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # Python球谐函数评估
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs.transpose(1, 2).contiguous(), dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = shs
    else:
        colors_precomp = override_color

    # 执行光栅化
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=extended_scaling,
        rotations=extended_rotation,
        cov3D_precomp=None
    )

    # 处理可见性过滤器
    visibility_filter = radii > 0
    if means3D.shape[0] > n_original:
        # 如果有分裂点，只返回原始点的可见性
        visibility_filter = visibility_filter[:n_original]
        radii = radii[:n_original]

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": visibility_filter,
        "radii": radii,
        "gaussian_stats": gaussian_stats
    }

def render_frequency_split(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None):
    """
    专门用于FrequencySplitGaussianModel的简洁渲染函数
    不包含渲染时分裂逻辑，分裂在密集化时进行
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor for gradients
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
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

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Handle covariance
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Handle colors
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}

"""
渲染初始化模块
包含各种渲染函数和网络GUI实现
"""
# 定义可导出的函数
__all__ = [
    "render",
    "render_frequency_split",
    "render_laplacian", 
    "render_fast_split",
    "network_gui"
]

# 导入network_gui模块
import gaussian_renderer.network_gui as network_gui
