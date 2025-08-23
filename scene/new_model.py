"""
scene/new_model.py
修复后的SplitGaussianModel，集成所有改进
"""

import torch
import numpy as np
from torch import nn
import os
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, strip_symmetric, build_scaling_rotation
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH
from utils.graphics_utils import BasicPointCloud
from random import randint

# 导入配置管理器
from config_manager import config_manager

class SplitGaussianModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.wave_activation = lambda x: x  # 直接传递
        self.shape_activation = lambda x, strength: torch.sigmoid(x * strength)

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._shape = torch.empty(0)
        self._wave = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.prune_shape_threshold = 0.2
        self.shape_strngth = 1.0
        self.setup_functions()
        
        # 从配置管理器获取参数
        config = config_manager.config
        self.use_wave = False  # 初始不使用wave
        self.use_splitting = config.use_splitting
        self._max_splits = config.max_splits
        self._split_factor = config.split_factor
        self._split_cache = {}
        self._split_cache_valid = False
        # 参数化分裂：可训练的沿主轴缩放系数（初始为1.0）
        self._split_sigma_alpha = nn.Parameter(torch.ones((0, 3), device="cuda"))

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._shape,
            self._wave,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self._shape,
         self._wave,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        # 确保参数化分裂系数尺寸与点数一致
        try:
            num = self._xyz.shape[0]
            self._split_sigma_alpha = nn.Parameter(torch.ones((num, 3), device="cuda").requires_grad_(True))
        except Exception:
            pass

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_wave(self):
        return self.wave_activation(self._wave)

    @property
    def get_shape(self):
        return self.shape_activation(self._shape, self.shape_strngth)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        shapes = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        
        # 重要修复：初始化wave为小的随机值而不是0
        config = config_manager.config
        wave = torch.randn((fused_point_cloud.shape[0], 3), device="cuda") * config.wave_init_noise

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._shape = nn.Parameter(shapes.requires_grad_(True))
        self._wave = nn.Parameter(wave.requires_grad_(True))
        # 初始化参数化分裂系数
        self._split_sigma_alpha = nn.Parameter(torch.ones((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.prune_shape_threshold = training_args.prune_shape_threshold if hasattr(training_args, 'prune_shape_threshold') else 0.2
        self.shape_strngth = training_args.shape_strngth if hasattr(training_args, 'shape_strngth') else 1.0
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 从配置管理器更新参数
        config = config_manager.config
        config_manager.update_config(
            use_splitting=training_args.use_splitting if hasattr(training_args, 'use_splitting') else True,
            max_splits=training_args.max_splits if hasattr(training_args, 'max_splits') else 10,
            split_factor=training_args.split_factor if hasattr(training_args, 'split_factor') else 1.6,
            wave_init_noise=training_args.wave_init_noise if hasattr(training_args, 'wave_init_noise') else 0.01,
            wave_lr=training_args.wave_lr if hasattr(training_args, 'wave_lr') else 0.01
        )
        
        self.use_splitting = config.use_splitting
        self._max_splits = config.max_splits
        self._split_factor = config.split_factor
        
        # 确保wave学习率正确设置
        wave_lr = config.wave_lr
        
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._shape], 'lr': training_args.shape_lr if hasattr(training_args, 'shape_lr') else 0.001, "name": "shape"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._wave], 'lr': wave_lr, "name": "wave"},
            {'params': [self._split_sigma_alpha], 'lr': training_args.scaling_lr, "name": "split_alpha"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('shape')
        # 添加wave属性
        for i in range(3):
            l.append('wave_{}'.format(i))
        return l

    def save_ply(self, path):
        """修复后的save_ply方法，确保所有属性正确保存"""
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        shape = self._shape.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        wave = self._wave.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, shape, wave), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
        
        # 打印保存信息
        print(f"[Save PLY] Saved {xyz.shape[0]} gaussians to {path}")
        print(f"  Features DC shape: {f_dc.shape}")
        print(f"  Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
        wave_norms = np.linalg.norm(wave, axis=1)
        try:
            from config_manager import config_manager
            thr = getattr(config_manager.config, 'wave_threshold', 1e-4)
        except Exception:
            thr = 1e-4
        print(f"  Wave active: {(wave_norms > thr).sum()}/{len(wave_norms)} (thr={thr})")

    def load_ply(self, path):
        """修复后的load_ply方法，确保所有属性正确加载"""
        plydata = PlyData.read(path)

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)
        
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        
        if len(extra_f_names) > 0:
            assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        else:
            features_extra = np.zeros((xyz.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 加载shape
        if "shape" in [p.name for p in plydata.elements[0].properties]:
            shapes = np.asarray(plydata.elements[0]["shape"])[..., np.newaxis]
        else:
            shapes = np.ones((xyz.shape[0], 1))

        # 加载wave
        wave_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("wave_")]
        if len(wave_names) > 0:
            wave_names = sorted(wave_names, key=lambda x: int(x.split('_')[-1]))
            wave = np.zeros((xyz.shape[0], len(wave_names)))
            for idx, attr_name in enumerate(wave_names):
                wave[:, idx] = np.asarray(plydata.elements[0][attr_name])
        else:
            # 如果没有wave，初始化为小的随机值
            config = config_manager.config
            wave = np.random.randn(xyz.shape[0], 3) * config.wave_init_noise

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shape = nn.Parameter(torch.tensor(shapes, dtype=torch.float, device="cuda").requires_grad_(True))
        self._wave = nn.Parameter(torch.tensor(wave, dtype=torch.float, device="cuda").requires_grad_(True))

        # 初始化参数化分裂系数为 1（与点数对齐）
        num = self._xyz.shape[0]
        self._split_sigma_alpha = nn.Parameter(torch.ones((num, 3), device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        
        # 打印加载信息
        print(f"[Load PLY] Loaded {xyz.shape[0]} gaussians from {path}")
        print(f"  Opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")
        wave_norms = np.linalg.norm(wave, axis=1)
        try:
            from config_manager import config_manager
            thr = getattr(config_manager.config, 'wave_threshold', 1e-4)
        except Exception:
            thr = 1e-4
        print(f"  Wave active: {(wave_norms > thr).sum()}/{len(wave_norms)} (thr={thr})")

    def invalidate_split_cache(self):
        """清除分裂缓存"""
        self._split_cache = {}
        self._split_cache_valid = False

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def get_split_data(self, iteration: int, max_iteration: int):
        """获取分裂数据用于渲染"""
        from gaussian_renderer.splits import compute_splits_precise
        
        progress = min(iteration / max_iteration, 1.0) if max_iteration > 0 else 0.0
        
        # 使用缓存机制（降低分裂计算频率）
        from config_manager import config_manager
        cfg = config_manager.config
        interval = getattr(cfg, 'split_compute_interval', 5)
        cache_key = f"{(iteration // max(1, interval))*interval}_{self._xyz.shape[0]}"
        if cache_key in self._split_cache and self._split_cache_valid:
            return self._split_cache[cache_key]
        
        # 计算分裂数据（优先精确实现，回退到旧实现由渲染层处理）
        split_data = compute_splits_precise(
            self,
            iteration=iteration,
            max_iteration=max_iteration,
            max_k=self._max_splits
        )
        # 回退：若精确分裂不可用，使用快速矢量化分裂以保证训练与渲染链路不断裂
        if split_data is None:
            try:
                from gaussian_renderer_optimized import vectorized_compute_splits_continuous_improved
                split_data = vectorized_compute_splits_continuous_improved(
                    self, max_splits_global=self._max_splits, progress=progress
                )
            except Exception:
                split_data = None
        
        # 缓存结果
        if split_data is not None:
            self._split_cache[cache_key] = split_data
            self._split_cache_valid = True
        
        return split_data

    # 添加密集化相关方法
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        # 保护性处理：当grad尚未填充时，跳过本次累积，避免NoneType错误
        if viewspace_point_tensor.grad is None:
            return
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """基于wave的智能密集化策略"""
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent
        )

        # 基于wave的策略
        if self.use_wave:
            wave_norms = torch.norm(self._wave[selected_pts_mask], dim=1)
            # 降低阈值，避免过早把高频候选全部判为低波
            from config_manager import config_manager
            wave_threshold = getattr(config_manager.config, 'wave_threshold', 1e-4)
            need_wave_mask = wave_norms < wave_threshold
            need_split_mask = ~need_wave_mask

            # 对低wave值的高斯增加wave
            if need_wave_mask.any():
                selected_indices = torch.where(selected_pts_mask)[0]
                wave_indices = selected_indices[need_wave_mask]
                with torch.no_grad():
                    self._wave[wave_indices] += torch.randn_like(self._wave[wave_indices]) * 0.1

            # 对高wave值的高斯进行分裂
            if need_split_mask.any():
                selected_indices = torch.where(selected_pts_mask)[0]
                split_indices = selected_indices[need_split_mask]
                self._perform_split(split_indices, N)
        else:
            # 标准分裂
            self._perform_split(torch.where(selected_pts_mask)[0], N)

    def _perform_split(self, indices, N=2):
        """执行高斯分裂"""
        if len(indices) == 0:
            return
            
        stds = self.get_scaling[indices].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[indices]).repeat(N, 1, 1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[indices].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[indices].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[indices].repeat(N, 1)
        new_features_dc = self._features_dc[indices].repeat(N, 1, 1)
        new_features_rest = self._features_rest[indices].repeat(N, 1, 1)
        new_opacity = self._opacity[indices].repeat(N, 1)
        new_shape = self._shape[indices].repeat(N, 1)
        new_wave = self._wave[indices].repeat(N, 1) * 0.5  # 分裂后降低wave

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                 new_opacity, new_scaling, new_rotation, new_shape, new_wave)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """克隆高斯球"""
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_shape = self._shape[selected_pts_mask]
        new_wave = self._wave[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                 new_opacities, new_scaling, new_rotation, new_shape, new_wave)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """密集化和剪枝"""
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 剪枝
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # 保护高wave值的高斯不被剪枝
        if self.use_wave:
            wave_norms = torch.norm(self._wave, dim=1)
            high_wave_mask = wave_norms > 0.5
            prune_mask = torch.logical_and(prune_mask, ~high_wave_mask)
        
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def prune_points(self, mask):
        """剪枝点"""
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._shape = optimizable_tensors["shape"]
        self._wave = optimizable_tensors["wave"]
        if "split_alpha" in optimizable_tensors:
            self._split_sigma_alpha = optimizable_tensors["split_alpha"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # 清除缓存
        self.invalidate_split_cache()

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities,
                            new_scaling, new_rotation, new_shape, new_wave):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "shape": new_shape,
            "wave": new_wave,
            # 新增的参数化分裂系数：为新增点初始化为 1
            "split_alpha": torch.ones((new_xyz.shape[0], 3), device="cuda")
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._shape = optimizable_tensors["shape"]
        self._wave = optimizable_tensors["wave"]
        if "split_alpha" in optimizable_tensors:
            self._split_sigma_alpha = optimizable_tensors["split_alpha"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # 清除缓存
        self.invalidate_split_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in tensors_dict:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], 
                                                        torch.zeros_like(tensors_dict[group["name"]])), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], 
                                                           torch.zeros_like(tensors_dict[group["name"]])), dim=0)
                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], tensors_dict[group["name"]]), dim=0).requires_grad_(True)
                    )
                    self.optimizer.state[group['params'][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(
                        torch.cat((group["params"][0], tensors_dict[group["name"]]), dim=0).requires_grad_(True)
                    )
                    optimizable_tensors[group["name"]] = group["params"][0]
        # 确保 split_alpha 若未被 optimizer param_groups 捕获，则直接在模型上拼接
        if "split_alpha" in tensors_dict and "split_alpha" not in optimizable_tensors:
            self._split_sigma_alpha = nn.Parameter(torch.cat((self._split_sigma_alpha, tensors_dict["split_alpha"]), dim=0).requires_grad_(True))
            optimizable_tensors["split_alpha"] = self._split_sigma_alpha
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # 对齐 mask 长度与当前参数长度
            param_len = group["params"][0].shape[0]
            if mask.shape[0] != param_len:
                if mask.shape[0] > param_len:
                    mask_adj = mask[:param_len]
                else:
                    pad = torch.zeros((param_len - mask.shape[0],), dtype=mask.dtype, device=mask.device)
                    mask_adj = torch.cat([mask, pad], dim=0)
            else:
                mask_adj = mask
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask_adj]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask_adj]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask_adj].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask_adj].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        # 确保 split_alpha 同步裁剪（若未在 optimizer 中）
        if hasattr(self, "_split_sigma_alpha") and "split_alpha" not in optimizable_tensors:
            # 与 xyz 长度对齐裁剪
            param_len = self._split_sigma_alpha.shape[0]
            if mask.shape[0] != param_len:
                if mask.shape[0] > param_len:
                    mask_adj2 = mask[:param_len]
                else:
                    pad2 = torch.zeros((param_len - mask.shape[0],), dtype=mask.dtype, device=mask.device)
                    mask_adj2 = torch.cat([mask, pad2], dim=0)
            else:
                mask_adj2 = mask
            self._split_sigma_alpha = nn.Parameter(self._split_sigma_alpha[mask_adj2].requires_grad_(True))
            optimizable_tensors["split_alpha"] = self._split_sigma_alpha
        return optimizable_tensors

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors