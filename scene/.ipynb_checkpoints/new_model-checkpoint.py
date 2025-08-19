import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, var_approx, var_generalized
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import math

class SplitGaussianModel:
    """
    继承自GaussianModel，添加分裂功能
    """
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

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._wave = torch.empty(0)   # 波向量，用于分裂
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        # 缓存机制
        self._cached_split_data = None
        self._cached_iteration = -1
        
        # 分裂统计
        self.split_statistics = {}

    def get_split_data(self, iteration, max_iteration, max_splits_global=None, progress=None):
        """
        获取分裂数据，使用缓存避免重复计算
        优化：添加内存清理
        """
        # 如果是同一迭代，返回缓存的结果
        if self._cached_iteration == iteration and self._cached_split_data is not None:
            return self._cached_split_data
    
        # 清理旧的缓存数据
        if self._cached_split_data is not None:
            del self._cached_split_data
            self._cached_split_data = None
            torch.cuda.empty_cache()
    
        # 计算进度
        if progress is None:
            progress = min(iteration / max_iteration, 1.0)
    
        # 使用默认的最大分裂数
        if max_splits_global is None:
            max_splits_global = self._max_splits
    
        # 重新计算并缓存
        from gaussian_renderer import vectorized_compute_splits_continuous_improved
    
        # 在计算前清理GPU缓存
        torch.cuda.empty_cache()
    
        self._cached_split_data = vectorized_compute_splits_continuous_improved(
            self, max_splits_global=max_splits_global, progress=progress
        )
        self._cached_iteration = iteration
    
        # 更新分裂统计
        if self._cached_split_data is not None and 'split_distribution' in self._cached_split_data:
            self.split_statistics[iteration] = self._cached_split_data['split_distribution']
    
        return self._cached_split_data

    
    def invalidate_split_cache(self):
        """
        使缓存失效，在高斯参数更新后调用
        """
        self._cached_split_data = None
        self._cached_iteration = -1

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._wave,
            self._split_factor,
            self._max_splits,
            self.use_splitting,
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
        self._wave,
        self._split_factor,
        self._max_splits,
        self.use_splitting,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
        return self._wave
    
    @property
    def get_split_factor(self):
        return self._split_factor
    
    @property
    def get_max_splits(self):
        return self._max_splits

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def compute_directional_covariance(self, original_cov, wave_dir, parallel_factor, perp_factor):
        """
        根据波向量方向计算方向性协方差变换
    
        参数:
            original_cov: 原始协方差矩阵 [N, 6]
            wave_dir: 归一化的波向量方向 [N, 3]
            parallel_factor: 波方向缩放因子
            perp_factor: 垂直方向缩放因子
        
        返回:
            transformed_cov: 变换后的协方差矩阵 [N, 6]
        """
        batch_size = wave_dir.shape[0]
        device = wave_dir.device
    
        # 将对称矩阵形式转换为完整的3x3矩阵
        cov_full = torch.zeros((batch_size, 3, 3), device=device)
    
        # 填充3x3矩阵 (从压缩的6个元素)
        cov_full[:, 0, 0] = original_cov[:, 0]
        cov_full[:, 1, 1] = original_cov[:, 3]
        cov_full[:, 2, 2] = original_cov[:, 5]
        cov_full[:, 0, 1] = cov_full[:, 1, 0] = original_cov[:, 1]
        cov_full[:, 0, 2] = cov_full[:, 2, 0] = original_cov[:, 2]
        cov_full[:, 1, 2] = cov_full[:, 2, 1] = original_cov[:, 4]
    
        # 构建波向量方向的投影矩阵
        wave_dir = wave_dir.view(batch_size, 3, 1)
        P_parallel = torch.bmm(wave_dir, wave_dir.transpose(1, 2))
    
        # 构建垂直方向的投影矩阵
        I = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        P_perp = I - P_parallel
    
        # 构建方向性缩放矩阵
        S = parallel_factor * P_parallel + perp_factor * P_perp
    
        # 应用变换: S * cov * S^T
        transformed_cov = torch.bmm(torch.bmm(S, cov_full), S.transpose(1, 2))
    
        # 将3x3矩阵转回6元素的压缩形式
        result = torch.zeros((batch_size, 6), device=device)
        result[:, 0] = transformed_cov[:, 0, 0]  # xx
        result[:, 1] = transformed_cov[:, 0, 1]  # xy
        result[:, 2] = transformed_cov[:, 0, 2]  # xz
        result[:, 3] = transformed_cov[:, 1, 1]  # yy
        result[:, 4] = transformed_cov[:, 1, 2]  # yz
        result[:, 5] = transformed_cov[:, 2, 2]  # zz
    
        return result
        
    def compute_split_scale_factor(self, wave_norm, split_index, base_scale_min):
        """
        计算分裂高斯球的协方差缩放因子
    
        理论：分裂后的高斯需要在波方向上压缩，以更好地逼近局部余弦变化
    
        参数:
            wave_norm (float): 波向量的范数 ||k||
            split_index (int): 分裂的索引（距离中心的半波长数）
            base_scale_min (float): 原高斯球的最小缩放
        
        返回:
            tuple: (parallel_factor, perpendicular_factor)
        """
        if split_index == 0:  # 中心点保持原始缩放
            return 1.0, 1.0
        
        # 波长
        wavelength = 2.0 * math.pi / (wave_norm + 1e-8)
    
        # 在波方向上的压缩因子
        # 使分裂的高斯在半波长内有效覆盖
        parallel_factor = min(0.8, wavelength / (4.0 * base_scale_min))
        parallel_factor = max(0.2, parallel_factor)  # 限制在[0.2, 0.8]
    
        # 垂直方向的因子（略微扩展以保持覆盖）
        perpendicular_factor = 1.0 / math.sqrt(parallel_factor)
    
        # 对于远离中心的分裂，可能需要额外调整
        distance_penalty = 1.0 + 0.05 * (split_index - 1)
        parallel_factor /= distance_penalty
    
        return parallel_factor, perpendicular_factor

    def compute_split_positions(self, wave_vector, max_scale, max_splits=None):
        """
        计算分裂位置和权重
        
        参数:
            wave_vector: 波向量 k
            max_scale: 原始高斯的最大尺度
            max_splits: 最大分裂数限制
            
        返回:
            positions: 分裂位置列表（相对于中心的偏移）
            weights: 对应的权重（考虑余弦和高斯衰减）
        """
        wave_norm = torch.norm(wave_vector)
        if wave_norm < 1e-6:
            return [], []
            
        wave_dir = wave_vector / wave_norm
        wavelength = 2 * math.pi / wave_norm.item()
        half_wavelength = wavelength / 2
        
        # 在3σ范围内的半波长数
        n_half_waves = int(math.ceil(3 * max_scale / half_wavelength))
        if max_splits is not None:
            n_half_waves = min(n_half_waves, 2 * max_splits)
        
        positions = []
        weights = []
        
        # 中心点（k=0）
        positions.append(torch.zeros(3, device=wave_vector.device))
        weights.append(1.0)
        
        # 其他分裂点
        for k in range(1, n_half_waves):
            distance = k * half_wavelength
            
            # 正向和负向
            for sign in [1, -1]:
                offset = sign * distance * wave_dir
                
                # 高斯衰减
                gaussian_weight = math.exp(-0.5 * (distance / max_scale) ** 2)
                
                # 余弦值（在kπ处）
                cos_value = (-1) ** k if sign > 0 else (-1) ** (k + 1)
                
                positions.append(offset)
                weights.append(gaussian_weight * abs(cos_value))
                
        return positions, weights

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        wave = torch.randn((fused_point_cloud.shape[0], 3), device="cuda") * 0.01  

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._wave = nn.Parameter(wave.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self.use_splitting = True  # 默认启用分裂
        self._max_splits = training_args.max_splits if hasattr(training_args, 'max_splits') else 10
        self._split_factor = training_args.split_factor if hasattr(training_args, 'split_factor') else -1.0
        wave_init_noise = training_args.wave_init_noise if hasattr(training_args, 'wave_init_noise') else 0.1
    
        # 改进的噪声添加逻辑
        if wave_init_noise > 0 and not hasattr(self, '_wave_initialized'):
            print(f"Adding initial noise (std={wave_init_noise}) to wave vectors.")
            with torch.no_grad():
                noise = torch.randn_like(self._wave) * wave_init_noise
                self._wave.data = torch.clamp(self._wave.data + noise, -0.1, 0.1)
            self._wave_initialized = True

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._wave], 'lr': training_args.wave_lr if hasattr(training_args, 'wave_lr') else 0.02, "name": "wave"}
            ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                  lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                  lr_delay_mult=training_args.position_lr_delay_mult,
                                                  max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # 添加wave属性
        for i in range(3):  # 假设wave是3D向量
            l.append('wave_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        wave = self._wave.detach().cpu().numpy()  # 保存wave到ply

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, wave), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 读取wave属性
        wave_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("wave_")]
        wave_names = sorted(wave_names, key = lambda x: int(x.split('_')[-1]))
        wave = np.zeros((xyz.shape[0], len(wave_names)))
        for idx, attr_name in enumerate(wave_names):
            wave[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._wave = nn.Parameter(torch.tensor(wave, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

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

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._wave = optimizable_tensors["wave"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # 清除缓存，因为高斯点已经改变
        self.invalidate_split_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # Check if the group name exists in the tensors_dict, otherwise skip
            if group["name"] not in tensors_dict:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_wave):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "wave": new_wave}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # Ensure wave is updated if it exists in the optimizer
        if "wave" in optimizable_tensors:
             self._wave = optimizable_tensors["wave"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # 清除缓存，因为高斯点已经改变
        self.invalidate_split_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        优化的分裂策略：优先增加wave而非分裂高斯
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 新增：检查wave状态，决定是增加wave还是分裂
        wave_norms = torch.norm(self._wave[selected_pts_mask], dim=1)
        
        # 策略：如果wave较小，优先增加wave；如果wave已经较大，才进行分裂
        wave_threshold = 0.5  # 可调参数
        need_wave_mask = wave_norms < wave_threshold
        need_split_mask = ~need_wave_mask
    
        # 对于需要增加wave的高斯
        if need_wave_mask.any():
            selected_indices = torch.where(selected_pts_mask)[0]
            wave_indices = selected_indices[need_wave_mask]
        
            with torch.no_grad():
                # 基于梯度方向增加wave
                if self._xyz.grad is not None:
                    grad_directions = self._xyz.grad[wave_indices]
                    grad_norms = torch.norm(grad_directions, dim=1, keepdim=True)
                    grad_directions = grad_directions / (grad_norms + 1e-8)
                
                    # 增加与梯度相关的wave分量
                    wave_increment = grad_directions * 0.1  # 可调参数
                    self._wave[wave_indices] += wave_increment
                else:
                    # 如果没有梯度信息，随机增加
                    self._wave[wave_indices] += torch.randn_like(self._wave[wave_indices]) * 0.05
    
        # 对于需要分裂的高斯（wave已经较大）
        if need_split_mask.any():
            selected_for_split = torch.zeros_like(selected_pts_mask)
            selected_indices = torch.where(selected_pts_mask)[0]
            split_indices = selected_indices[need_split_mask]
            selected_for_split[split_indices] = True
        
            # 只对这些高斯进行标准分裂
            if selected_for_split.any():
                stds = self.get_scaling[selected_for_split].repeat(N,1)
                means = torch.zeros((stds.size(0), 3),device="cuda")
                samples = torch.normal(mean=means, std=stds)
                rots = build_rotation(self._rotation[selected_for_split]).repeat(N,1,1)
                new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_for_split].repeat(N, 1)
            
                # 限制新高斯球的最小scale
                min_new_scale = 1e-5
                target_scaling = self.get_scaling[selected_for_split] / (0.8*N)
                target_scaling = torch.maximum(target_scaling, torch.full_like(target_scaling, min_new_scale))
                new_scaling = self.scaling_inverse_activation(target_scaling.repeat(N,1))
            
                new_rotation = self._rotation[selected_for_split].repeat(N,1)
                new_features_dc = self._features_dc[selected_for_split].repeat(N,1,1)
                new_features_rest = self._features_rest[selected_for_split].repeat(N,1,1)
                new_opacity = self._opacity[selected_for_split].repeat(N,1)

                # 智能wave继承：分裂后适度降低wave
                parent_wave = self._wave[selected_for_split]
                parent_wave_norm = torch.norm(parent_wave, dim=1)

                # 根据wave大小决定衰减因子
                decay_factor = torch.where(parent_wave_norm > 1.0, 
                                          torch.tensor(0.3, device='cuda', dtype=torch.float),
                                          torch.tensor(0.6, device='cuda', dtype=torch.float))
            
                decayed_parent_wave = parent_wave * decay_factor.unsqueeze(1)
                new_wave = decayed_parent_wave.repeat(N, 1)
            
                # 添加小的扰动
                perturbation = torch.randn_like(new_wave) * 0.02
                new_wave = new_wave + perturbation

                self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_wave)

                prune_filter = torch.cat((selected_for_split, torch.zeros(N * selected_for_split.sum(), device="cuda", dtype=bool)))
                self.prune_points(prune_filter)


    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # Also pass wave for cloning
        new_wave = self._wave[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_wave)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        优化的剪枝策略：考虑频率重要性
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 原有的剪枝条件
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    
        # 新增：频率感知的剪枝策略
        if hasattr(self, '_wave'):
            wave_norms = torch.norm(self._wave, dim=1)
        
            # 计算频率重要性分数
            # 考虑：wave大小、梯度累积、不透明度
            frequency_importance = wave_norms * 10.0  # wave贡献
        
            if self.xyz_gradient_accum is not None:
                grad_contribution = self.xyz_gradient_accum.squeeze() / (self.denom.squeeze() + 1e-8)
                frequency_importance += grad_contribution
        
            # 归一化重要性分数
            if frequency_importance.max() > 0:
                frequency_importance = frequency_importance / frequency_importance.max()
        
            # 保护高频重要的高斯不被剪枝
            high_freq_threshold = 0.3  # 重要性阈值
            important_for_freq = frequency_importance > high_freq_threshold
        
            # 更新剪枝掩码：排除频率重要的高斯
            prune_mask = prune_mask & ~important_for_freq
            
            # 额外的小高斯剪枝（但保护高频）
            min_scale_threshold = 1e-5
            tiny_gaussians = self.get_scaling.min(dim=1).values < min_scale_threshold
        
            # 只剪枝那些既小又不重要的高斯
            unimportant_tiny = tiny_gaussians & (frequency_importance < 0.1)
        
            if unimportant_tiny.any():
                print(f"[Pruning] Removing {unimportant_tiny.sum().item()} unimportant tiny gaussians")
        
            prune_mask = torch.logical_or(prune_mask, unimportant_tiny)
    
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        添加密集化统计信息 - 始终使用精确梯度
        """
        # 确保梯度存在
        if viewspace_point_tensor.grad is None:
            return
    
        # 计算2D屏幕空间的梯度范数
        screen_grad_norm = torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], 
            dim=-1, 
            keepdim=True
        )
    
        # 累积梯度和计数
        self.xyz_gradient_accum[update_filter] += screen_grad_norm
        self.denom[update_filter] += 1

    # 将以下方法添加到SplitGaussianModel类中
# 注意：这里只包含新增的方法，原有的方法保持不变

    def invalidate_split_cache(self):
        """
        使分裂缓存失效，在参数更新后调用
        """
        self._cached_split_data = None
        self._cached_iteration = -1

    def densify_and_prune_frequency_aware(self, max_grad, min_opacity, extent, max_screen_size, 
                                          iteration, max_iteration):
        """
        频率感知的密集化和剪枝策略
    
        在高频区域（大wave）更积极地密集化，更保守地剪枝
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
    
        # 计算频率重要性（基于wave大小）
        wave_norms = torch.norm(self._wave, dim=1)
        frequency_importance = torch.sigmoid(wave_norms * 2.0)  # 映射到[0, 1]
    
        # === 密集化部分 ===
    
        # 1. 克隆：小高斯满足梯度条件
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                             torch.max(self.get_scaling, dim=1).values <= self.percent_dense * extent)
    
        # 高频区域降低梯度阈值（更容易克隆）
        high_freq_mask = frequency_importance > 0.3
        high_freq_grad_mask = torch.norm(grads, dim=-1) >= max_grad * 0.7  # 降低30%阈值
        selected_pts_mask = torch.logical_or(selected_pts_mask, 
                                            torch.logical_and(high_freq_grad_mask, high_freq_mask))
    
        if selected_pts_mask.sum() > 0:
            new_xyz = self._xyz[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_wave = self._wave[selected_pts_mask]
        
            # 高频区域的克隆保持wave，低频区域减小wave
            freq_scale = 0.5 + 0.5 * frequency_importance[selected_pts_mask]
            new_wave = new_wave * freq_scale.unsqueeze(1)
        
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                      new_opacities, new_scaling, new_rotation, new_wave)
    
        # 2. 分裂：大高斯满足梯度条件
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                             torch.max(self.get_scaling, dim=1).values > self.percent_dense * extent)
    
        if selected_pts_mask.sum() > 0:
            # 特别处理高频区域的大高斯
            high_freq_split = torch.logical_and(selected_pts_mask, frequency_importance > 0.5)
        
            if high_freq_split.sum() > 0:
                # 沿wave方向分裂
                self._densify_and_split_along_wave(high_freq_split, extent)
        
            # 普通分裂
            normal_split = torch.logical_and(selected_pts_mask, ~high_freq_split)
            if normal_split.sum() > 0:
                self.densify_and_split(grads, max_grad, extent)
    
        # === 剪枝部分 ===
    
        # 基础剪枝条件
        prune_mask = (self.get_opacity < min_opacity).squeeze()
    
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    
        # 频率保护：高频重要的高斯不被剪枝
        frequency_protection = frequency_importance > 0.6
        prune_mask = torch.logical_and(prune_mask, ~frequency_protection)
    
        # 额外保护：正在学习的高斯（wave在增长）
        if hasattr(self, '_prev_wave_norms'):
            wave_growing = wave_norms > self._prev_wave_norms
            prune_mask = torch.logical_and(prune_mask, ~wave_growing)
    
        # 记录当前wave范数，用于下次比较
        self._prev_wave_norms = wave_norms.clone()
    
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def _densify_and_split_along_wave(self, mask, scene_extent):
        """
        沿wave方向分裂高斯
    
        高频区域的大高斯应该沿wave方向分裂，以更好地捕捉方向性细节
        """
        N = 2  # 分裂成2个
    
        selected_indices = torch.where(mask)[0]
    
        if selected_indices.numel() == 0:
            return
    
        # 获取需要分裂的高斯属性
        selected_xyz = self._xyz[mask]
        selected_wave = self._wave[mask]
        selected_scaling = self._scaling[mask]
        selected_rotation = self._rotation[mask]
        selected_features_dc = self._features_dc[mask]
        selected_features_rest = self._features_rest[mask]
        selected_opacity = self._opacity[mask]
        
        # 计算wave方向
        wave_norms = torch.norm(selected_wave, dim=1, keepdim=True)
        wave_dirs = selected_wave / (wave_norms + 1e-8)
    
        # 沿wave方向的偏移
        offset_scale = torch.min(selected_scaling, dim=1).values * 0.5
        offset_scale = offset_scale.unsqueeze(1)
    
        # 创建新高斯（正负方向各一个）
        new_xyz_list = []
        new_scaling_list = []
        new_rotation_list = []
        new_features_dc_list = []
        new_features_rest_list = []
        new_opacity_list = []
        new_wave_list = []
    
        for sign in [1, -1]:
            offset = sign * offset_scale * wave_dirs
            new_xyz = selected_xyz + offset
        
            # 缩小scale
            new_scaling = self.scaling_inverse_activation(
                self.scaling_activation(selected_scaling) / (0.8 * N)
            )
        
            # 保持rotation和其他属性
            new_rotation = selected_rotation
            new_features_dc = selected_features_dc
            new_features_rest = selected_features_rest
            new_opacity = selected_opacity
        
            # Wave稍微减小，但保持方向
            new_wave = selected_wave * 0.8
        
            new_xyz_list.append(new_xyz)
            new_scaling_list.append(new_scaling)
            new_rotation_list.append(new_rotation)
            new_features_dc_list.append(new_features_dc)
            new_features_rest_list.append(new_features_rest)
            new_opacity_list.append(new_opacity)
            new_wave_list.append(new_wave)
    
        # 合并所有新高斯
        new_xyz = torch.cat(new_xyz_list, dim=0)
        new_scaling = torch.cat(new_scaling_list, dim=0)
        new_rotation = torch.cat(new_rotation_list, dim=0)
        new_features_dc = torch.cat(new_features_dc_list, dim=0)
        new_features_rest = torch.cat(new_features_rest_list, dim=0)
        new_opacity = torch.cat(new_opacity_list, dim=0)
        new_wave = torch.cat(new_wave_list, dim=0)
    
        # 添加新高斯
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                  new_opacity, new_scaling, new_rotation, new_wave)
    
        # 删除原始高斯
        prune_filter = torch.zeros(self._xyz.shape[0], dtype=bool, device="cuda")
        prune_filter[mask] = True
        self.prune_points(prune_filter)

    def reset_opacity_selective(self, threshold=0.01):
        """
        选择性重置不透明度
    
        只重置低频区域（小wave）的不透明度，保护高频区域
        """
        wave_norms = torch.norm(self._wave, dim=1)
        
        # 只重置wave很小的高斯
        low_freq_mask = wave_norms < threshold
    
        opacities_new = self.inverse_opacity_activation(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
    
        # 只对低频高斯应用重置
        self._opacity.data[low_freq_mask] = opacities_new[low_freq_mask]

    def get_training_stats(self):
        """
        获取训练统计信息
        """
        wave_norms = torch.norm(self._wave, dim=1)
    
        stats = {
            'num_gaussians': self._xyz.shape[0],
            'wave_mean': wave_norms.mean().item(),
            'wave_max': wave_norms.max().item(),
            'wave_min': wave_norms.min().item(),
            'wave_active': (wave_norms > 0.01).sum().item(),
            'wave_active_ratio': (wave_norms > 0.01).sum().item() / wave_norms.shape[0],
            'opacity_mean': self.get_opacity.mean().item(),
            'scale_mean': self.get_scaling.mean().item(),
        }
    
        return stats

    def analyze_frequency_distribution(self):
        """
        分析频率分布，用于调试和可视化
        """
        wave_norms = torch.norm(self._wave, dim=1)
    
        # 将wave范数分成几个区间
        bins = [0, 0.1, 0.5, 1.0, 2.0, 5.0, float('inf')]
        distribution = {}
    
        for i in range(len(bins) - 1):
            mask = (wave_norms >= bins[i]) & (wave_norms < bins[i+1])
            count = mask.sum().item()
            distribution[f'{bins[i]}-{bins[i+1]}'] = count
    
        return distribution

    def save_frequency_map(self, path, iteration):
        """
        保存频率图，用于可视化
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    
        # 获取位置和wave信息
        positions = self._xyz.cpu().numpy()
        wave_norms = torch.norm(self._wave, dim=1).cpu().numpy()
    
        # 创建2D投影（俯视图）
        fig, ax = plt.subplots(figsize=(10, 10))
    
        # 根据wave大小着色
        scatter = ax.scatter(positions[:, 0], positions[:, 1], 
                            c=wave_norms, s=1, 
                            cmap='hot', vmin=0, vmax=2)
    
        plt.colorbar(scatter, label='Wave Magnitude')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Frequency Distribution at Iteration {iteration}')
    
        plt.savefig(f'{path}/frequency_map_{iteration}.png', dpi=150)
        plt.close()

    def print_split_statistics(self, iteration):
        if iteration in self.split_statistics:
            stats = self.split_statistics[iteration]
            total_splits = sum(stats.values())
            print(f"\n[Split Statistics at iteration {iteration}]")
            print(f"Total split gaussians: {total_splits}")
            for k, count in sorted(stats.items()):
                print(f"  k={k}: {count} gaussians")

class SplitLaplacianModel:
    """
    结合了GaussianModel中的分裂特性和LaplacianModel中的形状变化特性的模型
    """
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
        self.shape_activation = var_approx

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._shape = torch.empty(0)  # 从LaplacianModel继承
        self._wave = torch.empty(0)   # 从GaussianModel继承
        self._split_factor = 1.0      # 从GaussianModel继承
        self._max_splits = 0          # 从GaussianModel继承
        self.use_splitting = True    # 从GaussianModel继承
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.shape_prune_threshold = 0
        self.spatial_lr_scale = 0
        self.shape_strngth = 1.0
        self.setup_functions()
        
        # 缓存机制
        self._cached_split_data = None
        self._cached_iteration = -1
        
        # 分裂统计
        self.split_statistics = {}

    def get_split_data(self, iteration, max_iteration, max_splits_global=None, progress=None):
        """
        获取分裂数据，使用缓存避免重复计算
        
        Args:
            iteration: 当前迭代次数
            max_iteration: 最大迭代次数
            max_splits_global: 全局最大分裂数
            progress: 训练进度（可选，如果不提供则自动计算）
        
        Returns:
            分裂数据字典或None
        """
        # 如果是同一迭代，返回缓存的结果
        if self._cached_iteration == iteration and self._cached_split_data is not None:
            return self._cached_split_data
        
        # 计算进度
        if progress is None:
            progress = min(iteration / max_iteration, 1.0)
        
        # 使用默认的最大分裂数
        if max_splits_global is None:
            max_splits_global = self._max_splits
        
        # 重新计算并缓存
        # 注意：这里需要import相应的函数
        from gaussian_renderer import vectorized_compute_splits_continuous
        
        self._cached_split_data = vectorized_compute_splits_continuous(
            self, max_splits_global=max_splits_global, progress=progress
        )
        self._cached_iteration = iteration
        
        # 更新分裂统计
        if self._cached_split_data is not None and 'split_distribution' in self._cached_split_data:
            self.split_statistics[iteration] = self._cached_split_data['split_distribution']
        
        return self._cached_split_data
    
    def invalidate_split_cache(self):
        """
        使缓存失效，在高斯参数更新后调用
        """
        self._cached_split_data = None
        self._cached_iteration = -1

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
            self._split_factor,
            self._max_splits,
            self.use_splitting,
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
        self._split_factor,
        self._max_splits,
        self.use_splitting,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
    def get_shape(self):
        return self.shape_activation(self._shape, self.shape_strngth)
    
    @property
    def get_wave(self):
        return self._wave
    
    @property
    def get_split_factor(self):
        return self._split_factor
    
    @property
    def get_max_splits(self):
        return self._max_splits

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling * self.get_shape, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        shapes = torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        wave = torch.zeros((fused_point_cloud.shape[0], 3), device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._shape = nn.Parameter(shapes.requires_grad_(True))
        self._wave = nn.Parameter(wave.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.prune_shape_threshold = training_args.prune_shape_threshold
        self.shape_strngth = training_args.shape_strngth
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 设置分裂参数
        self.use_splitting = training_args.use_splitting if hasattr(training_args, 'use_splitting') else False
        self._max_splits = training_args.max_splits if hasattr(training_args, 'max_splits') else 0
        self._split_factor = training_args.split_factor if hasattr(training_args, 'split_factor') else 1.0
        wave_init_noise = training_args.wave_init_noise if hasattr(training_args, 'wave_init_noise') else 0.0
        
        # 改进的wave初始化（只在初始训练时）
        if wave_init_noise > 0 and not hasattr(self, '_wave_initialized'):
            print(f"Initializing wave vectors with controlled noise (std={wave_init_noise})")
            with torch.no_grad():
                # 使用更小的初始化，避免一开始就过度分裂
                noise = torch.randn_like(self._wave) * wave_init_noise
                # 确保初始wave不会太大
                noise = noise.clamp(-wave_init_noise * 3, wave_init_noise * 3)
                self._wave.data += noise
                
                # 打印初始化后的统计
                wave_norms = torch.norm(self._wave, dim=1)
                print(f"  Initial wave stats - Mean: {wave_norms.mean():.4f}, "
                      f"Max: {wave_norms.max():.4f}, Std: {wave_norms.std():.4f}")
            self._wave_initialized = True

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._shape], 'lr': training_args.shape_lr, "name": "shape"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._wave], 'lr': training_args.wave_lr if hasattr(training_args, 'wave_lr') else 0.0, "name": "wave"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        l.append('shape')
        # 添加wave属性
        for i in range(3):  # 假设wave是3D向量
            l.append('wave_{}'.format(i))
        return l

    def save_ply(self, path):
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

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_shape(self):
        shapes_new = torch.zeros((self.get_opacity.shape[0], 1), dtype=torch.float, device="cuda")
        optimizable_tensors = self.replace_tensor_to_optimizer(shapes_new, "shape")
        self._shape = optimizable_tensors["shape"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 尝试读取shape属性，如果不存在则初始化为0
        shape = np.zeros((xyz.shape[0], 1), dtype=np.float32)
        try:
            shape[:, 0] = np.asarray(plydata.elements[0]["shape"])
        except:
            pass

        # 尝试读取wave属性，如果不存在则初始化为0
        wave = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        wave_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("wave_")]
        if wave_names:
            wave_names = sorted(wave_names, key = lambda x: int(x.split('_')[-1]))
            for idx, attr_name in enumerate(wave_names):
                wave[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._shape = nn.Parameter(torch.tensor(shape, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._wave = nn.Parameter(torch.tensor(wave, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

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

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._shape = optimizable_tensors["shape"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._wave = optimizable_tensors["wave"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        
        # 清除缓存，因为高斯点已经改变
        self.invalidate_split_cache()

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # Check if the group name exists in the tensors_dict, otherwise skip
            if group["name"] not in tensors_dict:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_shapes, new_scaling, new_rotation, new_wave):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "shape": new_shapes,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "wave": new_wave}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._shape = optimizable_tensors["shape"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        # Ensure wave is updated if it exists in the optimizer
        if "wave" in optimizable_tensors:
             self._wave = optimizable_tensors["wave"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # 清除缓存，因为高斯点已经改变
        self.invalidate_split_cache()

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 如果没有选中的点，直接返回
        if not selected_pts_mask.any():
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_shape = self._shape[selected_pts_mask].repeat(N,1)
        
        # Inherit parent wave vector but apply decay to prevent infinite splitting
        decay_factor = 0.7
        parent_wave = self._wave[selected_pts_mask]  # [num_selected, 3]
        new_wave = parent_wave.repeat(N, 1) * decay_factor  # [num_selected * N, 3]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_shape, new_scaling, new_rotation, new_wave)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_shapes = self._shape[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        # Also pass wave for cloning
        new_wave = self._wave[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_shapes, new_scaling, new_rotation, new_wave)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        # 原有的剪枝条件
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # 智能的小高斯剪枝策略
        min_scale_threshold = 1e-4
        tiny_gaussians = self.get_scaling.max(dim=1).values < min_scale_threshold
        
        if tiny_gaussians.any():
            # 计算wave的梯度幅度（如果有梯度信息）
            wave_importance = torch.zeros(self._wave.shape[0], device='cuda')
            if self._wave.grad is not None:
                wave_importance = torch.norm(self._wave.grad, dim=1)
            
            # 也考虑wave本身的大小
            wave_norms = torch.norm(self._wave, dim=1)
            
            # 只剪枝那些既小又不重要的高斯
            # 条件：scale小 AND (wave小 OR wave梯度小)
            low_wave_threshold = 0.01
            low_wave_grad_threshold = 0.001
            
            unimportant_tiny = tiny_gaussians & (
                (wave_norms < low_wave_threshold) | 
                (wave_importance < low_wave_grad_threshold)
            )
            
            if unimportant_tiny.any():
                print(f"[Pruning] Removing {unimportant_tiny.sum().item()} unimportant tiny gaussians "
                      f"(out of {tiny_gaussians.sum().item()} tiny gaussians)")
            
            prune_mask = torch.logical_or(prune_mask, unimportant_tiny)
        
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    
    def size_prune(self, min_shape):
        prune_mask = (self.get_shape < min_shape).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, use_fast_render=False):
        if use_fast_render:
            # 如果有wave参数，则用wave范数影响estimated_grad
            if hasattr(self, '_wave') and self._wave is not None and self._wave.numel() > 0:
                wave_norms = torch.norm(self._wave[update_filter], dim=1, keepdim=True)
                # 你可以根据实际需要调整权重
                estimated_grad = wave_norms * 0.001 + 0.0001  # 基础梯度 + wave贡献
                self.xyz_gradient_accum[update_filter] += estimated_grad
                self.denom[update_filter] += 1
            else:
                # 没有wave参数时，回退到原有逻辑
                print("No wave parameter, using original gradient")
                self.xyz_gradient_accum[update_filter] += 0.0001
                self.denom[update_filter] += 1
        else:
            # 原有的基于viewspace_point_tensor.grad的统计方式
            print("Using original gradient")
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1

    def compute_split_scale_factor(self, wave_norm, split_index, base_scale_min):
        """
        理论计算分裂高斯球的协方差缩放因子
        
        参数:
            wave_norm (float): 波向量的范数
            split_index (int): 分裂的索引
            base_scale_min (float): 原高斯球的最小缩放
        
        返回:
            float: 协方差缩放因子
        """
        return 1.0 + (split_index**2 * math.pi**2) / (wave_norm**2 * base_scale_min)
    
    def print_split_statistics(self, iteration):
        """打印分裂统计信息"""
        if iteration in self.split_statistics:
            stats = self.split_statistics[iteration]
            total_splits = sum(stats.values())
            print(f"\n[Split Statistics at iteration {iteration}]")
            print(f"Total split gaussians: {total_splits}")
            for k, count in sorted(stats.items()):
                print(f"  k={k}: {count} gaussians")