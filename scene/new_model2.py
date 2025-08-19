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
        self._split_factor = 1.0      # 分裂因子
        self._max_splits = 0          # 最大分裂次数
        self.use_splitting = False    # 是否启用分裂
        self._grad_warning_printed = False  # 添加梯度警告标志
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

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

    def compute_directional_covariance(self, original_cov, wave_dir, split_factor):
        """
        计算方向性协方差变换，基于波向量方向和分裂因子
        
        Args:
            original_cov: 原始协方差矩阵 [B, 6] (OpenGL格式)
            wave_dir: 波向量方向 [B, 3]
            split_factor: 分裂缩放因子
        
        Returns:
            变换后的协方差矩阵 [B, 6]
        """
        batch_size = original_cov.shape[0]
        result = torch.zeros_like(original_cov)
        
        # 将OpenGL格式的协方差转为3x3矩阵
        for b in range(batch_size):
            # 提取协方差矩阵元素 (OpenGL格式: xx, yy, zz, xy, xz, yz)
            sx, sy, sz, xy, xz, yz = original_cov[b, 0], original_cov[b, 1], original_cov[b, 2], original_cov[b, 3], original_cov[b, 4], original_cov[b, 5]
            
            # 构建3x3协方差矩阵
            cov = torch.tensor([
                [sx, xy, xz],
                [xy, sy, yz],
                [xz, yz, sz]
            ], device=original_cov.device)
            
            # 获取波向量方向并归一化
            w_dir = wave_dir[b]
            w_norm = torch.norm(w_dir) + 1e-8
            w_dir = w_dir / w_norm
            
            # 构建波向量方向的投影矩阵 P_wave = w * w^T
            P_wave = torch.outer(w_dir, w_dir)
            
            # 构建垂直于波向量的投影矩阵 P_perp = I - P_wave
            P_perp = torch.eye(3, device=original_cov.device) - P_wave
            
            # 协方差变换系数:
            # 1. 波向量方向上缩小协方差 (乘以1/split_factor) - 使高斯在该方向变窄
            # 2. 垂直方向适当放大 (乘以sqrt(split_factor)) - 保持总体体积
            
            # 根据波向量大小动态调整变换强度
            w_magnitude = w_norm.item()
            if w_magnitude < 0.01:
                # 对非常小的波向量，增强变换效果
                enhancement = 1.2
            else:
                enhancement = 1.0
                
            # 计算变换系数
            scale_wave = 1.0 / (split_factor * enhancement)  # 波向量方向上收缩
            scale_perp = torch.sqrt(torch.tensor(split_factor, device=original_cov.device))  # 垂直方向扩张
            
            # 分解协方差到波向量方向和垂直方向
            cov_wave = torch.matmul(P_wave, torch.matmul(cov, P_wave))  # 波向量方向分量
            cov_perp = torch.matmul(P_perp, torch.matmul(cov, P_perp))  # 垂直方向分量
            cov_mixed = cov - cov_wave - cov_perp  # 混合项
            
            # 应用变换: 各方向分别缩放，保留混合项
            transformed_cov = cov_wave * scale_wave + cov_perp * scale_perp + cov_mixed
            
            # 确保对称性和正定性
            transformed_cov = 0.5 * (transformed_cov + transformed_cov.T)
            
            # 转回OpenGL格式 (xx, yy, zz, xy, xz, yz)
            result[b, 0] = transformed_cov[0, 0]  # xx
            result[b, 1] = transformed_cov[1, 1]  # yy
            result[b, 2] = transformed_cov[2, 2]  # zz
            result[b, 3] = transformed_cov[0, 1]  # xy
            result[b, 4] = transformed_cov[0, 2]  # xz
            result[b, 5] = transformed_cov[1, 2]  # yz
        
        return result

    def compute_split_scale_factor(self, wave_norm, split_index, base_scale_min):
        """
        计算分裂缩放因子，基于波向量大小和分裂次数
        """
        if self._split_factor < 0:
            # 使用改进的理论计算值
            # 基于波向量大小和分裂次数动态计算，采用更平滑的缩放函数
            k_sq = max(wave_norm ** 2, 1e-6)  # 增加数值稳定性
            base_scale_sq = max(base_scale_min ** 2, 1e-6)
            
            # 更平衡的缩放公式：在小波向量处不会产生过大的缩放，同时保持分裂效果
            mod_factor = 0.8  # 降低分母的影响程度，使缩放更稳定
            adaptive_term = (split_index**2 * (math.pi**2)) / (k_sq * base_scale_sq + 1e-6)
            adaptive_term = min(adaptive_term, 15.0)  # 限制最大缩放为15倍，防止数值不稳定
            
            # 添加非线性函数使缩放在大波向量处更加缓和
            if wave_norm > 1.0:
                adaptive_term = adaptive_term / (1.0 + 0.2 * (wave_norm - 1.0))
                
            scale_factor = 1.0 + mod_factor * adaptive_term
            return max(scale_factor, 1.0)  # 确保缩放因子不小于1.0
        else:
            # 使用固定值，但添加分裂索引的非线性项以更好地控制分裂
            return max(self._split_factor * (1.0 + 0.1 * split_index), 1.0)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, downsample_factor=1.0):
        """
        从点云创建高斯模型，支持下采样以减少初始点数量
        
        Args:
            pcd: 输入点云
            spatial_lr_scale: 空间学习率缩放
            downsample_factor: 下采样因子 (0.1-1.0), 1.0表示不下采样, 0.5表示保留50%的点
        """
        self.spatial_lr_scale = spatial_lr_scale
        
        # 点云下采样以减少初始高斯球数量，加速训练
        original_points = np.asarray(pcd.points)
        original_colors = np.asarray(pcd.colors)
        n_original = len(original_points)
        
        if downsample_factor < 1.0:
            n_keep = int(n_original * downsample_factor)
            n_keep = max(n_keep, 10000)  # 最少保留10000个点
            
            # 随机下采样 + 泊松盘采样的混合策略
            if n_keep < n_original:
                # 70%随机采样，30%均匀采样
                n_random = int(n_keep * 0.7)
                n_uniform = n_keep - n_random
                
                # 随机采样
                random_indices = np.random.choice(n_original, n_random, replace=False)
                
                # 均匀采样（空间分布更均匀）
                from sklearn.cluster import KMeans
                try:
                    kmeans = KMeans(n_clusters=n_uniform, random_state=42)
                    kmeans.fit(original_points)
                    # 找到每个聚类中心最近的点
                    from sklearn.metrics.pairwise import euclidean_distances
                    distances = euclidean_distances(kmeans.cluster_centers_, original_points)
                    uniform_indices = np.argmin(distances, axis=1)
                except:
                    # 如果聚类失败，使用均匀网格采样
                    step = n_original // n_uniform
                    uniform_indices = np.arange(0, n_original, step)[:n_uniform]
                
                # 合并索引并去重
                keep_indices = np.unique(np.concatenate([random_indices, uniform_indices]))
                
                # 如果去重后数量不够，补充随机点
                if len(keep_indices) < n_keep:
                    remaining = np.setdiff1d(np.arange(n_original), keep_indices)
                    additional = np.random.choice(remaining, n_keep - len(keep_indices), replace=False)
                    keep_indices = np.concatenate([keep_indices, additional])
                
                selected_points = original_points[keep_indices]
                selected_colors = original_colors[keep_indices]
                
                print(f"[点云下采样] 原始点数: {n_original:,} → 下采样后: {len(selected_points):,} (比例: {downsample_factor:.1%})")
            else:
                selected_points = original_points
                selected_colors = original_colors
                print(f"[点云处理] 保持原始点数: {n_original:,}")
        else:
            selected_points = original_points
            selected_colors = original_colors
            print(f"[点云处理] 使用全部点数: {n_original:,}")
        
        fused_point_cloud = torch.tensor(selected_points).float().cuda()
        fused_color = RGB2SH(torch.tensor(selected_colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        
        # 基于局部梯度方向的波向量初始化
        # 将点云转换为numpy数组以便计算KNN
        points_np = fused_point_cloud.cpu().numpy()
        num_points = points_np.shape[0]
        
        # 初始化波向量
        wave = torch.zeros((num_points, 3), device="cuda")
        
        # 如果点云规模太大，我们随机采样部分点进行KNN计算
        if num_points > 10000:
            sample_size = 10000
            sampled_indices = np.random.choice(num_points, sample_size, replace=False)
            points_sampled = points_np[sampled_indices]
        else:
            points_sampled = points_np
            sampled_indices = np.arange(num_points)
            
        try:
            # 尝试计算局部梯度方向
            from sklearn.neighbors import NearestNeighbors
            k = min(15, len(points_sampled)-1)  # KNN邻居数
            nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points_sampled)
            distances, indices = nbrs.kneighbors(points_sampled)
            
            # 对每个采样点计算局部梯度
            for i, idx in enumerate(sampled_indices):
                # 获取当前点的邻居
                neighbors = points_sampled[indices[i][1:]]  # 排除自身
                
                if len(neighbors) >= 3:  # 至少需要3个邻居才能确定平面
                    # 计算当前点的局部协方差矩阵
                    centered = neighbors - points_sampled[i]
                    cov = np.dot(centered.T, centered)
                    
                    # 对协方差矩阵进行特征分解
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        
                        # 最小特征值对应的特征向量表示法向量方向
                        normal = eigenvectors[:, 0]
                        
                        # 将法向量作为波向量的方向
                        wave[idx] = torch.tensor(normal, device="cuda")
                    except:
                        # 特征分解失败，使用随机方向
                        rand_dir = torch.randn(3, device="cuda")
                        wave[idx] = rand_dir / (torch.norm(rand_dir) + 1e-6)
            
            # 对于未采样的点，使用最近采样点的波向量
            if num_points > sample_size:
                remaining_indices = np.array([i for i in range(num_points) if i not in sampled_indices])
                remaining_points = points_np[remaining_indices]
                
                # 为剩余点找到最近的采样点
                nbrs_full = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points_sampled)
                _, nn_indices = nbrs_full.kneighbors(remaining_points)
                
                # 分配波向量
                for i, ri in enumerate(remaining_indices):
                    wave[ri] = wave[sampled_indices[nn_indices[i][0]]]
        except:
            # 如果KNN计算失败，则回退到随机初始化
            print("KNN计算失败，使用随机初始化波向量")
            for i in range(num_points):
                rand_dir = torch.randn(3, device="cuda")
                wave[i] = rand_dir / (torch.norm(rand_dir) + 1e-6)
        
        # 确保所有波向量都有合理的初始幅度范围 (0.001 to 0.05) - 调整范围更保守
        wave_norms = torch.norm(wave, dim=1, keepdim=True)
        wave_norms = torch.clamp(wave_norms, min=1e-6)  # 避免除以零
        wave_directions = wave / wave_norms  # 方向向量
        
        # 为每个波向量分配较小的随机幅度 (0.001 to 0.05) - 更保守的初始化
        target_magnitudes = 0.001 + torch.rand(num_points, device="cuda") * 0.049  # 0.001-0.05范围
        wave = wave_directions * target_magnitudes.unsqueeze(1)
        
        print(f"[Wave初始化] 波向量范围: {wave.min().item():.6f} 到 {wave.max().item():.6f}")
        print(f"[Wave初始化] 波向量平均范数: {torch.norm(wave, dim=1).mean().item():.6f}")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._wave = nn.Parameter(wave.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # 注意在training_setup中将添加实际的随机噪声缩放

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 确保分裂功能启用
        self.use_splitting = getattr(training_args, 'use_splitting', True)
        
        # 设置最大分裂次数
        if hasattr(training_args, 'max_splits'):
            self._max_splits = training_args.max_splits
        else:
            # 基于显存动态设置分裂次数上限
            try:
                allocated_gb = torch.cuda.memory_allocated() / 1024**3
                if allocated_gb > 12.0:
                    self._max_splits = 5  # 显存紧张时最多分裂5次
                elif allocated_gb > 8.0:
                    self._max_splits = 15  # 中等限制
                else:
                    self._max_splits = 25  # 相对宽松，最多分裂25次
            except:
                self._max_splits = 15  # 默认值
            
        print(f"[分裂配置] 分裂功能启用: {self.use_splitting}, 最大分裂次数: {self._max_splits}")
        
        # 设置分裂因子
        self._split_factor = getattr(training_args, 'split_factor', 2.0)
        wave_init_noise = getattr(training_args, 'wave_init_noise', 0.0)
        
        # 修复波向量噪声添加逻辑
        if wave_init_noise > 0 and (self.optimizer is None or 'wave' not in self.optimizer.state_dict()['state']):
            print(f"添加初始噪声(std={wave_init_noise})到波向量。")
            # 为每个波向量添加随机噪声，而不是简单乘以噪声
            noise = torch.randn_like(self._wave.data) * wave_init_noise
            self._wave.data = self._wave.data + noise  # 加法而不是乘法
            # 重新归一化以保持方向性但增加幅度变化
            wave_norms = torch.norm(self._wave.data, dim=1, keepdim=True)
            wave_norms = torch.clamp(wave_norms, min=1e-6)
            # 设置一个更小的波向量幅度范围 (0.001 to 0.05) - 与初始化一致
            target_magnitude = 0.001 + torch.rand_like(wave_norms.squeeze(1)) * 0.049  # 0.001-0.05范围
            self._wave.data = self._wave.data / wave_norms * target_magnitude.unsqueeze(1)
            self._wave.requires_grad_(True)
            print(f"[训练设置] 噪声后波向量范围: {self._wave.data.min().item():.6f} 到 {self._wave.data.max().item():.6f}")
            print(f"[训练设置] 噪声后波向量平均范数: {torch.norm(self._wave.data, dim=1).mean().item():.6f}")
            
        # 设置训练调度所需的属性
        self.wave_lr_init = training_args.wave_lr if hasattr(training_args, 'wave_lr') else 0.005
        self.wave_lr_final = self.wave_lr_init * 0.1  # 最终学习率为初始的10%
        self.wave_lr_delay_mult = 0.01  # 延迟因子
        self.wave_lr_max_steps = training_args.iterations if hasattr(training_args, 'iterations') else 30000

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._wave], 'lr': self.wave_lr_init, "name": "wave"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                  lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                  lr_delay_mult=training_args.position_lr_delay_mult,
                                                  max_steps=training_args.position_lr_max_steps)
        
        # 创建波向量学习率调度
        self.wave_scheduler_args = get_expon_lr_func(lr_init=self.wave_lr_init,
                                                    lr_final=self.wave_lr_final,
                                                    lr_delay_mult=self.wave_lr_delay_mult,
                                                    max_steps=self.wave_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            elif param_group["name"] == "wave":
                # 波向量使用专门的学习率调度
                wave_lr = self.wave_scheduler_args(iteration)
                param_group['lr'] = wave_lr
                
        # 返回位置学习率作为参考
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                return param_group['lr']
        return 0.0  # 如果找不到xyz参数组，返回0

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
        为分裂高斯模型定制的密集化和剪枝方法
        禁用 densify_and_split，因为分裂应该只在渲染时动态发生
        只保留克隆和剪枝功能
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # 只进行克隆，不进行分裂
        self.densify_and_clone(grads, max_grad, extent)

        # 注释掉分裂操作，因为分裂应该只在渲染时发生
        # self.densify_and_split(grads, max_grad, extent)  # 禁用！
        print(f"[SplitGaussianModel] 跳过传统分裂操作，分裂将在渲染时动态进行")

        # 保留剪枝功能
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, use_fast_render=False):
        """
        添加密度统计信息，在快速渲染模式下跳过梯度统计以避免警告
        
        Args:
            viewspace_point_tensor: 视点空间张量
            update_filter: 更新过滤器
            use_fast_render: 是否使用快速渲染模式
        """
        if use_fast_render:
            # 快速模式：使用启发式方法替代真实梯度统计
            self.denom[update_filter] += 1
            
            # 使用智能的替代梯度统计方法
            if hasattr(self, 'xyz_gradient_accum'):
                # 方法1：基于位置的变化和wave向量的活跃度
                if hasattr(self, '_wave') and self._wave is not None:
                    wave_norms = torch.norm(self._wave[update_filter], dim=1, keepdim=True)
                    # wave越大，estimated_grad越大，密集化倾向越强
                    estimated_grad = wave_norms * 0.001 + 0.0001  # 基础梯度 + wave贡献
                else:
                    # 如果没有wave，使用小的常数梯度
                    estimated_grad = torch.ones_like(self.xyz_gradient_accum[update_filter]) * 0.0002
                # 添加一些随机性以避免过度规律化
                noise = torch.randn_like(estimated_grad) * 0.00005
                estimated_grad = torch.clamp(estimated_grad + noise, min=0.0001, max=0.01)
                self.xyz_gradient_accum[update_filter] += estimated_grad
            return
        
        # 原始模式：进行完整的梯度统计
        viewspace_has_grad = False
        try:
            # 检查是否有梯度属性且梯度不为None
            if hasattr(viewspace_point_tensor, 'grad') and viewspace_point_tensor.grad is not None:
                viewspace_has_grad = True
        except Exception as e:
            # 捕获任何访问梯度时的异常
            if not hasattr(self, '_grad_warning_printed') or not self._grad_warning_printed:
                print(f"[梯度检查异常] {e}")
                self._grad_warning_printed = True
            viewspace_has_grad = False
        
        if viewspace_has_grad:
            # 正常添加梯度统计
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
            self.denom[update_filter] += 1
        else:
            # 如果梯度不存在，仍然更新分母以保持统计一致性
            self.denom[update_filter] += 1
            # 打印警告信息（只打印一次）
            if not hasattr(self, '_grad_warning_printed') or not self._grad_warning_printed:
                print("[警告] viewspace_point_tensor梯度不可用，跳过梯度统计。尝试在张量上调用retain_grad()...")
                self._grad_warning_printed = True
                # 尝试在张量上调用retain_grad以修复未来的调用
                try:
                    viewspace_point_tensor.retain_grad()
                except:
                    pass


class SplitLaplacianModel:
    """
    结合了GaussianModel中的分裂特性和LaplacianModel中的形状变化特性的模型
    暂时不使用，但保留类定义
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
        self.use_splitting = False    # 从GaussianModel继承
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.shape_prune_threshold = 0
        self.spatial_lr_scale = 0
        self.shape_strngth = 1.0
        self.setup_functions()

    # 其他方法省略，需要时再添加
    pass 