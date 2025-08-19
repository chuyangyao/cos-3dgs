#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 40_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 40_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.001
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 35_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        
        # Wave相关参数
        self.wave_lr = 0.01
        self.phase1_end = 10000
        self.phase2_end = 25000
        self.use_splitting = True
        self.max_splits = 5
        self.split_factor = 1.6
        self.lambda_wave_reg = 0.001
        self.wave_init_noise = 0.1
        
        # 高频增强参数（新增）
        self.enable_high_freq_enhancement = True  # 启用高频增强
        self.high_freq_loss_weight = 0.2  # 高频损失权重
        self.gradient_based_wave_init = True  # 基于梯度的wave初始化
        self.adaptive_densification = True  # 自适应密集化
        self.high_freq_densify_factor = 0.5  # 高频区域密集化因子(降低阈值的比例)
        self.texture_loss_weight = 0.1  # 纹理损失权重
        self.edge_loss_weight = 0.15  # 边缘损失权重
        self.protect_high_freq_gaussians = True  # 保护高频高斯不被剪枝
        self.freq_importance_threshold = 0.3  # 频率重要性阈值
        self.aggressive_split_threshold = 0.7  # 激进分裂的尺寸阈值因子
        self.min_gaussians_per_high_freq = 5  # 高频区域最小高斯数量
        self.high_freq_clone_boost = 2.0  # 高频区域克隆提升因子
        
        # 多尺度损失参数
        self.use_multi_scale_loss = True  # 使用多尺度损失
        self.multi_scale_levels = [1, 2, 4]  # 多尺度级别
        
        # 调试参数
        self.log_frequency_map_interval = 5000  # 保存频率图的间隔
        self.verbose_split_stats = False  # 详细分裂统计
        self.cache_split_data = True  # 缓存分裂数据
        
        # Wave动态调整参数
        self.wave_growth_rate = 0.02  # wave增长率
        self.max_wave_norm = 5.0  # 最大wave范数
        self.wave_sparsity_weight = 0.0001  # wave稀疏性权重
        
        super().__init__(parser, "Optimization Parameters")
    
    def get_phase_params(self, iteration):
        """
        根据迭代获取阶段特定参数
        """
        if iteration < self.phase1_end:
            return {
                'phase': 1,
                'use_wave': False,
                'use_splitting': False,
                'densify_interval': self.densification_interval,
                'wave_lr_scale': 0.0,
                'high_freq_enhancement': False
            }
        elif iteration < self.phase2_end:
            return {
                'phase': 2,
                'use_wave': True,
                'use_splitting': False,
                'densify_interval': self.densification_interval,
                'wave_lr_scale': 1.0,
                'high_freq_enhancement': self.enable_high_freq_enhancement
            }
        else:
            return {
                'phase': 3,
                'use_wave': True,
                'use_splitting': self.use_splitting,
                'densify_interval': int(self.densification_interval * 1.5),
                'wave_lr_scale': 0.5,
                'high_freq_enhancement': self.enable_high_freq_enhancement
            }
    
    def get_loss_weights(self, iteration):
        """
        获取动态损失权重
        """
        progress = min(iteration / self.iterations, 1.0)
        
        # 早期重视基础重建，后期重视细节
        base_weight = 1.0 - 0.3 * progress  # 1.0 -> 0.7
        detail_weight = 0.2 + 0.8 * progress  # 0.2 -> 1.0
        
        return {
            'l1': 0.8 * base_weight,
            'ssim': 0.2 * base_weight,
            'edge': self.edge_loss_weight * detail_weight if self.enable_high_freq_enhancement else 0.0,
            'texture': self.texture_loss_weight * detail_weight if self.enable_high_freq_enhancement else 0.0,
            'gradient': 0.1 * detail_weight,
            'wave_reg': self.lambda_wave_reg,
            'high_freq': self.high_freq_loss_weight * detail_weight if self.enable_high_freq_enhancement else 0.0
        }


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)