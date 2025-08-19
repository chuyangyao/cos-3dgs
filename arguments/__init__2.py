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
        self.exp_set = "00"
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
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0003
        
        # 频谱分裂密集化参数 - 重叠并行策略
        self.freq_densify_from_iter = 7_000  # 频谱分裂开始时机 (7k开始，与传统密集化并行)
        self.freq_densify_until_iter = 35_000  # 频谱分裂结束时机 (延长到35k)
        self.freq_densify_interval = 150  # 频谱分裂间隔 (适中，150轮)
        self.freq_split_protection_period = 800  # 新分裂点保护期 (800轮内不被剪枝)
        self.freq_split_warm_up_period = 6_000  # 热身期，此期间不进行频谱分裂 (前6k轮)
        
        # 并行期策略参数
        self.parallel_densify_start = 7_000  # 并行密集化开始
        self.parallel_densify_end = 15_000   # 并行密集化结束，之后仅频谱分裂
        self.freq_split_priority_ratio = 0.7  # 并行期间，70%倾向于频谱分裂
        
        # GES / Splitting 相关参数 (合并)
        self.shape_lr = 0.001
        self.shape_reset_interval = 1000 # Maybe remove if using SplitGaussianModel primarily
        self.shape_pruning_interval = 100 # Maybe remove if using SplitGaussianModel primarily
        self.prune_shape_threshold = 0.005 # Maybe remove if using SplitGaussianModel primarily
        self.prune_opacity_threshold = 0.005
        self.shape_strngth = 0.1 # For Laplacian component
        self.im_laplace_scale_factor = 0.2
        self.lambda_im_laplace = 0.4 # 从0.3增加到0.4，提高高频细节重建
        self.lambda_wave_reg: float = 0.003 # 从0.005减小到0.003，减少正则化强度，提高波向量自由度
        self.lambda_grad: float = 0.0 # Add parameter for gradient matching loss (Keep or remove? Let's keep for now, maybe rename)
        self.lambda_freq: float = 0.0 # Add parameter to potentially penalize/encourage wave norm (frequency)
        self.lambda_jacobian_wave: float = 0.5 # 从0.3增加到0.5，显著增强雅可比矩阵贡献
        self.lambda_grad_diff: float = 0.35 # 从0.2增加到0.35，大幅增强图像梯度差异损失
        self.wave_lr: float = 0.012 # 从0.008增加到0.012，加速波向量收敛
        self.wave_init_noise: float = 0.9 # 从0.7增加到0.9，增强方向指导的多样性
        
        # 添加分裂相关参数
        self.use_splitting: bool = False # 默认启用分裂功能
        self.split_factor: float = 2.0 # 分裂缩放因子，默认2.0
        self.max_splits: int = 15 # 每个高斯球最大分裂次数，默认15
        
        super().__init__(parser, "Optimization Parameters")

    def extract(self, args):
        g = super().extract(args)
        # 确保新的参数也被提取 (如果它们被命令行覆盖)
        g.use_splitting = getattr(args, "use_splitting", self.use_splitting) # 使用类定义的默认值

        # Extract all relevant params, ensuring defaults if not provided
        g.lambda_im_laplace = getattr(args, "lambda_im_laplace", self.lambda_im_laplace)
        g.lambda_wave_reg = getattr(args, "lambda_wave_reg", self.lambda_wave_reg)
        g.lambda_grad = getattr(args, "lambda_grad", self.lambda_grad)
        g.lambda_freq = getattr(args, "lambda_freq", self.lambda_freq)
        g.lambda_jacobian_wave = getattr(args, "lambda_jacobian_wave", self.lambda_jacobian_wave)
        g.lambda_grad_diff = getattr(args, "lambda_grad_diff", self.lambda_grad_diff)
        g.wave_lr = getattr(args, "wave_lr", self.wave_lr)
        g.wave_init_noise = getattr(args, "wave_init_noise", self.wave_init_noise)
        
        # 提取频谱分裂密集化参数
        g.freq_densify_from_iter = getattr(args, "freq_densify_from_iter", self.freq_densify_from_iter)
        g.freq_densify_until_iter = getattr(args, "freq_densify_until_iter", self.freq_densify_until_iter)
        g.freq_densify_interval = getattr(args, "freq_densify_interval", self.freq_densify_interval)
        g.freq_split_protection_period = getattr(args, "freq_split_protection_period", self.freq_split_protection_period)
        g.freq_split_warm_up_period = getattr(args, "freq_split_warm_up_period", self.freq_split_warm_up_period)
        
        # 提取并行期策略参数
        g.parallel_densify_start = getattr(args, "parallel_densify_start", self.parallel_densify_start)
        g.parallel_densify_end = getattr(args, "parallel_densify_end", self.parallel_densify_end)
        g.freq_split_priority_ratio = getattr(args, "freq_split_priority_ratio", self.freq_split_priority_ratio)
        
        # 添加分裂相关参数提取
        g.split_factor = getattr(args, "split_factor", self.split_factor)
        g.max_splits = getattr(args, "max_splits", self.max_splits)
        # Also extract other relevant params like shape_lr etc.
        g.shape_lr = getattr(args, "shape_lr", self.shape_lr)
        g.shape_strngth = getattr(args, "shape_strngth", self.shape_strngth)
        return g

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
