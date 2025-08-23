"""
config_manager.py
统一的配置管理器，解决变量跨文件定义问题
"""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class SplitConfig:
    """分裂相关的统一配置"""
    use_splitting: bool = True  # 默认启用分裂
    max_splits: int = 10
    split_factor: float = 1.6
    wave_lr: float = 0.01
    wave_init_noise: float = 0.01  # 重要：不能为0
    wave_threshold: float = 1e-4
    
    # 训练阶段控制
    phase1_end: int = 10000
    phase2_end: int = 25000
    
    # Wave正则化
    lambda_wave_reg: float = 0.001
    
    # 密集化相关
    densify_grad_threshold: float = 0.0002
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    opacity_reset_interval: int = 3000
    
    # 调试选项
    verbose: bool = True
    debug_interval: int = 100

    # 速度/精度平衡（新增）
    split_compute_interval: int = 5  # 分裂结果缓存迭代间隔
    split_fast_align: bool = True    # 快速对齐：复用原rotation，仅沿最接近主轴方向更新scale
    split_newton_steps: int = 1      # 牛顿步数（1步足够稳定，显著降开销）
    split_t_sigma_range: float = 1.5 # 驻点窗口（降低候选数量）
    split_top_ratio: float = 0.2     # 仅对top%高wave点进行分裂
    split_min_wave_norm: float = 1e-4 # 触发分裂的最小wave阈值（降低以激活分裂）
    split_max_extra_ratio: float = 0.15 # 分裂后额外点数不超过原始的比例，超出则跳过该次分裂
    split_skip_center_minN: int = 500000 # 当仅有中心项且N>=该值时直接跳过分裂，避免无效复制
    split_max_points_ratio: float = 1.10 # 分裂结果点数上限：超过 ratio*N 则视为过大，跳过
    # 高频阶段的激活建议（可在训练时动态覆盖）：
    split_top_ratio: float = 0.6
    lambda_wave_reg: float = 1e-4
    # 分裂模式：'precise' | 'parametric' | 'mixed'（训练/渲染用快速/参数化，导出用精确）
    split_mode: str = 'mixed'
    # 每次分裂调用的全局候选上限（按 wave 排序筛选），None 或 <=0 表示不限制
    split_max_active_per_call: int = 100000
    
    def __post_init__(self):
        """验证配置的合理性"""
        assert self.max_splits > 0, "max_splits must be positive"
        assert self.split_factor > 1.0, "split_factor must be > 1.0"
        assert 0 < self.wave_lr < 1.0, "wave_lr must be in (0, 1)"
        assert self.wave_init_noise > 0, "wave_init_noise must be > 0 to enable splitting"
        assert self.phase1_end < self.phase2_end, "phase1_end must be < phase2_end"

class ConfigManager:
    """全局配置管理器"""
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self._config = SplitConfig()
    
    @property
    def config(self) -> SplitConfig:
        return self._config
    
    def update_config(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                if self._config.verbose:
                    print(f"[Config] Updated {key} = {value}")
            else:
                print(f"[Config] Warning: Unknown config key: {key}")
    
    def get_phase(self, iteration: int) -> str:
        """根据迭代次数获取当前训练阶段"""
        if iteration <= self._config.phase1_end:
            return "low_freq"
        elif iteration <= self._config.phase2_end:
            return "mid_freq"
        else:
            return "high_freq"
    
    def should_use_splitting(self, iteration: int) -> bool:
        """判断当前迭代是否应该使用分裂"""
        if not self._config.use_splitting:
            return False
        # 只在高频阶段使用分裂
        return iteration > self._config.phase2_end
    
    def should_use_wave(self, iteration: int) -> bool:
        """判断当前迭代是否应该使用wave"""
        # 在中频和高频阶段使用wave
        return iteration > self._config.phase1_end
    
    def log_status(self, iteration: int, gaussians=None):
        """打印当前状态信息"""
        if not self._config.verbose:
            return
        
        if iteration % self._config.debug_interval != 0:
            return
        
        phase = self.get_phase(iteration)
        use_split = self.should_use_splitting(iteration)
        use_wave = self.should_use_wave(iteration)
        
        print(f"\n[Config Status] Iteration {iteration}")
        print(f"  Phase: {phase}")
        print(f"  Use Wave: {use_wave}")
        print(f"  Use Splitting: {use_split}")
        
        if gaussians is not None and hasattr(gaussians, '_wave') and use_wave:
            wave_norms = torch.norm(gaussians._wave, dim=1)
            active_waves = (wave_norms > self._config.wave_threshold).sum()
            print(f"  Wave Stats:")
            print(f"    Active: {active_waves}/{len(wave_norms)}")
            print(f"    Mean norm: {wave_norms.mean():.4f}")
            print(f"    Max norm: {wave_norms.max():.4f}")
            print(f"  Total Gaussians: {gaussians._xyz.shape[0]}")

# 全局配置实例
config_manager = ConfigManager()