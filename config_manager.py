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
    wave_threshold: float = 0.01
    
    # 训练阶段控制
    phase1_end: int = 10000
    phase2_end: int = 25000
    
    # Wave正则化
    lambda_wave_reg: float = 0.001
    
    # 密集化相关
    densify_grad_threshold: float = 0.0002
    densification_interval: int = 100
    densify_from_iter: int = 500
    densify_until_iter: int = 35000
    opacity_reset_interval: int = 3000
    
    # 调试选项
    verbose: bool = True
    debug_interval: int = 100
    
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