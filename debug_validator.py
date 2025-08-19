"""
debug_validator.py
用于调试和验证分裂功能、训练状态的模块
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
import time

class DebugValidator:
    """调试和验证工具"""
    
    def __init__(self, enable_timing: bool = True, enable_memory: bool = True):
        self.enable_timing = enable_timing
        self.enable_memory = enable_memory
        self.timing_records = {}
        self.memory_records = {}
        self.validation_history = []
        
    def validate_splitting(self, gaussians, iteration: int, verbose: bool = True) -> Dict[str, Any]:
        """验证分裂功能是否正常工作"""
        results = {
            'iteration': iteration,
            'splitting_enabled': False,
            'wave_enabled': False,
            'split_computation_valid': False,
            'issues': []
        }
        
        # 检查基本配置
        if hasattr(gaussians, 'use_splitting'):
            results['splitting_enabled'] = gaussians.use_splitting
        else:
            results['issues'].append("No 'use_splitting' attribute")
        
        if hasattr(gaussians, '_wave'):
            results['wave_enabled'] = True
            wave_norms = torch.norm(gaussians._wave, dim=1)
            results['wave_stats'] = {
                'total': len(wave_norms),
                'active': int((wave_norms > 0.01).sum()),
                'mean_norm': float(wave_norms.mean()),
                'max_norm': float(wave_norms.max()),
                'min_norm': float(wave_norms.min()),
            }
            
            # 检查wave是否被正确初始化（不应该全为0）
            if wave_norms.max() < 1e-6:
                results['issues'].append("Wave values too small (nearly zero) - check initialization!")
        else:
            results['issues'].append("No '_wave' attribute")
        
        # 测试分裂计算
        if results['wave_enabled'] and results['splitting_enabled']:
            try:
                start_time = time.time()
                
                # 尝试获取分裂数据
                if hasattr(gaussians, 'get_split_data'):
                    split_data = gaussians.get_split_data(iteration, 40000)
                    
                    if split_data is not None:
                        results['split_computation_valid'] = True
                        results['split_stats'] = {
                            'original_count': split_data['n_original'],
                            'split_count': split_data['split_xyz'].shape[0],
                            'split_ratio': split_data['split_xyz'].shape[0] / split_data['n_original'] if split_data['n_original'] > 0 else 0,
                            'computation_time': time.time() - start_time
                        }
                        
                        # 验证分裂数据的完整性
                        required_keys = ['split_xyz', 'split_opacity', 'split_scaling', 
                                       'split_rotation', 'split_features_dc', 'split_features_rest']
                        for key in required_keys:
                            if key not in split_data:
                                results['issues'].append(f"Missing key in split_data: {key}")
                    else:
                        results['issues'].append("get_split_data returned None")
                else:
                    results['issues'].append("No 'get_split_data' method")
                    
            except Exception as e:
                results['issues'].append(f"Split computation error: {str(e)}")
        
        # 打印验证结果
        if verbose:
            self._print_validation_results(results)
        
        # 记录历史
        self.validation_history.append(results)
        
        return results
    
    def _print_validation_results(self, results: Dict[str, Any]):
        """打印验证结果"""
        print(f"\n{'='*60}")
        print(f"[Validation] Iteration {results['iteration']}")
        print(f"{'='*60}")
        
        print(f"Splitting Enabled: {results['splitting_enabled']}")
        print(f"Wave Enabled: {results['wave_enabled']}")
        
        if 'wave_stats' in results:
            ws = results['wave_stats']
            print(f"Wave Statistics:")
            print(f"  Total: {ws['total']}")
            print(f"  Active (>0.01): {ws['active']} ({100*ws['active']/ws['total']:.1f}%)")
            print(f"  Mean norm: {ws['mean_norm']:.4f}")
            print(f"  Max norm: {ws['max_norm']:.4f}")
        
        if results['split_computation_valid']:
            ss = results['split_stats']
            print(f"Split Computation:")
            print(f"  Original: {ss['original_count']}")
            print(f"  Split: {ss['split_count']}")
            print(f"  Ratio: {ss['split_ratio']:.2f}x")
            print(f"  Time: {ss['computation_time']*1000:.1f}ms")
        
        if results['issues']:
            print(f"Issues Found:")
            for issue in results['issues']:
                print(f"  ❌ {issue}")
        else:
            print("✓ No issues found")
        
        print(f"{'='*60}\n")
    
    def check_memory_usage(self, tag: str = ""):
        """检查GPU内存使用情况"""
        if not self.enable_memory:
            return
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            
            self.memory_records[tag] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'timestamp': time.time()
            }
            
            print(f"[Memory {tag}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def validate_densification(self, gaussians, 
                              grads: Optional[torch.Tensor],
                              iteration: int,
                              verbose: bool = True) -> Dict[str, Any]:
        """验证密集化过程"""
        results = {
            'iteration': iteration,
            'total_gaussians_before': gaussians._xyz.shape[0],
            'issues': []
        }
        
        # 检查梯度累积
        if hasattr(gaussians, 'xyz_gradient_accum'):
            grad_accum = gaussians.xyz_gradient_accum
            results['gradient_stats'] = {
                'mean': float(grad_accum.mean()),
                'max': float(grad_accum.max()),
                'active': int((grad_accum > 0).sum())
            }
        else:
            results['issues'].append("No gradient accumulation found")
        
        # 检查密集化条件
        if hasattr(gaussians, 'percent_dense'):
            results['percent_dense'] = gaussians.percent_dense
        else:
            results['issues'].append("No percent_dense attribute")
        
        # 检查wave对密集化的影响
        if hasattr(gaussians, '_wave'):
            wave_norms = torch.norm(gaussians._wave, dim=1)
            high_wave_mask = wave_norms > 0.5
            results['wave_densification'] = {
                'high_wave_count': int(high_wave_mask.sum()),
                'should_affect_densification': True
            }
        
        if verbose:
            print(f"\n[Densification Check] Iteration {iteration}")
            print(f"  Total Gaussians: {results['total_gaussians_before']}")
            if 'gradient_stats' in results:
                gs = results['gradient_stats']
                print(f"  Gradient: mean={gs['mean']:.4f}, max={gs['max']:.4f}, active={gs['active']}")
            if 'wave_densification' in results:
                wd = results['wave_densification']
                print(f"  High wave Gaussians: {wd['high_wave_count']}")
            if results['issues']:
                print(f"  Issues: {', '.join(results['issues'])}")
        
        return results
    
    def check_pruning(self, gaussians, 
                     before_count: int,
                     after_count: int,
                     reason: str = "") -> Dict[str, Any]:
        """检查剪枝操作"""
        pruned = before_count - after_count
        
        results = {
            'before': before_count,
            'after': after_count,
            'pruned': pruned,
            'pruned_percent': 100 * pruned / before_count if before_count > 0 else 0,
            'reason': reason
        }
        
        if pruned > 0:
            print(f"[Pruning] {reason}")
            print(f"  Before: {before_count}")
            print(f"  After: {after_count}")
            print(f"  Pruned: {pruned} ({results['pruned_percent']:.1f}%)")
            
            # 检查是否错误地剪枝了高频高斯
            if hasattr(gaussians, '_wave'):
                wave_norms = torch.norm(gaussians._wave, dim=1)
                high_freq_count = (wave_norms > 0.5).sum()
                print(f"  Remaining high-freq Gaussians: {high_freq_count}")
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """生成完整的调试报告"""
        report = []
        report.append("="*80)
        report.append("DEBUG VALIDATION REPORT")
        report.append("="*80)
        
        # 验证历史
        if self.validation_history:
            report.append("\n## Validation History")
            for val in self.validation_history[-5:]:  # 最近5次
                report.append(f"\nIteration {val['iteration']}:")
                report.append(f"  Splitting: {val['splitting_enabled']}")
                report.append(f"  Wave: {val['wave_enabled']}")
                if 'wave_stats' in val:
                    ws = val['wave_stats']
                    report.append(f"  Active waves: {ws['active']}/{ws['total']}")
                if val['issues']:
                    report.append(f"  Issues: {', '.join(val['issues'])}")
        
        # 计时信息
        if self.timing_records:
            report.append("\n## Timing Information")
            for op, times in self.timing_records.items():
                avg_time = np.mean(times) * 1000  # ms
                report.append(f"  {op}: {avg_time:.2f}ms (avg over {len(times)} calls)")
        
        # 内存信息
        if self.memory_records:
            report.append("\n## Memory Usage")
            for tag, mem in self.memory_records.items():
                report.append(f"  {tag}: {mem['allocated_gb']:.2f}GB allocated")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")
        
        return report_text
    
    def time_operation(self, operation_name: str):
        """用于计时的上下文管理器"""
        class Timer:
            def __init__(self, validator, name):
                self.validator = validator
                self.name = name
                self.start_time = None
                
            def __enter__(self):
                if self.validator.enable_timing:
                    self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.validator.enable_timing and self.start_time:
                    elapsed = time.time() - self.start_time
                    if self.name not in self.validator.timing_records:
                        self.validator.timing_records[self.name] = []
                    self.validator.timing_records[self.name].append(elapsed)
        
        return Timer(self, operation_name)

# 全局调试验证器实例
debug_validator = DebugValidator()