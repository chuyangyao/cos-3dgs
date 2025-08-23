#!/usr/bin/env python3
"""
train_optimized_enhanced.py
集成所有修复的增强训练脚本
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import gc
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser, Namespace
import uuid

# 导入必要模块
from scene import Scene
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from scene.new_model import SplitGaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

# 导入修复模块
from config_manager import config_manager as CFG_MGR
from checkpoint_manager import CheckpointManager
from debug_validator import debug_validator
from gaussian_renderer_optimized import process_gradients_optimized

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def prepare_output_and_logger(args):
    """准备输出目录和日志"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)

    # 将控制台输出同时写入文件（完整训练日志）
    log_path = os.path.join(args.model_path, "train_full.log")
    class _Tee:
        def __init__(self, stream_a, filepath):
            self._a = stream_a
            self._f = open(filepath, 'a', buffering=1)
        def write(self, msg):
            try:
                self._a.write(msg)
            except Exception:
                pass
            try:
                self._f.write(msg)
            except Exception:
                pass
        def flush(self):
            try:
                self._a.flush()
            except Exception:
                pass
            try:
                self._f.flush()
            except Exception:
                pass
    try:
        sys.stdout = _Tee(sys.stdout, log_path)
        sys.stderr = _Tee(sys.stderr, log_path)
    except Exception as e:
        print(f"[Logger] Failed to tee stdout/stderr: {e}")
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    return tb_writer

def training_optimized_enhanced(dataset, opt, pipe, testing_iterations, saving_iterations, 
                                checkpoint_iterations, checkpoint, debug_from):
    """
    增强的优化训练函数，集成所有修复
    """
    
    # 准备输出
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # 创建模型（使用SplitGaussianModel）
    gaussians = SplitGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    # 更新全局配置
    CFG_MGR.update_config(
        use_splitting=opt.use_splitting if hasattr(opt, 'use_splitting') else True,
        max_splits=opt.max_splits if hasattr(opt, 'max_splits') else 10,
        wave_lr=opt.wave_lr if hasattr(opt, 'wave_lr') else 0.01,
        wave_init_noise=opt.wave_init_noise if hasattr(opt, 'wave_init_noise') else 0.01,
        phase1_end=opt.phase1_end if hasattr(opt, 'phase1_end') else 10000,
        phase2_end=opt.phase2_end if hasattr(opt, 'phase2_end') else 25000,
        densify_grad_threshold=opt.densify_grad_threshold,
        densification_interval=opt.densification_interval,
        densify_from_iter=opt.densify_from_iter,
        densify_until_iter=opt.densify_until_iter,
        opacity_reset_interval=opt.opacity_reset_interval,
        lambda_wave_reg=opt.lambda_wave_reg if hasattr(opt, 'lambda_wave_reg') else 0.001
    )
    
    # 设置训练
    gaussians.training_setup(opt)
    
    # 加载checkpoint（如果有）
    if checkpoint:
        try:
            load_info = scene.load_checkpoint(checkpoint_path=checkpoint)
            first_iter = load_info['iteration'] + 1
            print(f"Resumed from iteration {first_iter}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)
    
    # 背景设置
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 获取配置（数值从 config_manager.config 取；方法从 config_manager 调用）
    config = CFG_MGR.config
    
    # 训练循环
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations + 1), desc="Training")
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        progress_bar.update(1)
        
        iter_start.record()
        
        # 获取当前阶段（调试开关，默认关闭）
        if os.environ.get("DEBUG_FORCE_HIGH_FREQ", "0") == "1":
            phase = "high_freq"
        else:
            phase = CFG_MGR.get_phase(iteration)
        
        # 更新模型状态
        if phase == "high_freq":
            gaussians.use_wave = True
            gaussians.use_splitting = True
        else:
            gaussians.use_wave = CFG_MGR.should_use_wave(iteration)
            gaussians.use_splitting = CFG_MGR.should_use_splitting(iteration)
        
        # 阶段转换处理
        if iteration == config.phase1_end + 1:
            print(f"\n{'='*60}")
            print(f"[Phase 2] Starting Mid-Frequency Phase at iteration {iteration}")
            print(f"  Enabling wave parameters")
            print(f"{'='*60}")
            
            # 初始化wave（如果还没有初始化）
            if hasattr(gaussians, '_wave'):
                with torch.no_grad():
                    # 检查wave是否已经初始化
                    wave_norms = torch.norm(gaussians._wave, dim=1)
                    if wave_norms.max() < 1e-6:
                        # Wave还没有初始化，进行初始化
                        noise = torch.randn_like(gaussians._wave) * config.wave_init_noise
                        gaussians._wave.data = noise
                        print(f"  Wave initialized with noise level: {config.wave_init_noise}")
                    else:
                        print(f"  Wave already initialized, mean norm: {wave_norms.mean():.4f}")
            torch.cuda.empty_cache()
        
        elif iteration == config.phase2_end + 1:
            print(f"\n{'='*60}")
            print(f"[Phase 3] Starting High-Frequency Phase at iteration {iteration}")
            print(f"  Enabling splitting: {gaussians.use_splitting}")
            
            # 详细的调试信息
            print("[Phase 3 Debug Settings]")
            print(f"  use_splitting: {gaussians.use_splitting}")
            print(f"  config.use_splitting: {config.use_splitting}")
            if hasattr(gaussians, '_max_splits'):
                print(f"  max_splits: {gaussians._max_splits}")
            if hasattr(gaussians, '_wave'):
                wave_norms = torch.norm(gaussians._wave, dim=1)
                thr = getattr(CFG_MGR.config, 'wave_threshold', 1e-4)
                print(f"  Active waves: {(wave_norms > thr).sum()}/{len(wave_norms)} (thr={thr})")
            print(f"{'='*60}")
            # 提升高频阶段wave学习率，降低正则，点燃分裂
            for param_group in gaussians.optimizer.param_groups:
                if param_group["name"] == "wave":
                    param_group['lr'] *= 2.0
            CFG_MGR.update_config(lambda_wave_reg=max(1e-5, config.lambda_wave_reg * 0.5))
            torch.cuda.empty_cache()
        
        # 学习率调度
        gaussians.update_learning_rate(iteration)
        
        # Wave学习率调度（阶段2和3）
        if phase != "low_freq":
            for param_group in gaussians.optimizer.param_groups:
                if param_group["name"] == "wave":
                    if phase == "mid_freq":
                        base_lr = config.wave_lr * 0.5
                    else:
                        base_lr = config.wave_lr
                    
                    # Cosine衰减 + 下限
                    progress = (iteration - config.phase1_end) / (opt.iterations - config.phase1_end)
                    lr = base_lr * (0.5 * (1 + np.cos(np.pi * progress)))
                    lr_min_factor = 0.2  # 不低于基础学习率的20%
                    lr = max(lr, base_lr * lr_min_factor)
                    param_group['lr'] = lr
        
        # SH degree增加
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # 选择相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 调试模式
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # 渲染
        render_pkg = render(
            viewpoint_cam, gaussians, pipe, background,
            iteration=iteration,
            max_iteration=opt.iterations
        )
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        # 计算损失
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Wave正则化（如果在中高频阶段）
        if phase != "low_freq" and hasattr(gaussians, '_wave') and gaussians.use_wave:
            wave_reg = torch.norm(gaussians._wave, dim=1).mean()
            loss = loss + config.lambda_wave_reg * wave_reg
        
        # 反向传播
        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # 进度条更新
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.4f}"})
            
            # 密集化
            if iteration < opt.densify_until_iter:
                # 更新视空间点梯度
                # 若使用分裂，先将分裂屏幕点的梯度聚合回原始points
                if "split_viewspace_points" in render_pkg and render_pkg["split_viewspace_points"] is not None:
                    process_gradients_optimized(render_pkg, gaussians)
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # 执行密集化
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # 验证密集化前的状态
                    before_count = gaussians._xyz.shape[0]
                    
                    # 密集化和剪枝
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold
                    )
                    
                    # 验证密集化后的状态
                    after_count = gaussians._xyz.shape[0]
                    if before_count != after_count:
                        debug_validator.check_pruning(
                            gaussians, before_count, after_count, 
                            f"densification at iter {iteration}"
                        )
                
                # 重置不透明度
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()
            elif iteration == opt.densify_until_iter:
                print(f"[Densify] Stop densify/prune after iter {opt.densify_until_iter}")
            
            # 优化器步进
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            
            # 检查点保存
            if iteration in checkpoint_iterations:
                scene.save_checkpoint(iteration)
            
            # 定期保存
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, save_split_version=True)
            
            # 定期验证和状态输出
            if iteration % 500 == 0:
                # 验证分裂功能
                validation_result = debug_validator.validate_splitting(
                    gaussians, iteration, verbose=(iteration % 2000 == 0)
                )
                
                # 检查内存
                debug_validator.check_memory_usage(f"iter_{iteration}")
                
                # 打印配置状态
                CFG_MGR.log_status(iteration, gaussians)
            
            # 定期清理内存
            if iteration % 1000 == 0:
                torch.cuda.empty_cache()
                
                # 打印统计信息
                if config.verbose and phase != "low_freq":
                    wave_norms = torch.norm(gaussians._wave, dim=1) if hasattr(gaussians, '_wave') else None
                    if wave_norms is not None:
                        print(f"\n[Stats] Iteration {iteration}:")
                        print(f"  Gaussians: {gaussians._xyz.shape[0]}")
                        thr = getattr(CFG_MGR.config, 'wave_threshold', 1e-4)
                        print(f"  Active waves: {(wave_norms > thr).sum().item()} (thr={thr})")
                        print(f"  Mean wave norm: {wave_norms.mean().item():.4f}")
                        print(f"  Max wave norm: {wave_norms.max().item():.4f}")
            
            # TensorBoard记录
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('train_loss_patches/total', ema_loss_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/l1', Ll1.item(), iteration)
                tb_writer.add_scalar('total_points', gaussians._xyz.shape[0], iteration)
                tb_writer.add_scalar('phase', {"low_freq": 0, "mid_freq": 1, "high_freq": 2}[phase], iteration)
                
                if phase != "low_freq" and hasattr(gaussians, '_wave'):
                    wave_norms = torch.norm(gaussians._wave, dim=1)
                    tb_writer.add_scalar('wave/mean_norm', wave_norms.mean().item(), iteration)
                    tb_writer.add_scalar('wave/max_norm', wave_norms.max().item(), iteration)
                    thr = getattr(CFG_MGR.config, 'wave_threshold', 1e-4)
                    tb_writer.add_scalar('wave/active_count', (wave_norms > thr).sum().item(), iteration)
    
    # 训练结束
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    # 生成调试报告
    report_path = os.path.join(scene.model_path, "training_report.txt")
    debug_validator.generate_report(report_path)
    
    # 最终统计
    print(f"\nFinal Statistics:")
    print(f"  Total Gaussians: {gaussians._xyz.shape[0]}")
    if hasattr(gaussians, '_wave'):
        wave_norms = torch.norm(gaussians._wave, dim=1)
        print(f"  Active waves: {(wave_norms > 0.01).sum()}/{len(wave_norms)}")
        print(f"  Mean wave norm: {wave_norms.mean():.4f}")
    print(f"  Final loss: {ema_loss_for_log:.4f}")

if __name__ == "__main__":
    # 解析参数
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    # 初始化系统
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 启动训练
    training_optimized_enhanced(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint, args.debug_from
    )
    
    # 完成
    print("\nTraining complete!")