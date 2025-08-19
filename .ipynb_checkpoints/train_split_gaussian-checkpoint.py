from __future__ import annotations
import os
import datetime
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
# Need F for conv2d
import torch.nn.functional as F 
from gaussian_renderer import render, network_gui
import sys
from scene import Scene
from scene.new_model import SplitGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, apply_dog_filter
from utils.extra_utils import random_id
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import (
    compute_gradient_loss, compute_frequency_loss, 
    compute_wave_sparsity_loss, compute_wave_smoothness_loss,
    compute_adaptive_frequency_loss, compute_wave_diversity_loss,
    compute_wave_gradient_guidance_loss
)
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def compute_image_gradients(image_batch):
    """Computes image gradients (Sobel magnitude) for a batch of images."""
    # Ensure image is in float format
    image_batch = image_batch.float()
    # Convert to grayscale if it has 3 channels
    if image_batch.shape[1] == 3:
        # Standard RGB to grayscale conversion weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=image_batch.device).view(1, 3, 1, 1)
        grayscale_batch = (image_batch * weights).sum(dim=1, keepdim=True)
    elif image_batch.shape[1] == 1:
        grayscale_batch = image_batch
    else:
        raise ValueError(f"Unsupported number of channels: {image_batch.shape[1]}")

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image_batch.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image_batch.device).unsqueeze(0).unsqueeze(0)

    # Apply convolution
    grad_x = F.conv2d(grayscale_batch, sobel_x, padding=1)
    grad_y = F.conv2d(grayscale_batch, sobel_y, padding=1)
    
    # Return gradient magnitude
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8) # Add epsilon for stability
    return gradient_magnitude

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, log_to_wandb=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = SplitGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练进度")
    first_iter += 1
    
    # 简化损失权重
    lambda_dssim = opt.lambda_dssim
    lambda_freq = 0.1  # 简化的频率损失
    lambda_wave_reg = 0.01  # 增强wave正则化

    # 添加一个标志来跟踪训练是否已经发散
    training_diverged = False

    for iteration in range(first_iter, opt.iterations + 1):
        if training_diverged:
            print(f"[Warning] Training has diverged at iteration {iteration}, stopping...")
            break
            
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加SH的程度直到最大值
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        bg = background
        
        try:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, 
                              iteration=iteration, max_iteration=opt.iterations)
        except RuntimeError as e:
            print(f"[Error] Rendering failed at iteration {iteration}: {e}")
            training_diverged = True
            break
            
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # 如果使用分裂，创建一个新的tensor来存储梯度
        if gaussians.use_splitting and "split_viewspace_points" in render_pkg and render_pkg["split_viewspace_points"] is not None:
            viewspace_point_tensor_grad_holder = torch.zeros_like(viewspace_point_tensor, requires_grad=True)
        else:
            viewspace_point_tensor_grad_holder = viewspace_point_tensor
            
        # 确保需要梯度
        if not viewspace_point_tensor_grad_holder.requires_grad:
            viewspace_point_tensor_grad_holder = viewspace_point_tensor_grad_holder.detach().requires_grad_(True)
        
        try:
            viewspace_point_tensor_grad_holder.retain_grad()
        except:
            pass
            
        # 简化的损失计算
        gt_image = viewpoint_cam.original_image.cuda()
        
        # 基础损失
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)
        
        # 检查损失是否为NaN
        if torch.isnan(Ll1) or torch.isnan(ssim_loss):
            print(f"[Warning] NaN detected in loss at iteration {iteration}")
            training_diverged = True
            break
        
        # 简化的频率损失 - 只计算梯度差异来鼓励高频细节
        grad_loss = torch.tensor(0.0, device="cuda")
        if lambda_freq > 0 and iteration > 5000:  # 在早期训练后才开始
            # 简单的梯度损失
            img_grad_x = image[:, :, 1:] - image[:, :, :-1]
            img_grad_y = image[:, 1:, :] - image[:, :-1, :]
            gt_grad_x = gt_image[:, :, 1:] - gt_image[:, :, :-1]
            gt_grad_y = gt_image[:, 1:, :] - gt_image[:, :-1, :]
            
            grad_loss = (torch.abs(img_grad_x - gt_grad_x).mean() + 
                        torch.abs(img_grad_y - gt_grad_y).mean()) * 0.5
        
        # 增强的Wave正则化
        wave_reg_loss = torch.tensor(0.0, device="cuda")
        if lambda_wave_reg > 0:
            wave_norms = torch.norm(gaussians._wave, dim=1)
            
            # 基础L2正则化
            wave_reg_loss = wave_norms.mean()
            
            # 对大wave的额外惩罚
            large_wave_penalty = torch.relu(wave_norms - 2.0).mean() * 10.0
            wave_reg_loss = wave_reg_loss + large_wave_penalty
            
            # 防止wave变得过大
            max_wave_norm = wave_norms.max()
            if max_wave_norm > 10.0:
                print(f"[Warning] Max wave norm is {max_wave_norm.item():.2f} at iteration {iteration}")
                # 强制缩小wave
                with torch.no_grad():
                    gaussians._wave.data = gaussians._wave.data * (5.0 / max_wave_norm)
        
        # 组合总损失
        loss = Ll1 + lambda_dssim * ssim_loss + lambda_freq * grad_loss + lambda_wave_reg * wave_reg_loss
        
        # 检查总损失
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[Warning] Invalid loss detected at iteration {iteration}: {loss.item()}")
            training_diverged = True
            break
               
        loss.backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_([gaussians._xyz, gaussians._scaling, gaussians._rotation, 
                                       gaussians._opacity, gaussians._wave], max_norm=1.0)

        # 处理分裂高斯的梯度聚合
        if gaussians.use_splitting and "split_viewspace_points" in render_pkg and render_pkg["split_viewspace_points"] is not None:
            split_viewspace_points = render_pkg["split_viewspace_points"]
            split_info = render_pkg["split_info"]
            
            if split_viewspace_points.grad is not None:
                # 创建一个用于聚合梯度的tensor
                aggregated_grad = torch.zeros((split_info['n_original'], 3), 
                                            device='cuda', dtype=split_viewspace_points.grad.dtype)
                
                # 使用scatter_add将分裂高斯的梯度聚合回原始高斯
                aggregated_grad = aggregated_grad.scatter_add(0, 
                                                             split_info['original_indices'].unsqueeze(1).expand(-1, 3),
                                                             split_viewspace_points.grad[:, :3])
                
                # 手动设置梯度
                viewspace_point_tensor_grad_holder.grad = aggregated_grad

        iter_end.record()

        with torch.no_grad():
            # 检查wave是否有NaN
            if gaussians._wave.isnan().any():
                print(f"[ERROR] NaN detected in wave before optimizer step at iteration {iteration}")
                # 重置wave为小值
                gaussians._wave.data[gaussians._wave.isnan()] = 0.0
            
            # 梯度裁剪
            if gaussians._wave.grad is not None:
                # 检查梯度是否有NaN
                if gaussians._wave.grad.isnan().any():
                    print(f"[Warning] NaN in wave gradient at iteration {iteration}, setting to zero")
                    gaussians._wave.grad[gaussians._wave.grad.isnan()] = 0.0
                
                # 裁剪梯度
                max_grad_norm = 1.0
                grad_norm = gaussians._wave.grad.norm()
                if grad_norm > max_grad_norm:
                    gaussians._wave.grad *= max_grad_norm / grad_norm
            
            # 限制wave的最大值
            max_wave_norm = 10.0  # 根据wavelength计算，这对应约0.6的最小波长
            wave_norms = gaussians._wave.norm(dim=1)
            over_limit_mask = wave_norms > max_wave_norm
            if over_limit_mask.any():
                gaussians._wave.data[over_limit_mask] *= max_wave_norm / wave_norms[over_limit_mask].unsqueeze(1)

            # Progress bar update
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 详细的参数监控
            if tb_writer and iteration % 10 == 0:
                # 基础损失
                tb_writer.add_scalar('train_loss/l1_loss', Ll1.item(), iteration)
                tb_writer.add_scalar('train_loss/total_loss', loss.item(), iteration)
                if grad_loss.item() > 0:
                    tb_writer.add_scalar('train_loss/grad_loss', grad_loss.item(), iteration)
                if wave_reg_loss.item() > 0:
                    tb_writer.add_scalar('train_loss/wave_reg_loss', wave_reg_loss.item(), iteration)
                
                # 详细的Wave统计
                if hasattr(gaussians, '_wave'):
                    wave_norms = torch.norm(gaussians._wave, dim=1)
                    
                    # 基础统计
                    tb_writer.add_scalar('wave/mean_norm', wave_norms.mean().item(), iteration)
                    tb_writer.add_scalar('wave/max_norm', wave_norms.max().item(), iteration)
                    tb_writer.add_scalar('wave/min_norm', wave_norms.min().item(), iteration)
                    tb_writer.add_scalar('wave/std_norm', wave_norms.std().item(), iteration)
                    
                    # 分位数统计
                    tb_writer.add_scalar('wave/percentile_50', torch.quantile(wave_norms, 0.5).item(), iteration)
                    tb_writer.add_scalar('wave/percentile_90', torch.quantile(wave_norms, 0.9).item(), iteration)
                    tb_writer.add_scalar('wave/percentile_99', torch.quantile(wave_norms, 0.99).item(), iteration)
                    
                    # 活跃wave统计
                    active_waves = (wave_norms > 0.01).sum().item()
                    tb_writer.add_scalar('wave/active_count', active_waves, iteration)
                    tb_writer.add_scalar('wave/active_ratio', active_waves / wave_norms.shape[0], iteration)
                    
                    # Wave梯度统计
                    if gaussians._wave.grad is not None:
                        wave_grad_norms = torch.norm(gaussians._wave.grad, dim=1)
                        tb_writer.add_scalar('gradients/wave_grad_mean', wave_grad_norms.mean().item(), iteration)
                        tb_writer.add_scalar('gradients/wave_grad_max', wave_grad_norms.max().item(), iteration)
                
                # 缩放参数统计
                if hasattr(gaussians, '_scaling'):
                    scaling_mean = gaussians.get_scaling.mean(dim=1)
                    tb_writer.add_scalar('scaling/mean', scaling_mean.mean().item(), iteration)
                    tb_writer.add_scalar('scaling/max', scaling_mean.max().item(), iteration)
                    tb_writer.add_scalar('scaling/min', scaling_mean.min().item(), iteration)
                    
                    # 缩放梯度
                    if gaussians._scaling.grad is not None:
                        scaling_grad_norm = torch.norm(gaussians._scaling.grad, dim=1)
                        tb_writer.add_scalar('gradients/scaling_grad_mean', scaling_grad_norm.mean().item(), iteration)
                        tb_writer.add_scalar('gradients/scaling_grad_max', scaling_grad_norm.max().item(), iteration)
                
                # 不透明度统计
                if hasattr(gaussians, '_opacity'):
                    opacity = gaussians.get_opacity.squeeze()
                    tb_writer.add_scalar('opacity/mean', opacity.mean().item(), iteration)
                    tb_writer.add_scalar('opacity/min', opacity.min().item(), iteration)
                    tb_writer.add_scalar('opacity/max', opacity.max().item(), iteration)

                if hasattr(gaussians, 'print_split_statistics'):
                    gaussians.print_split_statistics(iteration)
                    
                    # 不透明度梯度
                    if gaussians._opacity.grad is not None:
                        opacity_grad_norm = torch.norm(gaussians._opacity.grad)
                        tb_writer.add_scalar('gradients/opacity_grad_norm', opacity_grad_norm.item(), iteration)
                
                # 位置梯度统计
                if gaussians._xyz.grad is not None:
                    xyz_grad_norm = torch.norm(gaussians._xyz.grad, dim=1)
                    tb_writer.add_scalar('gradients/xyz_grad_mean', xyz_grad_norm.mean().item(), iteration)
                    tb_writer.add_scalar('gradients/xyz_grad_max', xyz_grad_norm.max().item(), iteration)
                
                # 高斯数量统计
                tb_writer.add_scalar('gaussians/total_count', gaussians._xyz.shape[0], iteration)
                
                # 从render_pkg获取分裂统计
                if 'split_data' in render_pkg and render_pkg['split_data'] is not None:
                    split_data = render_pkg['split_data']
                    if 'n_splits' in split_data:
                        tb_writer.add_scalar('gaussians/split_count', split_data['n_splits'], iteration)
                        tb_writer.add_scalar('gaussians/split_ratio', 
                                           split_data['n_splits'] / gaussians._xyz.shape[0], iteration)
                    
                    # 如果有调试统计信息
                    if 'debug_stats' in split_data:
                        debug_stats = split_data['debug_stats']
                        if 'wave_stats' in debug_stats:
                            for key, value in debug_stats['wave_stats'].items():
                                tb_writer.add_scalar(f'debug/wave_{key}', value, iteration)
                        if 'scale_stats' in debug_stats:
                            for key, value in debug_stats['scale_stats'].items():
                                tb_writer.add_scalar(f'debug/scale_{key}', value, iteration)
                        if 'split_distribution' in debug_stats:
                            for k, count in debug_stats['split_distribution'].items():
                                tb_writer.add_scalar(f'debug/split_k_{k}', count, iteration)

            # 每1000次迭代打印详细统计
            if iteration % 1000 == 0:
                print(f"\n[ITER {iteration}] Detailed Statistics:")
                print(f"  Loss: {loss.item():.6f} (L1: {Ll1.item():.6f}, SSIM: {ssim_loss.item():.6f})")
                if hasattr(gaussians, '_wave'):
                    wave_norms = torch.norm(gaussians._wave, dim=1)
                    print(f"  Wave norms - Mean: {wave_norms.mean():.4f}, Max: {wave_norms.max():.4f}, "
                          f"Active: {(wave_norms > 0.01).sum().item()}/{wave_norms.shape[0]}")
                print(f"  Gaussians: {gaussians._xyz.shape[0]}")

            # Log and save
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # 密度调整
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中最大半径以进行剪枝
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                # 添加密度化统计
                if viewspace_point_tensor_grad_holder.grad is not None:
                    if viewspace_point_tensor_grad_holder.grad.shape[0] == visibility_filter.shape[0]:
                        gaussians.add_densification_stats(viewspace_point_tensor_grad_holder, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # 不透明度重置    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步进
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                
                # 清除分裂缓存，因为参数已更新
                if hasattr(gaussians, 'invalidate_split_cache'):
                    gaussians.invalidate_split_cache()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] 保存检查点".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
            # 简化的测试评估
            if iteration in testing_iterations:
                torch.cuda.empty_cache()
                validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                     {'name': 'train', 'cameras' : scene.getTrainCameras()})
                
                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0
                        psnr_test = 0
                        for idx, viewpoint in enumerate(config['cameras']):
                            try:
                                render_pkg = render(viewpoint, gaussians, pipe, background, 
                                                  iteration=iteration, max_iteration=opt.iterations)
                                image = render_pkg["render"]
                                gt_image = viewpoint.original_image.cuda()
                                l1_test += l1_loss(image, gt_image).mean().item()
                                psnr_test += psnr(image, gt_image).mean().item()
                            except RuntimeError as e:
                                print(f"[Error] Validation rendering failed: {e}")
                                continue
                        if l1_test > 0:
                            l1_test /= len(config['cameras'])
                            psnr_test /= len(config['cameras'])
                            print(f"\n[ITER {iteration}] 验证 {config['name']}: L1 = {l1_test:.6f}, PSNR = {psnr_test:.3f}")
                            if tb_writer:
                                tb_writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
                                tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # Ensure output path is relative to the project directory
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("输出文件夹: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 只使用 Tensorboard，完全移除 wandb
    tb_writer = None
    if TENSORBOARD_FOUND:
        # Log directory should be the model path
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard不可用：不记录进度")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, mask_loss, 
                   grad_loss, wave_reg_loss, freq_loss, wave_sparsity_loss, 
                   wave_smooth_loss, wave_diversity_loss, wave_guidance_loss,
                   elapsed, testing_iterations, 
                   scene : Scene, renderFunc, renderArgs, opt):
    if tb_writer:
        # Log main losses
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        
        # Log specific loss components
        if mask_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/mask_loss', mask_loss.item(), iteration)
        if grad_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/grad_loss', grad_loss.item(), iteration)
        if wave_reg_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/wave_reg_loss', wave_reg_loss.item(), iteration)
        if freq_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/freq_loss', freq_loss.item(), iteration)
        if wave_sparsity_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/wave_sparsity_loss', wave_sparsity_loss.item(), iteration)
        if wave_smooth_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/wave_smooth_loss', wave_smooth_loss.item(), iteration)
        if wave_diversity_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/wave_diversity_loss', wave_diversity_loss.item(), iteration)
        if wave_guidance_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/wave_guidance_loss', wave_guidance_loss.item(), iteration)

        # Log parameter statistics
        gaussians = scene.gaussians
        tb_writer.add_scalar('params/xyz_norm', gaussians._xyz.data.norm(), iteration)
        tb_writer.add_scalar('params/scaling_mean', gaussians.get_scaling.mean(), iteration)
        tb_writer.add_scalar('params/rotation_norm', gaussians._rotation.data.norm(), iteration)
        tb_writer.add_scalar('params/opacity_mean', gaussians.get_opacity.mean(), iteration)
        # 添加更详细的调试信息
        if hasattr(scene.gaussians, '_wave') and scene.gaussians._wave.numel() > 0:
            wave_data = scene.gaussians._wave.data
            wave_norms = torch.norm(wave_data, dim=1)
            
            # 记录wave统计信息
            tb_writer.add_scalar('debug/wave_nan_count', wave_data.isnan().sum().item(), iteration)
            tb_writer.add_scalar('debug/wave_inf_count', wave_data.isinf().sum().item(), iteration)
            tb_writer.add_scalar('debug/wave_percentile_90', torch.quantile(wave_norms, 0.9).item(), iteration)
            tb_writer.add_scalar('debug/wave_percentile_99', torch.quantile(wave_norms, 0.99).item(), iteration)
            tb_writer.add_scalar('debug/wave_std', wave_norms.std().item(), iteration)
            
            # 记录scale统计（确保是在exp之后）
            scales = scene.gaussians.get_scaling  # 这应该是exp之后的值
            tb_writer.add_scalar('debug/scale_min', scales.min().item(), iteration)
            tb_writer.add_scalar('debug/scale_max', scales.max().item(), iteration)
            tb_writer.add_scalar('debug/scale_mean', scales.mean().item(), iteration)
            
            # 记录分裂统计
            if iteration % 100 == 0:  # 每100步记录一次
                from gaussian_renderer import vectorized_compute_splits_continuous
                split_data = vectorized_compute_splits_continuous(
                    scene.gaussians, 
                    max_splits_global=scene.gaussians._max_splits, 
                    progress=iteration/opt.iterations
                )
                if split_data is not None:
                    # 统计每个k值的分裂数量
                    k_counts = {}
                    for k in range(1, 4):  # 统计k=1,2,3
                        k_counts[k] = 0
                    
                    # 这里需要根据实际的split_data结构来统计
                    # 假设我们可以通过某种方式获取每个k的数量
                    n_original = scene.gaussians._xyz.shape[0]
                    n_split = split_data['n_splits']
                    
                    tb_writer.add_scalar('debug/split_k_1', k_counts.get(1, 0), iteration)
                    tb_writer.add_scalar('debug/split_k_2', k_counts.get(2, 0), iteration)
                    tb_writer.add_scalar('debug/split_k_3', k_counts.get(3, 0), iteration)
        # Log feature norms
        tb_writer.add_scalar('params/f_dc_norm', gaussians._features_dc.data.norm(), iteration)
        tb_writer.add_scalar('params/f_rest_norm', gaussians._features_rest.data.norm(), iteration)

        tb_writer.add_scalar('scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        # Keep existing logging for shape, opacity, wave etc.
        if hasattr(scene.gaussians, 'get_shape'):  
            tb_writer.add_scalar('scene/small_points', (scene.gaussians.get_shape < 0.5).sum().item(), iteration)
            tb_writer.add_scalar('scene/average_shape', scene.gaussians.get_shape.mean().item(), iteration)
            
            if (scene.gaussians.get_shape >= 1.0).sum().item() > 0:
                tb_writer.add_scalar('scene/large_shapes', scene.gaussians.get_shape[scene.gaussians.get_shape >= 1.0].mean().item(), iteration)
            
            if (scene.gaussians.get_shape < 1.0).sum().item() > 0:
                tb_writer.add_scalar('scene/small_shapes', scene.gaussians.get_shape[scene.gaussians.get_shape < 1.0].mean().item(), iteration)
            
            if scene.gaussians._shape.grad is not None:
                tb_writer.add_scalar('scene/shape_grads', scene.gaussians._shape.grad.data.norm(2).item(), iteration)
        
        if scene.gaussians._opacity.grad is not None:
            tb_writer.add_scalar('scene/opacity_grads', scene.gaussians._opacity.grad.data.norm(2).item(), iteration)
        
        # Add splitting specific metrics if enabled
        if hasattr(scene.gaussians, 'use_splitting') and scene.gaussians.use_splitting:
            # Log mean wave norm
            wave_norms = scene.gaussians.get_wave.norm(dim=1)
            if wave_norms.numel() > 0:
                 tb_writer.add_scalar('scene/wave_norm_mean', wave_norms.mean().item(), iteration)
                 tb_writer.add_scalar('scene/wave_norm_max', wave_norms.max().item(), iteration)
                 tb_writer.add_scalar('scene/wave_norm_min', wave_norms.min().item(), iteration)
            # Log wave gradient norm 
            if scene.gaussians._wave.grad is not None:
                tb_writer.add_scalar('scene/wave_grads_norm', scene.gaussians._wave.grad.data.norm(2).item(), iteration)
            # Log number of active splitting points
            if hasattr(scene.gaussians, '_wave'):
                 active_splitters = (scene.gaussians._wave.norm(dim=1) > 0.01).sum().item()  # 更低的阈值
                 tb_writer.add_scalar('scene/active_splitters', active_splitters, iteration)

        # 测试集和训练集采样评估
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                 {'name': 'train', 'cameras' : scene.getTrainCameras()})
            
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0
                    psnr_test = 0
                    for idx, viewpoint in enumerate(config['cameras']):
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, 
                                              iteration=iteration, max_iteration=opt.iterations)
                        image = render_pkg["render"]
                        gt_image = viewpoint.original_image.cuda()
                        l1_test += l1_loss(image, gt_image).cpu().numpy()
                        psnr_test += psnr(image, gt_image).cpu().numpy()
                    l1_test /= len(config['cameras'])
                    psnr_test /= len(config['cameras'])
                    print("\n[ITER {}] 验证 {}: L1 = {:.6f}, PSNR = {:.3f}".format(iteration, config['name'], l1_test, psnr_test))
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test / len(config['cameras']))

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="训练分裂高斯模型")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--use_splitting', action='store_true', default=True, help='启用高斯分裂功能 (默认启用)')
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # 初始化系统状态(RNG)
    if hasattr(args, 'eval') and args.eval:
        # 评估模式下不添加时间戳和附加信息
        print("评估模式：使用原始模型路径 " + args.model_path)
    else:
        # 训练模式下添加时间戳
        exp_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        if args.model_path is None or args.model_path == "": # Check if path is empty
            args.model_path = "output"
        args.model_path = args.model_path + "/" + exp_id
        print("训练模式：优化分裂高斯模型 " + args.model_path)
        setup = vars(args)
        setup["exp_id"] = exp_id

    safe_state(args.quiet, args.seed)

    # 启动GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 完成
    print("\n分裂高斯模型训练完成。")