#!/usr/bin/env python
"""
改进的评估脚本
支持多种渲染方法的评估和比较
"""

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np

def readImages(renders_dir, gt_dir, max_images=None):
    """读取渲染和GT图像"""
    renders = []
    gts = []
    image_names = []
    
    print(f"正在读取图像:")
    print(f"  渲染目录: {renders_dir}")
    print(f"  GT目录: {gt_dir}")
    
    render_files = sorted(os.listdir(renders_dir))
    if max_images:
        render_files = render_files[:max_images]
    
    print(f"  找到 {len(render_files)} 个文件")
    
    for fname in render_files:
        render_path = renders_dir / fname
        gt_path = gt_dir / fname
        
        if not os.path.exists(render_path):
            print(f"  警告: 渲染文件不存在: {render_path}")
            continue
        
        if not os.path.exists(gt_path):
            print(f"  警告: GT文件不存在: {gt_path}")
            continue
        
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        
        # 转换为tensor并移到GPU
        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
        
        renders.append(render_tensor)
        gts.append(gt_tensor)
        image_names.append(fname)
    
    print(f"  成功加载 {len(renders)} 对图像")
    
    return renders, gts, image_names

def compute_metrics(renders, gts, image_names):
    """计算评估指标"""
    ssims = []
    psnrs = []
    lpipss = []
    
    for idx in tqdm(range(len(renders)), desc="计算指标"):
        # SSIM
        ssim_val = ssim(renders[idx], gts[idx])
        ssims.append(ssim_val)
        
        # PSNR
        psnr_val = psnr(renders[idx], gts[idx])
        psnrs.append(psnr_val)
        
        # LPIPS
        lpips_val = lpips(renders[idx], gts[idx], net_type='vgg')
        lpipss.append(lpips_val)
    
    # 计算平均值
    avg_ssim = torch.tensor(ssims).mean().item()
    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_lpips = torch.tensor(lpipss).mean().item()
    
    # 计算标准差
    std_ssim = torch.tensor(ssims).std().item()
    std_psnr = torch.tensor(psnrs).std().item()
    std_lpips = torch.tensor(lpipss).std().item()
    
    return {
        'ssim': {'mean': avg_ssim, 'std': std_ssim, 'values': torch.tensor(ssims).tolist()},
        'psnr': {'mean': avg_psnr, 'std': std_psnr, 'values': torch.tensor(psnrs).tolist()},
        'lpips': {'mean': avg_lpips, 'std': std_lpips, 'values': torch.tensor(lpipss).tolist()},
        'image_names': image_names
    }

def analyze_frequency_quality(renders, gts):
    """分析频率域的重建质量"""
    high_freq_psnrs = []
    low_freq_psnrs = []
    
    for render, gt in zip(renders, gts):
        # 转换到频域
        render_fft = torch.fft.rfft2(render.mean(dim=1, keepdim=True))
        gt_fft = torch.fft.rfft2(gt.mean(dim=1, keepdim=True))
        
        # 获取幅度谱
        render_mag = torch.abs(render_fft)
        gt_mag = torch.abs(gt_fft)
        
        # 分离高频和低频
        h, w = render_mag.shape[-2:]
        center_h, center_w = h // 4, w // 4
        
        # 低频区域（中心）
        low_freq_mask = torch.zeros_like(render_mag)
        low_freq_mask[..., :center_h, :center_w] = 1
        
        # 高频区域（边缘）
        high_freq_mask = 1 - low_freq_mask
        
        # 计算低频PSNR
        low_freq_diff = (render_mag * low_freq_mask - gt_mag * low_freq_mask) ** 2
        low_freq_mse = low_freq_diff.sum() / low_freq_mask.sum()
        low_freq_psnr = 20 * torch.log10(gt_mag.max() / torch.sqrt(low_freq_mse + 1e-8))
        low_freq_psnrs.append(low_freq_psnr.item())
        
        # 计算高频PSNR
        high_freq_diff = (render_mag * high_freq_mask - gt_mag * high_freq_mask) ** 2
        high_freq_mse = high_freq_diff.sum() / high_freq_mask.sum()
        high_freq_psnr = 20 * torch.log10(gt_mag.max() / torch.sqrt(high_freq_mse + 1e-8))
        high_freq_psnrs.append(high_freq_psnr.item())
    
    return {
        'low_freq_psnr': np.mean(low_freq_psnrs),
        'high_freq_psnr': np.mean(high_freq_psnrs),
        'freq_ratio': np.mean(high_freq_psnrs) / np.mean(low_freq_psnrs)
    }

def evaluate_scene(scene_dir, render_name="split_gaussian", iteration=None, verbose=True):
    """评估单个场景"""
    results = {}
    
    # 查找渲染目录
    test_dir = Path(scene_dir) / "test"
    
    # 支持多种目录结构
    possible_dirs = []
    
    if render_name:
        # 特定渲染方法
        method_dir = test_dir / render_name
        if method_dir.exists():
            renders_base = method_dir / "renders"
            gt_dir = method_dir / "gt"
            
            # 检查是否有迭代子目录
            if iteration:
                iter_dir = renders_base / f"iter_{iteration}"
                if iter_dir.exists():
                    possible_dirs.append((f"{render_name}_{iteration}", iter_dir, gt_dir))
            
            # 检查所有迭代目录
            if renders_base.exists():
                for iter_dir in renders_base.glob("iter_*"):
                    if iter_dir.is_dir():
                        iter_num = int(iter_dir.name.split("_")[1])
                        possible_dirs.append((f"{render_name}_{iter_num}", iter_dir, gt_dir))
                
                # 如果没有迭代子目录，直接使用renders目录
                if not possible_dirs and len(list(renders_base.glob("*.png"))) > 0:
                    possible_dirs.append((render_name, renders_base, gt_dir))
    
    # 评估每个找到的目录
    for method_name, renders_dir, gt_dir in possible_dirs:
        if not renders_dir.exists() or not gt_dir.exists():
            continue
        
        if verbose:
            print(f"\n评估: {method_name}")
        
        # 读取图像
        renders, gts, image_names = readImages(renders_dir, gt_dir)
        
        if not renders:
            if verbose:
                print(f"  跳过（没有有效图像）")
            continue
        
        # 计算基础指标
        metrics = compute_metrics(renders, gts, image_names)
        
        # 频率分析
        freq_analysis = analyze_frequency_quality(renders, gts)
        metrics['frequency'] = freq_analysis
        
        # 保存结果
        results[method_name] = metrics
        
        if verbose:
            print(f"  SSIM:  {metrics['ssim']['mean']:.4f} ± {metrics['ssim']['std']:.4f}")
            print(f"  PSNR:  {metrics['psnr']['mean']:.2f} ± {metrics['psnr']['std']:.2f}")
            print(f"  LPIPS: {metrics['lpips']['mean']:.4f} ± {metrics['lpips']['std']:.4f}")
            print(f"  低频PSNR: {freq_analysis['low_freq_psnr']:.2f}")
            print(f"  高频PSNR: {freq_analysis['high_freq_psnr']:.2f}")
            print(f"  高/低频比: {freq_analysis['freq_ratio']:.3f}")
    
    return results

def evaluate_multiple_scenes(scene_dirs, render_name="split_gaussian", iteration=None):
    """评估多个场景"""
    all_results = {}
    
    # 收集所有场景的结果
    for scene_dir in scene_dirs:
        scene_name = Path(scene_dir).name
        print(f"\n{'='*60}")
        print(f"场景: {scene_name}")
        print(f"{'='*60}")
        
        results = evaluate_scene(scene_dir, render_name, iteration)
        
        if results:
            all_results[scene_name] = results
            
            # 保存场景结果
            output_file = Path(scene_dir) / f"metrics_{render_name}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n结果保存到: {output_file}")
    
    # 计算平均指标
    if all_results:
        print(f"\n{'='*60}")
        print("所有场景平均指标")
        print(f"{'='*60}")
        
        # 收集所有方法
        all_methods = set()
        for scene_results in all_results.values():
            all_methods.update(scene_results.keys())
        
        # 计算每个方法的平均值
        for method in sorted(all_methods):
            ssims = []
            psnrs = []
            lpipss = []
            low_freq_psnrs = []
            high_freq_psnrs = []
            
            for scene_name, scene_results in all_results.items():
                if method in scene_results:
                    metrics = scene_results[method]
                    ssims.append(metrics['ssim']['mean'])
                    psnrs.append(metrics['psnr']['mean'])
                    lpipss.append(metrics['lpips']['mean'])
                    if 'frequency' in metrics:
                        low_freq_psnrs.append(metrics['frequency']['low_freq_psnr'])
                        high_freq_psnrs.append(metrics['frequency']['high_freq_psnr'])
            
            if ssims:
                print(f"\n方法: {method} ({len(ssims)}个场景)")
                print(f"  平均 SSIM:  {np.mean(ssims):.4f}")
                print(f"  平均 PSNR:  {np.mean(psnrs):.2f}")
                print(f"  平均 LPIPS: {np.mean(lpipss):.4f}")
                if low_freq_psnrs:
                    print(f"  平均 低频PSNR: {np.mean(low_freq_psnrs):.2f}")
                    print(f"  平均 高频PSNR: {np.mean(high_freq_psnrs):.2f}")
                    print(f"  平均 高/低频比: {np.mean(high_freq_psnrs)/np.mean(low_freq_psnrs):.3f}")
    
    return all_results

def compare_methods(scene_dir, methods=None):
    """比较不同渲染方法"""
    if methods is None:
        methods = ["split_gaussian", "split_gaussian_presplit", "ours", "ges"]
    
    print(f"\n比较场景: {scene_dir}")
    print("="*60)
    
    all_metrics = {}
    
    for method in methods:
        results = evaluate_scene(scene_dir, method, verbose=False)
        if results:
            # 取第一个结果（如果有多个迭代）
            first_key = list(results.keys())[0]
            all_metrics[method] = results[first_key]
    
    # 打印比较表格
    if all_metrics:
        print("\n方法比较:")
        print("-"*60)
        print(f"{'方法':<20} {'SSIM':<10} {'PSNR':<10} {'LPIPS':<10} {'高频PSNR':<10}")
        print("-"*60)
        
        for method, metrics in all_metrics.items():
            ssim_val = metrics['ssim']['mean']
            psnr_val = metrics['psnr']['mean']
            lpips_val = metrics['lpips']['mean']
            high_freq = metrics.get('frequency', {}).get('high_freq_psnr', 0)
            
            print(f"{method:<20} {ssim_val:<10.4f} {psnr_val:<10.2f} {lpips_val:<10.4f} {high_freq:<10.2f}")
    
    return all_metrics

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 设置命令行参数
    parser = ArgumentParser(description="改进的评估脚本")
    parser.add_argument('--model_paths', '-m', nargs="+", type=str, required=True,
                        help="模型路径列表")
    parser.add_argument('--render_name', type=str, default="split_gaussian",
                        help="渲染方法名称")
    parser.add_argument('--iteration', type=int, default=None,
                        help="指定迭代次数")
    parser.add_argument('--compare', action='store_true',
                        help="比较不同方法")
    parser.add_argument('--methods', nargs="+", type=str, default=None,
                        help="要比较的方法列表")
    
    args = parser.parse_args()
    
    print("="*60)
    print("开始评估")
    print(f"模型路径: {args.model_paths}")
    print(f"渲染名称: {args.render_name}")
    if args.iteration:
        print(f"迭代次数: {args.iteration}")
    print("="*60)
    
    if args.compare:
        # 比较模式
        for scene_dir in args.model_paths:
            compare_methods(scene_dir, args.methods)
    else:
        # 标准评估
        evaluate_multiple_scenes(args.model_paths, args.render_name, args.iteration)
    
    print("\n评估完成!")