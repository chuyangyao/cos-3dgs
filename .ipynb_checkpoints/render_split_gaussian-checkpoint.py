#!/usr/bin/env python
"""
分裂高斯模型的渲染脚本
支持两种模式：
1. 使用wave参数进行动态分裂渲染（默认）
2. 使用预分裂的标准高斯渲染（高效）
"""

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, export_splits_to_standard_gaussians
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.new_model import SplitGaussianModel, SplitLaplacianModel
from scene.gaussian_model import GaussianModel
from scene.laplacian_model import LaplacianModel
import json

def detect_model_type(model_path, iteration):
    """
    自动检测模型类型
    """
    point_cloud_path = os.path.join(
        model_path, 
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply"
    )
    
    split_path = os.path.join(
        model_path, 
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud_split.ply"
    )
    
    # 检查是否有分裂版本
    has_split_version = os.path.exists(split_path)
    
    # 检查是否有元信息
    meta_path = os.path.join(
        model_path, 
        "point_cloud",
        f"iteration_{iteration}",
        "split_info.json"
    )
    
    model_info = {
        'has_wave': False,
        'has_split': has_split_version,
        'model_type': 'gaussian',
        'meta': None
    }
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            model_info['meta'] = json.load(f)
            model_info['has_wave'] = True
            model_info['model_type'] = 'split_gaussian'
    
    return model_info

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               render_dir="split_gaussian", use_split_version=False, split_info=None):
    """
    渲染视图集合
    
    Args:
        use_split_version: 是否使用预分裂的版本
        split_info: 分裂信息（如果有）
    """
    # 创建输出目录
    if use_split_version:
        render_path = os.path.join(model_path, name, render_dir + "_presplit", "renders")
        gts_path = os.path.join(model_path, name, render_dir + "_presplit", "gt")
    else:
        iter_str = f"iter_{iteration}" if iteration > 0 else ""
        render_path = os.path.join(model_path, name, render_dir, "renders", iter_str)
        gts_path = os.path.join(model_path, name, render_dir, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 打印渲染信息
    if split_info and split_info['meta']:
        meta = split_info['meta']
        print(f"\n渲染信息:")
        print(f"  原始高斯数: {meta.get('original_gaussians', 'N/A')}")
        print(f"  分裂高斯数: {meta.get('split_gaussians', 'N/A')}")
        print(f"  最大wave范数: {meta.get('max_wave_norm', 0):.4f}")
        print(f"  平均wave范数: {meta.get('mean_wave_norm', 0):.4f}")
        print(f"  活跃wave数: {meta.get('active_waves', 0)}")

    print(f"\n渲染{len(views)}个视图到: {render_path}")
    
    # 渲染每个视图
    for idx, view in enumerate(tqdm(views, desc="渲染进度")):
        if hasattr(gaussians, 'use_splitting') and gaussians.use_splitting and not use_split_version:
            # 使用动态分裂渲染
            rendering = render(view, gaussians, pipeline, background, 
                             iteration=iteration, max_iteration=40000)["render"]
        else:
            # 使用标准渲染（无分裂）
            rendering = render(view, gaussians, pipeline, background)["render"]
            
        gt = view.original_image[0:3, :, :]
        
        # 保存渲染结果
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    print(f"渲染完成: {len(views)}个视图")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, render_dir="split_gaussian",
                use_presplit=False, export_standard=False):
    """
    渲染训练集和测试集
    
    Args:
        use_presplit: 使用预分裂的高斯（如果存在）
        export_standard: 导出标准高斯用于其他渲染器
    """
    with torch.no_grad():
        # 检测模型类型
        model_info = detect_model_type(dataset.model_path, iteration)
        
        print(f"\n模型信息:")
        print(f"  类型: {model_info['model_type']}")
        print(f"  包含wave: {model_info['has_wave']}")
        print(f"  有分裂版本: {model_info['has_split']}")
        
        # 根据模型类型加载
        if model_info['has_wave']:
            # 加载带wave的模型
            if 'laplacian' in model_info['model_type']:
                gaussians = SplitLaplacianModel(dataset.sh_degree)
            else:
                gaussians = SplitGaussianModel(dataset.sh_degree)
                
            # 如果要使用预分裂版本
            if use_presplit and model_info['has_split']:
                print("\n使用预分裂的高斯进行高效渲染...")
                # 加载预分裂版本
                split_path = os.path.join(
                    dataset.model_path,
                    "point_cloud",
                    f"iteration_{iteration}",
                    "point_cloud_split.ply"
                )
                
                # 创建标准高斯模型加载分裂版本
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, shuffle=False)
                gaussians.load_ply(split_path)
                gaussians.use_splitting = False  # 禁用动态分裂
            else:
                print("\n使用动态分裂渲染...")
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        else:
            # 标准高斯模型
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 导出标准高斯（如果需要）
        if export_standard and model_info['has_wave']:
            print("\n导出标准高斯模型...")
            output_path = scene.export_for_standard_rendering(iteration)
            print(f"导出完成: {output_path}")

        # 渲染训练集
        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, 
                      scene.getTrainCameras(), gaussians, pipeline, background, 
                      render_dir, use_presplit, model_info)

        # 渲染测试集
        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, 
                      scene.getTestCameras(), gaussians, pipeline, background, 
                      render_dir, use_presplit, model_info)

def compare_rendering_methods(dataset, iteration, pipeline):
    """
    比较不同渲染方法的性能
    """
    import time
    
    print("\n=== 渲染方法比较 ===")
    
    # 加载场景
    gaussians = SplitGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    test_views = scene.getTestCameras()[:5]  # 使用前5个测试视图
    
    results = {}
    
    # 1. 动态分裂渲染
    print("\n1. 动态分裂渲染...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    for view in test_views:
        _ = render(view, gaussians, pipeline, background, 
                  iteration=iteration, max_iteration=40000)["render"]
    
    torch.cuda.synchronize()
    dynamic_time = time.time() - start_time
    results['dynamic'] = {
        'time': dynamic_time,
        'fps': len(test_views) / dynamic_time,
        'gaussians': gaussians._xyz.shape[0]
    }
    
    # 2. 预分裂渲染（如果存在）
    split_path = os.path.join(
        dataset.model_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud_split.ply"
    )
    
    if os.path.exists(split_path):
        print("\n2. 预分裂渲染...")
        
        # 加载预分裂版本
        standard_gaussians = GaussianModel(dataset.sh_degree)
        standard_gaussians.load_ply(split_path)
        
        torch.cuda.synchronize()
        start_time = time.time()
        
        for view in test_views:
            _ = render(view, standard_gaussians, pipeline, background)["render"]
        
        torch.cuda.synchronize()
        presplit_time = time.time() - start_time
        results['presplit'] = {
            'time': presplit_time,
            'fps': len(test_views) / presplit_time,
            'gaussians': standard_gaussians._xyz.shape[0]
        }
    
    # 打印比较结果
    print("\n=== 性能比较结果 ===")
    for method, stats in results.items():
        print(f"\n{method}:")
        print(f"  高斯数: {stats['gaussians']}")
        print(f"  总时间: {stats['time']:.3f}秒")
        print(f"  FPS: {stats['fps']:.2f}")
    
    if 'presplit' in results:
        speedup = results['dynamic']['time'] / results['presplit']['time']
        print(f"\n预分裂加速比: {speedup:.2f}x")

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="分裂高斯渲染脚本")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_dir", default="split_gaussian", type=str, 
                        help="渲染输出目录名")
    parser.add_argument("--use_presplit", action="store_true",
                        help="使用预分裂的高斯进行高效渲染")
    parser.add_argument("--export_standard", action="store_true",
                        help="导出标准高斯模型")
    parser.add_argument("--compare", action="store_true",
                        help="比较不同渲染方法的性能")
    
    args = get_combined_args(parser)
    
    print(f"渲染模型: {args.model_path}")
    
    # 初始化系统状态
    safe_state(args.quiet)
    
    # 如果需要比较性能
    if args.compare:
        compare_rendering_methods(model.extract(args), args.iteration, pipeline.extract(args))
    else:
        # 正常渲染
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                   args.skip_train, args.skip_test, args.render_dir,
                   args.use_presplit, args.export_standard)
    
    print("\n渲染完成!")