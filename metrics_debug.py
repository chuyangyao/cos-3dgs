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
import traceback

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    
    # 打印调试信息
    print(f"正在读取图像:")
    print(f"  渲染目录: {renders_dir}")
    print(f"  GT目录: {gt_dir}")
    
    try:
        # 获取渲染文件列表
        render_files = sorted(os.listdir(renders_dir))
        print(f"  渲染文件数量: {len(render_files)}")
        
        # 获取GT文件列表
        gt_files = sorted(os.listdir(gt_dir))
        print(f"  GT文件数量: {len(gt_files)}")
        
        # 使用共同的文件列表
        common_files = sorted(set(render_files) & set(gt_files))
        print(f"  共同文件数量: {len(common_files)}")
        
        for fname in common_files:
            try:
                render_path = renders_dir / fname
                gt_path = gt_dir / fname
                
                # 检查文件是否存在
                if not os.path.exists(render_path):
                    print(f"  警告: 渲染文件不存在: {render_path}")
                    continue
                
                if not os.path.exists(gt_path):
                    print(f"  警告: GT文件不存在: {gt_path}")
                    continue
                
                render = Image.open(render_path)
                gt = Image.open(gt_path)
                renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
                gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
                image_names.append(fname)
            except Exception as e:
                print(f"  处理文件 {fname} 时出错: {str(e)}")
        
        print(f"  成功加载 {len(renders)} 对图像")
    except Exception as e:
        print(f"  读取图像时出错: {str(e)}")
        traceback.print_exc()
    
    return renders, gts, image_names

def find_render_dirs(scene_dir, render_name=None):
    """智能查找渲染目录和GT目录
    
    Args:
        scene_dir: 场景根目录
        render_name: 渲染方法名称
        
    Returns:
        render_dirs_list: 包含(method_name, renders_dir, gt_dir, iteration)的列表
    """
    render_dirs_list = []
    
    # 首先查找标准目录结构: scene_dir/test/render_name/
    test_dir = Path(scene_dir) / "test"
    
    if test_dir.exists():
        print(f"检查测试目录: {test_dir}")
        
        # 如果指定了渲染名称，则只查找对应目录
        if render_name:
            method_dirs = [test_dir / render_name]
        else:
            # 否则查找所有可能的方法目录
            method_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        
        for method_dir in method_dirs:
            if not method_dir.exists():
                continue
                
            method_name = method_dir.name
            renders_base = method_dir / "renders"
            gt_base = method_dir / "gt"
            
            print(f"  检查方法: {method_name}")
            print(f"    渲染目录: {renders_base}")
            print(f"    GT目录: {gt_base}")
            
            if renders_base.exists() and gt_base.exists():
                # 检查是否有不同迭代次数的子目录
                iter_dirs = sorted([d for d in renders_base.glob("iter_*") if d.is_dir()])
                
                if iter_dirs:
                    # 如果有迭代子目录，为每个迭代创建一个条目
                    for iter_dir in iter_dirs:
                        iter_num = int(iter_dir.name.split("_")[1])
                        
                        # GT目录也应该有对应的迭代子目录
                        gt_iter_dir = gt_base / iter_dir.name
                        
                        # 如果GT没有对应的迭代子目录，则使用基础GT目录
                        if gt_iter_dir.exists():
                            actual_gt_dir = gt_iter_dir
                        else:
                            actual_gt_dir = gt_base
                        
                        # 检查是否有文件
                        if len(list(iter_dir.glob("*.png"))) > 0:
                            render_dirs_list.append((f"{method_name}_{iter_num}", iter_dir, actual_gt_dir, iter_num))
                            print(f"    找到迭代 {iter_num}: {iter_dir}")
                else:
                    # 没有迭代子目录，直接使用renders目录
                    if len(list(renders_base.glob("*.png"))) > 0:
                        render_dirs_list.append((method_name, renders_base, gt_base, None))
                        print(f"    找到默认渲染目录")
    
    # 如果没有找到标准结构，尝试查找其他可能的路径
    if not render_dirs_list:
        print(f"未在标准路径找到渲染目录，尝试递归搜索...")
        
        # 递归查找包含renders和gt的目录
        for root, dirs, _ in os.walk(scene_dir):
            root_path = Path(root)
            
            # 跳过隐藏目录和缓存目录
            if any(part.startswith('.') for part in root_path.parts):
                continue
            
            if "renders" in dirs and "gt" in dirs:
                method_name = root_path.name  # 使用父目录名作为方法名
                
                if render_name and method_name != render_name:
                    continue  # 如果指定了渲染名称，只匹配对应名称
                
                renders_dir = root_path / "renders"
                gt_dir = root_path / "gt"
                
                if renders_dir.exists() and gt_dir.exists():
                    # 检查是否有不同迭代次数的子目录
                    iter_dirs = sorted([d for d in renders_dir.glob("iter_*") if d.is_dir()])
                    
                    if iter_dirs:
                        for iter_dir in iter_dirs:
                            iter_num = int(iter_dir.name.split("_")[1])
                            
                            # 检查GT对应的迭代目录
                            gt_iter_dir = gt_dir / iter_dir.name
                            actual_gt_dir = gt_iter_dir if gt_iter_dir.exists() else gt_dir
                            
                            if len(list(iter_dir.glob("*.png"))) > 0:
                                render_dirs_list.append((f"{method_name}_{iter_num}", iter_dir, actual_gt_dir, iter_num))
                                print(f"  找到: {method_name} 迭代 {iter_num}")
                    elif len(list(renders_dir.glob("*.png"))) > 0:
                        render_dirs_list.append((method_name, renders_dir, gt_dir, None))
                        print(f"  找到: {method_name}")
    
    # 最后尝试在场景目录直接查找
    if not render_dirs_list:
        renders_dir = Path(scene_dir) / "renders"
        gt_dir = Path(scene_dir) / "gt"
        
        if renders_dir.exists() and gt_dir.exists():
            method_name = render_name if render_name else "default"
            # 检查是否有不同迭代次数的子目录
            iter_dirs = sorted([d for d in renders_dir.glob("iter_*") if d.is_dir()])
            if iter_dirs:
                for iter_dir in iter_dirs:
                    iter_num = int(iter_dir.name.split("_")[1])
                    
                    # 检查GT对应的迭代目录
                    gt_iter_dir = gt_dir / f"iter_{iter_num}"
                    actual_gt_dir = gt_iter_dir if gt_iter_dir.exists() else gt_dir
                    
                    render_dirs_list.append((f"{method_name}_{iter_num}", iter_dir, actual_gt_dir, iter_num))
            elif len(list(renders_dir.glob("*.png"))) > 0:
                render_dirs_list.append((method_name, renders_dir, gt_dir, None))
    
    return render_dirs_list

def evaluate(model_paths, render_name=None):
    full_dict = {}
    per_view_dict = {}
    # 跟踪所有场景的平均指标
    all_metrics = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("=" * 80)
            print(f"场景: {scene_dir}")
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            # 使用智能目录查找
            render_dirs_list = find_render_dirs(scene_dir, render_name)
            
            if not render_dirs_list:
                print(f"警告: 在{scene_dir}中没有找到任何有效的渲染目录")
                continue
                
            print(f"找到{len(render_dirs_list)}个渲染结果:")
            for method, renders_dir, gt_dir, iter_num in render_dirs_list:
                iter_str = f" (迭代: {iter_num})" if iter_num is not None else ""
                print(f"  - {method}{iter_str}")
                print(f"      渲染: {renders_dir}")
                print(f"      GT: {gt_dir}")

            for method, renders_dir, gt_dir, iter_num in render_dirs_list:
                print("-" * 50)
                print(f"评估方法: {method}")
                if iter_num is not None:
                    print(f"迭代次数: {iter_num}")
                
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}

                renders, gts, image_names = readImages(renders_dir, gt_dir)
                
                if len(renders) == 0:
                    print(f"警告: 没有找到有效的图像对")
                    continue

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="指标评估进度"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                # 计算平均值
                avg_ssim = torch.tensor(ssims).mean().item()
                avg_psnr = torch.tensor(psnrs).mean().item()
                avg_lpips = torch.tensor(lpipss).mean().item()

                print(f"  SSIM : {avg_ssim:.4f}")
                print(f"  PSNR : {avg_psnr:.2f}")
                print(f"  LPIPS: {avg_lpips:.4f}")

                full_dict[scene_dir][method] = {
                    "SSIM": avg_ssim,
                    "PSNR": avg_psnr,
                    "LPIPS": avg_lpips
                }
                per_view_dict[scene_dir][method] = {
                    "SSIM": {name: ssim.item() for ssim, name in zip(ssims, image_names)},
                    "PSNR": {name: psnr.item() for psnr, name in zip(psnrs, image_names)},
                    "LPIPS": {name: lpips.item() for lpips, name in zip(lpipss, image_names)}
                }

                # 跟踪所有场景的指标
                if method not in all_metrics:
                    all_metrics[method] = {"SSIM": [], "PSNR": [], "LPIPS": []}
                all_metrics[method]["SSIM"].append(avg_ssim)
                all_metrics[method]["PSNR"].append(avg_psnr)
                all_metrics[method]["LPIPS"].append(avg_lpips)

        except Exception as e:
            print(f"处理场景 {scene_dir} 时出错: {str(e)}")
            traceback.print_exc()

    # 输出所有场景的平均结果
    if all_metrics:
        print("\n" + "=" * 80)
        print("所有场景的平均结果:")
        print("=" * 80)
        for method in all_metrics:
            avg_ssim = sum(all_metrics[method]["SSIM"]) / len(all_metrics[method]["SSIM"])
            avg_psnr = sum(all_metrics[method]["PSNR"]) / len(all_metrics[method]["PSNR"])
            avg_lpips = sum(all_metrics[method]["LPIPS"]) / len(all_metrics[method]["LPIPS"])
            print(f"{method}:")
            print(f"  平均 SSIM : {avg_ssim:.4f}")
            print(f"  平均 PSNR : {avg_psnr:.2f}")
            print(f"  平均 LPIPS: {avg_lpips:.4f}")
            print("-" * 40)

    # 保存结果
    output_dir = Path("./results/")
    output_dir.mkdir(exist_ok=True)
    
    # 使用render_name作为输出文件名的一部分
    suffix = f"_{render_name}" if render_name else ""
    
    with open(output_dir / f"per_view_metrics{suffix}.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=4)
        
    with open(output_dir / f"full_metrics{suffix}.json", 'w') as fp:
        json.dump(full_dict, fp, indent=4)
        
    print(f"\n结果已保存到 ./results/ 目录")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 设置命令行参数
    parser = ArgumentParser(description="训练脚本参数")
    parser.add_argument('--model_paths', '-m', nargs="+", type=str, required=True, 
                       help="要评估的模型路径列表")
    parser.add_argument('--render_name', type=str, default=None,
                       help="要评估的渲染方法名称 (如 'split_gaussian', 'ges' 等)，不指定则评估所有")
    args = parser.parse_args()
    
    print("=" * 80)
    print("开始评估")
    print(f"模型路径: {args.model_paths}")
    print(f"渲染名称: {args.render_name if args.render_name else '所有'}")
    print("=" * 80)
    
    evaluate(args.model_paths, args.render_name)