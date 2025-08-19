"""
scene/__init__.py
修复后的Scene类，添加checkpoint和改进的保存功能
"""

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.new_model import SplitGaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from torch import nn

# 导入checkpoint管理器
from checkpoint_manager import CheckpointManager
from config_manager import config_manager

class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 初始化checkpoint管理器
        self.checkpoint_mgr = CheckpointManager(self.model_path)

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(
                os.path.join(self.model_path, "input.ply"), 'wb'
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply"
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, save_split_version=False):
        """保存场景，包括原始版本和可选的预分裂版本"""
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        os.makedirs(point_cloud_path, exist_ok=True)
        
        # 保存原始版本
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        # 保存分裂版本（如果需要）
        if save_split_version and hasattr(self.gaussians, 'use_splitting') and self.gaussians.use_splitting:
            print(f"[Save] Creating split version at iteration {iteration}...")
            
            # 获取分裂数据
            if hasattr(self.gaussians, 'get_split_data'):
                split_data = self.gaussians.get_split_data(iteration, 40000)
                
                if split_data is not None:
                    split_path = os.path.join(point_cloud_path, "point_cloud_split.ply")
                    
                    # 创建临时的高斯模型来保存分裂版本
                    if isinstance(self.gaussians, SplitGaussianModel):
                        split_gaussians = SplitGaussianModel(self.gaussians.max_sh_degree)
                    elif isinstance(self.gaussians, SplitLaplacianModel):
                        split_gaussians = SplitLaplacianModel(self.gaussians.max_sh_degree)
                    else:
                        split_gaussians = GaussianModel(self.gaussians.max_sh_degree)
                    
                    print(f"[Export] Converting {self.gaussians._xyz.shape[0]} gaussians to {split_data['split_xyz'].shape[0]} split gaussians")
                    
                    # 设置分裂后的参数
                    split_gaussians._xyz = nn.Parameter(split_data['split_xyz'])
                    split_gaussians._features_dc = nn.Parameter(split_data['split_features_dc'])
                    split_gaussians._features_rest = nn.Parameter(split_data['split_features_rest'])
                    split_gaussians._opacity = nn.Parameter(split_data['split_opacity'])
                    split_gaussians._scaling = nn.Parameter(split_data['split_scaling'])
                    split_gaussians._rotation = nn.Parameter(split_data['split_rotation'])
                    
                    if hasattr(split_gaussians, '_shape'):
                        split_gaussians._shape = nn.Parameter(torch.ones_like(split_data['split_opacity']))
                    
                    # 预分裂版本不需要wave参数
                    if hasattr(split_gaussians, '_wave'):
                        split_gaussians._wave = nn.Parameter(torch.zeros((split_data['split_xyz'].shape[0], 3), device='cuda'))
                    
                    # 保存分裂版本
                    split_gaussians.save_ply(split_path)
                    print(f"[Save] Split version saved to {split_path}")
                    
                    # 保存分裂信息
                    split_info = {
                        'original_gaussians': split_data['n_original'],
                        'split_gaussians': split_data['split_xyz'].shape[0],
                        'split_ratio': split_data['split_xyz'].shape[0] / split_data['n_original'] if split_data['n_original'] > 0 else 0,
                        'max_splits': self.gaussians._max_splits if hasattr(self.gaussians, '_max_splits') else 10
                    }
                    
                    split_info_path = os.path.join(point_cloud_path, "split_info.json")
                    with open(split_info_path, 'w') as f:
                        json.dump(split_info, f, indent=2)
                    print(f"[Save] Split info saved to {split_info_path}")

    def save_checkpoint(self, iteration):
        """保存完整的checkpoint，包括优化器状态"""
        print(f"[Checkpoint] Saving checkpoint at iteration {iteration}")
        
        # 保存点云
        self.save(iteration, save_split_version=True)
        
        # 保存checkpoint
        scene_info = {
            'cameras_extent': self.cameras_extent,
            'model_path': self.model_path,
        }
        
        # 获取当前配置
        config_dict = {
            'use_splitting': config_manager.config.use_splitting,
            'max_splits': config_manager.config.max_splits,
            'wave_lr': config_manager.config.wave_lr,
            'phase1_end': config_manager.config.phase1_end,
            'phase2_end': config_manager.config.phase2_end,
        }
        
        checkpoint_path = self.checkpoint_mgr.save_checkpoint(
            self.gaussians,
            iteration,
            scene_info=scene_info,
            config=config_dict,
            save_full=True
        )
        
        print(f"[Checkpoint] Saved to {checkpoint_path}")
        
        # 清理旧的checkpoints
        self.checkpoint_mgr.cleanup_old_checkpoints(keep_latest=3)

    def load_checkpoint(self, checkpoint_path=None, iteration=None):
        """加载checkpoint"""
        load_info = self.checkpoint_mgr.load_checkpoint(
            self.gaussians,
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            load_optimizer=True
        )
        
        self.loaded_iter = load_info['iteration']
        
        # 恢复配置
        if load_info['config']:
            config_manager.update_config(**load_info['config'])
        
        return load_info

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]