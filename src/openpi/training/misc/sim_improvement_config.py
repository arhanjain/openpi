"""Sim-improvement configs and HDF5 data loading support."""

import dataclasses
import pathlib
from typing import Protocol, TypeAlias

from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.policies.ur_policy as ur_policy
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType



@dataclasses.dataclass(frozen=True)
class HDF5DataConfig:
    """
    Config for training on HDF5 datasets (e.g., robomimic-style datasets).
    
    Memory modes:
    - preload=True: Load all data into RAM at init (fastest, ~10-100x speedup, high memory)
    - preload=False, prefetch_demos>0: Background prefetching (balanced)
    - preload=False, prefetch_demos=0: Load on demand (slowest, lowest memory)
    """
    # The LeRobot repo id (used for asset lookup).
    repo_id: str = tyro.MISSING
    
    # Path to HDF5 file
    hdf5_file: str | None = None
    
    # HDF5 structure configuration
    image_key: str = "obs/agentview_image"
    wrist_image_key: str | None = "obs/eye_in_hand_image"
    state_key: str = "obs/robot0_eef_pos"  # Comma-separated for multiple keys
    action_key: str = "actions"
    demo_group_pattern: str = "data/demo_*"
    
    # Default prompt if not stored in HDF5
    default_prompt: str | None = None
    prompt_key: str | None = None  # HDF5 key for prompt, if stored in data
    
    # Image format
    image_hwc: bool = True  # If True, images are (H, W, C), else (C, H, W)
    
    # Memory/performance options
    preload: bool = True  # Load all data into RAM at initialization (fastest)
    prefetch_demos: int = 50  # Number of demos to prefetch if preload=False
    
    # Repack transform customization
    repack_transforms: tyro.conf.Suppress[_transforms.Group | None] = None
    
    # Data transform customization  
    # data_transforms_factory: tyro.conf.Suppress[GroupFactory | None] = None
    
    assets_dir: str | None = None
    asset_id: str | None = None

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        """Create a DataConfig from this HDF5DataConfig."""
        from openpi.training.config import DataConfig, ModelTransformFactory
        from openpi.training.hdf5_dataset import HDF5DatasetConfig, RepeatableHDF5Dataset
        import openpi.shared.download as _download
        import openpi.shared.normalize as _normalize
        import etils.epath as epath
        import logging
        
        # Load norm stats
        norm_stats = None
        asset_id = self.asset_id or self.repo_id
        if asset_id:
            data_assets_dir = str(epath.Path(self.assets_dir or assets_dirs) / asset_id)
            try:
                norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
            except FileNotFoundError:
                logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        
        # Default repack transform
        repack_transform = self.repack_transforms
        if repack_transform is None:
            repack_transform = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "observation/exterior_image_1_left": "observation/image",
                            "observation/wrist_image_left": "observation/wrist_image",
                            "observation/state": "observation/state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
        
        # # Default data transforms (using libero policy as template, can be customized)
        # if self.data_transforms_factory is not None:
        #     data_transforms = self.data_transforms_factory(model_config)
        # else:
        #     data_transforms = _transforms.Group(
        #         inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
        #         outputs=[libero_policy.LiberoOutputs()],
        #     )
        data_transforms = _transforms.Group(
            inputs=[ur_policy.URInputs(model_type=model_config.model_type)],
            outputs=[ur_policy.UROutputs()],
        )
        
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        
        assert self.hdf5_file is not None, "Need to set hdf5_file for HDF5 data loader."
        
        # Create HDF5 config
        hdf5_config = HDF5DatasetConfig(
            image_key=self.image_key,
            wrist_image_key=self.wrist_image_key,
            state_key=self.state_key,
            action_key=self.action_key,
            default_prompt=self.default_prompt,
            prompt_key=self.prompt_key,
            image_hwc=self.image_hwc,
            demo_group_pattern=self.demo_group_pattern,
        )
        
        # Capture config values for closure
        hdf5_file = self.hdf5_file
        preload = self.preload
        prefetch_demos = self.prefetch_demos
        
        # Create a dataset class that passes config and memory options
        class ConfiguredHDF5Dataset(RepeatableHDF5Dataset):
            def __init__(self, data_dir: str, batch_size: int, *, shuffle: bool = True, action_chunk_size: int = 16, **kwargs):
                # data_dir is ignored, we use hdf5_file from closure
                super().__init__(
                    data_dir=hdf5_file,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    action_chunk_size=action_chunk_size,
                    config=hdf5_config,
                    preload=preload,
                    prefetch_demos=prefetch_demos,
                    **kwargs,
                )
        
        return DataConfig(
            repo_id=self.repo_id if self.repo_id is not tyro.MISSING else None,
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            rlds_data_dir=self.hdf5_file,
            dataset_class=ConfiguredHDF5Dataset,
        )


def get_sim_improvement_configs():
    """Return sim-improvement training configs."""
    # Import here to avoid circular imports.
    from openpi.training.config import TrainConfig, FakeDataConfig, AssetsConfig
    
    return [
        TrainConfig(
            name="pi05_cubestack_test",
            model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
            data=HDF5DataConfig(
                repo_id="cubestack_small",
                hdf5_file="s3://tri-ml-datasets-uw2/arhanjain/rollout_datasets/cubestack_small.hdf5",

                assets_dir="./assets/pi05_cubestack_test",
                asset_id="cubestack_small",

                image_key="obs/vision/external_camera",
                wrist_image_key="obs/vision/wrist_camera",
                state_key="obs/vision/joint_pos",
                action_key="action",
                demo_group_pattern="data/demo_*",
                preload=True,
            ),
            batch_size=8,
            num_train_steps=200,
            save_interval=100,
            keep_period=100,
            save_train_state=False,
            overwrite=True,
            exp_name="debug_pi05",
            wandb_enabled=True,
            remote_checkpoint_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi",
        ),

        TrainConfig(
            name="pi05_cubestack",
            model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
            data=HDF5DataConfig(
                repo_id="cubestack_1k",
                hdf5_file="s3://tri-ml-datasets-uw2/arhanjain/rollout_datasets/cubestack_1k.hdf5",
                # hdf5_file="/home/arhanjain/projects/UWLab/rollout_dataset/cubestack_1k.hdf5",

                assets_dir="./assets/pi05_cubestack",
                asset_id="cubestack_1k",

                default_prompt="Stack the green cube on the blue cube",
                image_key="obs/vision/external_camera",
                wrist_image_key="obs/vision/wrist_camera",
                state_key="obs/vision/joint_pos",
                action_key="action",
                demo_group_pattern="data/demo_*",
                preload=True,
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_droid_jointpos/params",
            ),
            batch_size=256,
            num_train_steps=50_000,
            log_interval=100,
            save_interval=5000,
            keep_period=1000,
            num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
            save_train_state=False,

            overwrite=True,
            exp_name="v1",
            wandb_enabled=True,
            remote_checkpoint_dir="s3://tri-ml-datasets-uw2/arhanjain/openpi",
        ),
    ]

