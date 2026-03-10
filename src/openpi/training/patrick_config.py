"""Patrick-specific training configs using Zarr datasets.

These configs mirror the sim_improvement HDF5/MDS configs but use
`ZarrDataset` and `URPolicy` to train pi0.5 policies directly from Zarr.
"""

import dataclasses
import pathlib

import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.ur_policy as ur_policy
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from openpi.training.zarr_dataset import ZarrDatasetConfig, RepeatableZarrDataset


@dataclasses.dataclass(frozen=True)
class ZarrDataConfig:
    """Config for training on Patrick-style Zarr datasets."""

    # Root directory containing one or more *.zarr stores.
    zarr_root: str = tyro.MISSING

    # Optional override for default Zarr layout.
    zarr_config: ZarrDatasetConfig = dataclasses.field(default_factory=ZarrDatasetConfig)

    # Optional default prompt if the dataset does not provide one.
    default_prompt: str | None = None
    # If true, preload all episodes into RAM. Disable for large datasets.
    preload: bool = False

    # Assets for normalization (repo/asset id are used by the main config).
    repo_id: str = tyro.MISSING
    assets_dir: str | None = None
    asset_id: str | None = None

    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig):
        """Create a DataConfig wired up to `RepeatableZarrDataset` + UR policy."""
        from etils import epath
        from openpi.training.config import DataConfig, ModelTransformFactory
        import openpi.shared.download as _download
        import openpi.shared.normalize as _normalize
        import logging

        # Load norm stats (same convention as other configs).
        norm_stats = None
        asset_id = self.asset_id or self.repo_id
        if asset_id:
            data_assets_dir = str(epath.Path(self.assets_dir or assets_dirs) / asset_id)
            try:
                norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
            except FileNotFoundError:
                logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")

        # Repack from ZarrDataset outputs to URPolicy inputs.
        #
        # ZarrDataset yields samples of the form:
        #   {
        #     "actions": (T, action_dim),
        #     "observation": {
        #       "image": (H, W, 3),
        #       "side_image": (H, W, 3),
        #       "wrist_image": (H, W, 3),
        #       "state": (state_dim,),
        #     },
        #     "prompt": str,
        #   }
        #
        # URInputs expects:
        #   observation/exterior_image_1_left, observation/exterior_image_2_left,
        #   observation/wrist_image_left, observation/state, plus actions / prompt.
        repack_transforms = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/exterior_image_2_left": "observation/side_image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/state": "observation/state",
                        "actions": "actions",
                        # "prompt": "prompt",
                    }
                )
            ]
        )

        # UR policy data transforms (no extra delta conversion; state is already a flat vector).
        data_transforms = _transforms.Group(
            inputs=[ur_policy.URInputs(model_type=model_config.model_type)],
            outputs=[ur_policy.UROutputs()],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        assert (
            self.zarr_root is not tyro.MISSING
        ), "Need to set zarr_root for Zarr data loader."

        # Capture values for closure so the dataset matches the RLDS interface.
        zarr_root = self.zarr_root
        zarr_cfg = self.zarr_config
        preload = self.preload

        class ConfiguredZarrDataset(RepeatableZarrDataset):
            def __init__(
                self,
                data_dir: str,
                batch_size: int,
                *,
                shuffle: bool = True,
                action_chunk_size: int = 16,
                **kwargs,
            ):
                # data_dir is ignored; we use zarr_root from the closure.
                super().__init__(
                    data_dir=zarr_root,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    action_chunk_size=action_chunk_size,
                    config=zarr_cfg,
                    preload=preload,
                    **kwargs,
                )

        return DataConfig(
            repo_id=self.repo_id if self.repo_id is not tyro.MISSING else None,
            asset_id=asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type != _model.ModelType.PI0,
            rlds_data_dir=self.zarr_root,
            dataset_class=ConfiguredZarrDataset,
        )

    # keep only `create()` so model transforms are always included.


def get_patrick_configs():
    """Return Patrick training configs using Zarr + UR policy."""
    from openpi.training.config import TrainConfig

    # Example config; adjust paths/prompts as needed.
    return [
        TrainConfig(
            name="pi05_ur_from_zarr",
            model=pi0_config.Pi0Config(action_horizon=16, pi05=True),
            data=ZarrDataConfig(
                repo_id="patrick_zarr",
                zarr_root="/gpfs/scrubbed/pyin/datasets/2_20_26_peg_10k_ddc",
                default_prompt="Pick up the peg and insert it into the peghole.",
                assets_dir="/gpfs/scrubbed/arhanj/openpi-patrick/assets/pi05_ur_from_zarr",
                asset_id="patrick_zarr",
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params",
            ),
            batch_size=256,
            num_train_steps=50_000,
            log_interval=100,
            save_interval=5000,
            keep_period=1000,
            save_train_state=False,
            overwrite=True,
            exp_name="patrick_zarr_pi05",
            wandb_enabled=True,
            remote_checkpoint_dir=None,
        ),
    ]

