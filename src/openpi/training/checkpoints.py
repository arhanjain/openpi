from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import os
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


class DualCheckpointManager:
    def __init__(self, mngs: dict[str, ocp.CheckpointManager]):
        self.mng_assignments = mngs
        self.mngs = set(mngs.values())

    @property
    def directory(self) -> epath.Path:
        """Return the root checkpoint directory (from the first manager)."""
        return list(self.mngs)[0].directory

    def wait_until_finished(self):
        for mng in self.mngs:
            mng.wait_until_finished()

    def save(self, step: int, items: dict) -> list[epath.Path]:
        """Save checkpoint and return list of checkpoint directories that were saved."""
        saved_dirs = []
        for mng in self.mngs:
            items_single = {k: v for k, v in items.items() if self.mng_assignments[k] == mng}
            if items_single:
                mng.save(step, items_single)
                saved_dirs.append(mng.directory / str(step))
        return saved_dirs
    
    def restore(self, step: int, items: dict):
        restored = {}
        for mng in self.mngs:
            # items = {k: v for k, v in items.items() if self.mng_assignments[k] == mng}
            single_items = {k: v for k, v in items.items() if self.mng_assignments[k] == mng}
            restored_single = mng.restore(step, single_items)
            restored.update(restored_single)
        return restored

    def all_steps(self, read: bool = False):
        return set(step for mng in self.mngs for step in mng.all_steps(read))


# Type alias for checkpoint managers (supports both simple and dual)
CheckpointManager = ocp.CheckpointManager | DualCheckpointManager


def initialize_simple_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    """Initialize a simple checkpoint manager that only saves params and assets (no train state)."""
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[DualCheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr1 = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=None,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Only keeps latest train state checkpoint.
    train_states_dir = checkpoint_dir / "train_states"
    train_states_dir.mkdir(parents=True, exist_ok=True)
    mngr2 = ocp.CheckpointManager(
        train_states_dir,
        item_handlers={
            "train_state": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=None,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    mngr = DualCheckpointManager(
        {
            "assets": mngr1,
            "params": mngr1,
            "train_state": mngr2,
        }
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state_simple(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
) -> epath.Path:
    """Save checkpoint (params and assets only) and return the checkpoint directory."""
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        _, params = _split_params(state)
    items = {
        "assets": save_assets,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)
    return checkpoint_manager.directory / str(step)


def save_state(
    checkpoint_manager: DualCheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    save_train_state: bool = True,
) -> list[epath.Path]:
    """Save checkpoint and return list of checkpoint directories that were saved."""
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "params": {"params": params},
    }
    if save_train_state:
        items["train_state"] = train_state
    return checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: DualCheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])

def remote_sync(local_dir: str, remote_dir: str) -> bool:
    """Synchronously sync local directory to remote (S3) directory using boto3."""
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError:
        logging.error("boto3 not installed. Install with: pip install boto3")
        return False
    
    logging.info(f"Starting remote sync: {local_dir} -> {remote_dir}")
    
    # Parse S3 URI
    if not remote_dir.startswith("s3://"):
        logging.error(f"remote_dir must be an S3 URI (s3://bucket/prefix), got: {remote_dir}")
        return False
    
    remote_dir = remote_dir.rstrip("/")
    parts = remote_dir[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    try:
        s3 = boto3.client("s3")
        uploaded = 0
        
        for root, dirs, files in os.walk(local_dir):
            for filename in files:
                local_file = os.path.join(root, filename)
                # Compute relative path from local_dir
                rel_path = os.path.relpath(local_file, local_dir)
                s3_key = f"{prefix}/{rel_path}" if prefix else rel_path
                
                try:
                    s3.upload_file(local_file, bucket, s3_key)
                    uploaded += 1
                except ClientError as e:
                    logging.error(f"Failed to upload {local_file}: {e}")
                    return False
        
        logging.info(f"Successfully synced {uploaded} files to s3://{bucket}/{prefix}")
        return True
        
    except Exception as e:
        logging.error(f"Error during S3 sync: {e}")
        return False


class AsyncS3Uploader:
    """Manages async uploads to S3 after checkpoints are saved."""
    
    def __init__(self, checkpoint_manager: CheckpointManager, remote_dir: str, max_workers: int = 1):
        """
        Args:
            checkpoint_manager: The checkpoint manager to wait on before uploading.
                               Supports both ocp.CheckpointManager and DualCheckpointManager.
            remote_dir: S3 destination (e.g., "s3://bucket/checkpoints/").
            max_workers: Number of concurrent upload threads (1 is usually fine).
        """
        self.checkpoint_manager = checkpoint_manager
        self.remote_dir = remote_dir.rstrip("/")
        self.executor = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pending: list[futures.Future] = []
        logging.info(f"S3 uploader configured for: {self.remote_dir}")
    
    def schedule_upload(self, local_dir: epath.Path | str):
        """Schedule an async upload after the checkpoint manager finishes saving."""
        local_dir = str(local_dir)
        
        def upload_task():
            # Wait for any pending checkpoint saves to complete
            self.checkpoint_manager.wait_until_finished()
            # Then sync to S3
            remote_sync(local_dir, f"{self.remote_dir}/{epath.Path(local_dir).name}")
        
        future = self.executor.submit(upload_task)
        self._pending.append(future)
        # Clean up completed futures
        self._pending = [f for f in self._pending if not f.done()]
    
    def schedule_full_sync(self):
        """Schedule a full sync of the checkpoint directory."""
        local_dir = str(self.checkpoint_manager.directory)
        
        def upload_task():
            self.checkpoint_manager.wait_until_finished()
            remote_sync(local_dir, self.remote_dir)
        
        future = self.executor.submit(upload_task)
        self._pending.append(future)
        self._pending = [f for f in self._pending if not f.done()]
    
    def wait(self):
        """Wait for all pending uploads to complete."""
        for future in self._pending:
            future.result()
        self._pending = []
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor. Call at end of training."""
        if wait:
            self.wait()
        self.executor.shutdown(wait=wait)