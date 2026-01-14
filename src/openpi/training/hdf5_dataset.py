"""
HDF5-based data loader for robot demonstration datasets.
This loader supports common HDF5 formats used in robotics (e.g., robomimic-style datasets).
Provides a streaming interface compatible with openpi's data loading pipeline.

Features:
- Supports preloading all data into RAM to avoid disk I/O bottlenecks
- Background prefetching for memory-constrained scenarios
- Configurable HDF5 structure via HDF5DatasetConfig
"""

import dataclasses
import glob
import logging
import os
import queue
import threading
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import openpi.shared.download as _download

import h5py
import numpy as np
from tqdm import tqdm


def _load_demo_worker(args: tuple[str | Path, str, "HDF5DatasetConfig", int]) -> tuple[int, dict[str, Any] | None]:
    """Worker function for parallel demo loading. Must be at module level for pickling."""
    hdf5_file, demo_key, config, demo_idx = args
    try:
        with h5py.File(hdf5_file, "r") as f:
            demo_item = f[demo_key]
            if not isinstance(demo_item, h5py.Group):
                return demo_idx, None
            demo = demo_item
            
            # Load actions
            actions = np.array(demo[config.action_key])
            
            # Load main image
            image = np.array(demo[config.image_key])
            if not config.image_hwc:
                image = np.transpose(image, (0, 2, 3, 1))
            
            # Load wrist image if available
            wrist_image = None
            if config.wrist_image_key and config.wrist_image_key in demo.keys():
                wrist_image = np.array(demo[config.wrist_image_key])
                if not config.image_hwc:
                    wrist_image = np.transpose(wrist_image, (0, 2, 3, 1))
            
            # Load state
            state_keys = config.state_key.split(",")
            state_parts = []
            for key in state_keys:
                key = key.strip()
                if key in demo.keys():
                    state_parts.append(np.array(demo[key]))
            state = np.concatenate(state_parts, axis=-1) if state_parts else None
            
            # Compute trajectory length
            traj_len = min(len(actions), len(image))
            if wrist_image is not None:
                traj_len = min(traj_len, len(wrist_image))
            if state is not None:
                traj_len = min(traj_len, len(state))
            
            # Get prompt
            prompt = config.default_prompt
            if config.prompt_key and config.prompt_key in demo.keys():
                prompt_dataset = demo[config.prompt_key]
                if isinstance(prompt_dataset, h5py.Dataset):
                    prompt_data = prompt_dataset[()]
                    if isinstance(prompt_data, bytes):
                        prompt = prompt_data.decode("utf-8")
                    else:
                        prompt = str(prompt_data)
            elif "language_instruction" in demo.attrs:
                prompt_attr = demo.attrs["language_instruction"]
                if isinstance(prompt_attr, bytes):
                    prompt = prompt_attr.decode("utf-8")
                else:
                    prompt = str(prompt_attr)
            
            if prompt is None:
                prompt = ""
            
            return demo_idx, {
                "actions": actions,
                "observation": {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": state,
                },
                "prompt": prompt,
                "traj_len": traj_len,
            }
    except Exception as e:
        logging.warning(f"Error loading demo {demo_key}: {e}")
        return demo_idx, None


@dataclasses.dataclass
class HDF5DatasetConfig:
    """Configuration for HDF5 dataset structure.
    
    Defines the paths within the HDF5 file to extract observations, actions, and prompts.
    """
    # Path patterns within HDF5 file (relative to each demo group)
    image_key: str = "obs/agentview_image"
    wrist_image_key: str | None = "obs/eye_in_hand_image"
    state_key: str = "obs/robot0_eef_pos"  # Can be comma-separated for multiple keys to concat
    action_key: str = "actions"
    
    # Language instruction source
    # If None, will look for 'language_instruction' attribute on demo group
    # If string, will use as default prompt for all demos
    default_prompt: str | None = None
    prompt_key: str | None = None  # HDF5 key for prompt, if stored in data
    
    # Image preprocessing
    image_hwc: bool = True  # If True, images are stored as (H, W, C), else (C, H, W)
    
    # Demo group pattern (glob pattern for finding demo groups in HDF5)
    demo_group_pattern: str = "data/demo_*"


class HDF5Dataset:
    """HDF5 dataset loader for robot demonstration data.
    
    Supports loading from multiple HDF5 files, with configurable observation/action keys.
    Implements an iterator interface compatible with openpi's RLDS data loading pipeline.
    
    Memory modes:
    - preload=True: Load all data into RAM at init (fastest, highest memory)
    - preload=False, prefetch_demos>0: Background prefetching (balanced)
    - preload=False, prefetch_demos=0: Load on demand (slowest, lowest memory)
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,
        shuffle_buffer_size: int = 10_000,
        config: HDF5DatasetConfig | None = None,
        file_pattern: str = "*.hdf5",
        num_workers: int = 0,  # Unused, for API compatibility
        preload: bool = True,  # Load all data into RAM at initialization
        prefetch_demos: int = 2,  # Number of demos to prefetch if not preloading
        preload_workers: int = 8,  # Number of parallel workers for preloading
        drop_last: bool = True,  # Drop incomplete final batch (required for distributed training)
    ):
        """Initialize the HDF5 dataset.
        
        Args:
            data_dir: Path to the HDF5 file.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle samples.
            action_chunk_size: Number of future actions to include per sample.
            shuffle_buffer_size: Size of shuffle buffer for streaming.
            config: Configuration for HDF5 structure. If None, uses defaults.
            file_pattern: Unused, kept for API compatibility.
            num_workers: Unused, kept for API compatibility.
            preload: If True, load all data into RAM at initialization. Fastest but uses most memory.
            prefetch_demos: Number of demos to prefetch in background thread if preload=False.
            preload_workers: Number of parallel workers for preloading data (default 8).
            drop_last: If True (default), drop the last incomplete batch. Required for distributed training.
        """
        self.hdf5_file = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.action_chunk_size = action_chunk_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.config = config or HDF5DatasetConfig()
        self.preload = preload
        self.prefetch_demos = prefetch_demos
        self.preload_workers = preload_workers
        self.drop_last = drop_last
        
        # if self.hdf5_file.startswith("s3://"):
            # 
        self.hdf5_file = _download.maybe_download(self.hdf5_file)

        if not os.path.isfile(self.hdf5_file):
            raise ValueError(f"HDF5 file not found: {self.hdf5_file}")
        
        logging.info(f"Loading HDF5 file: {self.hdf5_file}")
        
        # Index all demos
        self._demo_index: list[str] = []  # demo_key
        self._total_steps = 0
        self._index_demos()
        
        logging.info(f"Indexed {len(self._demo_index)} demos with {self._total_steps} total steps")
        
        # Preload data into memory if requested
        self._preloaded_data: dict[int, dict[str, Any]] | None = None
        if self.preload:
            self._preload_all_data()
        
        # Prefetch state (for non-preload mode)
        self._prefetch_queue: queue.Queue | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._stop_prefetch = threading.Event()
    
    def _index_demos(self):
        """Index all demos in the HDF5 file."""
        try:
            with h5py.File(self.hdf5_file, "r") as f:
                demo_keys = self._find_demo_keys(f)
                for demo_key in demo_keys:
                    self._demo_index.append(demo_key)
                    # Count steps - handle nested keys like "obs/actions"
                    demo_group = f[demo_key]
                    if isinstance(demo_group, h5py.Group):
                        try:
                            # Navigate to the action dataset (handles nested paths)
                            action_data = demo_group[self.config.action_key]
                            if isinstance(action_data, h5py.Dataset):
                                self._total_steps += len(action_data)
                        except KeyError:
                            # Key not found, will be handled during loading
                            pass
        except Exception as e:
            logging.warning(f"Error indexing {self.hdf5_file}: {e}")
    
    def _preload_all_data(self):
        """Preload all demo data into RAM for fast access using parallel workers."""
        num_demos = len(self._demo_index)
        logging.info(f"Preloading {num_demos} demos into RAM using {self.preload_workers} workers...")
        self._preloaded_data = {}
        
        # Prepare arguments for worker processes
        worker_args = [
            (self.hdf5_file, demo_key, self.config, demo_idx)
            for demo_idx, demo_key in enumerate(self._demo_index)
        ]
        
        try:
            if self.preload_workers > 1:
                # Use multiprocessing for parallel loading
                with ProcessPoolExecutor(max_workers=self.preload_workers) as executor:
                    futures = [executor.submit(_load_demo_worker, args) for args in worker_args]
                    
                    with tqdm(total=num_demos, desc=f"Loading {os.path.basename(self.hdf5_file)}", unit="demo") as pbar:
                        for future in as_completed(futures):
                            demo_idx, demo_data = future.result()
                            if demo_data is not None:
                                self._preloaded_data[demo_idx] = demo_data
                            pbar.update(1)
            else:
                # Single-threaded fallback
                with h5py.File(self.hdf5_file, "r") as f:
                    for demo_idx, demo_key in enumerate(tqdm(self._demo_index, desc=f"Loading {os.path.basename(self.hdf5_file)}", unit="demo")):
                        demo_data = self._load_demo_from_file_handle(f, demo_key)
                        if demo_data is not None:
                            self._preloaded_data[demo_idx] = demo_data
        except Exception as e:
            logging.warning(f"Error loading demos from {self.hdf5_file}: {e}")
        
        # Recalculate total steps from preloaded data (more accurate than indexing)
        self._total_steps = sum(demo_data["traj_len"] for demo_data in self._preloaded_data.values())
        
        # Estimate memory usage
        total_bytes = 0
        for demo_data in self._preloaded_data.values():
            total_bytes += demo_data["actions"].nbytes
            total_bytes += demo_data["observation"]["image"].nbytes
            if demo_data["observation"]["wrist_image"] is not None:
                total_bytes += demo_data["observation"]["wrist_image"].nbytes
            if demo_data["observation"]["state"] is not None:
                total_bytes += demo_data["observation"]["state"].nbytes
        
        print(f"Preloaded {len(self._preloaded_data)} demos, {self._total_steps} steps ({total_bytes / 1e9:.2f} GB)")
    
    def _start_prefetch_thread(self, demo_indices: list[int]):
        """Start background thread to prefetch demos."""
        self._stop_prefetch.clear()
        prefetch_queue: queue.Queue[tuple[int, dict[str, Any]]] = queue.Queue(maxsize=self.prefetch_demos)
        self._prefetch_queue = prefetch_queue
        
        def prefetch_worker():
            for demo_idx in demo_indices:
                if self._stop_prefetch.is_set():
                    break
                demo_key = self._demo_index[demo_idx]
                demo_data = self._load_demo_from_disk(demo_key)
                if demo_data is not None:
                    try:
                        prefetch_queue.put((demo_idx, demo_data), timeout=1.0)
                    except queue.Full:
                        if self._stop_prefetch.is_set():
                            break
                        # Queue full, wait and retry
                        prefetch_queue.put((demo_idx, demo_data))
        
        self._prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self._prefetch_thread.start()
    
    def _stop_prefetch_thread(self):
        """Stop the prefetch thread."""
        if self._prefetch_thread is not None:
            self._stop_prefetch.set()
            self._prefetch_thread.join(timeout=2.0)
            self._prefetch_thread = None
            self._prefetch_queue = None
    
    def _get_demo(self, demo_idx: int) -> dict[str, Any] | None:
        """Get demo data, from preloaded cache or disk."""
        if self._preloaded_data is not None:
            return self._preloaded_data.get(demo_idx)
        
        demo_key = self._demo_index[demo_idx]
        return self._load_demo_from_disk(demo_key)
    
    def _find_demo_keys(self, f: h5py.File) -> list[str]:
        """Find all demo group keys in an HDF5 file."""
        demo_keys: list[str] = []
        
        # Parse the pattern to find demos
        pattern = self.config.demo_group_pattern
        parts = pattern.split("/")
        
        def search_group(group: h5py.Group | h5py.File, path_parts: list[str], current_path: str = ""):
            if not path_parts:
                demo_keys.append(current_path.lstrip("/"))
                return
            
            part = path_parts[0]
            remaining = path_parts[1:]
            
            if "*" in part:
                # Glob pattern - match against group keys
                import fnmatch
                for key in group.keys():
                    if fnmatch.fnmatch(key, part):
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            search_group(item, remaining, f"{current_path}/{key}")
            else:
                # Exact match
                if part in group.keys():
                    item = group[part]
                    if isinstance(item, h5py.Group):
                        search_group(item, remaining, f"{current_path}/{part}")
        
        search_group(f, parts)
        return demo_keys
    
    def _load_demo_from_disk(self, demo_key: str) -> dict[str, Any] | None:
        """Load a single demo from HDF5 file (opens and closes file)."""
        try:
            with h5py.File(self.hdf5_file, "r") as f:
                return self._load_demo_from_file_handle(f, demo_key)
        except Exception as e:
            logging.warning(f"Error loading demo {demo_key} from {self.hdf5_file}: {e}")
            return None
    
    def _load_demo_from_file_handle(self, f: h5py.File, demo_key: str) -> dict[str, Any] | None:
        """Load a single demo from an already-open HDF5 file handle."""
        try:
            demo_item = f[demo_key]
            if not isinstance(demo_item, h5py.Group):
                logging.warning(f"Demo {demo_key} is not a group")
                return None
            demo = demo_item
            
            # Load actions
            actions = np.array(demo[self.config.action_key])
            
            # Load main image
            image = np.array(demo[self.config.image_key])
            if not self.config.image_hwc:
                # Convert from (T, C, H, W) to (T, H, W, C)
                image = np.transpose(image, (0, 2, 3, 1))
            
            # Load wrist image if available
            wrist_image = None
            if self.config.wrist_image_key and self.config.wrist_image_key in demo.keys():
                wrist_image = np.array(demo[self.config.wrist_image_key])
                if not self.config.image_hwc:
                    wrist_image = np.transpose(wrist_image, (0, 2, 3, 1))
            
            # Load state (potentially concatenating multiple keys)
            state_keys = self.config.state_key.split(",")
            state_parts = []
            for key in state_keys:
                key = key.strip()
                if key in demo.keys():
                    state_parts.append(np.array(demo[key]))
            
            state = np.concatenate(state_parts, axis=-1) if state_parts else None
            
            # Use minimum length across all arrays to avoid index errors
            traj_len = min(len(actions), len(image))
            if wrist_image is not None:
                traj_len = min(traj_len, len(wrist_image))
            if state is not None:
                traj_len = min(traj_len, len(state))
            
            # Get prompt
            prompt = self.config.default_prompt
            if self.config.prompt_key and self.config.prompt_key in demo.keys():
                prompt_dataset = demo[self.config.prompt_key]
                if isinstance(prompt_dataset, h5py.Dataset):
                    prompt_data = prompt_dataset[()]
                    if isinstance(prompt_data, bytes):
                        prompt = prompt_data.decode("utf-8")
                    else:
                        prompt = str(prompt_data)
            elif "language_instruction" in demo.attrs:
                prompt_attr = demo.attrs["language_instruction"]
                if isinstance(prompt_attr, bytes):
                    prompt = prompt_attr.decode("utf-8")
                else:
                    prompt = str(prompt_attr)
            
            if prompt is None:
                prompt = ""
            
            return {
                "actions": actions,
                "observation": {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": state,
                },
                "prompt": prompt,
                "traj_len": traj_len,
            }
        except Exception as e:
            logging.warning(f"Error loading demo {demo_key}: {e}")
            return None
    
    def _chunk_actions(self, actions: np.ndarray, traj_len: int) -> np.ndarray:
        """Create action chunks for each timestep."""
        # For each timestep, get the next action_chunk_size actions
        action_chunks = []
        for t in range(traj_len):
            # Get indices for action chunk
            indices = np.arange(t, min(t + self.action_chunk_size, traj_len))
            # Pad with last action if needed
            if len(indices) < self.action_chunk_size:
                pad_len = self.action_chunk_size - len(indices)
                indices = np.concatenate([indices, np.full(pad_len, traj_len - 1)])
            chunk = actions[indices]
            action_chunks.append(chunk)
        return np.stack(action_chunks, axis=0)
    
    def _generate_samples(self):
        """Generator that yields individual samples from the dataset."""
        # Shuffle demo order if needed
        demo_indices = list(range(len(self._demo_index)))
        if self.shuffle:
            np.random.shuffle(demo_indices)
        
        # Use prefetching if not preloaded and prefetch_demos > 0
        use_prefetch = (self._preloaded_data is None and self.prefetch_demos > 0)
        
        if use_prefetch:
            self._start_prefetch_thread(demo_indices)
            try:
                yield from self._generate_samples_with_prefetch(demo_indices)
            finally:
                self._stop_prefetch_thread()
        else:
            yield from self._generate_samples_direct(demo_indices)
    
    def _generate_samples_direct(self, demo_indices: list[int]):
        """Generate samples by directly accessing demos (preloaded or on-demand)."""
        for demo_idx in demo_indices:
            demo_data = self._get_demo(demo_idx)
            
            if demo_data is None:
                continue
            
            yield from self._yield_samples_from_demo(demo_data)
    
    def _generate_samples_with_prefetch(self, demo_indices: list[int]):
        """Generate samples using prefetched demos from background thread."""
        demos_yielded = 0
        total_demos = len(demo_indices)
        prefetch_queue = self._prefetch_queue
        
        if prefetch_queue is None:
            return
        
        while demos_yielded < total_demos:
            try:
                demo_idx, demo_data = prefetch_queue.get(timeout=5.0)
                demos_yielded += 1
                yield from self._yield_samples_from_demo(demo_data)
            except queue.Empty:
                # Timeout - check if thread is still alive
                if self._prefetch_thread is None or not self._prefetch_thread.is_alive():
                    break
    
    def _yield_samples_from_demo(self, demo_data: dict[str, Any]):
        """Yield all samples from a single demo."""
        traj_len = demo_data["traj_len"]
        actions = self._chunk_actions(demo_data["actions"], traj_len)
        
        # Generate samples for each timestep
        timesteps = list(range(traj_len))
        if self.shuffle:
            np.random.shuffle(timesteps)
        
        for t in timesteps:
            sample = {
                "actions": actions[t],
                "observation": {
                    "image": demo_data["observation"]["image"][t],
                    "wrist_image": demo_data["observation"]["wrist_image"][t],
                    "state": demo_data["observation"]["state"][t],
                },
                "prompt": demo_data["prompt"],
            }
            
            yield sample
    
    def __iter__(self):
        """Iterate over batches of samples."""
        sample_generator = self._generate_samples()
        
        if self.shuffle:
            # Use a shuffle buffer
            buffer = []
            
            # Fill initial buffer
            for sample in sample_generator:
                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer_size:
                    break
            
            # Yield batches while refilling buffer
            batch = []
            for sample in sample_generator:
                # Pick random sample from buffer
                idx = np.random.randint(len(buffer))
                batch.append(buffer[idx])
                buffer[idx] = sample
                
                if len(batch) >= self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []
            
            # Drain remaining buffer
            np.random.shuffle(buffer)
            for sample in buffer:
                batch.append(sample)
                if len(batch) >= self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []
            
            # Yield final partial batch if any (unless drop_last=True)
            if batch and not self.drop_last:
                yield self._collate_batch(batch)
        else:
            # No shuffling - simple batching
            batch = []
            for sample in sample_generator:
                batch.append(sample)
                if len(batch) >= self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []
            
            # Yield final partial batch if any (unless drop_last=True)
            if batch and not self.drop_last:
                yield self._collate_batch(batch)
    
    def _collate_batch(self, samples: list[dict]) -> dict:
        """Collate a list of samples into a batch."""
        batch = {
            "actions": np.stack([s["actions"] for s in samples], axis=0),
            "observation": {
                "image": np.stack([s["observation"]["image"] for s in samples], axis=0),
            },
            "prompt": np.array([s["prompt"] for s in samples]),
        }
        
        # Add optional keys if present
        if "wrist_image" in samples[0]["observation"]:
            batch["observation"]["wrist_image"] = np.stack(
                [s["observation"]["wrist_image"] for s in samples], axis=0
            )
        
        if "state" in samples[0]["observation"]:
            batch["observation"]["state"] = np.stack(
                [s["observation"]["state"] for s in samples], axis=0
            )
        
        return batch
    
    def __len__(self) -> int:
        """Return the total number of samples (steps) in the dataset."""
        return self._total_steps


class RepeatableHDF5Dataset(HDF5Dataset):
    """HDF5 dataset that repeats indefinitely, suitable for training."""
    
    def __iter__(self):
        """Iterate over batches, repeating indefinitely."""
        while True:
            yield from super().__iter__()
    
    def __len__(self) -> int:
        """Return the total number of samples (steps) in the dataset."""
        return self._total_steps


# Convenience function to create dataset with common robotics formats
def create_robomimic_dataset(
    data_dir: str,
    batch_size: int,
    *,
    shuffle: bool = True,
    action_chunk_size: int = 16,
    default_prompt: str = "",
    preload: bool = True,
    prefetch_demos: int = 50,
    **kwargs,
) -> HDF5Dataset:
    """Create an HDF5 dataset configured for robomimic-style HDF5 files.
    
    Args:
        data_dir: Directory containing HDF5 files.
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        action_chunk_size: Action horizon.
        default_prompt: Default language instruction.
        preload: If True, load all data into RAM (fastest, ~10-100x speedup).
        prefetch_demos: Number of demos to prefetch if not preloading.
    """
    config = HDF5DatasetConfig(
        image_key="obs/agentview_image",
        wrist_image_key="obs/eye_in_hand_image",
        state_key="obs/robot0_eef_pos,obs/robot0_eef_quat,obs/robot0_gripper_qpos",
        action_key="actions",
        default_prompt=default_prompt,
        image_hwc=True,
        demo_group_pattern="data/demo_*",
    )
    return RepeatableHDF5Dataset(
        data_dir=data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_chunk_size,
        config=config,
        preload=preload,
        prefetch_demos=prefetch_demos,
        **kwargs,
    )

