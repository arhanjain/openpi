"""
Zarr-based data loader for UWLab-style robot demonstration datasets.

Expects a directory containing one or more *.zarr stores, each structured as:
    data/
        actions          (N, action_dim)   float32
        obs/
            front_rgb    (N, H, W, 3)      uint8
            side_rgb     (N, H, W, 3)      uint8
            wrist_rgb    (N, H, W, 3)      uint8
            arm_joint_pos          (N, 6)  float32
            end_effector_pose      (N, 6)  float32
            last_arm_action        (N, 6)  float32
            last_gripper_action    (N, 1)  float32
            ...
        dones            (N,)              float32
    meta/
        episode_ends     (num_episodes,)   int64

All episodes within a store are concatenated along the first axis;
`meta/episode_ends` gives the cumulative end index of each episode.
"""

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import openpi.shared.download as _download
except ImportError:
    _download = None


@dataclasses.dataclass
class ZarrDatasetConfig:
    # Mapping from output name -> zarr key for image observations.
    image_keys: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "image": "data/obs/front_rgb",
        "side_image": "data/obs/side_rgb",
        "wrist_image": "data/obs/wrist_rgb",
    })
    action_key: str = "data/actions"
    # Keys concatenated (in order) to form the state vector.
    state_keys: tuple[str, ...] = (
        "data/obs/end_effector_pose",
        "data/obs/arm_joint_pos",
        "data/obs/last_arm_action",
        "data/obs/last_gripper_action",
    )
    episode_ends_key: str = "meta/episode_ends"
    default_prompt: str = ""


class ZarrDataset:
    """Zarr dataset loader compatible with openpi's RLDS data-loading pipeline.

    Signature matches what `create_rlds_dataset` passes when `dataset_class` is set:
        __init__(data_dir, batch_size, *, shuffle, action_chunk_size)
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,
        shuffle_buffer_size: int = 10_000,
        config: ZarrDatasetConfig | None = None,
        preload: bool = True,
    ):
        import zarr

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.action_chunk_size = action_chunk_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.config = config or ZarrDatasetConfig()

        if _download is not None:
            data_dir = _download.maybe_download(data_dir)
        zarr_paths = sorted(Path(data_dir).glob("*.zarr"))
        if not zarr_paths:
            raise ValueError(f"No .zarr stores found in {data_dir}")

        logging.info(f"Found {len(zarr_paths)} zarr stores in {data_dir}")

        self._stores: list[zarr.Group] = []
        self._episodes: list[tuple[int, int, int]] = []  # (store_idx, start, end)
        self._total_steps = 0

        for store_path in zarr_paths:
            store_idx = len(self._stores)
            store = zarr.open(str(store_path), mode="r")
            self._stores.append(store)

            ep_ends: np.ndarray = store[self.config.episode_ends_key][:]
            prev = 0
            for end in ep_ends:
                end = int(end)
                if end > prev:
                    self._episodes.append((store_idx, prev, end))
                    self._total_steps += end - prev
                prev = end

            logging.info(
                f"  {store_path.name}: {len(ep_ends)} episodes, "
                f"{int(ep_ends[-1]) if len(ep_ends) else 0} steps"
            )

        logging.info(f"Total: {len(self._episodes)} episodes, {self._total_steps} steps")

        self._preloaded: list[dict[str, Any]] | None = None
        if preload:
            self._preload_all()

    def _load_episode_from_store(self, store, start: int, end: int) -> dict[str, Any]:
        cfg = self.config
        demo: dict[str, Any] = {
            "actions": np.array(store[cfg.action_key][start:end]),
            "prompt": cfg.default_prompt,
        }
        for name, key in cfg.image_keys.items():
            if key in store:
                demo[name] = np.array(store[key][start:end])
        state_parts = []
        for key in cfg.state_keys:
            if key in store:
                state_parts.append(np.array(store[key][start:end]))
        if state_parts:
            demo["state"] = np.concatenate(state_parts, axis=-1)
        return demo

    def _preload_all(self):
        logging.info("Preloading episodes into RAM ...")
        self._preloaded = []
        for store_idx, start, end in tqdm(self._episodes, desc="Preloading", unit="ep"):
            self._preloaded.append(
                self._load_episode_from_store(self._stores[store_idx], start, end)
            )
        total_bytes = sum(
            v.nbytes for d in self._preloaded for v in d.values() if isinstance(v, np.ndarray)
        )
        logging.info(f"Preloaded {len(self._preloaded)} episodes ({total_bytes / 1e9:.2f} GB)")

    def _load_episode(self, ep_idx: int) -> dict[str, Any]:
        if self._preloaded is not None:
            return self._preloaded[ep_idx]
        store_idx, start, end = self._episodes[ep_idx]
        return self._load_episode_from_store(self._stores[store_idx], start, end)

    def _chunk_actions(self, actions: np.ndarray) -> np.ndarray:
        T = len(actions)
        chunks = []
        for t in range(T):
            idxs = np.arange(t, min(t + self.action_chunk_size, T))
            if len(idxs) < self.action_chunk_size:
                idxs = np.concatenate([idxs, np.full(self.action_chunk_size - len(idxs), T - 1)])
            chunks.append(actions[idxs])
        return np.stack(chunks, axis=0)

    def _generate_samples(self):
        ep_indices = list(range(len(self._episodes)))
        if self.shuffle:
            np.random.shuffle(ep_indices)

        image_names = list(self.config.image_keys.keys())

        for ep_idx in ep_indices:
            demo = self._load_episode(ep_idx)
            traj_len = len(demo["actions"])
            action_chunks = self._chunk_actions(demo["actions"])

            timesteps = list(range(traj_len))
            if self.shuffle:
                np.random.shuffle(timesteps)

            for t in timesteps:
                obs: dict[str, Any] = {}
                for name in image_names:
                    if name in demo:
                        obs[name] = demo[name][t]
                if "state" in demo:
                    obs["state"] = demo["state"][t]

                yield {
                    "actions": action_chunks[t],
                    "observation": obs,
                    "prompt": demo["prompt"],
                }

    def _collate_batch(self, samples: list[dict]) -> dict:
        obs0 = samples[0]["observation"]
        batch_obs: dict[str, np.ndarray] = {}
        for key in obs0:
            batch_obs[key] = np.stack([s["observation"][key] for s in samples])
        return {
            "actions": np.stack([s["actions"] for s in samples]),
            "observation": batch_obs,
            "prompt": np.array([s["prompt"] for s in samples]),
        }

    def __iter__(self):
        gen = self._generate_samples()

        if self.shuffle:
            buf: list[dict] = []
            for sample in gen:
                buf.append(sample)
                if len(buf) >= self.shuffle_buffer_size:
                    break

            batch: list[dict] = []
            for sample in gen:
                idx = np.random.randint(len(buf))
                batch.append(buf[idx])
                buf[idx] = sample
                if len(batch) >= self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []

            np.random.shuffle(buf)
            for sample in buf:
                batch.append(sample)
                if len(batch) >= self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []
        else:
            batch = []
            for sample in gen:
                batch.append(sample)
                if len(batch) >= self.batch_size:
                    yield self._collate_batch(batch)
                    batch = []

    def __len__(self) -> int:
        return self._total_steps


class RepeatableZarrDataset(ZarrDataset):
    """Zarr dataset that repeats indefinitely."""

    def __iter__(self):
        while True:
            yield from super().__iter__()


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ds = ZarrDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        action_chunk_size=args.action_chunk_size,
        preload=not args.no_preload,
    )

    print(f"\nDataset length: {len(ds)} steps")
    for i, batch in enumerate(ds):
        print(f"\nBatch {i}:")
        for k, v in batch.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    print(f"  observation/{k2}: shape={v2.shape} dtype={v2.dtype}")
            elif isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            else:
                print(f"  {k}: {v[:2]}...")
        if i >= 2:
            break
