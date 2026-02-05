"""
MosaicML Streaming (MDS) data loader for VLA training.

This loader supports streaming from S3 or local MDS datasets with:
- Smart local caching for S3 data
- Deterministic shuffling
- Elastic resume (exact sample position)
- Automatic shard handling

See: https://docs.mosaicml.com/projects/streaming/en/stable/
"""

import dataclasses
import logging
from enum import Enum, auto
from typing import Any

import numpy as np

import openpi.shared.download as _download

from PIL import Image
from io import BytesIO
from streaming.base.format.mds.encodings import Encoding, _encodings


class ActionSpace(Enum):
    """Action space for MDS dataset."""
    JOINT_POSITION = auto()


class JPEG95(Encoding):
    """Store PIL image as JPEG with quality 95."""

    def encode(self, obj: Any) -> bytes:
        buf = BytesIO()
        obj.save(buf, format="JPEG", quality=95)
        return buf.getvalue()

    def decode(self, data: bytes) -> Any:
        return Image.open(BytesIO(data))


_encodings["jpeg95"] = JPEG95



@dataclasses.dataclass
class MDSDatasetConfig:
    """Configuration for MDS dataset structure.

    Defines the column names within the MDS dataset.
    """
    # Image column names
    external_cam_key: str = "external_cam"
    wrist_cam_key: str = "wrist_cam"

    # State/proprioception column names
    state_key: str | None = "state"
    arm_joint_pos_key: str | None = None
    gripper_joint_pos_key: str | None = None

    # Action column name
    action_key: str = "action_chunk"

    # Prompt/language instruction
    prompt_key: str | None = "prompt"
    default_prompt: str | None = None

    # Action space configuration
    action_space: ActionSpace = ActionSpace.JOINT_POSITION


class MDSDataset:
    """MDS dataset loader for robot demonstration data.

    Streams data from MDS format (local or S3) with smart caching.
    Implements an iterator interface compatible with openpi's data loading pipeline.

    Features:
    - Deterministic shuffle (reproducible across runs)
    - S3 streaming with intelligent local caching
    - Elastic resume (restart from exact sample)
    """

    def __init__(
        self,
        remote: str,
        batch_size: int,
        *,
        local: str = "/tmp/mds_cache",
        shuffle: bool = True,
        action_chunk_size: int = 16,
        config: MDSDatasetConfig | None = None,
        predownload: int | None = None,
        cache_limit: str | None = None,
        num_canonical_nodes: int | None = None,
        drop_last: bool = True,
    ):
        """Initialize the MDS dataset.

        Args:
            remote: Path to MDS dataset (local dir or s3://bucket/prefix).
            batch_size: Number of samples per batch.
            local: Local cache directory for downloaded shards.
            shuffle: Whether to shuffle samples (deterministically).
            action_chunk_size: Expected action chunk size (for validation).
            config: Configuration for MDS structure. If None, uses defaults.
            predownload: Number of samples to predownload. None for auto.
            cache_limit: Max cache size (e.g., "10gb"). None for unlimited.
            num_canonical_nodes: Number of canonical nodes for distributed training.
            drop_last: If True (default), drop the last incomplete batch.
        """
        # Import streaming here to not make it mandatory
        try:
            from streaming import StreamingDataset
        except ImportError:
            raise ImportError(
                "MosaicML Streaming is required for MDS datasets. "
                "Install with: pip install mosaicml-streaming"
            )

        self.remote = remote
        self.local = local
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.action_chunk_size = action_chunk_size
        self.config = config or MDSDatasetConfig()
        self.drop_last = drop_last

        logging.info(f"Loading MDS dataset from: {remote}")
        logging.info(f"Local cache: {local}")

        # Create streaming dataset
        streaming_kwargs = {
            "remote": remote,
            "local": local,
            "shuffle": shuffle,
            "batch_size": batch_size,
        }

        if predownload is not None:
            streaming_kwargs["predownload"] = predownload
        if cache_limit is not None:
            streaming_kwargs["cache_limit"] = cache_limit
        if num_canonical_nodes is not None:
            streaming_kwargs["num_canonical_nodes"] = num_canonical_nodes

        self._streaming_dataset = StreamingDataset(**streaming_kwargs)
        self._total_samples = len(self._streaming_dataset)

        logging.info(f"MDS dataset loaded with {self._total_samples} samples")

    def _process_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Process a single sample from the streaming dataset."""
        config = self.config

        # Load images (MDS stores as PNG bytes, StreamingDataset auto-decodes to numpy)
        external_image = sample[config.external_cam_key]
        wrist_image = sample.get(config.wrist_cam_key)

        # Ensure images are numpy arrays with correct shape (H, W, C)
        if isinstance(external_image, bytes):
            # Fallback if not auto-decoded
            import io
            from PIL import Image
            external_image = np.array(Image.open(io.BytesIO(external_image)))

        if wrist_image is not None and isinstance(wrist_image, bytes):
            import io
            from PIL import Image
            wrist_image = np.array(Image.open(io.BytesIO(wrist_image)))

        # Load actions and truncate to action_chunk_size
        actions = sample[config.action_key]
        if isinstance(actions, np.ndarray):
            actions = actions.astype(np.float32)
        else:
            actions = np.array(actions, dtype=np.float32)

        # Truncate to model's action horizon if needed
        if actions.shape[0] > self.action_chunk_size:
            actions = actions[:self.action_chunk_size]

        # # Load state / joint positions
        # arm_jp = None
        # gripper_jp = None
        # state = None

        # if config.arm_joint_pos_key and config.arm_joint_pos_key in sample:
        #     arm_jp = np.array(sample[config.arm_joint_pos_key], dtype=np.float32)
        # if config.gripper_joint_pos_key and config.gripper_joint_pos_key in sample:
        #     gripper_jp = np.array(sample[config.gripper_joint_pos_key], dtype=np.float32)

        # if config.state_key and config.state_key in sample:
        #     state = np.array(sample[config.state_key], dtype=np.float32)
        #     # If we don't have explicit arm/gripper keys, split from state
        #     if arm_jp is None and gripper_jp is None and state is not None:
        #         # Assume state is [arm_joints..., gripper]
        #         arm_jp = state[:-1]
        #         gripper_jp = state[-1:]

        arm_jp = sample["obs.vision.arm_joint_pos"]
        gripper_jp = sample["obs.vision.gripper_pos"]

        # Get prompt
        prompt = config.default_prompt or ""
        if config.prompt_key and config.prompt_key in sample:
            prompt_val = sample[config.prompt_key]
            if isinstance(prompt_val, bytes):
                prompt = prompt_val.decode("utf-8")
            else:
                prompt = str(prompt_val)

        result = {
            "actions": actions,
            "observation": {
                "image": external_image,
                "wrist_image": wrist_image,
                "arm_jp": arm_jp,
                "gripper_jp": gripper_jp,
            },
            "prompt": prompt,
        }

        return result

    def _generate_samples(self):
        """Generator that yields individual samples from the dataset."""
        for sample in self._streaming_dataset:
            yield self._process_sample(sample)

    def __iter__(self):
        """Iterate over batches of samples."""
        batch = []
        for sample in self._generate_samples():
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
        if "wrist_image" in samples[0]["observation"] and samples[0]["observation"]["wrist_image"] is not None:
            batch["observation"]["wrist_image"] = np.stack(
                [s["observation"]["wrist_image"] for s in samples], axis=0
            )

        if "arm_jp" in samples[0]["observation"]:
            batch["observation"]["arm_jp"] = np.stack(
                [s["observation"]["arm_jp"] for s in samples], axis=0
            )

        if "gripper_jp" in samples[0]["observation"]:
            batch["observation"]["gripper_jp"] = np.stack(
                [s["observation"]["gripper_jp"] for s in samples], axis=0
            )

        return batch

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._total_samples


class RepeatableMDSDataset(MDSDataset):
    """MDS dataset that repeats indefinitely, suitable for training."""

    def __iter__(self):
        """Iterate over batches, repeating indefinitely."""
        while True:
            yield from super().__iter__()

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self._total_samples
