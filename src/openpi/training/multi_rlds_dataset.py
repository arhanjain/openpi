"""
RLDS-based data loader for DROID.
While openpi typically uses LeRobot's data loader, it is not currently scalable enough for larger datasets like DROID.
Thus, we provide a data loader example here that uses the RLDS data format.
The data loader also applies a few DROID-specific data filters / transformations.
"""

from enum import Enum
from enum import auto
from typing import List, Tuple
from openpi.training.droid_rlds_dataset import DroidRldsDataset
from pathlib import Path
import json
import logging
import tqdm
import openpi.shared.download as download


class DroidActionSpace(Enum):
    """Action space for DROID dataset."""

    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


class MultiRldsDataset(DroidRldsDataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        *,  # Force keyword-only arguments
        shuffle: bool = True,
        action_chunk_size: int = 16,
        # We default to joint position actions, since they allow policy evaluation in simulation.
        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,
        # Reduce this if you are running out of memory, but careful -- below ~100k shuffling is not sufficiently random.
        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        num_parallel_calls: int = -1,  # -1 == tf.data.AUTOTUNE -- hack to not import tf at top level
        datasets: List[Tuple[str, float]] = [],
        filter_dict_path: str = None,
    ):
        # Import tensorflow here to not make it mandatory in case RLDS data loader is not used.
        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds
        # Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
        tf.config.set_visible_devices([], "GPU")

        assert sum([weight for ds_name, weight in datasets]) == 1.0, "Dataset weights must add up to 1.0"

        def prepare_dataset_og(ds_name):
            builder = tfds.builder(ds_name, data_dir=data_dir)
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

            # Filter out any unsuccessful trajectories -- we use the file name to check this
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            def restructure(traj):
                """Reformat observation and action keys, sample language instruction."""
                # Important: we use joint *position* action space -- easier to simulate!
                actions = tf.concat(
                    (
                        (
                            traj["action_dict"]["joint_position"]
                        ),
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )
                # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
                # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_img = traj["observation"]["wrist_image_left"]
                # Randomly sample one of the three language instructions
                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]


                return {
                    "actions": actions,
                    "observation": {
                        "image": exterior_img,
                        "wrist_image": wrist_img,
                        "joint_position": traj["observation"]["joint_position"],
                        "gripper_position": traj["observation"]["gripper_position"],
                    },
                    "prompt": instruction,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """Splits episode into action chunks."""
                traj_len = tf.shape(traj["actions"])[0]

                # For each step in the trajectory, construct indices for the next n actions
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # Cap to length of the sequence --> final chunks will repeat the last action
                # This makes sense, since we are using absolute joint + gripper position actions
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # # Filter out frames where actions are idle. Must be done after flattening, as filter should apply per-frame.
            def filter_idle(traj):
                """Filter out chunks with idle actions.
                --> we filter if at least first half of chunk does not move.
                """
                if action_space == DroidActionSpace.JOINT_POSITION:
                    # Compute delta to first position in action chunk
                    return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

            dataset = dataset.filter(filter_idle)


            # Decode images: RLDS saves encoded images, only decode now for efficiency
            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            dataset = dataset.frame_map(decode_images, num_parallel_calls)
            # Shuffle
            dataset = dataset.shuffle(shuffle_buffer_size)

            return dataset


        def prepare_dataset(ds_name):
            builder = tfds.builder(ds_name, data_dir=data_dir)
            dataset = dl.DLataset.from_rlds(builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads)

            # kill the broken/empty trajectories
            dataset = dataset.filter(
                lambda traj: tf.shape(traj["action"])[0] > 0
            )

            # Filter out any unsuccessful trajectories -- we use the file name to check this
            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )

            # Repeat dataset so we never run out of data.
            # dataset = dataset.repeat()
            # Load the filter dictionary if provided.
            # The filter dictionary is a JSON file that maps episode keys to ranges of frames to sample
            # (e.g.,
            # {
            #     "<episode key>": [[0, 100], [200, 300]]
            # }
            # means keep frames 0-99 and 200-299).
            if ds_name == "droid":
                cached_filter_dict_path = download.maybe_download(filter_dict_path)
                with Path(cached_filter_dict_path).open("r") as f:
                    filter_dict = json.load(f)

                logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                keys_tensor = []
                values_tensor = []

                for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                    for start, end in ranges:
                        for t in range(start, end):
                            frame_key = f"{episode_key}--{t}"
                            keys_tensor.append(frame_key)
                            values_tensor.append(True)
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                )
                logging.info("Filter hash table initialized")
            else:
                def add_dummy_recording(traj):
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"] = tf.constant(
                        "[RECORDING_FOLDERPATH]", dtype=tf.string
                    )
                    return traj
                dataset = dataset.map(add_dummy_recording)
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            def restructure(traj):
                """Reformat observation and action keys, sample language instruction."""
                # Important: we use joint *position* action space -- easier to simulate!
                actions = tf.concat(
                    (
                        (
                            traj["action_dict"]["joint_position"]
                            # if action_space == DroidActionSpace.JOINT_POSITION
                            # else traj["action_dict"]["joint_velocity"]
                        ),
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )
                # Randomly samples one of the two exterior images in DROID during training (we only train with one at a time).
                # Note: the "left" refers to the left camera in the stereo pair, we only train on the left camera.
                exterior_img = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_img = traj["observation"]["wrist_image_left"]
                # Randomly sample one of the three language instructions
                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]

                traj_len = tf.shape(traj["action"])[0]
                indices = tf.as_string(tf.range(traj_len))
                # Data filtering:
                # Compute a uniquely-identifying step ID by concatenating the recording folderpath, file path,
                # and each step's time step index. This will index into the filter hash table, and if it returns true,
                # then the frame passes the filter.
                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + indices
                )
                passes_filter = self.filter_table.lookup(step_id)


                return {
                    "actions": actions,
                    "observation": {
                        "image": exterior_img,
                        "wrist_image": wrist_img,
                        "joint_position": traj["observation"]["joint_position"],
                        "gripper_position": traj["observation"]["gripper_position"],
                    },
                    "prompt": instruction,
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):
                """Splits episode into action chunks."""
                traj_len = tf.shape(traj["actions"])[0]

                # For each step in the trajectory, construct indices for the next n actions
                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )

                # Cap to length of the sequence --> final chunks will repeat the last action
                # This makes sense, since we are using absolute joint + gripper position actions
                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)

                # Gather the actions for each chunk
                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)

            # Flatten: map from trajectory dataset to dataset of individual action chunks
            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)

            # # Filter out frames where actions are idle. Must be done after flattening, as filter should apply per-frame.
            def filter_idle(traj):
                """Filter out chunks with idle actions.
                --> we filter if at least first half of chunk does not move.
                """
                if action_space == DroidActionSpace.JOINT_POSITION:
                    # Compute delta to first position in action chunk
                    return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2] - traj["actions"][:1]) > 1e-3)
                return tf.reduce_any(tf.abs(traj["actions"][: action_chunk_size // 2]) > 1e-3)

            # Filter data that doesn't pass the filter
            def filter_from_dict(frame):
                return frame["passes_filter"]
            # Remove "passes_filter" key from output
            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            if ds_name == "droid":
                dataset = dataset.filter(filter_from_dict)
                dataset = dataset.map(remove_passes_filter)
            else:
                dataset = dataset.filter(filter_idle)
                dataset = dataset.map(remove_passes_filter)


            # Decode images: RLDS saves encoded images, only decode now for efficiency
            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            dataset = dataset.frame_map(decode_images, num_parallel_calls)
            # Shuffle
            dataset = dataset.shuffle(shuffle_buffer_size)

            return dataset
        
        print('_____________________')
        print(datasets)
        print(action_space)
        print('_____________________')

        ds_streams = [prepare_dataset(ds_name) for ds_name, _ in datasets]
        weights = [weight for _, weight in datasets]

        final_ds = dl.DLataset.sample_from_datasets(ds_streams, weights=weights)
        final_ds = final_ds.batch(batch_size)

        # Note =>> Seems to reduce memory usage without affecting speed?
        final_ds = final_ds.with_ram_budget(1)

        self.dataset = final_ds
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):
        # This is the approximate number of samples in DROID after filtering.
        # Easier to hardcode than to iterate through the dataset and compute it.
        return 20_000_000
