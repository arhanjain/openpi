import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
print(sys.path)
from dataclasses import dataclass, replace
from train import main as train_main, _config  
from datetime import datetime

import wandb
import wandb.util
import tyro
import submitit
import socket
import subprocess
import json


@dataclass
class ExecutorConfig:
    timeout: int = 60 * 24
    partition: str = "gpu-h200"
    cluster: str = "slurm"
    account: str = "weirdlab"
    nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_task: int = 32
    memory: int = 800 # GB
    slurm_array_parallelism: int = 256
    qos: str = "normal"


@dataclass
class Args:
    executor: ExecutorConfig
    name: str


class TrainJob(submitit.helpers.Checkpointable):
    def __init__(self):
        pass

    def __call__(self, train_cfg: _config.TrainConfig):
        train_main(train_cfg)


    def checkpoint(self, *args, **kwargs):
        print("-------------------")
        print("Checkpointing")
        print("-------------------")

        return super().checkpoint(*args, **kwargs)

if __name__ == "__main__":
    args = tyro.cli(Args)
    date, time_ = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%I:%M:%S-%p")

    # log folder
    folder = Path(f"submitit-logs") / date / f"{args.name}-{time_}"
    folder.mkdir(parents=True, exist_ok=True)

    # # Define executor
    ex = submitit.AutoExecutor(
        folder=folder, slurm_max_num_timeout=3, cluster=args.executor.cluster
    )
    ex.update_parameters(
            name = "openpi-cotraining",
            slurm_partition=args.executor.partition,
            slurm_account=args.executor.account,
            timeout_min=args.executor.timeout,
            nodes=1,
            gpus_per_node=args.executor.gpus_per_node,
            cpus_per_task=args.executor.cpus_per_task,
            mem_gb=args.executor.memory,
            slurm_array_parallelism=args.executor.slurm_array_parallelism,
            slurm_qos=args.executor.qos,
            # slurm_gres=f"gpu:{args.executor.gpus_per_node}",
            )
    print(f" >>> Executing on cluster: {args.executor.cluster} <<< ")

    TRAIN_CONFIGS = [
        # "pi05_droid_jointpos_fullfinetune",
        # "pi0_fast_droid_jointpos_fullfinetune",
        # "pi0_droid_jointpos_fullfinetune",
        # "pi0_droid_jointpos_100k_fullfinetune",
        # "paligemma_binning_droid_jointpos_fullfinetune",
        "pi05_droid_libero_fullfinetune",
        "pi0_fast_droid_libero_fullfinetune",
        "pi0_droid_libero_fullfinetune",
        "pi0_droid_libero_100k_fullfinetune",
        "paligemma_binning_droid_libero_fullfinetune",
    ]
    # VISUAL_FIDELITY = [
    #     ("splat", "droid_ood_cotrain_robotsplat_dataset:2.0.0"),
    #     ("raytraced", "droid_ood_cotrain_raytraced_dataset:2.0.0"),
    #     ("nightmare", "droid_ood_cotrain_nightmare_dataset:2.0.0"),
    # ]

    jobs = []
    with ex.batch():
        for config in TRAIN_CONFIGS:
            train_cfg = _config._CONFIGS_DICT[config]
            train_cfg = replace(
                train_cfg,
                # overwrite=True, 
                resume=True,
                exp_name=f"{args.name}",
                )

            job = ex.submit(TrainJob(), train_cfg)
            jobs.append(job)
    print(f"Submitted {len(jobs)} jobs.")

    # Wait till all jobs are queued.
    import time
    while len(jobs) > 0:
        time.sleep(1)
        jobs = [j for j in jobs if j.state != "RUNNING"]








