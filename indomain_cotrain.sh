#! /bin/bash
set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

uv run --group rlds scripts/train.py pi05_droid_jointpos_fullfinetune --no-save-train-state --overwrite --exp-name indomain-cotrain
rsync -e "ssh -S /tmp/mux-socket" -av ./checkpoints arhanj@klone.hyak.uw.edu:~/arhan/openpi/checkpoints &

uv run --group rlds scripts/train.py pi0_fast_droid_jointpos_fullfinetune --no-save-train-state --overwrite --exp-name indomain-cotrain
rsync -e "ssh -S /tmp/mux-socket" -av ./checkpoints arhanj@klone.hyak.uw.edu:~/arhan/openpi/checkpoints &

uv run --group rlds scripts/train.py pi0_droid_jointpos_fullfinetune --no-save-train-state --overwrite --exp-name indomain-cotrain
rsync -e "ssh -S /tmp/mux-socket" -av ./checkpoints arhanj@klone.hyak.uw.edu:~/arhan/openpi/checkpoints &

uv run --group rlds scripts/train.py paligemma_binning_droid_jointpos_fullfinetune --no-save-train-state --overwrite --exp-name indomain-cotrain
rsync -e "ssh -S /tmp/mux-socket" -av ./checkpoints arhanj@klone.hyak.uw.edu:~/arhan/openpi/checkpoints &

wait