# WSM adapters for ScoreHMR

These scripts keep the upstream ScoreHMR training path intact while adapting
WSV-stored SMPLer-X annotations into ScoreHMR's expected `.npz` dataset and
PARE feature cache format.

Typical debug flow:

```bash
cd third_party/ScoreHMR

python wsm_adapters/build_wsv_smpl_dataset.py \
  --train-root /common/users/sf895/wave2pixel/AVIS_train_results \
  --val-root /common/users/sf895/wave2pixel/AVIS_val_results \
  --max-videos 8 \
  --frame-stride 8

python wsm_adapters/cache_wsv_pare_feats.py \
  --dataset data/datasets/wsv_smpl_train.npz \
  --out cache/pare/wsv_smpl_train.npz

python wsm_adapters/train_wsv_smpl.py \
  --name wsv_smpl_debug \
  --train-dataset data/datasets/wsv_smpl_train.npz \
  --train-pare cache/pare/wsv_smpl_train.npz \
  --total-steps 1000
```

The main semantic difference from the official ScoreHMR path is the annotation
source: SMPL-like axis-angle parameters are extracted from WSV channels 72-87
instead of ScoreHMR's original training datasets.

For normal WSM training, prefer the repository launcher:

```bash
cd /research/cbim/vast/sf895/code/WSMs/WSM_v1
FORCE_REBUILD=1 FORCE_RECACHE=1 RESET_RUN=1 \
  bash scripts/launch_scripts/scorehmr_wsv_smpl_train.sh
```

The launcher starts a detached tmux session by default and writes the full
pipeline log, including dataset build, PARE cache, and training, under:

```text
third_party/ScoreHMR/logs/launch_logs/
```

Useful overrides:

- `RUN_NAME`: ScoreHMR run name.
- `DATASET_PREFIX`: generated dataset/cache prefix.
- `MAX_VIDEOS`, `FRAME_STRIDE`, `MAX_FRAMES_PER_VIDEO`: subset controls.
- `FORCE_REBUILD=1`: rebuild WSV-derived `.npz` datasets.
- `FORCE_RECACHE=1`: recache PARE image features.
- `RESET_RUN=1`: remove old logs/checkpoints/visuals for the same run.
- `DETACH=0`: run in the foreground instead of tmux.

The adapter maps WSV SMPLer-X information into the official ScoreHMR SMPL
parameter space:

- WSV joint ids `0-21` become SMPL `global_orient + body_pose[0:21]`.
- WSV hand-root ids `22` and `37` approximate SMPL's terminal `L_hand` and
  `R_hand` body joints.
- WSV channels `78-87` are used as SMPL betas when `--use-betas` is enabled.

This keeps the official 144D/154D ScoreHMR diffusion target unchanged. WSV
hands, face, translation, vertices, and bbox marker channels are intentionally
not included yet because they require extending the ScoreHMR output parameter
space beyond SMPL body pose and shape.
