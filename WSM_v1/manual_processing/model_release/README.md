# WSM v1 model release

`publish_wsm_v1_models.py` selects, audits, sanitizes, and uploads the official
WSM v1 TunnelingNet runs to the Hugging Face model repository
`World-Snapshot/WSM_v1`.

Selection comes from the dense recommendation registry and the sparse/custom
`recommended_checkpoint` fields. Unique deployed dense Heads carrying a quality
warning remain included; unselected ablations, failed runs, and superseded runs
are omitted.

## Three release tiers

1. **Direct inference files** retain their original paths. This tier contains
   every selected recommended checkpoint, `resolved_config.yaml`,
   `resolved_runtime_config.json`, and active categorical/audio inference
   sidecars selected by their manifests. It can be used without unpacking.
2. **Continued-training materials** contain every other retained artifact:
   trainer states, final and intermediate checkpoints, statistics, logs, curves,
   and ancillary training records. They are stored in deterministic,
   uncompressed TAR parts with a 14 GiB source-size target. Every part is checked
   to remain below 15 GiB after it is built.
3. **Excluded artifacts** are not published. This includes
   `pseudo-visualization`, `pseudo-visualization-inference`,
   `wsg-reader-visualization`, `wsg-reader-visualization-inference`, generic
   `visuals` directories, top-level sparse `step_*` previews, local caches,
   symbolic links, and host-bearing TensorBoard event files.

The TAR files are intentionally not compressed. Packaging reduces repository
file count while avoiding expensive, low-yield compression of weights, NPZ
arrays, and images. Archive members retain their paths under `train_log/`,
`train_log_sparse/`, and `train_log_custom/`.

The source directories are never rewritten. Every text-like file is sanitized
in memory during upload. The audit covers local absolute paths, user and host
identifiers, institutional/cluster domains, IP addresses, scheduler identity
fields, email addresses, and credential signatures. Supported PyTorch states
are rewritten safely when necessary; any unresolved sensitive binary stops the
release before upload.

## Plan, audit, and upload

```bash
cd WSM_v1
python manual_processing/model_release/publish_wsm_v1_models.py plan
python manual_processing/model_release/publish_wsm_v1_models.py audit --verbose
python manual_processing/model_release/publish_wsm_v1_models.py upload --verbose
```

The uploader defaults to private repositories. For the licensed public release,
resume with `--allow-public-repo`; the script verifies that the repository
already contains `LICENSE` before uploading any artifact.

The upload is resumable through Git-ignored state in `train_log/`. Migrating
from the incomplete legacy layout requires one upload with `--recreate-repo`;
subsequent restarts omit that flag and resume the new layout.

Download only the directly usable inference tier:

```bash
hf download World-Snapshot/WSM_v1 --repo-type model \
  --exclude "continued_training_materials/*" --local-dir WSM_v1
```

Download and restore the complete release:

```bash
hf download World-Snapshot/WSM_v1 --repo-type model --local-dir WSM_v1
python WSM_v1/restore_wsm_v1.py \
  --parts-dir WSM_v1/continued_training_materials --output WSM_v1
```
