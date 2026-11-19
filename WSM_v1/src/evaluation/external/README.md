# External paper benchmark

`run_benchmark.py` is the public one-stop entry point for preparing data and
released methods, running inference, computing metrics, and writing paper-ready
tables. Downloads, method environments, checkpoints, predictions, logs, and
reproducibility locks live in one ignored workspace below
`src/evaluation_output/external/`.

## Quick start

```bash
python src/evaluation/external/run_benchmark.py init \
  --config src/evaluation_output/external/configs/benchmark.json

# Edit the generated config: set the candidate manifest, WSM checkpoint and
# command, then enable only the methods that should run.
python src/evaluation/external/run_benchmark.py doctor \
  --config src/evaluation_output/external/configs/benchmark.json --dry-run

python src/evaluation/external/run_benchmark.py all \
  --config src/evaluation_output/external/configs/benchmark.json
```

By default, `evaluate` and `all` compute only the quality metrics reported in
paper Tables 6 and 7. This is the recommended reproduction profile and keeps
the public outputs aligned with the paper. To additionally compute every
diagnostic metric, bootstrap confidence intervals, and any enabled FID/CLIP
metrics, pass:

```bash
python src/evaluation/external/run_benchmark.py evaluate \
  --config src/evaluation_output/external/configs/benchmark.json \
  --extended-evaluation
```

The selected profile is recorded as `metric_profile` in `audit.json`. An
extended run replaces the selected method/task rows in the same tables, so use
a separate workspace when both paper-facing and diagnostic outputs must be
kept. The default quality columns are the union of those displayed across
Tables 6 and 7 for each directed task:

- RGB to relative depth: MAE, RMSE, PSNR, Acc.05/.10/.20, Pearson, and Spearman;
- edge plus reference RGB to RGB: edge-density error, F0.5, edge precision,
  MAE, MSE, PSNR, and SSIM;
- normal to edge: accuracy, balanced accuracy, F1, F2, false-positive and
  false-negative rates, specificity, negative predictive value, IoU, and recall;
- relative depth to normal: mean/median angular error and accuracy below
  11.25, 22.5, and 30 degrees;
- edge to human-part segmentation: pixel accuracy, mIoU, human/foreground IoU,
  mean class accuracy, and macro-F1.

### Independent public annotations (Table 8)

`evaluate_independent_postrelease.py` reproduces the paper-facing independent-
annotation comparison. Its default profile evaluates only the frozen directed
tasks and quality columns used for the paper-facing report. Additional task
directions and diagnostic metric families are opt-in and written to a separate
output directory; they are not silently merged into the paper profile.

```bash
python src/evaluation/external/evaluate_independent_postrelease.py
```

The default CSV files are written below
`metrics/independent_annotations/`. Additional diagnostic tasks and metric
families are opt-in and are kept separate from the paper reproduction:

```bash
python src/evaluation/external/evaluate_independent_postrelease.py \
  --extended-evaluation
```

Extended outputs are written below `metrics/independent_annotations_extended/`.
This switch is an explicit diagnostic mode, not a different sample-selection
rule: it uses the same frozen manifests and records the expanded profile in
the output metadata.

The stages are `doctor`, `prepare-data`, `prepare-methods`, `infer`, `evaluate`,
`report`, and `all`. Downloads resume when the underlying provider supports it.
The runner records the frozen sample-manifest hash, requested and resolved source
revisions, model locations, method environments, and completed stages.

Downloaded repositories, virtual environments, model directories, and the
configured Hugging Face cache are disposable after a completed run. Predictions,
tables, logs, sample manifests, `evaluation_contract.json`, and
`reproducibility_lock.json` can be retained independently. A later
`prepare-methods` or `all` run recreates missing dependencies from the pinned
registry metadata; it does not require archived local environments or weights.

The versioned `benchmark_heldout24.example.json` contains no machine paths and is
also the source used by the `init` command. Generated configurations belong in
`src/evaluation_output/`; do not replace the public example with a resolved local
configuration.

## Components

- `run_benchmark.py`: orchestration and reproducibility locks.
- `select_heldout24.py`: deterministic sample selection and deduplication.
- `dataset_selection_plan.example.json`: versioned selection-policy example.
- `evaluate_predictions.py`: framework-independent coverage checks and metrics.
- `adapters/`: canonical prediction exporters for released methods.
- `official_registry.json`: source, checkpoint, environment, license, and protocol
  metadata.
- `recipes/`: explicitly scoped reconstruction helpers for individual paper runs.

The adapter registry is authoritative for supported methods. A method without a
verified adapter can provide an explicit argv-style `inference.command` in the
local configuration. Each method keeps its released VAE, tokenizer, backbone,
checkpoint, and dependency environment.

## Frozen held-out-24 protocol

Use a candidate JSONL pool with at least 24 eligible records for each evaluated
task. Each record needs `sample_id`, `task`, `target_path`, the appropriate input
or condition path, and the selection metadata required by the example config.

The selector rejects missing files, possible training overlap, exact duplicate
targets, and near-duplicate images. It then maximizes stratum coverage with a fixed
seed and records a hash audit. Within a task, every compared method must cover the
same frozen sample IDs.

For semantic estimation, the primary contract is the WSM MediaPipe channel-103
six-class ontology: background, hair, body-skin, face-skin, clothes, and
others/accessories. A human-parsing baseline is admissible only with a frozen,
auditable mapping into these classes. Unmappable labels are ignored rather than
guessed. Binary foreground/background evaluation is not part of this benchmark.

## Canonical predictions and normal encodings

Every inference environment exports a JSONL prediction manifest. Canonical values
are RGB images, two-dimensional class IDs, scalar depth arrays, or three-component
normal fields. `evaluate_predictions.py` verifies exact sample coverage before
computing cross-method metrics.

Normal targets and predictions may independently declare one of:

- `rgb_01`: RGB visualization values mapped from `[0, 1]` to `[-1, 1]`;
- `xyz` or `physical_xyz`: physical vector components, used without value decoding;
- `wsg_linear_01`: WSG-normalized values decoded using three channel ranges;
- `wsg_log_01`: log-normalized WSG values decoded using the same ranges.

Use `target_normal_encoding` and `target_normal_channel_ranges` on sample records,
and `normal_encoding` and `normal_channel_ranges` on prediction records. Protocol
values provide backward-compatible defaults. Vector fields are resized in floating
point and normalized only after decoding.

## Results and paper-specific recipes

Formal results are written below
`src/evaluation_output/external/<benchmark_name>/`. Temporary smoke tests may use
`debug/`; neither location is tracked. Paper-reported values must use
`inference_regime=reported_only` and remain separate from locally recomputed
rankings.

The crop-aligned AVIS depth reconstruction used for the paper is retained under
`recipes/paper_heldout24/`. It includes protocol-specific preparation,
post-processing, and metric helpers, while cluster scheduling and local GPU or
environment paths remain outside the public repository.

## WorldSnapshot Horizon release

`export_horizon_dataset.py` builds the portable dataset published at
[World-Snapshot/WorldSnapshot-Horizon](https://huggingface.co/datasets/World-Snapshot/WorldSnapshot-Horizon).
It copies the exact final AVIS held-out, post-release, and independent-public-
annotation manifests; removes machine-specific paths; extracts only the required
WSG channels; preserves the exact evaluated RGB frames and item-level Wikimedia
provenance; and writes a checksum manifest. Full source videos are referenced by
their original URLs instead of being duplicated in the release.

```bash
python src/evaluation/external/export_horizon_dataset.py \
  --output src/evaluation_output/external/worldsnapshot_horizon_release_v1 \
  --wsv-workers 4
```

The exporter refuses to write into a non-empty destination. Model predictions and
downloaded baseline assets are not included in the dataset release.
