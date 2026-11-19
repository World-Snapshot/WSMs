# WSM evaluation

This package contains quantitative evaluation code, split by comparison scope.

## Internal entry points

Repository imports and command-line jobs use the implementations directly:

- internal/channel_head_evaluator.py: evaluate and select WSM Channel-Head checkpoints.
- internal/representation_alignment_evaluator.py: compare exactly invertible target
  representations under the frozen backbone before task-specific training.
- internal/representation_alignment_train.py: evaluation-only runtime wrapper for
  training the predeclared target-coordinate variants while leaving the production
  trainer and all of its defaults unchanged.
- internal/representation_alignment_native_normal_eval.py: resumable held-out
  evaluation of archived alignment checkpoints after inversion to native surface
  normals, using angular error and fixed-threshold normal accuracy.
- internal/workspace_comparison.py: compare two internal WSM evaluation workspaces.
- internal/recommended_registry_export.py: freeze the recommended Head registry.

There are no duplicate root-level compatibility wrappers.

## Directory layout

- `internal/`: WSM-only checkpoint discovery, inference metrics, workspace comparison,
  and registry export.
- `external/`: WSM-versus-published-method benchmark orchestration and unified metrics.
- `external/recipes/`: versioned helpers for reconstructing specific paper experiments;
  these are not additional public entry points.
- `docs/`: evaluation notes and research records.
- `legacy/`: historical `.orig` snapshots; not executable entry points.

Generated evaluation results belong under src/evaluation_output/; temporary tests may use debug/, but formal external benchmark outputs are namespaced under src/evaluation_output/external/.

The versioned sparse-Head paper results are consolidated in
`docs/native_sparse_head_results_20260904.json`. This is the canonical record for
the metric names, evaluation splits, recommended checkpoints, and full-precision
values summarized in the supplementary table.

## Internal versus external evaluation

Internal paper quantitative evaluation has one main entry point: internal/channel_head_evaluator.py. It performs checkpoint discovery, inference, decoded/latent/categorical metrics, resumable runs, and CSV/JSON/LaTeX tables. The other internal files are auxiliary commands for workspace comparison and registry export, not separate metric definitions.

External evaluation answers how WSM compares with official released baselines on a
frozen third-party held-out sample manifest. Each method keeps its official VAE,
tokenizer, backbone, checkpoint, and dependency environment. External inference must
export canonical predictions; `external/evaluate_predictions.py` then enforces exact
sample coverage and computes cross-method metrics with one implementation.
Its default paper profile computes only the quality metrics reported in Tables 6
and 7. Pass `--extended-evaluation` to the one-stop runner to compute the complete
diagnostic metric set, bootstrap confidence intervals, and enabled distribution
metrics. The chosen profile is recorded in the evaluation audit.

The one-stop external entry point is external/run_benchmark.py. It performs:

1. license and disk-space preflight;
2. dataset download or gated-dataset verification;
3. official repository and pinned-checkpoint preparation;
4. isolated baseline inference;
5. canonical prediction conversion;
6. exact sample-ID audit, metrics, and paper tables.

The paper protocol freezes exactly 24 shared samples per task. Every method that
supports a task receives the same 24 sample IDs. external/select_heldout24.py
performs deterministic recency/contamination filtering, exact and perceptual
deduplication, stratified selection, and writes a hash audit before inference.
FID is intentionally disabled for this small sample count; aggregate metrics include
deterministic bootstrap 95% confidence intervals in the extended profile.

Quick start:

    python src/evaluation/external/run_benchmark.py init \
      --config src/evaluation_output/external/configs/benchmark.json

    python src/evaluation/external/run_benchmark.py doctor \
      --config src/evaluation_output/external/configs/benchmark.json --dry-run

    python src/evaluation/external/run_benchmark.py all \
      --config src/evaluation_output/external/configs/benchmark.json

Edit the generated config first: point external_manifest.manifest at a candidate
JSONL pool, fill the WSM checkpoint/inference command, and enable only verified
baselines. Public or licensed fallback datasets require explicit license
acknowledgement. The runner records resolved Git commits, model directories, the
frozen sample-manifest hash, and stage completion in the benchmark workspace.

`external/benchmark_heldout24.example.json` is the versioned public template.
Machine-specific resolved configurations, environments, downloaded repositories,
weights, predictions, and tables remain below the ignored `src/evaluation_output/`
tree. This separation keeps local resumability without publishing cluster paths or
generated artifacts.

Paper-reported numbers must be marked `reported_only` and remain in a separate table
from locally rerun predictions. In particular, no ranking may mix numbers copied from
a paper with numbers recomputed on a different sample manifest.

Run the lightweight contract tests before publishing evaluation changes:

    python -m unittest discover -s src/evaluation/tests -v
