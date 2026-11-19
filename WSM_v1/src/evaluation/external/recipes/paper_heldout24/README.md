# Paper held-out-24 recipes

This directory preserves the experiment-specific post-processing used for the
paper's crop-aligned AVIS depth comparison. These scripts are reproducibility
recipes, not alternative public entry points. New evaluations should start from
`../../run_benchmark.py`.

The scripts intentionally encode the frozen paper protocol, including the
24-sample coverage requirement, WSM depth-head variants, and the documented
near/far convention corrections for individual released baselines. They accept
workspace paths on the command line and do not contain user, environment, or GPU
paths.

Typical order:

```bash
python prepare_crop_aligned_depth.py \
  --workspace /path/to/main/benchmark/workspace \
  --latent-cache /path/to/wsm/latent/cache \
  --output-root /path/to/crop-aligned/workspace

# Run official adapters against output-root/samples.jsonl. Each adapter writes
# its canonical JSONL manifest below output-root/predictions/.

python evaluate_crop_aligned_depth.py \
  --root /path/to/crop-aligned/workspace
```

`finalize_predictions.py` converts the exact archived paper-run layouts into
canonical manifests. `evaluate_depth_protocols.py` compares the four frozen WSM
absolute/relative depth value-space variants. Both are retained so reported
tables can be reconstructed, but neither is needed for a new benchmark.

Cluster scheduling, virtual-environment locations, and GPU allocation are local
operational choices. They are deliberately excluded from this public recipe.
