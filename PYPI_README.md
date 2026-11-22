# WSM: World Snapshot Models

**World Snapshot Models (WSMs) are an ideal world model designed to unify,
expand, and ultimately transcend Computer Vision.** The first implementation is
WSM v1, codename Augustus.

To train the world itself, we define world frames and use
[World Snapshot Graphics (`.wsg`)](https://github.com/World-Snapshot/wsg) to
encode RGB, geometry, motion, pose, physics, materials, semantics, audio, and
other information into a unified 256-channel image. This realizes the
[Unified Law for Visual Tasks](https://github.com/World-Snapshot/Unified-Law-for-Visual-Tasks):
**if all information can be encoded into the image itself, then two-dimensional
image generation becomes lossless world generation.**

WSM v1 is the first large-scale implementation of this law. **A single
generative model validates 1,148 visual tasks and provides a trained baseline
for every task.** Capabilities previously scattered across different models,
datasets, and research communities coexist in the same system, while 99.99% of
earlier visual models were built for one task. Papers with Code recorded 1,477
visual or vision-related tasks representing the existing field; all 1,477 are
compatible with WSM interfaces. Among the 1,148 evaluated tasks, an estimated
655–1,056 are entirely new, allowing WSM to **expand the recorded Computer
Vision task space by 44.3–71.5%.**

The four WSM interfaces are the minimum common training form of the complete
visual task space. At this higher level, classification, localization,
detection, segmentation, estimation, reconstruction, restoration, completion,
registration, correspondence, tracking, retrieval, indexing, prediction,
generation, editing, rendering, simulation, planning, and control all belong
to these four interface categories. Thousands of concrete tasks can therefore
be trained through the same four unified interfaces.

This scale is enabled by **TunnelingNet**. It preserves a shared FLUX.2
generative backbone while learning an independently trainable compact tunnel
for each directed task. Every tunnel can be trained, loaded, improved,
separated, or newly added without retraining the task bank from scratch.
Compared with parameter-matched multimodal and unified-model baselines,
TunnelingNet reveals recognizable structure 77–100 times earlier and requires
19.0–38.5 times fewer optimizer updates. Each task adds only a
12.51M-parameter tunnel; with the complete task bank loaded, WSM v1 forms an
18B-class model rather than multiplying the backbone thousands of times.

The `wsms` package makes this complete task space directly callable through
`WSMPipeline`, command-line discovery, and qualified Dense, Sparse, and
Custom Channel Heads.

## Installation

```bash
pip install wsms
```

The [one-click WSM notebook](https://github.com/World-Snapshot/WSMs/blob/main/src/examples/WSM_Quickstart.ipynb)
turns the discovery and inference examples below into an executable GPU
workflow.

WSM v1 inference requires a CUDA-capable GPU and currently runs in
`torch.bfloat16`. If pip selects a PyTorch build that does not match the local
CUDA driver, install the appropriate PyTorch wheel first and then install
`wsms`.

Optional interfaces can be installed without changing the API:

```bash
pip install "wsms[audio]"     # audio-specific decoding support
pip install "wsms[webui]"     # ControlWorld/WebUI dependencies
pip install "wsms[training]"  # training and continuation utilities
pip install "wsms[full]"      # all optional components
```

The pip distribution contains the WSM pipeline, task catalog, modality
contracts, and the source required to run released TunnelingNets. Model weights
are distributed separately as the single `WSM_v1.wsm` checkpoint on Hugging
Face and are never embedded in the Python package.

### Install directly from GitHub

The public Git repository can also be installed directly. Dependencies declared
in `pyproject.toml` are installed automatically:

```bash
pip install "wsms @ git+https://github.com/World-Snapshot/WSMs.git@v0.2.0"
```

Replace the release tag with a specific commit when exact reproducibility is
required.

## Model access

The default model repository is:

```text
World-Snapshot/WSM_v1
```

The model repository is public, so downloading released checkpoints does not
require authentication. A Hugging Face login remains optional for users who
want authenticated Hub access:

```bash
hf auth login
```

When present, the pipeline uses the cached Hugging Face identity automatically.
It also supports the standard `HF_TOKEN` environment variable and an explicit
`token=...` argument. Avoid storing tokens in source files or shell history.

## Three levels of discovery

WSM separates the public task relationship from its concrete implementation:

1. A **tunnel** is an ordered input-to-output mapping, such as
   `0-2 -> 36-38`.
2. A **Head** is one implementation of that mapping.
3. A **family** qualifies the Head namespace: `dense`, `sparse`, or
   `custom`.

Head indices repeat across families. Consequently, a bare value such as
`000003` is intentionally rejected. Use `dense:000003`,
`sparse:000003`, or `custom:000003` when pinning an implementation.

### Command-line discovery

```bash
wsms list
wsms tunnel list
wsms head list
wsms info
```

`wsms list` shows the package's top-level capabilities. `wsms tunnel list`
shows all user-facing mappings, while `wsms head list` shows the qualified
implementation behind each mapping.

Examples:

```bash
# RGB -> surface normals
wsms tunnel list --input 0-2 --output 36-38

# Ordered multi-input route
wsms tunnel list --input 39 --input 0-2 --output 90-92

# Only Sparse implementations
wsms head list --source sparse

# Machine-readable discovery
wsms tunnel list --output Universal_Pose --json
```

Filters preserve input order because `A + B -> C` and `B + A -> C` can
represent different routing contracts.

### Python discovery

```python
from wsms import WSMPipeline

tasks = WSMPipeline.list_tasks(
    inputs=["0-2"],
    output="36-38",
)
print(tasks[0]["task_id"])
print(tasks[0]["head"])
print(tasks[0]["condition_description"])
print(tasks[0]["target_description"])
```

The catalog combines all released and recommended Dense, Sparse, and Custom
Heads. It can be queried without downloading model weights.

## Python inference

Input-output selection is the recommended public API:

```python
import torch
from wsms import WSMPipeline

pipe = WSMPipeline.from_pretrained(
    "World-Snapshot/WSM_v1",
    inputs=["0-2"],
    output="36-38",
    torch_dtype=torch.bfloat16,
    device="cuda:0",
)

result = pipe(
    "input.png",
    num_inference_steps=10,
    seed=20260812,
    size=256,
)
result.images[0].save("normal.png")
```

`from_pretrained` first reads the compact index of `WSM_v1.wsm`, then fetches
only the embedded FLUX.2 Klein 4B runtime and the selected TunnelingNet byte
ranges. The backbone is cached once and shared across subsequent tasks; the
other 1,157 TunnelingNets are not downloaded or loaded.

### Multi-input tunnels

Pass a dictionary keyed by modality ID:

```python
pipe = WSMPipeline.from_pretrained(
    "World-Snapshot/WSM_v1",
    inputs=["39", "0-2"],
    output="90-92",
    device="cuda:0",
)

result = pipe(
    {
        "39": "curvature.npz",
        "0-2": "reference.png",
    }
)
```

An ordered list or tuple is also accepted, but a dictionary makes the routing
contract more explicit.

### Pinning a concrete Head

Use the input-output interface for ordinary use. A qualified Head is available
for exact reproduction:

```python
pipe = WSMPipeline.from_pretrained(
    "World-Snapshot/WSM_v1",
    head="custom:000000",
    device="cuda:0",
)
```

### Local and offline loading

A local single-file WSM checkpoint can replace the Hub repository:

```python
pipe = WSMPipeline.from_pretrained(
    "/models/WSM_v1.wsm",
    inputs=["0-2"],
    output="36-38",
    device="cuda:0",
)
```

Legacy release directories and direct task checkpoints remain supported, with
`base_model=...` available for those layouts. Set `cache_dir` to control where
the selected single-file members are materialized and reused.

## Command-line inference

The command-line interface uses the same catalog and `WSMPipeline` runtime:

```bash
wsms infer \
  --input 0-2=input.png \
  --target 36-38 \
  --device cuda:0 \
  --steps 10 \
  --seed 20260812 \
  --output-dir outputs/rgb_to_normal
```

Multi-input example:

```bash
wsms infer \
  --input 39=curvature.npz \
  --input 0-2=reference.png \
  --target 90-92 \
  --device cuda:0 \
  --output-dir outputs/curvature_reference_to_target
```

To pin an implementation, replace `--target` with a qualified selector such
as `--head dense:000424`. Run `wsms infer --help` for all runtime options.

## Accepted inputs

The Python pipeline accepts:

- paths to PNG/JPEG and WSG/NPZ inputs;
- `PIL.Image.Image` objects;
- NumPy arrays;
- PyTorch tensors.

RGB arrays are materialized as images. Non-RGB arrays are materialized as
modality-scoped NPZ files and then pass through the same normalization and
encoding contract used by the canonical WSM inference script.

## Outputs

Calling the pipeline returns `WSMPipelineOutput`:

```python
result.images       # list[PIL.Image.Image], primary visual outputs
result.wsg          # generated WSG array when available
result.task_id      # resolved family and input-output mapping
result.artifacts    # files retained when output_dir is provided
result.metadata     # task, seed, steps, size, and runtime manifest
```

Without `output_dir`, temporary intermediate files are removed after the
in-memory result is loaded. With `output_dir`, the generated image, WSG,
manifest, and related runtime artifacts are retained.

## Special modality assets

Some geometry and human-body visualizations require separately licensed assets.
Set these only when the selected tunnel needs them:

- `WSM_SMPLX_MODEL_ROOT`: directory containing SMPL-X model assets;
- `WSM_HUMAN_MODELS_PYTHONPATH`: optional Python helper root for SMPL-X;
- `WSM_SHAPENET_TEXTURE_ROOT`: ShapeNet texture directory.

These third-party assets are not distributed by `wsms`.

## Naming compatibility

`WSMPipeline` follows Python and Diffusers class naming conventions.
`WSM_Pipeline` remains available as a compatibility alias:

```python
from wsms import WSMPipeline, WSM_Pipeline

assert WSMPipeline is WSM_Pipeline
```

## Links

- [Project](https://world-snapshot.github.io/WSMs/)
- [Source](https://github.com/World-Snapshot/WSMs)
- [Model repository](https://huggingface.co/World-Snapshot/WSM_v1)
- [Datasets](https://huggingface.co/World-Snapshot/datasets)
- [Documentation](https://world-snapshot.github.io/doc/index.html)
- [Paper](https://arxiv.org/abs/2611.24917)
