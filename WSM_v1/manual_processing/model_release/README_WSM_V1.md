---
library_name: wsms
pipeline_tag: image-to-image
tags:
- world-snapshot-model
- computer-vision
- image-to-image
- diffusers
- sparse-activation
- multi-task
license: other
license_name: wsm-non-commercial-research-license
license_link: https://huggingface.co/World-Snapshot/WSM_v1/blob/main/LICENSE
---

# WSM v1: World Snapshot Model

[[Paper]](https://arxiv.org/abs/2611.24917) |
[[Project Page]](https://world-snapshot.github.io/WSMs/) |
[[Code]](https://github.com/World-Snapshot/WSMs) |
[[Documentation]](https://world-snapshot.github.io/doc/index.html) |
[[Datasets]](https://huggingface.co/World-Snapshot/datasets) |
[[Post-training Materials]](https://huggingface.co/World-Snapshot/WSM_v1-Training-Resources) |
[[ControlWorld WebUI]](https://github.com/World-Snapshot/ControlWorld) |
[[Live Demo]](https://huggingface.co/spaces/FangSen9000/WSM-ControlWorld)

WSM v1, codename **Augustus**, is the first large-scale implementation of the
Unified Law for Visual Tasks. It organizes visual tasks as transformations
among image-like World Snapshot representations and realizes them through one
shared generative model with built-in, sparsely activated TunnelingNets. The
paper trains and evaluates 1,148 visual tasks and provides a trained baseline
for every evaluated task.

WSM uses [World Snapshot Graphics (`.wsg`)](https://github.com/World-Snapshot/wsg)
to place RGB, geometry, motion, pose, physics, materials, semantics, audio, and
other World Information in a common 256-channel address space. TunnelingNet
then expands a frozen visual generation model with compact, independently
trainable task paths.

## One Model Checkpoint

The complete inference release is the single checkpoint **`WSM_v1.wsm`**:

- **46,643,677,777 bytes** (43.44 GiB);
- the complete frozen FLUX.2 Klein 4B runtime;
- built-in, sparsely activated TunnelingNets spanning Dense, Sparse, and Custom
  visual-task families;
- a terminal task index, byte offsets, lengths, and SHA-256 checksums.

The file is monolithic for distribution but sparse in execution. The `wsms`
pipeline first reads its small header and terminal index using HTTP Range
requests. For each request, the model activates the corresponding internal
tunnel and materializes only the byte ranges required by that route. The other
routes remain inactive and are neither downloaded nor loaded. The shared
backbone is cached once and reused by later tasks. Users therefore interact
with one checkpoint, one Pipeline, and one model interface.

## Installation

```bash
pip install -U wsms
```

For the full training repository:

```bash
git clone --recursive https://github.com/World-Snapshot/WSMs.git
```

## Discover Tasks Without Downloading Weights

```python
from wsms import WSMPipeline

tasks = WSMPipeline.list_tasks(inputs=["0-2"], output="36-38")
print(tasks[0]["task_id"])
print(tasks[0]["head"])
print(tasks[0]["condition_description"])
print(tasks[0]["target_description"])
```

The packaged catalog covers all released Dense, Sparse, and Custom task
families. Input order is preserved because `A + B -> C` and `B + A -> C` can
represent different routing contracts.

## Python Inference

Select a task by its input and output modalities:

```python
import torch
from wsms import WSMPipeline

pipe = WSMPipeline.from_pretrained(
    "World-Snapshot/WSM_v1",
    inputs=["0-2"],       # RGB
    output="36-38",      # surface normals
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

Pin an exact released implementation when needed:

```python
pipe = WSMPipeline.from_pretrained(
    "World-Snapshot/WSM_v1",
    head="dense:000001",
    device="cuda:0",
)
```

For a multi-input route, pass a modality-keyed dictionary:

```python
pipe = WSMPipeline.from_pretrained(
    "World-Snapshot/WSM_v1",
    inputs=["39", "0-2"],
    output="90-92",
    device="cuda:0",
)

result = pipe({
    "39": "curvature.npz",
    "0-2": "reference.png",
})
```

## Local, Offline, and Legacy Loading

The same API accepts a downloaded checkpoint:

```python
pipe = WSMPipeline.from_pretrained(
    "/models/WSM_v1.wsm",
    inputs=["0-2"],
    output="36-38",
    device="cuda:0",
)
```

`cache_dir=...` controls where selected members are materialized. The pipeline
automatically recognizes a local `.wsm` file, a directory containing
`WSM_v1.wsm`, the unified Hub repository, a legacy split-layout Hub repository,
or a direct legacy task checkpoint. Existing local workflows therefore remain
supported by the same public API.

## Command-Line Inference

```bash
wsms infer \
  --input 0-2=input.png \
  --target 36-38 \
  --device cuda:0 \
  --steps 10 \
  --seed 20260812 \
  --output-dir outputs/rgb_to_normal
```

Use `wsms tunnel list` to discover mappings and `wsms head list` to inspect
concrete implementations.

## ControlWorld and the Live Demo

[ControlWorld](https://github.com/World-Snapshot/ControlWorld) is the reference
WebUI. It selects a route from the WSG task graph, validates and encodes its
inputs, activates the required internal tunnel, performs inference, and
visualizes the native result through WSG Reader. The hosted
[Live Demo](https://huggingface.co/spaces/FangSen9000/WSM-ControlWorld) uses the
same inference code and the same `WSM_v1.wsm` checkpoint.

## Models, Data, and Post-training Materials

This repository is intentionally inference-oriented. It contains one model
checkpoint plus its model card and license.

- **Datasets and processed latent data:**
  [World-Snapshot datasets](https://huggingface.co/World-Snapshot/datasets)
- **Training states, optimizer states, intermediate checkpoints, logs, curves,
  and continued-training resources:**
  [`WSM_v1-Training-Resources`](https://huggingface.co/World-Snapshot/WSM_v1-Training-Resources)
- **Training, evaluation, conversion, and inference code:**
  [World-Snapshot/WSMs](https://github.com/World-Snapshot/WSMs)
- **Complete executable notebook:**
  [`WSM_Quickstart.ipynb`](https://github.com/World-Snapshot/WSMs/blob/main/src/examples/WSM_Quickstart.ipynb)

Keeping post-training records separate gives ordinary users one clean model
artifact while preserving the complete material needed for inspection,
continued training, and reproducibility.

## License

**In short:** You may use, study, modify, train, evaluate, and extend the
Rutgers-owned WSM software and model artifacts for noncommercial research,
experimentation, development, education, and personal projects. Commercial use
requires separate permission from Rutgers.

The Rutgers-owned software and model artifacts in this release—including
trained or fine-tuned weights, checkpoints, adapters, `.pth` files, and other
model-weight files—are released under the **Rutgers Non-commercial Research
License (RU-NCRL)**. The complete governing terms are in the repository's
[`LICENSE`](LICENSE) file.

For commercial-use rights, contact **compliance@research.rutgers.edu**, copy
**dnm@cs.rutgers.edu** and **sen.fang@rutgers.edu**, and reference
**Rutgers Docket #2027-020**.

The embedded FLUX.2 Klein 4B components retain their Apache 2.0 license.
Other third-party components, dependencies, and notices retain their respective
licenses. No rights under the WSM patent application are granted except as
expressly stated in the RU-NCRL.

## Citation

```bibtex
@misc{fang2026unifiedlawvisualtasks,
  title={Unified Law for Visual Tasks (The Four Laws of Computer Vision)},
  author={Sen Fang and Yalin Feng and Yancheng Zhang and Yanxin Zhang and Zonglin Lyu and Hongbin Zhong and Liwei Che and Rui Wu and Chen Chen and Dimitris N. Metaxas},
  year={2026},
  eprint={2611.24917},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.24917}
}
```
