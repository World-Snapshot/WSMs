---
library_name: wsms
pipeline_tag: image-to-image
tags:
- world-snapshot-model
- computer-vision
- diffusers
- sparse-activation
license: other
---

# WSM v1

WSM v1 is distributed as one model checkpoint, `WSM_v1.wsm`. The checkpoint
contains the complete frozen FLUX.2 Klein 4B runtime and 1,158 TunnelingNets for
the released Dense, Sparse, and Custom visual tasks. Its embedded task index and
aligned byte ranges allow the public pipeline to read only the shared backbone
and the selected TunnelingNet; the remaining task parameters are not loaded.

```python
from wsms import WSMPipeline

pipe = WSMPipeline.from_pretrained(
    "World-Snapshot/WSM_v1",
    inputs=["0-2"],
    output="36-38",
    device="cuda:0",
)
result = pipe("input.png", num_inference_steps=10)
result.images[0].save("result.png")
```

The first inference materializes the embedded backbone and selected task in the
local cache. Later tasks reuse the backbone and add only their selected
TunnelingNet. A local checkpoint can be used with the same API:

```python
pipe = WSMPipeline.from_pretrained(
    "/models/WSM_v1.wsm",
    head="dense:000001",
    device="cuda:0",
)
```

Training states, optimizer states, intermediate checkpoints, logs, and
continued-training material are maintained separately in
[`World-Snapshot/WSM_v1-Training-Resources`](https://huggingface.co/World-Snapshot/WSM_v1-Training-Resources).

See the [WSMs repository](https://github.com/World-Snapshot/WSMs) for the Python
package, task discovery, input contracts, and complete examples. WSM components
are released under the repository's non-commercial research license. Embedded
third-party components retain their original licenses and notices.
