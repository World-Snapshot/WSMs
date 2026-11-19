# Manual Processing

This directory contains the data preparation tools that remain useful for
building or extending WSM v1. Production entry points are grouped by purpose;
completed one-off migrations and repair investigations live under `archive/`.

## Current tools

- `flux2_preprocessing/vae_latents/`: build the dense FLUX.2 frame-latent
  cache. `precompute_wsg_flux_frame_latents.py` is the main encoder.
  `reencode_coarse_categorical_modalities.py` maintains the reduced output
  label space for categorical targets 111, 131, and 136. The current release
  remaps categorical sidecars and retrains refiners; it does not re-encode VAE
  latents or retrain the base Channel Heads.
- `flux2_preprocessing/cache/`: current cache launchers and builders for WSG
  frames, text embeddings, XPose, Universal Pose, and sparse audio-semantic
  overlays. The MP-100 Universal Pose encoder stores paired `0-2` and
  `Universal_Pose` FLUX.2 latents; rectangular images are intentionally
  stretched to the square training canvas and retain their original size for
  aspect-ratio restoration after decoding. Its `_8gpu.sh` launcher provides
  resumable eight-GPU preprocessing.
- `flux2_preprocessing/configs/`: dataset configurations consumed by the
  FLUX.2 preprocessing entry points.
- `flux2_preprocessing/audit_dense_channel_head_pairs.py`: verify that cached
  input and target modalities have aligned samples before dense training.
- `audio/`: build the sparse EnCodec audio dataset used by sparse tasks.
- `normalization/`: compute WSG normalization metadata.
- `modality_metadata/`: generate and inspect the WSG v1 modality inventory.
- `model_release/`: select the official dense, sparse, and custom Channel Head
  runs, remove redundant per-step visualizations, audit private metadata, and
  publish the remaining records to the private Hugging Face model repository.

## Archive

`archive/` preserves completed migrations, repair scripts, and diagnostic
experiments for provenance. They are not part of the normal user workflow and
may refer to historical data layouts. In particular:

- `categorical_latent_repairs_2026/`: the original 111/131/195 latent repair
  and visualization utilities.
- `avis_reindex_repairs_2026/`: the completed AVIS frame-order and channel-195
  repair pipeline.
- `flux2_cache_repairs_2026/`: one-time cache corrections for specific
  modalities or datasets.
- `legacy_wan_univid/`: preprocessing for the retired Wan/UniVid backbone.

New reusable preprocessing should be placed in the appropriate active
subdirectory. One-time repair scripts should be added directly to `archive/`
with enough context to identify the affected data version.
