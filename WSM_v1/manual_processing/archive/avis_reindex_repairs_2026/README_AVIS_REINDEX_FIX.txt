AVIS WSV frame-order and channel-195 semantic repair
====================================================

Background
----------
Some AVIS WSV files were built with frame arrays ordered by lexicographic crop-frame file names instead of the natural AVIS JSON frame order. This made frame-indexed modalities appear temporally inconsistent. Most frame-level dense modalities (RGB, normal, edge, etc.) were already correct for their physical frame content, but their logical index needed to be reinterpreted after the WSV repair. Channel 195 (audio semantic / semantic instance mask) was worse: it had been injected using the wrong frame alignment, so its content had to be rebuilt.

Final repair strategy
---------------------
1. Repair each AVIS WSV in one expensive read/write pass:
   - open the WSV once;
   - inject channel 195 using the legacy lexicographic crop index mapped to the AVIS natural JSON frame;
   - save the WSV in natural AVIS JSON frame order;
   - extract cropped raw channel-195 planes needed for the existing FLUX latent cache.

2. Rebuild only channel-195 FLUX latent tensors:
   - read extracted raw channel-195 planes from data/avis_ch195_repair_parts;
   - VAE-encode them;
   - write them back to their original shard positions in shards/195.

3. Rekey the dense latent cache metadata:
   - tensor row order is intentionally not moved;
   - sample_ids, sample_sources.frame_keys, frame_keys, and frame_indices are rewritten from legacy lexicographic crop indices to natural AVIS JSON indices;
   - this makes each existing latent row refer to the correct natural frame index.

Important invariant
-------------------
After the repair, training code should rely on sample_ids / frame_keys / frame_indices, not on assuming that shard row order is physically sorted by video frame. The row content and metadata now agree. Re-running the metadata rekey on the same cache root is unsafe because the mapping is old-index -> natural-index and is not idempotent. The script writes .avis_frame_rekey_after_reindex_done.json and refuses to run again unless --force is passed.

Primary scripts
---------------
WSV repair and channel-195 extraction:
  /research/cbim/vast/sf895/code/wsg/main/pipeline30c_reindex_avis_and_extract_ch195.py

Fill a small number of missing channel-195 extraction records directly from AVIS JSON:
  manual_processing/archive/avis_reindex_repairs_2026/fill_missing_avis_ch195_parts_from_json.py

Rebuild channel-195 FLUX latent cache from extracted planes:
  manual_processing/archive/avis_reindex_repairs_2026/rebuild_avis_ch195_cache_from_parts.py

Rekey dense latent cache metadata after WSV repair:
  manual_processing/archive/avis_reindex_repairs_2026/rekey_avis_flux_cache_after_reindex.py

Debug visualization:
  manual_processing/archive/avis_reindex_repairs_2026/debug_visualize_avis_wsv_long.py

Completed repair record
-----------------------
Dense cache root:
  data/wsg_flux2_frame_latents_dense_multidataset

Channel-195 rebuild completed with:
  replaced: 50927
  samples: 50927

Dense metadata rekey dry-run reported:
  files: 322
  changed: 2122447

Intermediate extracted channel-195 parts:
  data/avis_ch195_repair_parts

This directory was deleted after successful rebuild and validation.
