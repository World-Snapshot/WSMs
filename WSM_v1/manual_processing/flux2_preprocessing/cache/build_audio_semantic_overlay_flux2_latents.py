#!/usr/bin/env python
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[3] if '__file__' in globals() else Path.cwd()

COLORS = {
    1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (255, 255, 0),
    5: (255, 0, 255), 6: (0, 255, 255), 7: (255, 128, 0), 8: (128, 255, 0),
    9: (128, 0, 255), 10: (255, 0, 128), 11: (255, 192, 128), 12: (128, 128, 255),
    13: (255, 255, 128), 14: (128, 255, 128), 15: (128, 0, 0), 16: (170, 255, 195),
    17: (128, 128, 0), 18: (255, 215, 180), 19: (0, 0, 128), 20: (128, 128, 128),
    21: (255, 255, 255), 22: (20, 120, 20), 23: (120, 20, 20), 24: (20, 20, 120),
    25: (120, 80, 20), 26: (20, 120, 120),
}


def safe_modality_name(value: str) -> str:
    return str(value).replace('-', '_')


def shard_path(root: Path, modality: str, shard_id: int, num_shards: int) -> Path:
    return root / 'shards' / safe_modality_name(modality) / f'shard_{shard_id:05d}_of_{num_shards:05d}.pt'


def audio_ids(paths: list[Path]) -> set[str]:
    ids = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for wsv_path, frame_key in zip(data['wsv_path'].astype(str), data['frame_key'].astype(str)):
                ids.add(f'{wsv_path}::{frame_key}-{frame_key}')
    return ids


def parse_shard_ids(value: str, num_shards: int) -> list[int]:
    raw = str(value or "").strip()
    if not raw:
        return list(range(num_shards))
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    seen = set()
    deduped = []
    for idx in out:
        if idx < 0 or idx >= num_shards:
            raise ValueError(f"shard id {idx} out of range [0,{num_shards})")
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def parse_sample_id(sample_id: str) -> tuple[str, str] | None:
    if '::' not in sample_id:
        return None
    wsv, rest = sample_id.split('::', 1)
    frame_key = rest.split('-', 1)[0]
    return wsv, frame_key


def crop_frame(frame: np.ndarray, crop_info: dict | None) -> np.ndarray:
    if not crop_info:
        return frame
    patch_hw = int(crop_info['patch_hw'])
    crop_y = int(crop_info['crop_y'])
    crop_x = int(crop_info['crop_x'])
    h, w = frame.shape[:2]
    if h < patch_hw or w < patch_hw:
        pad_width = ((0, max(0, patch_hw - h)), (0, max(0, patch_hw - w)), (0, 0))
        frame = np.pad(frame, pad_width, mode='constant', constant_values=-1.0)
    return frame[crop_y:crop_y + patch_hw, crop_x:crop_x + patch_hw, :]


class WSVFrameCache:
    def __init__(self, max_open: int = 8) -> None:
        self.max_open = max(1, int(max_open))
        self._items: OrderedDict[str, np.lib.npyio.NpzFile] = OrderedDict()

    def get(self, wsv_path: str, frame_key: str) -> np.ndarray | None:
        item = self._items.get(wsv_path)
        if item is None:
            item = np.load(wsv_path, allow_pickle=False)
            self._items[wsv_path] = item
            while len(self._items) > self.max_open:
                _, old = self._items.popitem(last=False)
                old.close()
        else:
            self._items.move_to_end(wsv_path)
        if frame_key not in item:
            return None
        return item[frame_key].astype(np.float32, copy=False)

    def close(self) -> None:
        while self._items:
            _, item = self._items.popitem(last=False)
            item.close()


def semantic_overlay(rgb: torch.Tensor, frame: np.ndarray | None, sample_name: str, crop_info: dict | None) -> torch.Tensor | None:
    # rgb: [3,H,W] in [-1,1]. Replace valid mask pixels with opaque label color.
    try:
        if frame is None:
            return None
        if frame.ndim != 3 or frame.shape[-1] <= 195:
            return None
        frame = crop_frame(frame, crop_info)
        sem = frame[..., 195]
        valid = np.isfinite(sem) & (sem > 0.0) & (sem != -1.0)
        labels = np.rint(sem).astype(np.int32, copy=False)
        if not valid.any():
            return None
        h, w = rgb.shape[-2:]
        if sem.shape[:2] != (h, w):
            img = Image.fromarray(labels.astype(np.int32), mode='I')
            labels = np.asarray(img.resize((w, h), resample=Image.Resampling.NEAREST), dtype=np.int32)
            valid_img = Image.fromarray(valid.astype(np.uint8) * 255, mode='L')
            valid = np.asarray(valid_img.resize((w, h), resample=Image.Resampling.NEAREST)) > 0
        overlay = rgb.detach().float().cpu().clone()
        color = np.zeros((3, h, w), dtype=np.float32)
        for label, rgb255 in COLORS.items():
            m = valid & (labels == label)
            if m.any():
                color[:, m] = (np.asarray(rgb255, dtype=np.float32)[:, None] / 127.5) - 1.0
        unknown = valid & ~np.isin(labels, list(COLORS))
        if unknown.any():
            color[:, unknown] = (np.asarray((255, 128, 0), dtype=np.float32)[:, None] / 127.5) - 1.0
        mask = torch.from_numpy(valid).unsqueeze(0)
        overlay[:, mask[0]] = torch.from_numpy(color[:, mask[0].numpy()])
        return overlay.clamp(-1, 1)
    except Exception as exc:
        print(f'[overlay-cache] skip {sample_name}: {exc}', flush=True)
        return None


def decode_batch(pipe, latents: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    # latents [B,32,1,60,60] -> [B,3,480,480]
    if latents.ndim == 5:
        latents = latents[:, :, 0]
    with torch.no_grad():
        return pipe.vae.decode(latents.to(device=device, dtype=dtype)).sample.detach().cpu().float()


def encode_batch(pipe, images: torch.Tensor, device: str, dtype: torch.dtype, cache_dtype: torch.dtype) -> torch.Tensor:
    with torch.no_grad():
        latents = pipe.vae.encode(images.to(device=device, dtype=dtype)).latent_dist.sample()
    return latents.detach().cpu().to(cache_dtype).unsqueeze(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--source-cache-dir', required=True)
    ap.add_argument('--output-cache-dir', required=True)
    ap.add_argument('--output-modality', default='195_overlay_rgb')
    ap.add_argument('--rgb-modality', default='0-2')
    ap.add_argument('--mask-modality', default='195')
    ap.add_argument('--audio-npz', action='append', default=[])
    ap.add_argument('--num-shards', type=int, default=8)
    ap.add_argument('--shard-ids', default='', help='Comma/range list of physical shard ids to build, e.g. 0,2,4-7. Empty builds all.')
    ap.add_argument('--skip-existing', action='store_true')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--wsv-cache-size', type=int, default=8)
    ap.add_argument('--progress-every', type=int, default=10)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dtype', default='bf16', choices=('bf16','fp16','fp32'))
    args = ap.parse_args()

    source = Path(args.source_cache_dir)
    out_root = Path(args.output_cache_dir)
    out_dir = out_root / 'shards' / safe_modality_name(args.output_modality)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = audio_ids([Path(p) for p in args.audio_npz]) if args.audio_npz else None
    print(f'[overlay-cache] wanted_ids={len(wanted) if wanted is not None else "all"}', flush=True)

    torch_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    pipe = Flux2KleinPipeline.from_pretrained(args.model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    pipe.vae.to(args.device).eval().requires_grad_(False)

    total = 0
    shard_ids = parse_shard_ids(args.shard_ids, args.num_shards)
    print(f'[overlay-cache] shard_ids={shard_ids}', flush=True)
    wsv_cache = WSVFrameCache(args.wsv_cache_size)
    try:
      for shard_id in shard_ids:
        out_path = shard_path(out_root, args.output_modality, shard_id, args.num_shards)
        if args.skip_existing and out_path.exists():
            print(f'[overlay-cache] skip existing {out_path}', flush=True)
            continue
        rgb_path = shard_path(source, args.rgb_modality, shard_id, args.num_shards)
        mask_path = shard_path(source, args.mask_modality, shard_id, args.num_shards)
        rgb = torch.load(rgb_path, map_location='cpu', weights_only=False)
        mask = torch.load(mask_path, map_location='cpu', weights_only=False)
        rgb_by_id = {sid: i for i, sid in enumerate(rgb['sample_ids'])}
        selected = []
        for i, sid in enumerate(mask['sample_ids']):
            if wanted is not None and sid not in wanted:
                continue
            j = rgb_by_id.get(sid, -1)
            if j >= 0:
                selected.append((sid, i, j))
        print(f'[overlay-cache] shard {shard_id}: selected={len(selected)}', flush=True)
        out_latents=[]; out_meta=defaultdict(list)
        for start in range(0, len(selected), args.batch_size):
            chunk = selected[start:start+args.batch_size]
            rgb_lat = torch.stack([rgb['latents'][j] for _,_,j in chunk], dim=0)
            rgb_img = decode_batch(pipe, rgb_lat, args.device, torch_dtype)
            overlay_imgs=[]; kept=[]
            for img, (sid, mi, rj) in zip(rgb_img, chunk):
                parsed = parse_sample_id(sid)
                if parsed is None:
                    continue
                crop_infos = mask.get('crop_infos') or rgb.get('crop_infos')
                crop_info = crop_infos[mi] if crop_infos is not None else None
                frame = wsv_cache.get(parsed[0], parsed[1])
                overlay = semantic_overlay(img, frame, sid, crop_info)
                if overlay is None:
                    continue
                overlay_imgs.append(overlay)
                kept.append((sid, mi, rj))
            if not overlay_imgs:
                continue
            encoded = encode_batch(pipe, torch.stack(overlay_imgs, dim=0), args.device, torch_dtype, torch.bfloat16)
            out_latents.append(encoded)
            for sid, mi, rj in kept:
                for key in ('sample_ids','sample_sources','video_ids','frame_keys','frame_indices','crop_infos','shard_indices','source_shard_indices'):
                    values = mask.get(key)
                    if isinstance(values, (list, tuple)):
                        out_meta[key].append(values[mi])
                out_meta['rgb_source_indices'].append(int(rj))
            if args.progress_every > 0 and (start // args.batch_size + 1) % args.progress_every == 0:
                done = start + len(chunk)
                kept_count = sum(x.shape[0] for x in out_latents)
                print(f'[overlay-cache] shard {shard_id}: batch {done}/{len(selected)} kept={kept_count}', flush=True)
        result = dict(mask)
        n = sum(x.shape[0] for x in out_latents)
        result['latents'] = torch.cat(out_latents, dim=0) if out_latents else torch.empty((0,32,1,60,60), dtype=torch.bfloat16)
        for key, vals in out_meta.items():
            result[key] = vals
        result['sample_ids'] = out_meta.get('sample_ids', [])
        result['num_samples'] = n
        result['global_num_samples'] = n
        result['modality'] = args.output_modality
        result['derived_from'] = {'rgb_modality': args.rgb_modality, 'mask_modality': args.mask_modality, 'method': 'opaque semantic color overlay on decoded RGB, re-encoded with Flux2 VAE'}
        tmp = out_path.with_suffix(out_path.suffix + '.tmp')
        torch.save(result, tmp)
        os.replace(tmp, out_path)
        total += n
        print(f'[overlay-cache] saved {out_path} samples={n}', flush=True)
    finally:
        wsv_cache.close()
    (out_root / 'README_audio_semantic_overlay.txt').write_text(
        'Derived target cache for sparse Channel_Head_000008. Original channel 195 latents are not modified.\n'
        'Each target is decoded RGB with opaque AVIS audio semantic mask colors overlaid, then re-encoded by FLUX.2 VAE.\n', encoding='utf-8')
    print(f'[overlay-cache] done total={total} output={out_root}', flush=True)

if __name__ == '__main__':
    main()
