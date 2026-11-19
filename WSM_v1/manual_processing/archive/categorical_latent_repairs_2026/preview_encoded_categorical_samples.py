#!/usr/bin/env python3
"""Compare source RGB, categorical GT, and decoded newly encoded latents."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from manual_processing.archive.categorical_latent_repairs_2026.preview_categorical_gt import rgb_image
from manual_processing.archive.categorical_latent_repairs_2026.reencode_channel_111_latents import Channel111Dataset
from manual_processing.archive.categorical_latent_repairs_2026.reencode_channel_195_latents import frame_from_open_wsv
from src.dataset.wsg_dataset import apply_resize_crop_info
from wsg_reader.categorical_visualization import (
    AVIS_CATEGORY_COLORS,
    CHANNEL_111_ID_TO_CATEGORY,
    categorical_ids_to_rgb,
    channel195_semantic_map,
    dense_channel111_category_map,
)

CACHE = Path('/common/users/sf895/WSMs/WSM_v1/data/wsg_flux2_frame_latents_dense_multidataset/shards')
STAGING = Path('/common/users/sf895/WSMs/WSM_v1/data/channel_195_rebuild_staging/channel_195_20260731_141321')
MODEL = Path('/common/users/sf895/WSM_models/black-forest-labs/FLUX.2-klein-4B')


def load(path: Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False)


def samples_111():
    selected = {}
    for shard_path in sorted((CACHE/'111').glob('*.pt')):
        payload = load(shard_path)
        for i, source in enumerate(payload['sample_sources']):
            name = str(source.get('dataset_name', ''))
            group = 'ShapeNetSem' if 'ShapeNetSem' in name else ('Articulation' if 'Articulation' in name else '')
            if group and group not in selected:
                crop = payload['crop_infos'][i]
                with np.load(source['path']) as npz:
                    raw = np.asarray(npz['data'] if 'data' in npz else npz['wsg_data'])
                frame = apply_resize_crop_info(raw, crop)
                resolver = Channel111Dataset([source], [crop], (-1.0, 417.0), 'linear')
                class_id = resolver._category_id(frame[...,111], source['path'])
                labels = dense_channel111_category_map(frame, require_alpha=False, category_id=class_id, allow_category_mask_fallback=True)
                selected[group] = {
                    'rgb': rgb_image(frame),
                    'gt': categorical_ids_to_rgb(labels),
                    'latent': payload['latents'][i].clone(),
                    'title': f'111 encoded | {group} | ID {class_id}: {CHANNEL_111_ID_TO_CATEGORY[class_id]}',
                }
            if len(selected)==2:
                return [selected['ShapeNetSem'], selected['Articulation']]
    raise RuntimeError('Could not locate two channel-111 samples')


def sample_195_from_checkpoint(worker: int, source_index: int):
    manifest = json.load((STAGING/'manifests'/f'worker_{worker:02d}.json').open())
    source = manifest['sources'][source_index]
    checkpoint = load(STAGING/'parts'/f'worker_{worker:02d}_checkpoints'/f'source_{source_index:05d}.pt')
    for shard, entry in checkpoint['entries'].items():
        if not entry['positions'].numel():
            continue
        row = int(entry['positions'][0])
        record = next(item for item in source['records'] if item['shard']==shard and int(item['row'])==row)
        with np.load(source['path'], allow_pickle=False) as npz:
            frame = frame_from_open_wsv(npz, source['path'], record['frame_key'])
        frame = apply_resize_crop_info(frame, record['crop'])
        labels = channel195_semantic_map(frame)
        ids = sorted(int(v) for v in np.unique(labels) if v != -1)
        return {
            'rgb': rgb_image(frame),
            'gt': categorical_ids_to_rgb(labels, AVIS_CATEGORY_COLORS),
            'latent': entry['latents'][0].clone(),
            'title': f"195 encoded | worker {worker} source {source_index} | {record['frame_key']} | IDs {ids}",
        }
    raise RuntimeError(f'Checkpoint has no encoded rows: worker {worker} source {source_index}')


def decode(vae, latent, device, dtype):
    if latent.ndim == 4:
        latent = latent[:,0]
    with torch.no_grad():
        decoded = vae.decode(latent.unsqueeze(0).to(device=device, dtype=dtype)).sample[0]
    return (((decoded[:3].float().cpu()+1.0)*127.5).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8))


def labeled(image, label, size=480):
    im=Image.fromarray(image).resize((size,size), Image.Resampling.NEAREST if label=='Discrete GT' else Image.Resampling.BILINEAR)
    draw=ImageDraw.Draw(im); font=ImageFont.load_default(size=17)
    draw.rectangle((0,0,150,27),fill=(0,0,0)); draw.text((7,5),label,fill=(255,255,255),font=font)
    return im


def render(rows, output):
    tile=480; margin=18; title_h=42; width=tile*3+margin*4; height=margin+(title_h+tile+margin)*len(rows)
    canvas=Image.new('RGB',(width,height),(245,247,250)); draw=ImageDraw.Draw(canvas); font=ImageFont.load_default(size=18)
    for index,row in enumerate(rows):
        y=margin+index*(title_h+tile+margin); draw.text((margin,y),row['title'],fill=(17,24,39),font=font)
        iy=y+title_h
        for col,(arr,label) in enumerate(((row['rgb'],'Source RGB'),(row['gt'],'Discrete GT'),(row['decoded'],'Decoded latent'))):
            canvas.paste(labeled(arr,label,tile),(margin+col*(tile+margin),iy))
    output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output,optimize=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,default=ROOT/'asset/categorical_encoded_samples_111_195.png'); ap.add_argument('--device',default='cuda:0'); args=ap.parse_args()
    rows=samples_111()
    # Use checkpoints already persisted before the final resume.
    rows.extend([sample_195_from_checkpoint(0,0),sample_195_from_checkpoint(1,1)])
    dtype=torch.bfloat16
    vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=dtype)
    if hasattr(vae.config,'force_upcast'): vae.config.force_upcast=False
    vae=vae.to(device=args.device,dtype=dtype).eval()
    for row in rows: row['decoded']=decode(vae,row['latent'],args.device,dtype)
    render(rows,args.output); print(args.output.resolve())

if __name__=='__main__': main()
