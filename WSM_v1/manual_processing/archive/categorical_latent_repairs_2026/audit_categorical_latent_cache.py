#!/usr/bin/env python3
"""Audit actual categorical latent caches against aligned source GT."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from manual_processing.archive.categorical_latent_repairs_2026.preview_categorical_gt import rgb_image
from manual_processing.archive.categorical_latent_repairs_2026.reencode_channel_111_latents import Channel111Dataset
from manual_processing.archive.categorical_latent_repairs_2026.reencode_channel_195_latents import frame_from_open_wsv, frame_key
from src.dataset.wsg_dataset import apply_resize_crop_info
from wsg_reader.categorical_visualization import (AVIS_CATEGORY_COLORS,CHANNEL_111_COLORS,CHANNEL_111_ID_TO_CATEGORY,categorical_ids_to_rgb,categorical_rgb_to_ids,channel195_semantic_map,dense_channel111_category_map)

CACHE=Path('/common/users/sf895/WSMs/WSM_v1/data/wsg_flux2_frame_latents_dense_multidataset/shards')
OLD_111=Path('/common/users/sf895/WSMs/WSM_v1/data/wsg_flux2_frame_latents_dense_multidataset/backups/111_before_range_fix_20260731_045646')
OLD_195=Path('/common/users/sf895/WSMs/WSM_v1/data/wsg_flux2_frame_latents_dense_multidataset/backups/195_before_discrete_rgb_20260731_185420')
MODEL=Path('/common/users/sf895/WSM_models/black-forest-labs/FLUX.2-klein-4B')

def load(p): return torch.load(p,map_location='cpu',weights_only=False)
def dominant(labels):
 v=np.rint(labels[labels!=-1]).astype(np.int32)
 if not v.size:return -1
 x,c=np.unique(v,return_counts=True);return int(x[np.argmax(c)])

def collect111(count):
 out=[]; seen=set(); groups={'ShapeNetSem':0,'Articulation':0}; resolver=None
 for sp in sorted((CACHE/'111').glob('*.pt')):
  d=load(sp)
  for i,s in enumerate(d['sample_sources']):
   name=str(s.get('dataset_name','')); group='ShapeNetSem' if 'ShapeNetSem' in name else ('Articulation' if 'Articulation' in name else '')
   if not group or groups[group]>=count//2: continue
   crop=d['crop_infos'][i]
   with np.load(s['path']) as z: raw=np.asarray(z['data'] if 'data' in z else z['wsg_data'])
   frame=apply_resize_crop_info(raw,crop)
   if resolver is None: resolver=Channel111Dataset([s],[crop],(-1.,417.),'linear')
   cid=resolver._category_id(frame[...,111],s['path'])
   if cid in seen: continue
   labels=dense_channel111_category_map(frame,require_alpha=False,category_id=cid,allow_category_mask_fallback=True)
   out.append({'channel':111,'rgb':rgb_image(frame),'labels':labels,'gt':categorical_ids_to_rgb(labels,CHANNEL_111_COLORS),'latent':d['latents'][i].clone(),'sample_id':d['sample_ids'][i],'cache_shard':sp.name,'title':f'{group} | ID {cid}: {CHANNEL_111_ID_TO_CATEGORY[cid]}'})
   seen.add(cid);groups[group]+=1
   if len(out)>=count:return out
 raise RuntimeError(f'Only found {len(out)} channel-111 audit samples')

def collect195(count):
 out=[];seen_classes=set();seen_sources=set()
 for sp in sorted((CACHE/'195').glob('*.pt')):
  d=load(sp)
  for i,s in enumerate(d['sample_sources']):
   path=str(s.get('path',''))
   if not path or path in seen_sources:continue
   seen_sources.add(path); key=str(d['frame_keys'][i]); crop=d['crop_infos'][i]
   try:
    with np.load(path,allow_pickle=False) as z: frame=frame_from_open_wsv(z,path,key)
   except (KeyError,OSError,ValueError):continue
   frame=apply_resize_crop_info(frame,crop);labels=channel195_semantic_map(frame);cid=dominant(labels)
   if cid<0:continue
   ids=sorted(int(v) for v in np.unique(labels) if v!=-1)
   out.append({'channel':195,'rgb':rgb_image(frame),'labels':labels,'gt':categorical_ids_to_rgb(labels,AVIS_CATEGORY_COLORS),'latent':d['latents'][i].clone(),'sample_id':d['sample_ids'][i],'title':f'AVIS | {key} | IDs {ids}'})
   seen_classes.add(cid)
   if len(out)>=count:return out
 raise RuntimeError(f'Only found {len(out)} channel-195 audit samples with distinct dominant classes')

def attach_latents_by_sample_id(items, cache_dir, source_field, target_field, required=True):
 wanted={str(x['sample_id']):x for x in items};found=set()
 for sp in sorted(cache_dir.glob('*.pt')):
  d=load(sp)
  for i,sid in enumerate(d['sample_ids']):
   sid=str(sid)
   if sid in wanted:
    wanted[sid][target_field]=d[source_field][i].clone();found.add(sid)
  del d
  if len(found)==len(wanted):break
 missing=sorted(set(wanted)-found)
 if missing and required:raise RuntimeError(f'{cache_dir} is missing {len(missing)} selected sample IDs: {missing}')
 return missing

def attach_latents_from_corresponding_shard(items, cache_dir, target_field, required=False):
 missing=[]
 groups={}
 for item in items:groups.setdefault(item['cache_shard'],[]).append(item)
 for shard_name,group in groups.items():
  path=cache_dir/shard_name
  if not path.is_file():
   missing.extend(str(x['sample_id']) for x in group);continue
  d=load(path);lookup={str(sid):i for i,sid in enumerate(d['sample_ids'])}
  for item in group:
   pos=lookup.get(str(item['sample_id']))
   if pos is None:missing.append(str(item['sample_id']))
   else:item[target_field]=d['latents'][pos].clone()
  del d
 if missing and required:raise RuntimeError(f'{cache_dir} is missing {len(missing)} selected sample IDs: {missing}')
 return missing

def decode(vae,items,device,dtype):
 jobs=[]
 for item in items:
  jobs.append((item,'latent','decoded'))
  if 'old_latent' in item:jobs.append((item,'old_latent','old_decoded'))
  if 'rgb_latent' in item:jobs.append((item,'rgb_latent','rgb_decoded'))
  if 'alpha_latent' in item:jobs.append((item,'alpha_latent','alpha_decoded'))
  for channel in ('126','128','131','133','136'):
   source=f'physical_{channel}_latent'
   if source in item:jobs.append((item,source,f'physical_{channel}_decoded'))
 for start in range(0,len(jobs),4):
  batch=jobs[start:start+4];xs=[]
  for item,source,_target in batch:
   x=item[source];xs.append(x[:,0] if x.ndim==4 else x)
  with torch.no_grad(): y=vae.decode(torch.stack(xs).to(device=device,dtype=dtype)).sample
  for (item,_source,target),z in zip(batch,y):item[target]=(((z[:3].float().cpu()+1)*127.5).clamp(0,255).permute(1,2,0).numpy().astype(np.uint8))

def metric(item):
 colors=CHANNEL_111_COLORS if item['channel']==111 else AVIS_CATEGORY_COLORS
 pred=categorical_rgb_to_ids(item['decoded'],colors);gt=item['labels'];gf=gt!=-1;pf=pred!=-1
 inter=np.count_nonzero(gf&pf);union=np.count_nonzero(gf|pf)
 iou=inter/max(union,1);acc=np.count_nonzero((pred==gt)&gf)/max(np.count_nonzero(gf),1);fpr=np.count_nonzero(pf&~gf)/max(np.count_nonzero(~gf),1)
 return {'foreground_iou':iou,'foreground_class_accuracy':acc,'background_false_positive_rate':fpr,'gt_foreground_ratio':float(gf.mean()),'pred_foreground_ratio':float(pf.mean())}

def tile(arr,label,size=320,nearest=False):
 im=Image.fromarray(arr).resize((size,size),Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR);dr=ImageDraw.Draw(im);font=ImageFont.load_default(size=15);dr.rectangle((0,0,175,24),fill='black');dr.text((5,4),label,fill='white',font=font);return im

def sheet(items,path):
 size=320;gap=14;head=54;cols=(9 if items and items[0]['channel']==111 and 'old_decoded' in items[0] else (4 if items and 'old_decoded' in items[0] else 3));w=size*cols+gap*(cols+1);h=gap+len(items)*(head+size+gap);im=Image.new('RGB',(w,h),(245,247,250));dr=ImageDraw.Draw(im);font=ImageFont.load_default(size=16)
 for n,x in enumerate(items):
  y=gap+n*(head+size+gap);m=x['metrics'];dr.text((gap,y),f"{n+1}. {x['title']} | IoU {m['foreground_iou']:.3f} | class {m['foreground_class_accuracy']:.3f} | bg-FP {m['background_false_positive_rate']:.3f}",fill=(17,24,39),font=font);iy=y+head
  if 'old_decoded' in x and x['channel']==111:
   alpha=x.get('alpha_decoded')
   if alpha is None:alpha=np.zeros_like(x['gt'])
   cells=[(x['rgb_decoded'],'0-2 RGB cache',False),(alpha,'3 Alpha cache' if 'alpha_decoded' in x else '3 Alpha cache MISSING',False),(x['old_decoded'],'Old 111 cache',False),(x['decoded'],'New 111 cache',False)]
   for channel in ('126','128','131','133','136'):
    value=x.get(f'physical_{channel}_decoded')
    cells.append((value if value is not None else np.zeros_like(x['gt']),f'{channel} cache' if value is not None else f'{channel} cache MISSING',False))
  elif 'old_decoded' in x:
   cells=((x['rgb_decoded'],'0-2 cache decoded',False),(x['gt'],'WSV 195 GT',True),(x['old_decoded'],'Old 195 cache',False),(x['decoded'],'New 195 cache',False))
  else:
   cells=((x['rgb'],'Source RGB',False),(x['gt'],'Discrete GT',True),(x['decoded'],'Decoded latent',False))
  for c,(a,l,q) in enumerate(cells):im.paste(tile(a,l,size,q),(gap+c*(size+gap),iy))
 path.parent.mkdir(parents=True,exist_ok=True);im.save(path,optimize=True)

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--count',type=int,default=8)
 ap.add_argument('--out-dir',type=Path,default=ROOT/'asset/categorical_cache_audit_8')
 ap.add_argument('--device',default='cuda:0')
 ap.add_argument('--only-channel',choices=('111','195','both'),default='both')
 a=ap.parse_args()
 items111=collect111(a.count) if a.only_channel in ('111','both') else []
 items195=collect195(a.count) if a.only_channel in ('195','both') else []
 if items111:
  attach_latents_from_corresponding_shard(items111,CACHE/'0_2','rgb_latent',required=True)
  attach_latents_from_corresponding_shard(items111,CACHE/'3','alpha_latent')
  attach_latents_from_corresponding_shard(items111,OLD_111,'old_latent',required=True)
  for channel in ('126','128','131','133','136'):
   attach_latents_from_corresponding_shard(items111,CACHE/channel,f'physical_{channel}_latent')
 if items195:
  attach_latents_by_sample_id(items195,CACHE/'0_2','latents','rgb_latent')
  attach_latents_by_sample_id(items195,OLD_195,'latents','old_latent')
 dtype=torch.bfloat16
 vae=AutoencoderKLFlux2.from_pretrained(str(MODEL),subfolder='vae',torch_dtype=dtype)
 if hasattr(vae.config,'force_upcast'):vae.config.force_upcast=False
 vae=vae.to(a.device,dtype=dtype).eval();decode(vae,items111+items195,a.device,dtype)
 records=[]
 for x in items111+items195:x['metrics']=metric(x);records.append({'channel':x['channel'],'sample_id':x['sample_id'],'title':x['title'],**x['metrics']})
 if items111:sheet(items111,a.out_dir/'channel_111_old_vs_new_8.png')
 if items195:sheet(items195,a.out_dir/'channel_195_audit_8.png')
 (a.out_dir/'metrics.json').write_text(json.dumps(records,indent=2)+'\n')
 print(a.out_dir.resolve())
if __name__=='__main__':main()
