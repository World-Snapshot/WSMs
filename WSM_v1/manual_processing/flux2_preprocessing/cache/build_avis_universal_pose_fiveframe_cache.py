#!/usr/bin/env python3
"""Build the AVIS five-frame RGB/Universal-Pose annotation staging cache."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import torch

ROLES = ("first", "quarter", "middle", "three_quarter", "last")

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--source-cache",type=Path,default=Path("data/wsg_flux2_frame_latents_dense_multidataset"))
    p.add_argument("--output-cache",type=Path,default=None,help="Defaults to --source-cache; Universal_Pose is an optional modality in the same latent library.")
    p.add_argument("--num-shards",type=int,default=8)
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()

def stable_shard(dataset:str,video:str,n:int)->int:
    return int.from_bytes(hashlib.sha1(f"{dataset}\x1f{video}".encode()).digest()[:8],"big")%n

def select_positions(n:int)->list[int]:
    if n<5: raise ValueError(f"Need >=5 indexed frames, found {n}")
    desired=[round((n-1)*i/4) for i in range(5)]; used=set(); out=[]
    for target in desired:
        value=next(x for x in sorted(range(n),key=lambda x:(abs(x-target),x)) if x not in used)
        used.add(value); out.append(value)
    return out

def save(payload:dict[str,Any],path:Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,tmp); tmp.replace(path)

def main():
    a=args(); src=a.source_cache.expanduser().resolve(); out=(a.output_cache.expanduser().resolve() if a.output_cache is not None else src)
    paths=sorted((src/"shards/0_2").glob("*.pt"))
    if not paths: raise FileNotFoundError(src/"shards/0_2")
    if list((out/"shards/Universal_Pose").glob("*.pt")) and not a.overwrite:
        raise FileExistsError(f"{out} already contains shards; pass --overwrite")
    groups=defaultdict(list); payloads={}
    for si,path in enumerate(paths):
        d=torch.load(path,map_location="cpu",mmap=True,weights_only=False); payloads[si]=d
        n=len(d["sample_ids"])
        fields=zip(d["sample_ids"],d.get("sample_sources") or [{}]*n,d.get("video_ids") or [""]*n,d.get("frame_indices") or [-1]*n,d.get("frame_keys") or [""]*n,d.get("crop_infos") or [None]*n)
        for ri,(sid,source,video,frame,key,crop) in enumerate(fields):
            dataset=str(source.get("dataset_name") or "")
            if not dataset.startswith("AVIS_"): continue
            groups[(dataset,str(video))].append(dict(sample_id=str(sid),dataset_name=dataset,video_id=str(video),frame_index=int(frame),frame_key=str(key),sample_source=dict(source),crop_info=crop,source_shard=si,source_row=ri))
    selected=[]
    for (dataset,video),records in sorted(groups.items()):
        records.sort(key=lambda x:(x["frame_index"],x["sample_id"]))
        for role,pos in zip(ROLES,select_positions(len(records))):
            r=dict(records[pos]); r.update(selection_role=role,selection_index=pos,video_indexed_frame_count=len(records),output_shard=r["source_shard"]); selected.append(r)
    by=defaultdict(list)
    for r in selected: by[r["output_shard"]].append(r)
    for records in by.values(): records.sort(key=lambda x:(x["dataset_name"],x["video_id"],x["frame_index"],x["sample_id"]))
    manifest=[]; total=len(selected)
    for shard in range(a.num_shards):
        records=by.get(shard,[])
        rgb=torch.stack([payloads[r["source_shard"]]["latents"][r["source_row"]].clone() for r in records])
        sources=[]
        for r in records:
            s=dict(r["sample_source"]); s.update(split=r["dataset_name"].removeprefix("AVIS_").removesuffix("_results"),universal_pose_annotation_selection=r["selection_role"],universal_pose_annotation_source="custom_head_000000_gradient_best"); sources.append(s)
        base=dict(schema_version=1,cache_purpose="AVIS five-frame Universal Pose annotation and reference-conditioned Head 1 training",source_cache=str(src),num_samples=len(records),global_num_samples=total,num_shards=a.num_shards,shard_id=shard,num_frames=1,latent_num_frames=1,data_type="bfloat16",dtype="bfloat16",cache_dtype="bfloat16",sample_ids=[r["sample_id"] for r in records],sample_sources=sources,video_ids=[r["video_id"] for r in records],frame_keys=[r["frame_key"] for r in records],frame_indices=[r["frame_index"] for r in records],crop_infos=[r["crop_info"] for r in records],selection_roles=[r["selection_role"] for r in records],source_shard_indices=[r["source_shard"] for r in records],source_rows=[r["source_row"] for r in records])
        name=f"shard_{shard:05d}_of_{a.num_shards:05d}.pt"
        save({**base,"modality":"Universal_Pose","annotation_state":"placeholder_pending_head0_inference","latents":torch.zeros_like(rgb)},out/"shards/Universal_Pose"/name)
        for oi,r in enumerate(records):
            manifest.append({k:r[k] for k in ("sample_id","dataset_name","video_id","frame_index","frame_key","selection_role","selection_index","video_indexed_frame_count","source_shard","source_row","output_shard")}|{"output_row":oi})
        print(f"[avis-pose-staging] shard={shard}/{a.num_shards} samples={len(records)}",flush=True)
    manifest.sort(key=lambda x:(x["dataset_name"],x["video_id"],x["frame_index"]))
    doc=dict(schema_version=1,selection=dict(description="Existing indexed frames nearest first, quarter, middle, three-quarter, and last positions",roles=list(ROLES),frames_per_video=5),source_cache=str(src),output_cache=str(out),videos=len(groups),samples=total,dataset_video_counts=dict(Counter(k[0] for k in groups)),dataset_sample_counts=dict(Counter(x["dataset_name"] for x in manifest)),head0_checkpoint="train_log_custom/Channel_Head_000000/20260828_024500_mp100_instance_crop_gradient_v1/ckpt/0000940/pytorch_lora_weights.safetensors",records=manifest)
    manifest_path=out/"annotations/Universal_Pose/avis_fiveframe_manifest.json"
    manifest_path.parent.mkdir(parents=True,exist_ok=True)
    manifest_path.write_text(json.dumps(doc,indent=2,ensure_ascii=False)+"\n")
    print(f"[avis-pose-staging] complete videos={len(groups)} samples={total}",flush=True)
if __name__=="__main__": main()
