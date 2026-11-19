import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset.wsg_dataset import WSGModalityDataset
from src.models import ModelManager
from src.prompters import WanPrompter


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return Path(project_root) / path


def _load_config(config_path: str | Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_t5_path(model_params: dict) -> str:
    model_paths = model_params.get("model_paths")
    if model_paths is None:
        raise ValueError("model.params.model_paths is required to locate Wan text encoder")
    paths = json.loads(model_paths) if isinstance(model_paths, str) else model_paths
    for path in paths:
        path_str = str(path)
        if "t5" in path_str.lower() or "umt5" in path_str.lower():
            return str(_resolve_project_path(path_str))
    raise ValueError(f"Could not find a T5 path in model_paths: {paths}")


def _build_dataset(config: dict) -> WSGModalityDataset:
    dataset_cfg = config["dataset"]
    if dataset_cfg["name"] != "WSGModalityDataset":
        raise ValueError("precompute_wsg_text_embeddings.py expects dataset.name = WSGModalityDataset")
    params = dict(dataset_cfg["params"])
    for key in ("modality_config_path", "normalization_cache_path"):
        if key in params and params[key] is not None:
            params[key] = str(_resolve_project_path(params[key]))
    return WSGModalityDataset(**params)


def _load_prompts(config_path: Path, latent_cache_dir: Path, config: dict) -> dict:
    prompts_path = latent_cache_dir / "prompts.pt"
    if prompts_path.exists():
        return torch.load(prompts_path, map_location="cpu", weights_only=False)

    dataset = _build_dataset(config)
    sample_ids = [sample["sample_id"] for sample in dataset.samples]
    prompts = [sample.get("prompt", "") for sample in dataset.samples]
    payload = {
        "config": str(config_path),
        "num_samples": len(sample_ids),
        "global_num_samples": len(sample_ids),
        "sample_ids": sample_ids,
        "prompts": prompts,
    }
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, prompts_path)
    return payload


def _filter_prompts_to_latent_samples(prompt_payload: dict, latent_cache_dir: Path) -> dict:
    available_sample_ids = set()
    for path in sorted((latent_cache_dir / "shards").glob("*/shard_*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        available_sample_ids.update(str(sample_id) for sample_id in payload.get("sample_ids", []))
        del payload
    if not available_sample_ids:
        raise RuntimeError(f"No latent shard sample_ids found under {latent_cache_dir / 'shards'}")

    sample_ids = list(prompt_payload["sample_ids"])
    prompts = list(prompt_payload["prompts"])
    filtered = [(sample_id, prompt) for sample_id, prompt in zip(sample_ids, prompts) if str(sample_id) in available_sample_ids]
    missing = sorted(available_sample_ids - {str(sample_id) for sample_id, _ in filtered})
    if missing:
        raise ValueError(f"Prompt cache is missing latent sample_ids: {missing[:10]}")
    return {
        **prompt_payload,
        "num_samples": len(filtered),
        "sample_ids": [sample_id for sample_id, _ in filtered],
        "prompts": [prompt for _, prompt in filtered],
        "filtered_to_latent_samples": True,
    }


def _load_prompter(t5_path: str, tokenizer_path: str, device: str, torch_dtype: torch.dtype) -> WanPrompter:
    manager = ModelManager(torch_dtype=torch_dtype, device=device)
    manager.load_model(t5_path, model_names=["wan_video_text_encoder"], device=device, torch_dtype=torch_dtype)
    text_encoder = manager.fetch_model("wan_video_text_encoder")
    if text_encoder is None:
        raise RuntimeError(f"Failed to load wan_video_text_encoder from {t5_path}")
    text_encoder.eval()
    prompter = WanPrompter(tokenizer_path=tokenizer_path)
    prompter.fetch_models(text_encoder)
    return prompter


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wsm_v1_augustus_train.yaml")
    parser.add_argument("--latent-cache-dir", default="data/wsg_vae_latents_avis_train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--tokenizer-path", default="models/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--filter-to-latent-samples",
        action="store_true",
        help="Encode only prompts whose sample_ids are present in latent shards.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard-id < num-shards")

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    config_path = _resolve_project_path(args.config)
    latent_cache_dir = _resolve_project_path(args.latent_cache_dir)
    if args.num_shards == 1:
        embeddings_path = latent_cache_dir / "text_embeddings.pt"
    else:
        embeddings_path = (
            latent_cache_dir
            / "text_embeddings"
            / f"shard_{args.shard_id:05d}_of_{args.num_shards:05d}.pt"
        )
    config = _load_config(config_path)
    prompt_payload = _load_prompts(config_path, latent_cache_dir, config)
    if args.filter_to_latent_samples:
        prompt_payload = _filter_prompts_to_latent_samples(prompt_payload, latent_cache_dir)
    prompts = list(prompt_payload["prompts"])
    sample_ids = list(prompt_payload["sample_ids"])
    shard_positions = [
        position
        for position, sample_id in enumerate(sample_ids)
        if int.from_bytes(hashlib.sha1(str(sample_id).encode("utf-8")).digest()[:8], "big") % args.num_shards == args.shard_id
    ]
    prompts = [prompts[position] for position in shard_positions]
    sample_ids = [sample_ids[position] for position in shard_positions]
    unique_prompts = list(dict.fromkeys(prompts))

    existing_payload = {}
    if args.skip_existing and embeddings_path.exists():
        existing_payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
        existing_sample_ids = set(str(sample_id) for sample_id in existing_payload.get("sample_ids", []))
        if all(str(sample_id) in existing_sample_ids for sample_id in sample_ids):
            print(f"skip existing text embeddings: {embeddings_path}")
            return
    existing_unique_prompts = list(existing_payload.get("unique_prompts", []))
    existing_embeddings = existing_payload.get("text_embeddings")
    existing_prompt_to_index = {prompt: index for index, prompt in enumerate(existing_unique_prompts)}
    missing_unique_prompts = [prompt for prompt in unique_prompts if prompt not in existing_prompt_to_index]

    t5_path = _select_t5_path(config["model"]["params"])
    tokenizer_path = str(_resolve_project_path(args.tokenizer_path))
    prompter = _load_prompter(t5_path, tokenizer_path, args.device, torch_dtype)

    encoded_batches = []
    for start in tqdm(range(0, len(missing_unique_prompts), args.batch_size), desc="encode WSG prompts", dynamic_ncols=True):
        batch_prompts = missing_unique_prompts[start:start + args.batch_size]
        emb = prompter.encode_prompt(batch_prompts, positive=True, device=args.device)
        encoded_batches.append(emb.detach().cpu().to(dtype=torch_dtype))

    embedding_parts = []
    if existing_embeddings is not None:
        embedding_parts.append(existing_embeddings.to(dtype=torch_dtype))
    if encoded_batches:
        embedding_parts.append(torch.cat(encoded_batches, dim=0))
    if not embedding_parts:
        raise RuntimeError("No text embeddings were loaded or encoded")
    text_embeddings = torch.cat(embedding_parts, dim=0)
    all_unique_prompts = existing_unique_prompts + missing_unique_prompts
    prompt_to_index = {prompt: index for index, prompt in enumerate(all_unique_prompts)}
    prompt_indices = torch.tensor([prompt_to_index[prompt] for prompt in prompts], dtype=torch.long)
    payload = {
        "config": str(config_path),
        "t5_path": t5_path,
        "tokenizer_path": tokenizer_path,
        "dtype": args.dtype,
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "num_samples": len(prompts),
        "sample_ids": sample_ids,
        "prompts": prompts,
        "unique_prompts": all_unique_prompts,
        "prompt_indices": prompt_indices,
        "text_embeddings": text_embeddings,
        "filtered_to_latent_samples": bool(args.filter_to_latent_samples),
    }
    tmp_path = embeddings_path.with_suffix(embeddings_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(embeddings_path)
    print(f"saved text embeddings: {tuple(text_embeddings.shape)} -> {embeddings_path}")


if __name__ == "__main__":
    main()
