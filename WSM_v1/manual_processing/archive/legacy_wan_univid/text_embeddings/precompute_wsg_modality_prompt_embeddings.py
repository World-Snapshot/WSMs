import argparse
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

from src.dataset.wsg_dataset import LORA_AND_MULTI_LANES_PREFIX, WSGModalityDataset, parse_channel_key
from src.models import ModelManager
from src.prompters import WanPrompter


MODALITY_ROLES = ("condition", "target")
PROMPT_FORMAT_VERSIONS = {
    "numeric": "modality_id_channels_and_role_numeric_v1",
    "semantic": "modality_id_channels_semantic_description_and_role_v2",
}


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


def _load_modalities(modality_config_path: Path, include_conditions: bool = True) -> list[str]:
    with open(modality_config_path, "r", encoding="utf-8") as f:
        raw = json.load(f).get("wsg_v1", {})
    if raw and all(str(key).startswith(LORA_AND_MULTI_LANES_PREFIX) for key in raw):
        modalities: list[str] = []
        for group in raw.values():
            for target, spec in group.items():
                if str(target).startswith("__"):
                    continue
                modalities.append(str(target))
                if include_conditions:
                    modalities.extend(str(modality) for modality in (spec.get("conditions") or {}))
        return list(dict.fromkeys(modalities))

    graph = WSGModalityDataset._load_modality_graph(modality_config_path)
    modalities = list(graph.keys())
    if include_conditions:
        for modality in list(modalities):
            modalities.extend((graph.get(modality, {}).get("conditions") or {}).keys())
    return list(dict.fromkeys(str(modality) for modality in modalities))


def _load_modality_descriptions(modality_config_path: Path) -> dict[str, str]:
    with open(modality_config_path, "r", encoding="utf-8") as f:
        raw = json.load(f).get("wsg_v1", {})
    if raw and all(str(key).startswith(LORA_AND_MULTI_LANES_PREFIX) for key in raw):
        descriptions: dict[str, str] = {}
        for group in raw.values():
            for target, spec in group.items():
                if str(target).startswith("__"):
                    continue
                description = str(spec.get("description") or "").strip()
                if description and str(target) not in descriptions:
                    descriptions[str(target)] = description
        return descriptions

    graph = WSGModalityDataset._load_modality_graph(modality_config_path)
    return {
        str(modality): str(spec.get("description") or "").strip()
        for modality, spec in graph.items()
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


def _load_existing_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu", weights_only=False)
    role_embeddings = payload.get("modality_text_embeddings_by_role")
    legacy_embeddings = payload.get("modality_text_embeddings")
    if not isinstance(role_embeddings, dict) and not isinstance(legacy_embeddings, dict):
        return {}
    return payload


def _modality_prompt(modality: str, description: str, role: str, prompt_style: str) -> str:
    if prompt_style == "semantic" and not description:
        raise ValueError(f"Missing semantic description for modality {modality}")
    if role not in MODALITY_ROLES:
        raise ValueError(f"Unsupported modality role: {role}")
    if prompt_style not in PROMPT_FORMAT_VERSIONS:
        raise ValueError(f"Unsupported modality prompt style: {prompt_style}")
    channels = ", ".join(str(channel) for channel in parse_channel_key(modality))
    if prompt_style == "numeric":
        return channels
    instruction = (
        "This modality is provided as conditioning input."
        if role == "condition"
        else "This modality is the generation target."
    )
    fields = [f"modality_id: {modality}", f"channels: {channels}", f"semantic_description: {description}"]
    fields.extend((f"role: {role}", f"instruction: {instruction}"))
    return "; ".join(fields)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wsm_v1_augustus_train_latent_1.3b.yaml")
    parser.add_argument("--latent-cache-dir", default="data/wsg_vae_latents_avis_train")
    parser.add_argument("--modality-config", default="configs/wsg_v1_modalities_small.json")
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Explicit modality IDs to encode instead of loading them from --modality-config.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer-path", default="models/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl")
    parser.add_argument("--prompt-style", default="semantic", choices=sorted(PROMPT_FORMAT_VERSIONS))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    config_path = _resolve_project_path(args.config)
    latent_cache_dir = _resolve_project_path(args.latent_cache_dir)
    modality_config_path = _resolve_project_path(args.modality_config)
    embeddings_path = latent_cache_dir / f"modality_prompt_embeddings_{args.prompt_style}.pt"
    modalities = list(dict.fromkeys(str(modality) for modality in args.modalities)) if args.modalities else _load_modalities(modality_config_path)
    descriptions = _load_modality_descriptions(modality_config_path)
    missing_descriptions = [
        modality
        for modality in modalities
        if args.prompt_style == "semantic" and not descriptions.get(modality)
    ]
    if missing_descriptions:
        raise ValueError(
            f"{modality_config_path} is missing semantic descriptions for modalities: {missing_descriptions}"
        )
    prompts_by_role = {
        role: {
            modality: _modality_prompt(modality, descriptions.get(modality, ""), role, args.prompt_style)
            for modality in modalities
        }
        for role in MODALITY_ROLES
    }
    existing_payload = _load_existing_cache(embeddings_path)
    existing_embeddings_by_role = dict(existing_payload.get("modality_text_embeddings_by_role") or {})
    existing_prompts_by_role = dict(existing_payload.get("modality_prompts_by_role") or {})
    stale_role_modalities = [
        (role, modality)
        for role in MODALITY_ROLES
        for modality in modalities
        if modality not in (existing_embeddings_by_role.get(role) or {})
        or (existing_prompts_by_role.get(role) or {}).get(modality) != prompts_by_role[role][modality]
    ]

    if args.skip_existing and not stale_role_modalities:
        print(f"skip existing modality prompt embeddings: {embeddings_path}")
        return

    config = _load_config(config_path)
    t5_path = _select_t5_path(config["model"]["params"])
    tokenizer_path = str(_resolve_project_path(args.tokenizer_path))
    prompter = _load_prompter(t5_path, tokenizer_path, args.device, torch_dtype)

    modality_text_embeddings_by_role = {
        role: {
            modality: (existing_embeddings_by_role.get(role) or {})[modality]
            for modality in modalities
            if modality in (existing_embeddings_by_role.get(role) or {})
            and (role, modality) not in stale_role_modalities
        }
        for role in MODALITY_ROLES
    }
    encode_role_modalities = stale_role_modalities if args.skip_existing else [
        (role, modality)
        for role in MODALITY_ROLES
        for modality in modalities
    ]
    for start in tqdm(
        range(0, len(encode_role_modalities), args.batch_size),
        desc="encode role-aware WSG modality prompts",
        dynamic_ncols=True,
    ):
        batch_role_modalities = encode_role_modalities[start:start + args.batch_size]
        batch_prompts = [prompts_by_role[role][modality] for role, modality in batch_role_modalities]
        embeddings = prompter.encode_prompt(batch_prompts, positive=True, device=args.device)
        embeddings = embeddings.detach().cpu().to(dtype=torch_dtype)
        for offset, (role, modality) in enumerate(batch_role_modalities):
            modality_text_embeddings_by_role[role][modality] = embeddings[offset]

    # Keep the legacy fields as target-role aliases for older cache readers.
    modality_text_embeddings = modality_text_embeddings_by_role["target"]
    prompts = prompts_by_role["target"]

    payload = {
        "config": str(config_path),
        "modality_config_path": None if args.modalities else str(modality_config_path),
        "t5_path": t5_path,
        "tokenizer_path": tokenizer_path,
        "dtype": args.dtype,
        "prompt_style": args.prompt_style,
        "prompt_format_version": PROMPT_FORMAT_VERSIONS[args.prompt_style],
        "modalities": modalities,
        "modality_roles": list(MODALITY_ROLES),
        "modality_prompts_by_role": prompts_by_role,
        "modality_text_embeddings_by_role": modality_text_embeddings_by_role,
        "modality_prompts": prompts,
        "modality_text_embeddings": modality_text_embeddings,
    }
    latent_cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = embeddings_path.with_suffix(embeddings_path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(embeddings_path)
    print(
        f"saved role-aware modality prompt embeddings: "
        f"{len(modality_text_embeddings) * len(MODALITY_ROLES)} role/modality pairs -> {embeddings_path}"
    )


if __name__ == "__main__":
    main()
