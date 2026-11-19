import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.wsg_dataset import WSGNormalizationStore, flatten_wsg_modality_graph, parse_channel_key


def channels_from_modality_config(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    graph = data.get("wsg_v1")
    if not isinstance(graph, dict):
        raise ValueError(f"{path} must contain top-level key 'wsg_v1'")
    graph = flatten_wsg_modality_graph(graph)

    channels = []
    for modality, spec in graph.items():
        channels.extend(parse_channel_key(modality))
        for condition in (spec.get("conditions") or {}).keys():
            channels.extend(parse_channel_key(condition))
    return sorted(set(channels))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", action="append", required=True)
    parser.add_argument("--modality-config", default="configs/wsg_v1_modalities_dense.json")
    parser.add_argument("--cache", default="configs/wsg_dataset_info.json")
    parser.add_argument("--channels", default=None, help="Comma/range list such as 36-38,40. Defaults to modality config.")
    parser.add_argument("--include-rgb", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    modality_config = Path(args.modality_config)
    if not modality_config.is_absolute():
        modality_config = project_root / modality_config
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = project_root / cache_path

    if args.channels:
        channels = parse_channel_key(args.channels)
    else:
        channels = channels_from_modality_config(modality_config)
    if not args.include_rgb:
        channels = [channel for channel in channels if channel not in (0, 1, 2)]
    channels = sorted(set(channels))
    print(f"channels: {channels}")

    store = WSGNormalizationStore(cache_path, show_progress=True)
    dataset_paths = [str(Path(path).expanduser().resolve()) for path in args.dataset_path]
    for channel in channels:
        channel_range = store.get_channel_range(channel, dataset_paths)
        print(f"channel {channel}: {channel_range}")
    print(f"updated {cache_path}")


if __name__ == "__main__":
    main()
