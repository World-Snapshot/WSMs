from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsm_adapters.compat import apply_legacy_smpl_compat

apply_legacy_smpl_compat()

from score_hmr.configs import model_config
from score_hmr.datasets.image_dataset import ImageDataset
from score_hmr.models.model_utils import load_pare
from score_hmr.utils import recursive_to


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_cfg = model_config()
    dataset = ImageDataset(model_cfg, args.dataset, "/", train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    pare = load_pare(model_cfg.SMPL).to(device)
    pare.eval()

    betas, pose, pred_cam, pose_feats, cam_shape_feats = [], [], [], [], []
    for batch in tqdm(dataloader, desc=f"cache PARE {Path(args.dataset).name}"):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = pare(batch["img"])
        betas.append(out["pred_shape"].cpu().numpy())
        pose.append(out["pred_pose"].cpu().numpy())
        pred_cam.append(out["pred_cam"].cpu().numpy())
        pose_feats.append(out["pose_feats"].cpu().numpy())
        cam_shape_feats.append(out["cam_shape_feats"].cpu().numpy())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        pose_feats=np.concatenate(pose_feats),
        cam_shape_feats=np.concatenate(cam_shape_feats),
        pred_betas=np.concatenate(betas),
        pred_pose=np.concatenate(pose),
        pred_cam=np.concatenate(pred_cam),
    )
    print(f"Saved PARE cache -> {out_path}")


if __name__ == "__main__":
    main()
