from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsm_adapters.compat import apply_legacy_smpl_compat

apply_legacy_smpl_compat()

from score_hmr import GaussianDiffusion, Trainer
import score_hmr.trainer as scorehmr_trainer_module
from score_hmr.configs import model_config
from score_hmr.datasets.image_dataset import ImageDataset
from score_hmr.models import FC
from wsm_adapters.scorehmr_bbox import FCWithBBox, GaussianDiffusionWithBBox


def make_loader(cfg, dataset_file: str, pare_file: str, batch_size: int, shuffle: bool, num_workers: int, train: bool):
    dataset = ImageDataset(
        cfg,
        dataset_file,
        "/",
        train=train,
        pare_preds_file=pare_file,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=train,
        num_workers=num_workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--train-dataset", default="data/datasets/wsv_smpl_train.npz")
    parser.add_argument("--val-dataset", default="data/datasets/wsv_smpl_val.npz")
    parser.add_argument("--train-pare", default="cache/pare/wsv_smpl_train.npz")
    parser.add_argument("--val-pare", default="cache/pare/wsv_smpl_val.npz")
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-size-val", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-freq", type=int, default=500)
    parser.add_argument("--val-freq", type=int, default=100)
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    parser.add_argument("--output-dir", default="logs")
    parser.add_argument("--clean-log-layout", action="store_true", help="Use --output-dir as this run directory for ScoreHMR outputs.")
    parser.add_argument(
        "--use-betas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train ScoreHMR on SMPL pose plus WSV betas. Use --no-use-betas for pose-only.",
    )
    parser.add_argument(
        "--use-bbox",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also diffuse normalized bbox parameters cx,cy,w,h from WSV person boxes.",
    )
    args = parser.parse_args()

    scorehmr_trainer_module.RESULTS_DIR = str(Path(args.output_dir) / "scorehmr") if args.clean_log_layout else args.output_dir
    torch.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    cfg = model_config()
    cfg.defrost()
    cfg.MODEL.USE_BETAS = bool(args.use_betas)
    cfg.MODEL.DENOISING_MODEL.IMG_FEATS = "pare"
    cfg.EXTRA.LOAD_IMG_FEATS = "pare"
    cfg.EXTRA.LOAD_PREDICTIONS = "pare"
    cfg.TRAIN.TOTAL_STEPS = int(args.total_steps)
    cfg.TRAIN.BATCH_SIZE = int(args.batch_size)
    cfg.TRAIN.BATCH_SIZE_VAL = int(args.batch_size_val)
    cfg.TRAIN.NUM_WORKERS = int(args.num_workers)
    cfg.TRAIN.LOG_FREQ = int(args.log_freq)
    cfg.TRAIN.CHECKPOINT_FREQ = int(args.checkpoint_freq)
    cfg.freeze()

    train_loader = make_loader(cfg, args.train_dataset, args.train_pare, args.batch_size, True, args.num_workers, train=True)
    val_loader = make_loader(cfg, args.val_dataset, args.val_pare, args.batch_size_val, False, args.num_workers, train=False)

    if args.use_bbox:
        model = FCWithBBox(cfg).to(device)
        diffusion = GaussianDiffusionWithBBox(cfg, model, device=device).to(device)
    else:
        model = FC(cfg).to(device)
        diffusion = GaussianDiffusion(cfg, model, device=device).to(device)
    trainer_name = "run" if args.clean_log_layout else args.name
    trainer = Trainer(
        cfg,
        trainer_name,
        diffusion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        validation_freq=args.val_freq,
    )
    trainer.train()


if __name__ == "__main__":
    main()
