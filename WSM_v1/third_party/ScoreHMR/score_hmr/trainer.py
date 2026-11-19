import os
from tqdm import tqdm
from typing import Dict
import numpy as np
from yacs.config import CfgNode
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from ema_pytorch import EMA
from constants import CHECKPOINT_DIR, RESULTS_DIR
from score_hmr.models import SMPL
from score_hmr.utils.train_utils import cycle, filter_based_on_pose
from score_hmr.utils import recursive_to, prepare_smpl_params, StandarizeImageFeatures
from score_hmr.utils.skeleton_renderer import SkeletonRenderer



class Trainer(object):
    def __init__(
        self,
        cfg: CfgNode,
        name: str,
        diffusion_model: torch.nn.Module,
        num_samples: int = 1,
        train_dataloader: torch.utils.data.DataLoader = None,
        val_dataloader: torch.utils.data.DataLoader = None,
        validation_freq: int = None,
        validation_seed: int = 0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.name = name
        self.model = diffusion_model
        self.device = diffusion_model.device
        self.use_betas = diffusion_model.use_betas
        self.use_bbox = getattr(diffusion_model, "use_bbox", False)
        self.bbox_dim = int(getattr(diffusion_model, "bbox_dim", 0)) if self.use_bbox else 0
        self.num_samples = num_samples
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.step = 0
        self.train_num_steps = self.cfg.TRAIN.TOTAL_STEPS
        self.validation_freq = int(validation_freq or self.cfg.TRAIN.LOG_FREQ)
        self.validation_seed = int(validation_seed)
        self.best_val_loss = float("inf")

        # Set up optimizer.
        self.opt = Adam(diffusion_model.parameters(), lr=self.cfg.TRAIN.LR, betas=(0.9, 0.99))

        # Keep an EMA copy of the model weights.
        self.ema = EMA(diffusion_model, beta=0.995, update_every=10)

        # Set up directories for tensorboard and checkpoints.
        self.ckpt_folder = os.path.join(os.path.join(RESULTS_DIR, "checkpoints", name))
        tb_folder = os.path.join(os.path.join(RESULTS_DIR, "tensorboard", name))
        os.makedirs(self.ckpt_folder, exist_ok=True)
        os.makedirs(tb_folder, exist_ok=True)
        self.tb_writter = SummaryWriter(tb_folder)
        self.metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
        if not os.path.isfile(self.metrics_path):
            with open(self.metrics_path, "w") as metrics_file:
                metrics_file.write("step,val_loss,best_val_loss\n")

        # Set up renderer for TB visualizations.
        self.renderer = SkeletonRenderer(self.cfg)

        # Set up class to standarize image features.
        self.img_feat_standarizer = StandarizeImageFeatures(
            backbone=self.cfg.MODEL.DENOISING_MODEL.IMG_FEATS,
            use_betas=self.use_betas,
            device=self.device,
        )

        # Set up SMPL model.
        smpl_cfg = {k.lower(): v for k, v in dict(cfg.SMPL).items()}
        self.smpl = SMPL(**smpl_cfg).to(self.device)

    def save(self, milestone: int, filename: str = None, val_loss: float = None) -> None:
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        if val_loss is not None:
            data["val_loss"] = float(val_loss)
        checkpoint_name = filename or f"model-step_{self.step:07d}.pt"
        torch.save(data, os.path.join(self.ckpt_folder, checkpoint_name))

    def load(self, milestone: int, name: str = 'score_hmr', use_default: bool = False) -> None:
        ckpt_dir = f"{CHECKPOINT_DIR}/{name}" if use_default else self.ckpt_folder
        candidates = (
            os.path.join(ckpt_dir, f"model-{milestone}.pt"),
            os.path.join(ckpt_dir, f"model-step_{milestone:07d}.pt"),
            os.path.join(ckpt_dir, f"model-step_{milestone * int(self.cfg.TRAIN.LOG_FREQ):07d}.pt"),
        )
        checkpoint = next((path for path in candidates if os.path.isfile(path)), candidates[0])
        data = torch.load(checkpoint, map_location=self.device)
        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])
        self.best_val_loss = float(data.get("best_val_loss", self.best_val_loss))

    def _reduce_diffusion_loss(self, simple_loss_raw: torch.Tensor, batch: Dict) -> torch.Tensor:
        if self.use_betas:
            has_gt_pose = batch["has_smpl_params"]["body_pose"].bool()
            has_gt_shape = batch["has_smpl_params"]["betas"].bool()
            pose_end = simple_loss_raw.shape[1] - 10 - self.bbox_dim
            pose_loss = simple_loss_raw[:, :pose_end][has_gt_pose].reshape(-1).mean()
            shape_loss = simple_loss_raw[:, pose_end:pose_end + 10][has_gt_shape].reshape(-1).mean()
            loss = pose_loss + shape_loss
        else:
            has_gt_pose = batch["has_smpl_params"]["body_pose"].bool()
            pose_end = simple_loss_raw.shape[1] - self.bbox_dim
            loss = simple_loss_raw[:, :pose_end][has_gt_pose].reshape(-1).mean()
        if self.use_bbox:
            has_bbox = batch.get("has_bbox", torch.ones(simple_loss_raw.shape[0], device=self.device)).bool()
            if torch.any(has_bbox):
                loss = loss + simple_loss_raw[:, -self.bbox_dim:][has_bbox].reshape(-1).mean()
        return loss

    def compute_validation_loss(self) -> float:
        """Evaluate EMA diffusion loss with repeatable timesteps and noise."""
        if self.val_dataloader is None:
            return float("inf")
        ema_model = self.ema.ema_model
        was_training = ema_model.training
        ema_model.eval()
        weighted_loss = 0.0
        sample_count = 0
        cuda_devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(self.validation_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed(self.validation_seed)
                with torch.no_grad():
                    for batch in self.val_dataloader:
                        batch = filter_based_on_pose(recursive_to(batch, self.device))
                        simple_loss_raw, _, _ = ema_model(batch)
                        loss = self._reduce_diffusion_loss(simple_loss_raw, batch)
                        batch_size = int(batch["img_feats"].shape[0])
                        weighted_loss += float(loss.detach().cpu()) * batch_size
                        sample_count += batch_size
        finally:
            ema_model.train(was_training)
        return weighted_loss / sample_count if sample_count else float("inf")

    def _record_validation(self, val_loss: float) -> None:
        self.tb_writter.add_scalar("val/loss", val_loss, self.step)
        with open(self.metrics_path, "a") as metrics_file:
            metrics_file.write(f"{self.step},{val_loss:.10e},{self.best_val_loss:.10e}\n")

    def train(self) -> None:
        self.dl = cycle(self.train_dataloader)
        self.val_dl = cycle(self.val_dataloader)

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                batch = recursive_to(next(self.dl), self.device)

                # Keep entries with SMPL pose annotations.
                batch = filter_based_on_pose(batch)

                self.opt.zero_grad()

                # Compute training loss.
                simple_loss_raw, _, _ = self.model(batch)
                simple_loss = self._reduce_diffusion_loss(simple_loss_raw, batch)
                loss = simple_loss

                # Optimizer step.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                pbar.set_description(f"loss: {loss.item():.4f}")
                self.step += 1
                self.ema.to(self.device)
                self.ema.update()

                if self.step != 0 and self.step % self.validation_freq == 0:
                    val_loss = self.compute_validation_loss()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save(self.step // self.validation_freq, filename="model-best.pt", val_loss=val_loss)
                        print(f"[smpl] step={self.step} new_best_val_loss={val_loss:.10e}", flush=True)
                    else:
                        print(
                            f"[smpl] step={self.step} val_loss={val_loss:.10e} "
                            f"best_val_loss={self.best_val_loss:.10e}",
                            flush=True,
                        )
                    self._record_validation(val_loss)

                if self.step != 0 and self.step % self.cfg.TRAIN.LOG_FREQ == 0:
                    output_train = {
                        "losses": {"p_loss": simple_loss.detach()}
                    }

                    ## Validation

                    # Get validation batch.
                    batch_val = recursive_to(next(self.val_dl), self.device)
                    batch_size_val = batch_val["img_feats"].size(0)
                    num_samples_val = self.cfg.TRAIN.NUM_VAL_SAMPLES
                    bs_times_samples_val = batch_size_val * num_samples_val

                    cond_feats_val = batch_val["img_feats"]
                    cond_feats_val = self.img_feat_standarizer(cond_feats_val)
                    cond_feats_val = (
                        cond_feats_val.view(batch_size_val, 1, -1)
                        .repeat(1, num_samples_val, 1)
                        .reshape(bs_times_samples_val, -1)
                    )

                    # Sample from the diffusion model.
                    self.ema.ema_model.eval()
                    with torch.no_grad():
                        sampling_output = self.ema.ema_model.sample(
                            batch_val,
                            cond_feats_val,
                            bs_times_samples_val,
                        )
                        x_0 = sampling_output['x_0']
                        if self.use_bbox:
                            pred_bbox_params = (
                                x_0[:, -self.bbox_dim:]
                                .reshape(batch_size_val, num_samples_val, self.bbox_dim)
                                .clamp(-1.0, 1.0)
                                .add(1.0)
                                .mul(0.5)
                            )
                            x_0 = x_0[:, :-self.bbox_dim]
                    self.ema.ema_model.train()

                    # Run the SMPL model.
                    pred_smpl_params = prepare_smpl_params(
                        x_0,
                        num_samples_val,
                        use_betas=self.use_betas,
                        betas_min = self.model.betas_min if self.use_betas else None,
                        betas_max = self.model.betas_max if self.use_betas else None,
                        pred_betas=batch_val["pred_betas"] if not self.use_betas else None,
                    )
                    pred_smpl_params["global_orient"] = pred_smpl_params["global_orient"].reshape(batch_size_val * num_samples_val, -1, 3, 3)
                    pred_smpl_params["body_pose"] = pred_smpl_params["body_pose"].reshape(batch_size_val * num_samples_val, -1, 3, 3)
                    pred_smpl_params["betas"] = pred_smpl_params["betas"].reshape(batch_size_val * num_samples_val, -1)
                    smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, pose2rot=False )
                    pred_keypoints_3d = smpl_output.joints

                    pred_cam = batch_val["pred_cam"].unsqueeze(1).repeat(1, num_samples_val, 1)
                    focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size_val, num_samples_val, 2, device=self.device, dtype=pred_cam.dtype)
                    pred_cam_t = torch.stack(
                        [
                            pred_cam[:, :, 1],
                            pred_cam[:, :, 2],
                            2 * focal_length[:, :, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, :, 0] + 1e-9),
                        ],
                        dim=-1,
                    )

                    output_val = {
                        "pred_cam_t": pred_cam_t,
                        "pred_keypoints_3d": pred_keypoints_3d.reshape(batch_size_val, num_samples_val, -1, 3),
                    }
                    if self.use_bbox:
                        output_val["pred_bbox_params"] = pred_bbox_params

                    # Log to tensorboard.
                    milestone = self.step // self.cfg.TRAIN.LOG_FREQ
                    self.visuals_logging(batch, output_train, milestone, train=True)
                    self.visuals_logging(batch_val, output_val, milestone, train=False)

                # Save periodic checkpoints independently from visualization logging.
                if self.step != 0 and self.step % self.cfg.TRAIN.CHECKPOINT_FREQ == 0:
                    self.save(self.step // self.cfg.TRAIN.CHECKPOINT_FREQ)

                pbar.update(1)

    def visuals_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True) -> None:
        """
        Function to perform tensorboard logging.
        Args:
            batch: Dictionary with the dataset labels.
            output: Dictionary with model predictions.
            step_count: training iteration iterations number.
            train: flag to indicate train or val mode.
        """
        if train:
            losses = output["losses"]
            for loss_name, val in losses.items():
                self.tb_writter.add_scalar("train/" + loss_name, val.detach().item(), step_count)
            return

        images = batch["img"]
        batch_size = images.size(0)
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1)
        images = 255 * images.permute(0, 2, 3, 1).cpu().numpy()

        num_images = min(batch_size, 4)
        num_samples_per_image = min(self.cfg.TRAIN.NUM_VAL_SAMPLES, 8)

        gt_keypoints_2d = batch["keypoints_2d"]
        gt_keypoints_3d = batch["keypoints_3d"]

        pred_keypoints_3d = output["pred_keypoints_3d"]
        pred_cam_t = output["pred_cam_t"]

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        predictions = self.renderer(
            pred_keypoints_3d[:num_images, :num_samples_per_image],
            gt_keypoints_3d[:num_images],
            2 * gt_keypoints_2d[:num_images],
            images=images[:num_images],
            camera_translation=pred_cam_t[:num_images, :num_samples_per_image],
        )

        self.tb_writter.add_image("val/predictions", predictions.transpose((2, 0, 1)), step_count)
        visual_folder = os.path.join(RESULTS_DIR, "visuals", self.name)
        os.makedirs(visual_folder, exist_ok=True)
        visual_img = np.clip(predictions * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(visual_img).save(
            os.path.join(
                visual_folder,
                f"val_predictions_step_{step_count:06d}x{self.cfg.TRAIN.LOG_FREQ}.png",
            )
        )
