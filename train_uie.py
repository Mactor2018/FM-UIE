"""Train a Conditional Flow Matching model for Underwater Image Enhancement.

Based on examples/images/cifar10/train_cifar10_ddp.py from TorchCFM.

Usage (single GPU):
    python train_uie.py

Usage (2-GPU DDP):
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        train_uie.py --parallel True --batch_size 32
"""

import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torchvision.utils import save_image
from tqdm import trange

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

from datasets.uie_dataset import PairedUIEDataset

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS

# Paths
flags.DEFINE_string("data_root", "/home/zyr/datasets/UIE/256/referenced",
                    "Root directory containing UIEBR/ and LSUI/")
flags.DEFINE_string("output_dir", "/data/RL-UIE/training/SFT",
                    "Directory for checkpoints, logs, and val results")
flags.DEFINE_string("resume", "", "Path to checkpoint for resuming training")

# Model
flags.DEFINE_string("model", "otcfm", "FM variant: otcfm|icfm|fm|si")
flags.DEFINE_integer("num_channel", 128, "Base channel count of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, "Learning rate")
flags.DEFINE_float("grad_clip", 1.0, "Gradient norm clipping")
flags.DEFINE_integer("total_steps", 400001, "Total training steps")
flags.DEFINE_integer("warmup", 5000, "LR warmup steps")
flags.DEFINE_integer("batch_size", 32, "Total batch size across all GPUs")
flags.DEFINE_integer("num_workers", 4, "DataLoader workers per process")
flags.DEFINE_float("ema_decay", 0.9999, "EMA decay rate")
flags.DEFINE_bool("parallel", False, "Enable DDP multi-GPU training")
flags.DEFINE_string("master_addr", "localhost", "DDP master address")
flags.DEFINE_string("master_port", "12355", "DDP master port")
flags.DEFINE_bool("use_amp", True, "Enable mixed-precision training")

# Validation
flags.DEFINE_integer("val_step", 5000, "Validate every N steps")
flags.DEFINE_integer("nfe", 20, "Number of Euler steps for validation sampling")

# Wandb
flags.DEFINE_string("wandb_project", "UIE-FlowMatching", "Wandb project name")
flags.DEFINE_string("wandb_run", "", "Wandb run name (auto-generated if empty)")


# ---------------------------------------------------------------------------
# Helpers (adapted from utils_cifar.py)
# ---------------------------------------------------------------------------

def setup_ddp(rank, world_size, master_addr, master_port):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def ema(source, target, decay):
    """Exponential Moving Average update."""
    src = source.state_dict()
    tgt = target.state_dict()
    for key in src.keys():
        tgt[key].data.copy_(tgt[key].data * decay + src[key].data * (1 - decay))


def infiniteloop(dataloader):
    """Infinite iterator over a dataloader."""
    while True:
        for batch in dataloader:
            yield batch


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


# ---------------------------------------------------------------------------
# Conditional UNet Wrapper
# ---------------------------------------------------------------------------

class ConditionalUNet(nn.Module):
    """UNet for conditional flow matching (image-to-image).

    Input:  6 channels (3ch x_t + 3ch x_cond)
    Output: 3 channels (predicted vector field v_t)
    """

    def __init__(self, image_size=256, num_channels=128, num_res_blocks=2,
                 channel_mult=None, attention_resolutions="32,16",
                 dropout=0.1, num_heads=4, num_head_channels=64):
        super().__init__()

        self.unet = UNetModelWrapper(
            dim=(6, image_size, image_size),
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
        )

        # --- Fix out_channels: 6 → 3 (Scheme A) ---
        # UNetModelWrapper sets out_channels = dim[0] = 6 by default.
        # We replace the final Conv2d to output only 3 channels.
        old_conv = self.unet.out[-1]
        new_conv = nn.Conv2d(
            old_conv.in_channels, 3,
            kernel_size=3, padding=1,
        )
        # Zero-init (matches the original zero_module pattern)
        nn.init.zeros_(new_conv.weight)
        nn.init.zeros_(new_conv.bias)
        self.unet.out[-1] = new_conv

        # --- Explicit sanity checks ---
        first_conv = self.unet.input_blocks[0][0]
        assert first_conv.in_channels == 6, \
            f"First conv must accept 6ch, got {first_conv.in_channels}"
        assert self.unet.out[-1].out_channels == 3, \
            f"Output conv must produce 3ch, got {self.unet.out[-1].out_channels}"

    def forward(self, t, xt, x_cond):
        """
        Args:
            t:      (B,) flow time in [0, 1]
            xt:     (B, 3, H, W) noisy interpolation state
            x_cond: (B, 3, H, W) degraded underwater image (condition)
        Returns:
            vt:     (B, 3, H, W) predicted vector field
        """
        x_in = torch.cat([xt, x_cond], dim=1)   # (B, 6, H, W)
        return self.unet(t, x_in)                # (B, 3, H, W)


# ---------------------------------------------------------------------------
# Sampling / Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_euler(model, x_cond, steps=20):
    """Generate enhanced images via Euler ODE integration.

    Args:
        model:  ConditionalUNet (or DDP-unwrapped)
        x_cond: (B, 3, H, W) degraded images in [-1, 1]
        steps:  number of Euler integration steps (NFE)
    Returns:
        x1_hat: (B, 3, H, W) enhanced images in [-1, 1]
    """
    model.eval()
    B, C, H, W = x_cond.shape
    # IMPORTANT: initial noise is 3-channel (same as GT), NOT 6-channel
    xt = torch.randn(B, C, H, W, device=x_cond.device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((B,), i * dt, device=x_cond.device)
        vt = model(t, xt, x_cond)
        xt = xt + vt * dt

    model.train()
    return xt.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(model, val_loader, device, nfe, step, savedir, dataset_name="val"):
    """Run validation: sample, compute metrics, save images.

    Returns dict of metric averages.
    """
    from torchmetrics.image import (
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)

    sample_imgs = []

    for batch_idx, (x_cond, x_gt) in enumerate(val_loader):
        x_cond = x_cond.to(device)
        x_gt = x_gt.to(device)

        x_pred = sample_euler(model, x_cond, steps=nfe)

        # Convert from [-1, 1] to [0, 1] for metrics
        pred_01 = (x_pred + 1.0) / 2.0
        gt_01 = (x_gt + 1.0) / 2.0

        psnr_metric.update(pred_01, gt_01)
        ssim_metric.update(pred_01, gt_01)
        lpips_metric.update(pred_01, gt_01)

        # Save first batch as sample images
        if batch_idx == 0:
            cond_01 = (x_cond + 1.0) / 2.0
            n = min(8, x_cond.size(0))
            for i in range(n):
                sample_imgs.extend([cond_01[i], pred_01[i], gt_01[i]])

    # Compute final metrics
    results = {
        f"{dataset_name}/psnr": psnr_metric.compute().item(),
        f"{dataset_name}/ssim": ssim_metric.compute().item(),
        f"{dataset_name}/lpips": lpips_metric.compute().item(),
    }

    # Save sample grid: [cond | pred | gt] rows
    if sample_imgs:
        img_dir = os.path.join(savedir, "val_samples")
        os.makedirs(img_dir, exist_ok=True)
        grid = torch.stack(sample_imgs)
        save_image(grid, os.path.join(img_dir, f"{dataset_name}_step_{step}.png"),
                   nrow=3, padding=2)

    return results


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def train(rank, world_size, argv):
    is_main = (rank == 0) if isinstance(rank, int) else True

    if FLAGS.parallel and world_size > 1:
        batch_size_per_gpu = FLAGS.batch_size // world_size
        setup_ddp(rank, world_size, FLAGS.master_addr, FLAGS.master_port)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    device = rank if FLAGS.parallel and world_size > 1 else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- Datasets -----
    train_datasets = []
    val_datasets = {}
    for ds_name in ["UIEBR", "LSUI"]:
        base = os.path.join(FLAGS.data_root, ds_name)
        train_datasets.append(PairedUIEDataset(
            os.path.join(base, "train_ud"),
            os.path.join(base, "train_gt"),
            augment=True,
        ))
        val_datasets[ds_name] = PairedUIEDataset(
            os.path.join(base, "val_ud"),
            os.path.join(base, "val_gt"),
            augment=False,
        )

    train_dataset = ConcatDataset(train_datasets)
    print(f"[Rank {rank}] Training samples: {len(train_dataset)}")

    sampler = DistributedSampler(train_dataset) if FLAGS.parallel else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    train_iter = infiniteloop(train_loader)

    val_loaders = {
        name: DataLoader(ds, batch_size=batch_size_per_gpu,
                         shuffle=False, num_workers=2, pin_memory=True)
        for name, ds in val_datasets.items()
    }

    # ----- Model -----
    net_model = ConditionalUNet(
        image_size=256,
        num_channels=FLAGS.num_channel,
        num_res_blocks=2,
        attention_resolutions="32,16",
        dropout=0.1,
        num_heads=4,
        num_head_channels=64,
    ).to(device)

    ema_model = copy.deepcopy(net_model)

    model_size = sum(p.numel() for p in net_model.parameters())
    if is_main:
        print(f"Model params: {model_size / 1e6:.2f} M")

    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    if FLAGS.parallel and world_size > 1:
        net_model = DistributedDataParallel(net_model, device_ids=[rank])
        # NOTE: ema_model is NOT wrapped in DDP — it only stores weight
        # averages and never participates in forward/backward passes.
        # Wrapping it would waste VRAM and risk state_dict key mismatches.

    # ----- FM Matcher -----
    sigma = 0.0
    fm_classes = {
        "otcfm": ExactOptimalTransportConditionalFlowMatcher,
        "icfm": ConditionalFlowMatcher,
        "fm": TargetConditionalFlowMatcher,
        "si": VariancePreservingConditionalFlowMatcher,
    }
    if FLAGS.model not in fm_classes:
        raise ValueError(f"Unknown FM model: {FLAGS.model}")
    FM = fm_classes[FLAGS.model](sigma=sigma)

    # ----- AMP -----
    use_amp = FLAGS.use_amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ----- Output dirs -----
    savedir = os.path.join(FLAGS.output_dir, FLAGS.model)
    ckpt_dir = os.path.join(savedir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(savedir, "val_samples"), exist_ok=True)

    # ----- Resume -----
    start_step = 0
    if FLAGS.resume and os.path.isfile(FLAGS.resume):
        ckpt = torch.load(FLAGS.resume, map_location=device)
        net_model.load_state_dict(ckpt["net_model"])
        ema_model.load_state_dict(ckpt["ema_model"])
        optim.load_state_dict(ckpt["optim"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        if is_main:
            print(f"Resumed from step {start_step}")

    # ----- Wandb -----
    if is_main:
        import wandb
        run_name = FLAGS.wandb_run or f"{FLAGS.model}_ch{FLAGS.num_channel}"
        wandb.init(
            project=FLAGS.wandb_project,
            name=run_name,
            config={
                "model": FLAGS.model,
                "lr": FLAGS.lr,
                "batch_size": FLAGS.batch_size,
                "num_channel": FLAGS.num_channel,
                "ema_decay": FLAGS.ema_decay,
                "total_steps": FLAGS.total_steps,
                "warmup": FLAGS.warmup,
                "nfe": FLAGS.nfe,
                "use_amp": use_amp,
            },
            dir=savedir,
            resume="allow",
        )

    # ----- Training Loop -----
    nan_count = 0

    with trange(start_step, FLAGS.total_steps, dynamic_ncols=True,
                disable=not is_main) as pbar:
        for step in pbar:
            optim.zero_grad(set_to_none=True)

            x_cond, x1 = next(train_iter)
            x_cond = x_cond.to(device)
            x1 = x1.to(device)

            # x0 = pure noise (3 channels, matching GT)
            x0 = torch.randn_like(x1)

            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

            # Forward with AMP
            # DDP's __call__ correctly delegates to module.forward,
            # so model(t, xt, x_cond) works for both plain and DDP models.
            with torch.amp.autocast("cuda", enabled=use_amp):
                vt = net_model(t, xt, x_cond)
                loss = F.mse_loss(vt, ut)

            # NaN safety check BEFORE backward
            if not torch.isfinite(loss):
                nan_count += 1
                if is_main:
                    print(f"\n[WARNING] Step {step}: non-finite loss "
                          f"({loss.item():.4f}), skip #{nan_count}")
                if nan_count > 20:
                    raise RuntimeError("Too many NaN losses, aborting.")
                optim.zero_grad(set_to_none=True)
                continue

            nan_count = 0  # reset on successful step

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip
            )

            # AMP may skip step if grads are inf
            old_scale = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            new_scale = scaler.get_scale()

            if new_scale < old_scale and is_main:
                print(f"\n[WARNING] Step {step}: grad overflow, "
                      f"scale {old_scale:.0f} → {new_scale:.0f}")

            sched.step()
            # EMA update: use the unwrapped model as source
            source_model = net_model.module if FLAGS.parallel and world_size > 1 else net_model
            ema(source_model, ema_model, FLAGS.ema_decay)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{sched.get_last_lr()[0]:.2e}")

            # Log to wandb
            if is_main and step % 50 == 0:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else 0,
                    "train/lr": sched.get_last_lr()[0],
                    "train/amp_scale": scaler.get_scale(),
                }, step=step)

            # ----- Validation & Checkpoint -----
            if FLAGS.val_step > 0 and step > 0 and step % FLAGS.val_step == 0:
                # All ranks must reach this barrier together to prevent
                # NCCL timeout when only rank 0 runs validation.
                if FLAGS.parallel and world_size > 1:
                    torch.distributed.barrier()

                if is_main:
                    ema_model.eval()

                    all_metrics = {}
                    for ds_name, vl in val_loaders.items():
                        metrics = validate(
                            ema_model, vl, device,
                            nfe=FLAGS.nfe, step=step,
                            savedir=savedir, dataset_name=ds_name,
                        )
                        all_metrics.update(metrics)

                    ema_model.train()

                    import wandb
                    wandb.log(all_metrics, step=step)

                    print(f"\n[Val @ step {step}] " +
                          " | ".join(f"{k}: {v:.4f}" for k, v in all_metrics.items()))

                    # Save checkpoint (every validation, never overwrite)
                    torch.save(
                        {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "optim": optim.state_dict(),
                            "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(),
                            "step": step,
                            "metrics": all_metrics,
                        },
                        os.path.join(ckpt_dir, f"checkpoint_step_{step}.pt"),
                    )

                # All ranks wait for rank 0 to finish validation + saving
                if FLAGS.parallel and world_size > 1:
                    torch.distributed.barrier()

    if is_main:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv):
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if FLAGS.parallel and world_size > 1:
        rank = int(os.getenv("RANK", 0))
        train(rank=rank, world_size=world_size, argv=argv)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(rank=device, world_size=1, argv=argv)


if __name__ == "__main__":
    app.run(main)
