"""Batch inference for UIE Flow Matching model.

Usage:
    python inference_uie.py \
        --checkpoint /data/RL-UIE/training/SFT/otcfm/checkpoints/checkpoint_step_100000.pt \
        --input_dir /home/zyr/datasets/UIE/256/referenced/UIEBR/test_ud \
        --output_dir /data/RL-UIE/training/SFT/otcfm/test_results/UIEBR \
        --nfe 50
"""

import os

import torch
from absl import app, flags
from PIL import Image
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint", "", "Path to model checkpoint (.pt)")
flags.DEFINE_string("input_dir", "", "Directory of degraded input images")
flags.DEFINE_string("output_dir", "", "Directory for enhanced output images")
flags.DEFINE_integer("nfe", 50, "Number of Euler steps for ODE integration")
flags.DEFINE_integer("batch_size", 8, "Inference batch size")
flags.DEFINE_integer("num_channel", 128, "UNet base channels (must match training)")
flags.DEFINE_bool("use_ema", True, "Use EMA model weights")


def build_model(device, num_channels=128):
    """Build ConditionalUNet (import from train_uie.py)."""
    from train_uie import ConditionalUNet
    model = ConditionalUNet(
        image_size=256,
        num_channels=num_channels,
        num_res_blocks=2,
        attention_resolutions="32,16",
        dropout=0.0,  # No dropout at inference
        num_heads=4,
        num_head_channels=64,
    ).to(device)
    return model


@torch.no_grad()
def sample_euler(model, x_cond, steps=50):
    """Euler ODE integration from noise to enhanced image."""
    B, C, H, W = x_cond.shape
    xt = torch.randn(B, C, H, W, device=x_cond.device)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((B,), i * dt, device=x_cond.device)
        vt = model(t, xt, x_cond)
        xt = xt + vt * dt

    return xt.clamp(-1.0, 1.0)


def main(argv):
    assert FLAGS.checkpoint, "--checkpoint is required"
    assert FLAGS.input_dir, "--input_dir is required"
    assert FLAGS.output_dir, "--output_dir is required"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(FLAGS.output_dir, exist_ok=True)

    # Build model and load weights
    model = build_model(device, FLAGS.num_channel)
    ckpt = torch.load(FLAGS.checkpoint, map_location=device)

    key = "ema_model" if FLAGS.use_ema else "net_model"
    state_dict = ckpt[key]
    # Strip DDP 'module.' prefix if present
    clean_sd = {}
    for k, v in state_dict.items():
        clean_sd[k.replace("module.", "")] = v
    model.load_state_dict(clean_sd)
    model.eval()

    step = ckpt.get("step", "unknown")
    print(f"Loaded {key} from step {step}")
    print(f"NFE: {FLAGS.nfe}, Input: {FLAGS.input_dir}")

    # Collect input images
    fnames = sorted([
        f for f in os.listdir(FLAGS.input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"Found {len(fnames)} images")

    # Process in batches
    for i in tqdm(range(0, len(fnames), FLAGS.batch_size), desc="Inference"):
        batch_fnames = fnames[i: i + FLAGS.batch_size]

        imgs = []
        for fname in batch_fnames:
            img = Image.open(os.path.join(FLAGS.input_dir, fname)).convert('RGB')
            tensor = TF.to_tensor(img) * 2.0 - 1.0  # normalize to [-1, 1]
            imgs.append(tensor)

        x_cond = torch.stack(imgs).to(device)
        x_pred = sample_euler(model, x_cond, steps=FLAGS.nfe)

        # Convert back to [0, 1] and save
        x_pred_01 = (x_pred + 1.0) / 2.0

        for j, fname in enumerate(batch_fnames):
            out_path = os.path.join(FLAGS.output_dir, fname)
            save_image(x_pred_01[j], out_path)

    print(f"Done! Results saved to {FLAGS.output_dir}")


if __name__ == "__main__":
    app.run(main)
