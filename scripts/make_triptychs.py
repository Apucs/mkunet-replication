"""Save qualitative image|ground-truth|prediction triptychs for a trained checkpoint.

Automatically picks the worst, median, and best test images by DICE (or takes
explicit filenames via --names). Prediction overlap is color-coded:
green = correct plant (TP), red = missed plant (FN), blue = false alarm (FP).

Usage (from repo root):
    python scripts/make_triptychs.py --checkpoint model_pth/<run>/<run>-best.pth \
        --test_path ./data/polyp/target/SugarBeets/test/ --out_dir figures/sugarbeets
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from train_polyp import dice_coefficient
from scripts.eval_by_session import NET_CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--test_path', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--network', default='MK_UNet')
    ap.add_argument('--img_size', type=int, default=352)
    ap.add_argument('--names', nargs='*', default=None,
                    help='specific test image filenames; default = worst/median/best by DICE')
    args = ap.parse_args()

    model = MK_UNet(num_classes=1, in_channels=3, channels=NET_CONFIGS[args.network])
    state = torch.load(args.checkpoint, map_location='cuda')
    state = {k: v for k, v in state.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state)
    model.cuda().eval()

    loader = get_loader(
        image_root=f'{args.test_path}/images/', gt_root=f'{args.test_path}/masks/',
        batchsize=8, trainsize=args.img_size, shuffle=False, split='test', color_image=True,
    )

    # pass 1: per-image dice + cached binary predictions at original resolution
    results = {}  # name -> (dice, pred_binary ndarray HxW)
    with torch.no_grad():
        for images, gts, original_shapes, names in loader:
            images, gts = images.cuda(), gts.cuda().float()
            preds = model(images)
            preds = preds[0] if isinstance(preds, list) else preds
            for i in range(len(images)):
                h, w = int(original_shapes[0][i]), int(original_shapes[1][i])
                p = F.interpolate(preds[i].unsqueeze(0), size=(h, w),
                                  mode='bilinear', align_corners=False).sigmoid().squeeze()
                p = (p - p.min()) / (p.max() - p.min() + 1e-8)
                g = F.interpolate(gts[i].unsqueeze(0), size=(h, w), mode='nearest').squeeze()
                pb, gb = (p >= 0.5).float(), (g >= 0.2).float()
                d = dice_coefficient(pb, gb).item()
                results[Path(names[i]).name] = (d, pb.cpu().numpy().astype(bool))

    if args.names:
        chosen = args.names
    else:
        ranked = sorted(results, key=lambda n: results[n][0])
        chosen = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        d, pred = results[name]
        img = np.array(Image.open(Path(args.test_path) / 'images' / name).convert('RGB'))
        gt = np.array(Image.open(Path(args.test_path) / 'masks' / name).convert('L')) > 50
        overlay = img.copy()
        overlay[pred & gt] = [0, 200, 0]        # true positive: green
        overlay[~pred & gt] = [220, 0, 0]       # missed plant: red
        overlay[pred & ~gt] = [0, 90, 220]      # false alarm: blue
        gt_rgb = np.stack([gt.astype(np.uint8) * 255] * 3, axis=-1)
        strip = np.concatenate([img, gt_rgb, overlay], axis=1)
        fname = out / f"{Path(name).stem}_dice{d:.3f}.png"
        Image.fromarray(strip).save(fname)
        print(f"{fname}  (DICE {d:.4f})")


if __name__ == '__main__':
    main()
