"""Per-image test DICE broken down by SugarBeets recording session (growth stage).

Mirrors the DICE computation of train_polyp.py's test() exactly (same transforms,
same resize-to-original, same 0.5/0.2 thresholds), but reports per-session means
instead of one aggregate, using the session prefix embedded in each filename.

Usage (from repo root):
    python scripts/eval_by_session.py --checkpoint model_pth/<run_id>/<run_id>-best.pth \
        --test_path ./data/polyp/target/SugarBeets/test/
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from train_polyp import dice_coefficient, iou

NET_CONFIGS = {
    'MK_UNet_T': [4, 8, 16, 24, 32],
    'MK_UNet_S': [8, 16, 32, 48, 80],
    'MK_UNet':   [16, 32, 64, 96, 160],
    'MK_UNet_M': [32, 64, 128, 192, 320],
    'MK_UNet_L': [64, 128, 256, 384, 512],
}


def session_of(name):
    name = Path(name).name
    if name.startswith('CKA_weeds'):
        return 'CKA_weeds'
    parts = name.split('_')
    return '_'.join(parts[:2])  # e.g. CKA_160421


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--test_path', default='./data/polyp/target/SugarBeets/test/')
    ap.add_argument('--network', default='MK_UNet')
    ap.add_argument('--img_size', type=int, default=352)
    args = ap.parse_args()

    model = MK_UNet(num_classes=1, in_channels=3, channels=NET_CONFIGS[args.network],  kernel_sizes=[3,5,7])
    state = torch.load(args.checkpoint, map_location='cuda')
    # drop thop profiler bookkeeping buffers that cal_params_flops() left in the checkpoint
    state = {k: v for k, v in state.items()
             if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state)
    model.cuda().eval()

    loader = get_loader(
        image_root=f'{args.test_path}/images/', gt_root=f'{args.test_path}/masks/',
        batchsize=8, trainsize=args.img_size, shuffle=False, split='test', color_image=True,
    )

    per_session = defaultdict(list)
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
                per_session[session_of(names[i])].append(
                    (dice_coefficient(pb, gb).item(), iou(pb, gb).item()))

    print(f"{'session':<14} {'n':>4} {'DICE':>8} {'IoU':>8}")
    all_d, all_i = [], []
    for sess in sorted(per_session):
        ds = [d for d, _ in per_session[sess]]
        ious = [j for _, j in per_session[sess]]
        all_d += ds; all_i += ious
        print(f"{sess:<14} {len(ds):>4} {sum(ds)/len(ds):>8.4f} {sum(ious)/len(ious):>8.4f}")
    print(f"{'OVERALL':<14} {len(all_d):>4} {sum(all_d)/len(all_d):>8.4f} {sum(all_i)/len(all_i):>8.4f}")


if __name__ == '__main__':
    main()
