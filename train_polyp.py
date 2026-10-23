import os
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def structure_loss(pred, mask, w=1):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (w * (wbce + wiou)).mean()

def sas_weights(target_1h, sas):
    """Scale-aware [B, C] weights: (a_ref/object_area)^alpha capped at w_max;
    small objects weigh more. Absent classes get neutral weight 1."""
    a_ref, alpha, w_max = sas
    area = target_1h.sum(dim=(2, 3))                               # [B, C] px per class
    u = (a_ref / area.clamp(min=1.0)).pow(alpha).clamp(max=w_max)  # small -> heavy
    return torch.where(area > 0, u, torch.ones_like(u))

def multiclass_dice_loss(logits, target, num_classes, smooth=1e-6, sas=None):
    """Macro soft dice over foreground classes (background class 0 excluded).

    logits: [B, C, H, W] raw model output; target: [B, H, W] long.

    sas: optional (a_ref, alpha, w_max) tuple enabling scale-aware weighting
    of the per-(image, class) terms (see sas_weights).
    """
    probs = F.softmax(logits, dim=1)
    target_1h = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    inter = (probs * target_1h).sum(dim=(2, 3))                    # [B, C]
    total = probs.sum(dim=(2, 3)) + target_1h.sum(dim=(2, 3))      # [B, C]
    dice = (2 * inter + smooth) / (total + smooth)                 # [B, C]

    if sas is None:
        return 1 - dice[:, 1:].mean()

    # u.sum() > 0 always (absent classes weigh 1)
    u = sas_weights(target_1h, sas)[:, 1:]
    return (u * (1 - dice[:, 1:])).sum() / u.sum()

def ce_dice_loss(logits, target, num_classes, sas=None):
    """Multiclass loss: cross entropy + macro foreground dice (scale-aware
    when sas is given; CE stays uniform). target: [B, H, W] long."""
    return F.cross_entropy(logits, target) + multiclass_dice_loss(logits, target, num_classes, sas=sas)

def measure_aref(mask_root, num_classes, img_size):
    """Geometric mean per-image foreground-class area of a mask folder,
    rescaled to img_size x img_size equivalent pixels. Runs once at startup."""
    import cv2 as _cv2
    log_areas = []
    for f in sorted(os.listdir(mask_root)):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue
        m = _cv2.imread(os.path.join(mask_root, f), _cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"measure_aref: unreadable mask file {os.path.join(mask_root, f)}")
        scale = (img_size * img_size) / (m.shape[0] * m.shape[1])
        for c in range(1, num_classes):
            a = (m == c).sum() * scale
            if a >= 1:
                log_areas.append(np.log(a))
    assert log_areas, f"no foreground objects found in {mask_root}"
    return float(np.exp(np.mean(log_areas)))

def multiclass_structure_loss(logits, target, num_classes, w=1, sas=None):
    """Faithful multiclass port of structure_loss: boundary-weighted CE +
    boundary-weighted soft IoU over foreground classes.

    For num_classes=2 this reduces exactly to the binary structure_loss.
    logits: [B, C, H, W]; target: [B, H, W] long.

    sas: optional (a_ref, alpha, w_max) tuple. When given, each (image,
    foreground class) IoU term is weighted by (a_ref/area)^alpha, capped at
    w_max (scale-aware supervision); the CE term stays boundary-weighted only.
    """
    # 1. Predictions and ground truth
    probs = F.softmax(logits, dim=1)
    gt = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()

    # 2. Boundary weights (pad the label map with class 0 before one-hot,
    #    so "outside the image" counts as background in every channel)
    padded = F.pad(target, (15, 15, 15, 15), value=0)
    padded_gt = F.one_hot(padded, num_classes).permute(0, 3, 1, 2).float()

    local_avg = F.avg_pool2d(padded_gt, kernel_size=31, stride=1)
    weights = 1 + 5 * torch.abs(local_avg - gt)

    # 3. Weighted Cross-Entropy (each pixel weighted at its true class)
    ce = F.cross_entropy(logits, target, reduction="none")
    pixel_weights = (weights * gt).sum(dim=1)

    wce = (ce * pixel_weights).sum((1, 2)) / pixel_weights.sum((1, 2))

    # 4. Weighted Soft IoU over foreground classes
    overlap = probs * gt
    inter = (overlap * weights).sum((2, 3))
    union = ((probs + gt - overlap) * weights).sum((2, 3))

    wiou = 1 - (inter + 1) / (union + 1)

    if sas is None:
        wiou = wiou[:, 1:].mean(dim=1)
        # 5. Final loss
        return (w * (wce + wiou)).mean()

    # Scale-aware variant: object-size weights on the per-class IoU terms.
    # u.sum() > 0 always (absent classes weigh 1)
    u = sas_weights(gt, sas)[:, 1:]
    wiou = (u * wiou[:, 1:]).sum() / u.sum()
    return (w * (wce.mean() + wiou))

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def test(model, path, dataset, opt):
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root,
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image
    )

    DSC, IOU, total_images = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack
            images = images.to(DEVICE)
            gts = gts.to(DEVICE).float()

            ress = model(images)
            # Take the primary output
            predictions = ress[0] if isinstance(ress, list) else ress
            
            for i in range(len(images)):
                # Note: original_shapes in some loaders is [W, H], in others [H, W]
                # We ensure it matches your specific data loader's return order
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Prediction Resize (Bilinear for soft maps)
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_resized = pred_resized.sigmoid().squeeze()
                
                # 2. Local Normalization
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                # 3. GT Resize (NEAREST to maintain binary mask integrity)
                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

                #print(pred_resized.shape, gt_resized.shape, g.shape)

                # 4. Binary Thresholding
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float() 

                # Applying original thresholding (0.5 for pred, 0.2 for target)
                total_images += 1

                DSC += dice_coefficient(input_binary, target_binary).item()
                IOU += iou(input_binary, target_binary).item()

    return DSC / total_images, IOU / total_images, total_images

def test_multiclass(model, path, dataset, opt):
    """Per-class evaluation via argmax over class channels.

    Returns (mean_fg_dice, per_class_dice, per_class_iou, n_images).
    Per-class scores are averaged only over images whose GT contains that
    class, so an absent class neither inflates nor deflates the average.
    """
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()

    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root,
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image,
        num_classes=opt.num_classes
    )

    C = opt.num_classes
    dice_sum, iou_sum, class_n = np.zeros(C), np.zeros(C), np.zeros(C, dtype=int)
    total_images = 0

    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack
            images = images.to(DEVICE)
            gts = gts.to(DEVICE)

            ress = model(images)
            predictions = ress[0] if isinstance(ress, list) else ress   # [B, C, h, w]

            for i in range(len(images)):
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])

                # Resize logits (bilinear on soft scores), then argmax -> label map
                p = predictions[i].unsqueeze(0)
                p = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_map = p.argmax(dim=1).squeeze(0)                   # [H, W]

                # Resize GT label map with nearest to keep class indices intact
                g = gts[i].unsqueeze(0).float()
                gt_map = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze().long()

                for c in range(C):
                    if (gt_map == c).any():
                        pb = (pred_map == c).float()
                        gb = (gt_map == c).float()
                        dice_sum[c] += dice_coefficient(pb, gb).item()
                        iou_sum[c] += iou(pb, gb).item()
                        class_n[c] += 1
                total_images += 1

    per_class_dice = np.divide(dice_sum, class_n, out=np.zeros(C), where=class_n > 0)
    per_class_iou = np.divide(iou_sum, class_n, out=np.zeros(C), where=class_n > 0)
    mean_fg_dice = per_class_dice[1:].mean()
    return mean_fg_dice, per_class_dice, per_class_iou, total_images

def train(train_loader, model, optimizer, epoch, opt, model_name):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot
    
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1, 1.25] 
    total_step = len(train_loader)

    for i, pack in enumerate(train_loader, start=1):
        batch_images, batch_gts = pack
        batch_images = batch_images.to(DEVICE)
        batch_gts = batch_gts.float().to(DEVICE)
        for rate in size_rates:
            optimizer.zero_grad()
            # each rate must resize from the originals, not the previous rate's output
            images, gts = batch_images, batch_gts

            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='nearest')

            out = model(images)
            outs = out if isinstance(out, list) else [out]

            # per-head weights 1/2^i, unnormalized so p4 matches the baseline scale
            stage_w = [0.5 ** i for i in range(len(outs))]

            if opt.num_classes == 1:
                loss = sum(w * structure_loss(o, gts) for w, o in zip(stage_w, outs))
            else:
                # CE wants [B, H, W] long class indices (no channel dim)
                targets = gts.squeeze(1).long()
                if opt.sas:
                    # areas scale by rate^2 under multi-scale resizing; a_ref follows
                    a_ref_cur = opt.sas_aref * (images.shape[-1] / opt.img_size) ** 2
                    sas_cfg = (a_ref_cur, opt.sas_alpha, opt.sas_wmax)
                else:
                    sas_cfg = None
                if opt.mc_loss == 'structure':
                    base = lambda o: multiclass_structure_loss(o, targets, opt.num_classes, sas=sas_cfg)
                else:
                    base = lambda o: ce_dice_loss(o, targets, opt.num_classes, sas=sas_cfg)
                loss = sum(w * base(o) for w, o in zip(stage_w, outs))

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
                
        if i % 100 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
        
    total_train_time += (time.time() - epoch_start)
    
    # Save Last
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation and Testing
    epoch_results = {}
    for ds in ['test', 'val']:
        if opt.num_classes == 1:
            d_dice, d_iou, _ = test(model, opt.test_path, ds, opt)
            msg = f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}'
        else:
            d_dice, d_cls, i_cls, _ = test_multiclass(model, opt.test_path, ds, opt)
            cls_str = ' '.join(f'c{c}:{d_cls[c]:.4f}' for c in range(opt.num_classes))
            msg = f'Epoch: {epoch}, Dataset: {ds}, MeanFG Dice: {d_dice:.4f}, per-class [{cls_str}]'
        epoch_results[ds] = d_dice
        logging.info(msg)
        print(msg)
        dict_plot[ds].append(d_dice)

    # Check if Best Validation Dice
    if epoch_results['val'] > best:
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        best = epoch_results['val']
        test_dice_at_best_val = epoch_results['test'] # Track test dice at peak val
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='MK_UNet',
                        choices=['MK_UNet_T', 'MK_UNet_S', 'MK_UNet', 'MK_UNet_M', 'MK_UNet_L'])
    parser.add_argument('--dataset', type=str, default='SugarBeets')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='1 = binary (sigmoid + structure_loss); >1 = multiclass (softmax + CE/dice)')
    parser.add_argument('--mc_loss', type=str, default='ce_dice', choices=['ce_dice', 'structure'],
                        help='multiclass loss: ce_dice (CE + macro dice) or structure (boundary-weighted CE + soft IoU)')
    parser.add_argument('--deep_supervision', action='store_true',
                        help='train on all four prediction heads (p4..p1), weights 1.0/0.5/0.25/0.125')
    parser.add_argument('--sas', action='store_true',
                        help='scale-aware supervision: weight each (image, class) dice term by (a_ref/area)^alpha')
    parser.add_argument('--sas_aref', type=str, default='auto',
                        help="'auto' = geometric mean foreground-class area of the train set "
                             "(at img_size scale), or an explicit pixel value")
    parser.add_argument('--sas_alpha', type=float, default=0.5)
    parser.add_argument('--sas_wmax', type=float, default=10.0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005) # base learning rate is 0.0005 for CosineAnnealingLR and 0.0001 for no scheduler
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=300)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--train_path', type=str, default='')
    parser.add_argument('--test_path', type=str, default='')
    parser.add_argument('--train_save', type=str, default='')
    opt = parser.parse_args()

    dataset_name = opt.dataset
    if not opt.train_path:
        opt.train_path = f'./data/polyp/target/{dataset_name}/train/'
    if not opt.test_path:
        opt.test_path = f'./data/polyp/target/{dataset_name}/'

    if opt.sas:
        assert opt.num_classes > 1, "--sas is implemented for multiclass losses"
        if opt.sas_aref == 'auto':
            opt.sas_aref = measure_aref(f'{opt.train_path}/masks/', opt.num_classes, opt.img_size)
            print(f"SAS: a_ref = {opt.sas_aref:.0f} px "
                  f"(geometric mean foreground-class area at {opt.img_size}x{opt.img_size})")
        else:
            opt.sas_aref = float(opt.sas_aref)

    # Network configuration mapping
    NET_CONFIGS = {
        'MK_UNet_T': [4, 8, 16, 24, 32],
        'MK_UNet_S': [8, 16, 32, 48, 80],
        'MK_UNet':   [16, 32, 64, 96, 160],
        'MK_UNet_M': [32, 64, 128, 192, 320],
        'MK_UNet_L': [64, 128, 256, 384, 512]
    }

    # Handling Spelling Mistakes or Invalid Choices
    # We use a case-insensitive match or check if it exists in our dictionary
    chosen_net = opt.network
    if chosen_net not in NET_CONFIGS:
        print(f"WARNING: Network '{chosen_net}' not found. Defaulting to 'MK_UNet'.")
        chosen_net = 'MK_UNet'

    for run in [1,2,3,4,5]:
        dict_plot = {'val': [], 'test': []}
        best = 0.0
        test_dice_at_best_val = 0.0
        total_train_time = 0

        timestamp = time.strftime('%H%M%S')
        cls_tag = f"c{opt.num_classes}_" if opt.num_classes > 1 else ""
        fix_tag = "fx_"  # fixed multi-scale protocol; untagged runs used the released loop
        loss_tag = "siou_" if (opt.num_classes > 1 and opt.mc_loss == 'structure') else ""
        ds_tag = ("ds_" if opt.deep_supervision else "") + ("sas_" if opt.sas else "")
        run_id = (f"{dataset_name}_{chosen_net}_{fix_tag}{cls_tag}{loss_tag}{ds_tag}bs{opt.batchsize}_lr{opt.lr}_"
                      f"e{opt.epoch}_aug{opt.augmentation}_run{run}_t{timestamp}")
        opt.train_save = f'./model_pth/{run_id}/'
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)
        
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                            format='[%(asctime)s] %(message)s', force=True)

        if opt.network != chosen_net:
            logging.warning(f"User input '{opt.network}' was invalid. Fallback to '{chosen_net}' used.")

        # Build model
        channels = NET_CONFIGS[chosen_net]
        model = MK_UNet(num_classes=opt.num_classes, in_channels=3, channels=channels,
                        deep_supervision=opt.deep_supervision)

        '''if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)'''

        model.to(DEVICE)

        print(f"Network: {chosen_net} | Channels: {channels} | Classes: {opt.num_classes} | Device: {DEVICE}")
        cal_params_flops(model, opt.img_size, logging)
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size,
            shuffle=True, augmentation=opt.augmentation, split='train', color_image=opt.color_image,
            num_classes=opt.num_classes
        )

        for epoch in range(1, opt.epoch + 1):
            #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, model, optimizer, epoch, opt, run_id)
            scheduler.step()
        # FINAL SUMMARY
        
        summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
                   f"Best Val Dice: {best:.4f}\n"
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
        print(summary)
        logging.info(summary)
        
        
