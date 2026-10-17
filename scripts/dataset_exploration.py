import os
import numpy as np
from PIL import Image
from pathlib import Path

# resolve the archive relative to THIS FILE, so it works from any working directory
HERE = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(HERE, '..', '..', 'datasets', 'sugarbeets_raw',
                    'ijrr_sugarbeets_2016_annotations')
assert os.path.isdir(root), f'archive not found at: {os.path.abspath(root)}'

total_rgb = total_imap = 0
for s in sorted(os.listdir(root)):
    rgb_dir = os.path.join(root, s, 'images', 'rgb')
    imap_dir = os.path.join(root, s, 'annotations', 'dlp', 'iMapCleaned')
    rgb = len(os.listdir(rgb_dir)) if os.path.isdir(rgb_dir) else 0
    imap = len(os.listdir(imap_dir)) if os.path.isdir(imap_dir) else 0
    print(f'{s:14s} rgb={rgb:5d}  annotated={imap:5d}')
    total_rgb += rgb
    total_imap += imap
print(f'TOTAL rgb={total_rgb}  annotated={total_imap}')

sample = os.path.join(root, 'CKA_160421', 'annotations', 'dlp', 'iMapCleaned',
                      '1461247063_539566720.png')
im = np.array(Image.open(sample))
print(im.dtype, im.shape, np.unique(im))

# check unique values of the first N iMap files in one folder
folder = os.path.join(root, 'CKA_weeds', 'annotations', 'dlp', 'iMapCleaned')
limit = 10
print(f'\nunique values of first {limit} files in {os.path.basename(folder)}:')
for f in sorted(os.listdir(folder))[-200:]:
    a = np.array(Image.open(os.path.join(folder, f)))
    print(f'{f:35s} {list(np.unique(a))}')


img_name = Path("/mnt/E2AE16C1AE168DE3/research/mr_tt/datasets/sugarbeets_raw/ijrr_sugarbeets_2016_annotations/CKA_160421/annotations/dlp/iMapCleaned/1461247064_539602853.png")
print(img_name.name)

# # vegetation-pixel distribution: how many plant pixels per annotated frame?
# # (this is what justifies MIN_FG_PIXELS = 100 in convert_sugarbeets.py)
# fg_limit = 500   # frames to scan; set to None to scan all 12,340 (takes a few minutes)
# buckets = {'0 (empty)': 0, '1-99': 0, '100-999': 0, '1000-9999': 0, '>=10000': 0}
# scanned = 0
# for s in sorted(os.listdir(root)):
#     imap_dir = os.path.join(root, s, 'annotations', 'dlp', 'iMapCleaned')
#     if not os.path.isdir(imap_dir):
#         continue
#     for f in sorted(os.listdir(imap_dir)):
#         if fg_limit is not None and scanned >= fg_limit:
#             break
#         n = int((np.array(Image.open(os.path.join(imap_dir, f))) > 0).sum())
#         if n == 0:
#             buckets['0 (empty)'] += 1
#         elif n < 100:
#             buckets['1-99'] += 1
#         elif n < 1000:
#             buckets['100-999'] += 1
#         elif n < 10000:
#             buckets['1000-9999'] += 1
#         else:
#             buckets['>=10000'] += 1
#         scanned += 1

# print(f'\nvegetation pixels per frame ({scanned} frames scanned):')
# for k, v in buckets.items():
#     print(f'  {k:12s} {v:6d}')