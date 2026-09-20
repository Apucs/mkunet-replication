# MK-UNet: Codebase Exploration and Experimental Verification

## 1. Objective

In this study, the original implementation of [MK-UNet](https://arxiv.org/pdf/2509.18493) was explored, and reproducing the reported benchmarking result on `ClinicDB` and `ColonDB` was attempted. Alongside of trying to reproduce the reported result, the identical architecture was trained on a non-biomedical domain (`SugarBeets2016`, from the guidelines' dataset menu) data to study cross-domain behavior.

## 2. Environment and Setup Notes

- GPU : NVIDIA H100 (GCP) - see `env_gpu.txt`
- Python : 3.10
- PyTorch / CUDA : 2.14 / 13.0
- Env records : `env_gpu.txt`, `env_freeze.txt`

## 3. Experiments

- All the experiments had been conducted with released code-defaults

- Model no of parameters: script reports `0.3156 M params` (paper: `0.316 M`)

- Publihsed numbers have been reproduced

- Training command used: 
`CUDA_VISIBLE_DEVICES=0 python -W ignore train_polyp.py --network MK_UNet
`

### 3.1. Experiment 1: Benchmark Replication

- Data: pre-split `ClinicDB` and `ColonDB` (Google Drive links from the original README).
- Metric: test DICE at best-val epoch, mean ± std over 5 runs (paper protocol).

| Benchmark | Images (train/val/test) | Paper | This work | Time/run |
|---|---|---|---|---|
| ClinicDB | 489 / 61 / 62 | 93.48 | 93.53 ± 0.33 | ~2,145 s |
| ColonDB | 303 / 38 / 38 | 90.01 | 89.48 ± 2.38 | ~1,602 s |

> [N.B]:  Though we used different hardware and software, both experiments achieved the paper's reported scores with small deviations. ColonDB had the larger deviation due to the low number of test images (38), hence larger noise. Overall the replication was successful.

### 3.2. Experiment 2: Cross-Domain (SugarBeets2016)

#### Dataset Details:

- Source data: Bonn IJRR annotated subset 12,330 RGB/annotation pairs
- 19 field sessions (2016-04-21 to 2016-05-27) + weed-species session, 1296×966 px.

#### MK-UNet Compatible Data Preparation 

- Script: `scripts/convert_sugarbeets.py`, deterministic, seed 42
- Binarization: label > 0 considered vegetation (crop + weed merged), 0 is for soil
- Subsample `2,500 frames`[frames with <100 vegetation px excluded (116 frames = 0.9 % of data)]
- split: `2,000 / 250 / 250`
- Mean vegetation fraction: `1.69 %` (compare: polyps typically cover far more of the frame)

#### Result (identical training arguments):

| Dataset | This work (5-run) | Time/run |
|---|---|---|
| SugarBeets2016 | 81.69 ± 0.86 | ~7,830 s |

#### Growth-stage breakdown (best run, run 4; source: `eval_by_session_run4.txt`):

- Script: `scripts/eval_by_session.py`

| Session (recording date) | n | Test DICE | Test IoU |
|---|---|---|---|
| CKA_160421 (Apr 21) | 13 | 0.6115 | 0.4450 |
| CKA_160426 (Apr 26) | 4 | 0.6713 | 0.5123 |
| CKA_160427 (Apr 27) | 16 | 0.7732 | 0.6324 |
| CKA_160428 (Apr 28) | 4 | 0.7718 | 0.6289 |
| CKA_160429 (Apr 29) | 16 | 0.7799 | 0.6423 |
| CKA_160502 (May 02) | 3 | 0.7838 | 0.6482 |
| CKA_160503 (May 03) | 35 | 0.8103 | 0.6836 |
| CKA_160504 (May 04) | 12 | 0.8221 | 0.6986 |
| CKA_160505 (May 05) | 15 | 0.7979 | 0.6660 |
| CKA_160506 (May 06) | 6 | 0.8509 | 0.7421 |
| CKA_160509 (May 09) | 20 | 0.8370 | 0.7236 |
| CKA_160510 (May 10) | 17 | 0.8316 | 0.7151 |
| CKA_160511 (May 11) | 4 | 0.9111 | 0.8371 |
| CKA_160512 (May 12) | 7 | 0.9193 | 0.8512 |
| CKA_160513 (May 13) | 6 | 0.9254 | 0.8625 |
| CKA_160517 (May 17) | 11 | 0.9151 | 0.8488 |
| CKA_160518 (May 18) | 4 | 0.9523 | 0.9090 |
| CKA_160523 (May 23) | 38 | 0.9182 | 0.8540 |
| CKA_160527 (May 27) | 6 | 0.9572 | 0.9183 |
| CKA_weeds (Jun, weed species) | 13 | 0.8141 | 0.6875 |
| **Overall** | **250** | **0.8313** | **0.7217** |

## 4. Per-run test DICE (at best-val epoch)

| Run | ClinicDB | ColonDB | SugarBeets |
|---|---|---|---|
| 1 | 0.9315 | 0.8655 | 0.8185 |
| 2 | 0.9356 | 0.8874 | 0.8114 |
| 3 | 0.9400 | 0.9256 | 0.8112 |
| 4 | 0.9331 | 0.8840 | 0.8313 |
| 5 | 0.9363 | 0.9116 | 0.8123 |
| **mean ± std** | **0.9353 ± 0.0033** | **0.8948 ± 0.0238** | **0.8169 ± 0.0086** |