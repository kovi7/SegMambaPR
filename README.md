# SegMambaPR

Code for the MSc thesis *Deep learning-based pancreatic tumor segmentation for resectability
classification* (Bioinformatics, University of Warsaw, 2026).

SegMambaPR segments the pancreas, the tumor (PDAC), the veins, the arteries and the common bile
duct on portal-venous CT. PDAC detection is read from the predicted mask. A geometric rule turns
the predicted tumor-vessel contact into a DPCG resectability category. The rule is exploratory,
not a clinical tool.

The model is SegMamba-V2 with four additions:
- tabular gating by age, sex and scanner type (used in training only);
- a vessel-band cross-entropy weight;
- an interface margin on the tumor logit;
- PanTS pretraining.

## Installation

Python 3.10+ and a CUDA GPU (`mamba-ssm`):

```bash
pip install -r requirements.txt     # or: uv sync
```

## Data

No data are included. You need:
- [PANORAMA](https://panorama.grand-challenge.org/) CT scans, with labels from
  [DIAGNijmegen/panorama_labels](https://github.com/DIAGNijmegen/panorama_labels);
- `clinical_information.xlsx`, `PANORAMA_verification.csv` and `PANORAMA_test.txt` (274 test cases);
- [PanTS](https://github.com/MrGiovanni/PanTS), for pretraining only.

Replace `/path/to/panorama` and `/path/to/PanTS` in the configs with your paths. The label files
keep the stored ids. The pancreatic duct is merged into the pancreas during training:

| Label | Background | PDAC | Veins | Arteries | Pancreas | Pancreatic duct | Bile duct |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Stored value | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| Training class | 0 | 1 | 2 | 3 | 4 | 4 | 5 |

```bash
python tools/convert_panorama.py --input_dir /path/to/panorama --output_dir /path/to/panorama/converted
python tools/reference_labels.py --label_dir /path/to/panorama/converted/labels --out_csv /path/to/panorama/dpcg_labels.csv
python tools/convert_pants.py --pants_root /path/to/PanTS/data --out_root /path/to/PanTS/converted
```

The first command resamples the scans to 0.8 x 0.8 x 3.0 mm. The second derives the reference
resectability categories from the ground-truth masks. The third converts PanTS to the same grid.

## Usage

The scripts in `scripts/` submit themselves to SLURM. Add your account and partition to their
headers first. Each script describes its arguments in its header.

```bash
# split: 1,668 train / 294 val / 274 test
IMAGE_DIR=... LABEL_DIR=... CLINICAL_XLSX=... TEST_LIST=... DPCG_CSV=... VERIFICATION_CSV=... \
  bash scripts/prepare_panorama_manifest.sh

# training (chained stages), then inference on test and validation
bash scripts/submit_training_sequence.sh --model segmamba-pr-tabgated-band
C=configs/panorama/segmamba_pr_tabgated_band.yaml; O=output/panorama/segmamba_pr_tabgated_band
bash scripts/infer.sh $C best test $O/inference_test_after_train3_forced_missing_best --force_missing_tabular
bash scripts/infer.sh $C best val  $O/inference_val_best --force_missing_tabular

# 50 mm location filter, with the resectability rule (fixed and fitted on validation)
bash scripts/run_roi_filter.sh apply 50 segmamba_pr_tabgated_band

# tumor probability in rings around the arteries
bash scripts/infer.sh $C best test $O/inference_probs --force_missing_tabular --save_prob_classes 1
bash scripts/run_band_mass.sh $C $O/inference_probs/probs
```

Gated configurations are tested without metadata (`--force_missing_tabular`). Their checkpoint
is also selected without metadata.

**PanTS pretraining.**
1. Build the manifest (`PANTS_MODE=all bash scripts/prepare_pants_manifest.sh`). The pretraining
   config validates on a fixed 150-case subset (`data/pants_manifest_all_val150.json`).
2. Pretrain for 120 epochs:
   `bash scripts/submit_training_sequence.sh --model pants-all-pretrain --train-only 8`.
3. Copy the final `last.ckpt` to `pretrained/pants/segmamba_pr_pants_all/pants_init.ckpt`.
4. Fine-tune `segmamba_pr_tabgated_band_iface_pants`. Run the first stage with
   `--ignore_mismatched_shapes`, which skips only the output layer, and resume the later stages
   without it.

## Configurations

| Thesis | Config (`configs/panorama/`) |
|---|---|
| CNN U-Net / MONAI U-Net / Swin UNETR | `cnn_unet` / `monai_unet` / `swinunetr` |
| SegMambaV2 (image only) | `segmamba_pr` |
| + gating (gated reference) | `segmamba_pr_tabgated` |
| + fusion / + spatial fusion | `segmamba_pr_tabgated_adf` / `_sdf` |
| + stem / + stem, repaired | `segmamba_pr_tabgated_dwconv` / `_msdw` |
| + text / + separable / + aggregator | `segmamba_pr_tabgated_text` / `_sep` / `_aggregator` |
| + band loss / + interface margin | `segmamba_pr_tabgated_band` / `_iface` |
| + band loss, interface margin | `segmamba_pr_tabgated_band_iface` |
| **SegMambaPR** (+ PanTS pretraining) | `segmamba_pr_tabgated_band_iface_pants` |

`+ text` needs embeddings from `tools/precompute_tabular_text.py`.

## License and acknowledgements

The code is released under the [Apache License 2.0](LICENSE). Attributions are in [NOTICE](NOTICE).
The backbone follows [SegMamba](https://github.com/ge-xing/SegMamba) and
[SegMamba-V2](https://github.com/ge-xing/SegMamba-V2). The gating adapts
[TabAttention](https://github.com/SanoScience/Tab-Attention).

The datasets keep their own licences, and none of their data or derived files are included
here:
- PANORAMA (images and labels): CC BY-NC 4.0;
- PanTS: CC BY-NC-ND 4.0.
