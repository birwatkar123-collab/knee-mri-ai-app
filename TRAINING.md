# Training the RSNA Knee Classifier Head

The Kaggle DINOv2 Small file:

```text
models/dinov2-small/pytorch_model.bin
```

is only the image feature encoder. To produce real 12-label knee outputs, train a classifier head on top of DINOv2 features using RSNA knee training data.

## Required training data

Place these files under a training data folder:

```text
data/
  train.csv
  train_series.csv
  train/
    <study_id>/
      <series_id>/
        *.dcm
```

The label CSV must contain one row per study and the 12 target columns:

```text
ACL_tear
MCL_tear
medial_meniscus_tear
lateral_meniscus_tear
medial_cartilage_loss
lateral_cartilage_loss
pf_cartilage_loss
effusion
synovitis
bakers_cyst
bone_contusion
fracture
```

## Why training is not running locally yet

This workspace currently contains only:

```text
D:\kaggle RSNA\data\sample_submission.csv
```

It does not contain RSNA training DICOMs or labels. Also, this Windows machine currently blocks local PyTorch DLL loading, so real training should be run in Kaggle or another GPU Python environment.

## Recommended first training target

Start with a frozen DINOv2 encoder and train only a small classifier head:

```text
middle DICOM slices -> DINOv2 CLS embeddings -> average study embedding -> 12 sigmoid outputs
```

After that works, improve with:

- sequence-aware aggregation
- multiple slices per series
- plane/sequence slots
- fine-tuning final DINOv2 blocks
- threshold calibration

## Command

```bash
python train_rsna_head.py --data-dir /kaggle/input/rsna-knee-abnormality-detection --output weights/rsna_dinov2_head.pt
```
