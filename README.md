# Knee MRI Abnormality Detection App

Prototype Streamlit app for testing knee MRI abnormality detection from DICOM studies.

The app accepts a single DICOM file or a ZIP of DICOM slices, displays study/series metadata and a preview slice, then can run one of two model paths:

- **Pillar-0 knee fine-tune**: a 3D MRI model with a trained 12-label classification head.
- **DINOv2 feature test**: a 2D image-feature pipeline useful for checking DICOM upload/model plumbing.

> Research demo only. This project is not a medical device and must not be used as a clinical diagnosis system.

## Supported Outputs

The Pillar-0 fine-tune predicts 12 RSNA knee finding scores:

1. ACL injury
2. MCL injury
3. Medial meniscus tear
4. Lateral meniscus tear
5. Medial-compartment osteoarthritis
6. Lateral-compartment osteoarthritis
7. Patellofemoral osteoarthritis
8. Joint effusion
9. Synovitis
10. Baker's cyst
11. Bone contusion
12. Fracture

## Current Status

Working locally:

- DICOM and DICOM ZIP upload
- DICOM metadata/series detection
- MRI slice preview
- DINOv2 encoder test path
- Pillar-0 knee fine-tune inference path
- CSV export of prediction scores

Tested with a 17-slice Kaggle RSNA test series ZIP on CPU. CPU inference can take several minutes.

## Project Structure

```text
knee-mri-ai-app/
  app.py
  requirements.txt
  src/
    dicom_io.py
    labels.py
    model.py
    pillar_model.py
  models/
    dinov2-small/
    pillar-knee/
  sample_dicoms/
```

Large files under `models/` and `sample_dicoms/` are intentionally ignored by Git.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Model Files

This repository should not include large model checkpoints. Download/place them locally:

```text
models/pillar-knee/pillar_ft_a.final.pt
models/dinov2-small/pytorch_model.bin
```

The Pillar model also requires official Pillar-0 custom code/config from:

```text
YalaLab/Pillar0-BreastMRI
```

In this local prototype those files are expected under:

```text
external/pillar-breastmri-hf/
```

Access to the base Pillar-0 BreastMRI repo may require accepting its Hugging Face gated model terms.

## Credits

This project builds on the following work:

- **RSNA Knee Abnormality Detection competition**: source task and DICOM data format.
- **Pillar-0 BreastMRI / Atlas backbone** by YalaLab: base radiology foundation model.
- **Pillar-0 Knee MRI Fine-tune** by `fabotelli` on Kaggle: checkpoint with trained `Linear(1152, 12)` head for RSNA knee labels.
- **DINOv2** by Meta AI: optional image feature encoder path.
- **Kaggle**: model hosting and sample RSNA test DICOM access.
- **Hugging Face**: Pillar-0 model code/config hosting.
- **Streamlit**, **PyTorch**, **pydicom**, and supporting Python libraries.

Please follow the license terms of each upstream model/dataset. The Pillar derivative retains the base model's Educational Community License 2.0 terms as described in its model card. DINOv2 model-card material indicates non-commercial licensing for the referenced checkpoint.

## Deployment Notes

Free hosting tiers are usually not enough for this full app because the Pillar model is large and 3D MRI inference is CPU/RAM heavy.

Recommended paths:

- GitHub for source code.
- Hugging Face Spaces for an ML demo, preferably with sufficient CPU/RAM or GPU.
- Keep model weights out of Git and download them at startup or attach them as private/gated assets.

## Safety Notice

Outputs are model scores, not medical diagnoses. A qualified clinician/radiologist must interpret MRI findings.
