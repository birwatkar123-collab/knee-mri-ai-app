from functools import lru_cache
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.labels import TARGET_LABELS


PILLAR_MODEL_PATH = Path("models/pillar-knee/pillar_ft_a.final.pt")
PILLAR_CODE_PATH = Path("external/pillar-breastmri-hf")


def predict_with_pillar(dicom_files: list[Path]) -> tuple[dict[str, float], str]:
    if not PILLAR_MODEL_PATH.exists():
        return _neutral_scores(), f"Pillar knee checkpoint not found: {PILLAR_MODEL_PATH}"
    if not PILLAR_CODE_PATH.exists():
        return _neutral_scores(), f"Pillar source code not found: {PILLAR_CODE_PATH}"

    try:
        volume = _load_dicom_volume(dicom_files)
        if volume is None:
            return _neutral_scores(), "No readable DICOM pixel data was available."
        if volume.shape[0] < 4:
            return (
                _neutral_scores(),
                "Pillar knee model needs a multi-slice MRI series or study ZIP. "
                f"Only {volume.shape[0]} readable DICOM slice(s) were uploaded. "
                "Use DINOv2 feature test for a single-slice pipeline check.",
            )
        model, head = _load_pillar_model()
        with torch.no_grad():
            inputs = _preprocess_volume(volume)
            features = model.extract_vision_feats(image={"breast_mr": inputs})
            logits = head(features)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        scores = {
            label: round(float(prob), 4)
            for label, prob in zip(TARGET_LABELS, probs)
        }
        return scores, "Pillar-0 knee fine-tune loaded. Scores come from the trained 12-label head."
    except Exception as exc:
        return _neutral_scores(), f"Could not run Pillar knee model: {exc}"


@lru_cache(maxsize=1)
def _load_pillar_model() -> tuple[nn.Module, nn.Linear]:
    package_name = "pillar_breastmri_hf"
    package_path = PILLAR_CODE_PATH.resolve()
    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            package_path / "__init__.py",
            submodule_search_locations=[str(package_path)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    config_module = importlib.import_module(f"{package_name}.config_clip_mmatlas")
    model_module = importlib.import_module(f"{package_name}.modeling_clip_mmatlas")
    CLIPMultimodalAtlasConfig = config_module.CLIPMultimodalAtlasConfig
    CLIPMultimodalAtlas = model_module.CLIPMultimodalAtlas

    checkpoint = torch.load(str(PILLAR_MODEL_PATH), map_location="cpu")

    config = CLIPMultimodalAtlasConfig.from_json_file(
        str(PILLAR_CODE_PATH / "config.json")
    )
    model = CLIPMultimodalAtlas(config)
    model_state = {
        key.replace("model.", "", 1): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(model_state, strict=True)
    model.eval()

    head = nn.Linear(1152, len(TARGET_LABELS))
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    return model, head


def _load_dicom_volume(dicom_files: list[Path]) -> np.ndarray | None:
    slices = []
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim > 2:
                arr = arr[0]
            position = _slice_position(ds)
            slices.append((position, arr))
        except Exception:
            continue

    if not slices:
        return None

    slices.sort(key=lambda item: item[0])
    volume = np.stack([arr for _, arr in slices], axis=0)
    return volume


def _slice_position(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    iop = getattr(ds, "ImageOrientationPatient", None)
    if ipp is not None and iop is not None and len(ipp) >= 3 and len(iop) >= 6:
        row = np.array(iop[:3], dtype=np.float32)
        col = np.array(iop[3:6], dtype=np.float32)
        normal = np.cross(row, col)
        return float(np.dot(np.array(ipp[:3], dtype=np.float32), normal))
    return float(getattr(ds, "InstanceNumber", 0) or 0)


def _preprocess_volume(volume: np.ndarray) -> torch.Tensor:
    volume = volume.astype(np.float32)
    volume -= float(volume.min())
    max_value = float(volume.max())
    if max_value > 0:
        volume /= max_value

    mean = float(volume.mean())
    std = float(volume.std()) or 1.0
    volume = (volume - mean) / std

    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(384, 384, 192),
        mode="trilinear",
        align_corners=False,
    )
    tensor = tensor.repeat(1, 3, 1, 1, 1)
    return tensor


def _neutral_scores() -> dict[str, float]:
    return {label: 0.5 for label in TARGET_LABELS}
