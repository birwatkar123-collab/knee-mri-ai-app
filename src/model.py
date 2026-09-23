import numpy as np
from PIL import Image

from src.labels import TARGET_LABELS


DINOV2_MODEL_DIR = "models/dinov2-small"
LOCAL_DINOV2_WEIGHTS = f"{DINOV2_MODEL_DIR}/pytorch_model.bin"


def predict_with_dinov2(preview_image: Image.Image | None) -> tuple[dict[str, float], str]:
    """Run the supplied DINOv2 encoder and map features to demo scores.

    The local `pytorch_model.bin` is a DINOv2 backbone checkpoint. It produces
    image embeddings, not the 12 medical labels directly. A trained RSNA
    classifier head must replace `_demo_head` for real predictions.
    """
    if preview_image is None:
        return _neutral_scores(), "No readable image slice was available."

    try:
        import torch
        from transformers import Dinov2Config, Dinov2Model
    except Exception as exc:
        return _neutral_scores(), f"Could not import PyTorch/Transformers: {exc}"

    from pathlib import Path

    weights_path = Path(LOCAL_DINOV2_WEIGHTS)
    if not weights_path.exists():
        return _neutral_scores(), f"DINOv2 weights not found: {LOCAL_DINOV2_WEIGHTS}"

    try:
        config = Dinov2Config(
            image_size=518,
            patch_size=14,
            num_channels=3,
            hidden_size=384,
            num_hidden_layers=12,
            num_attention_heads=6,
            intermediate_size=1536,
            qkv_bias=True,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            layer_norm_eps=1e-6,
            layerscale_value=1.0,
            use_swiglu_ffn=False,
        )
        model = Dinov2Model(config)
        state_dict = torch.load(str(weights_path), map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        pixel_values = _preprocess_image(preview_image)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        scores = _demo_head(cls_embedding)
        note = "Kaggle DINOv2 Small encoder loaded from models/dinov2-small/pytorch_model.bin. Final label scores use an untrained demo head."
        if missing or unexpected:
            note += f" Missing keys: {len(missing)}. Unexpected keys: {len(unexpected)}."
        return scores, note
    except Exception as exc:
        return _neutral_scores(), f"Could not run local DINOv2 encoder: {exc}"


def _demo_head(embedding: np.ndarray) -> dict[str, float]:
    centered = embedding - float(np.mean(embedding))
    scale = float(np.std(centered)) or 1.0
    normalized = centered / scale
    values = []
    for index in range(len(TARGET_LABELS)):
        start = (index * 29) % max(len(normalized) - 16, 1)
        value = float(np.mean(normalized[start : start + 16]))
        prob = 1.0 / (1.0 + np.exp(-value))
        values.append(round(float(np.clip(prob, 0.01, 0.99)), 4))
    return dict(zip(TARGET_LABELS, values))


def _neutral_scores() -> dict[str, float]:
    return {label: 0.5 for label in TARGET_LABELS}


def _preprocess_image(image: Image.Image):
    import torch

    resized = image.convert("RGB").resize((518, 518), Image.BICUBIC)
    array = np.asarray(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    array = (array - mean) / std
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)
