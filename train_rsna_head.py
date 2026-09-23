import argparse
from pathlib import Path


TARGET_COLUMNS = [
    "ACL_tear",
    "MCL_tear",
    "medial_meniscus_tear",
    "lateral_meniscus_tear",
    "medial_cartilage_loss",
    "lateral_cartilage_loss",
    "pf_cartilage_loss",
    "effusion",
    "synovitis",
    "bakers_cyst",
    "bone_contusion",
    "fracture",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the 12-label RSNA knee classifier head on top of DINOv2 features."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("weights/rsna_dinov2_head.pt"))
    args = parser.parse_args()

    train_csv = args.data_dir / "train.csv"
    train_series_csv = args.data_dir / "train_series.csv"
    train_images = args.data_dir / "train"

    missing = [
        path
        for path in [train_csv, train_series_csv, train_images]
        if not path.exists()
    ]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Missing required RSNA training data:\n"
            f"{missing_text}\n\n"
            "Download or attach the RSNA training CSVs and DICOM folder, then rerun."
        )

    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "PyTorch could not be imported in this environment. "
            "Run training on Kaggle GPU or fix local PyTorch first.\n"
            f"Import error: {exc}"
        ) from exc

    raise SystemExit(
        "Training data was found, but the full trainer is not implemented yet. "
        "Next step: add DICOM dataset loading, DINOv2 feature extraction, and a 12-label head."
    )


if __name__ == "__main__":
    main()
