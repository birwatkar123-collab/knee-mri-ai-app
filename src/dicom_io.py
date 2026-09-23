from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pydicom
from PIL import Image


@dataclass
class SeriesSummary:
    series_uid: str
    series_description: str
    plane_hint: str
    sequence_hint: str
    num_slices: int


@dataclass
class StudySummary:
    study_uid: str
    series: list[SeriesSummary]


def extract_dicom_zip(zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = output_dir / member.filename
            if not target.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError("ZIP contains an unsafe path.")
        archive.extractall(output_dir)
    return output_dir


def find_dicom_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file()]
    dicom_files = []
    for path in files:
        if path.suffix.lower() == ".dcm":
            dicom_files.append(path)
            continue
        try:
            pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            dicom_files.append(path)
        except Exception:
            pass
    return dicom_files


def build_study_summary(dicom_files: list[Path]) -> StudySummary:
    grouped: dict[str, list[pydicom.dataset.FileDataset]] = {}
    study_uid = ""

    for path in dicom_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue

        study_uid = study_uid or str(getattr(ds, "StudyInstanceUID", ""))
        series_uid = str(getattr(ds, "SeriesInstanceUID", path.parent.as_posix()))
        grouped.setdefault(series_uid, []).append(ds)

    series = []
    for uid, datasets in sorted(grouped.items(), key=lambda item: _series_sort_key(item[1])):
        first = datasets[0]
        description = str(getattr(first, "SeriesDescription", ""))
        protocol = str(getattr(first, "ProtocolName", ""))
        combined = f"{description} {protocol}".strip()
        series.append(
            SeriesSummary(
                series_uid=uid,
                series_description=description or protocol or "Unnamed series",
                plane_hint=_infer_plane(first, combined),
                sequence_hint=_infer_sequence(combined),
                num_slices=len(datasets),
            )
        )

    return StudySummary(study_uid=study_uid, series=series)


def load_preview_image(dicom_files: list[Path]) -> Image.Image | None:
    for path in dicom_files:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim > 2:
                arr = arr[0]
            arr -= float(arr.min())
            max_value = float(arr.max())
            if max_value > 0:
                arr /= max_value
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        except Exception:
            continue
    return None


def _series_sort_key(datasets: list[pydicom.dataset.FileDataset]) -> tuple[int, str]:
    first = datasets[0]
    number = int(getattr(first, "SeriesNumber", 9999) or 9999)
    description = str(getattr(first, "SeriesDescription", ""))
    return number, description


def _infer_plane(ds: pydicom.dataset.FileDataset, text: str) -> str:
    lower = text.lower()
    if "sag" in lower:
        return "Sagittal"
    if "cor" in lower:
        return "Coronal"
    if "ax" in lower or "tra" in lower:
        return "Axial"
    orientation = getattr(ds, "ImageOrientationPatient", None)
    if orientation and len(orientation) >= 6:
        normal = np.cross(np.array(orientation[:3], dtype=float), np.array(orientation[3:6], dtype=float))
        axis = int(np.argmax(np.abs(normal)))
        return ["Sagittal", "Coronal", "Axial"][axis]
    return "Unknown"


def _infer_sequence(text: str) -> str:
    lower = text.lower()
    fat_sat = any(term in lower for term in ["fs", "fat", "spair", "stir"])
    if "t1" in lower:
        base = "T1"
    elif "pd" in lower:
        base = "PD"
    elif "t2" in lower:
        base = "T2"
    else:
        base = "Unknown"
    if fat_sat and base != "Unknown":
        return f"{base} fat-suppressed"
    if fat_sat:
        return "Fat-suppressed"
    return base
