from pathlib import Path
import tempfile
import zipfile

import pandas as pd
import streamlit as st

from src.dicom_io import build_study_summary, extract_dicom_zip, find_dicom_files, load_preview_image
from src.labels import TARGET_LABELS
from src.model import LOCAL_DINOV2_WEIGHTS, predict_with_dinov2
from src.pillar_model import PILLAR_MODEL_PATH, predict_with_pillar


st.set_page_config(page_title="Knee MRI AI", page_icon="🦵", layout="wide")

st.title("Knee MRI AI")
st.caption("Prototype decision-support workflow for complete knee MRI DICOM studies.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload knee MRI DICOM file or ZIP", type=["dcm", "dicom", "zip"])
    model_choice = st.radio(
        "Model",
        ["Pillar-0 knee fine-tune", "DINOv2 feature test"],
        index=0,
    )
    st.caption(f"Pillar checkpoint: {PILLAR_MODEL_PATH}")
    st.caption(f"DINOv2 weights: {LOCAL_DINOV2_WEIGHTS}")
    st.info(
        "For a real study-level test, upload a complete MRI ZIP. A single DICOM slice can be used only to test the model pipeline."
    )

st.subheader("Workflow")
workflow_cols = st.columns(5)
for col, label in zip(
    workflow_cols,
    ["DICOM import", "Series detection", "Preprocessing", "Model inference", "12 scores"],
):
    col.metric(label, "Ready" if uploaded else "Waiting")

if not uploaded:
    st.write("Upload a DICOM file or DICOM ZIP to inspect the study and run the model pipeline.")
    st.stop()

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    upload_path = tmp_path / uploaded.name
    upload_path.write_bytes(uploaded.getbuffer())

    if upload_path.suffix.lower() == ".zip":
        try:
            input_dir = extract_dicom_zip(upload_path, tmp_path / "dicom")
        except zipfile.BadZipFile:
            st.error("The uploaded file is not a valid ZIP archive.")
            st.stop()
    else:
        input_dir = tmp_path / "dicom"
        input_dir.mkdir(parents=True, exist_ok=True)
        dicom_path = input_dir / uploaded.name
        dicom_path.write_bytes(upload_path.read_bytes())

    dicom_files = find_dicom_files(input_dir)
    if not dicom_files:
        st.error("No DICOM files were found in the uploaded ZIP.")
        st.stop()

    summary = build_study_summary(dicom_files)

    st.subheader("MRI Import")
    c1, c2, c3 = st.columns(3)
    c1.metric("DICOM files", len(dicom_files))
    c2.metric("Detected series", len(summary.series))
    c3.metric("Study UID", summary.study_uid[-12:] if summary.study_uid else "Unknown")

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Detected Series")
        rows = [
            {
                "Series": item.series_description or item.series_uid[-8:],
                "Plane hint": item.plane_hint,
                "Sequence hint": item.sequence_hint,
                "Slices": item.num_slices,
            }
            for item in summary.series
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with right:
        st.subheader("Preview")
        preview = load_preview_image(dicom_files)
        if preview is None:
            st.warning("Could not render a preview slice from these files.")
        else:
            st.image(preview, caption="Representative DICOM slice", use_container_width=True)

    st.subheader("AI Finding Scores")
    if model_choice == "Pillar-0 knee fine-tune":
        st.info("Running the 3D Pillar model on CPU can take a while, especially for full MRI studies.")
        predictions, model_note = predict_with_pillar(dicom_files)
    else:
        predictions, model_note = predict_with_dinov2(preview)
    st.warning(model_note)
    if model_choice == "DINOv2 feature test":
        st.caption(
            "DINOv2 is being used as the image feature encoder. Real medical probabilities require a trained classifier/aggregation head."
        )
    else:
        st.caption("Pillar-0 uses the downloaded knee fine-tune checkpoint with a trained 12-label linear head.")
    result_rows = [
        {"Finding": label, "Score": predictions[label], "Percent": f"{predictions[label] * 100:.1f}%"}
        for label in TARGET_LABELS
    ]
    results = pd.DataFrame(result_rows)

    for _, row in results.iterrows():
        col_a, col_b = st.columns([0.45, 0.55])
        col_a.write(row["Finding"])
        col_b.progress(float(row["Score"]), text=row["Percent"])

    with st.expander("Export results"):
        st.download_button(
            "Download CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="knee_mri_ai_scores.csv",
            mime="text/csv",
        )

st.caption("Not a diagnosis. For research/prototype use only.")
