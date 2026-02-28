import json
import requests
import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit.logger import get_logger

FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"
FASTAPI_WINE_MODEL_LOCATION = Path(__file__).resolve().parents[1] / 'backend' / 'model' / 'wine_model.pkl'
METRICS_PATH = Path(__file__).resolve().parents[1] / 'backend' / 'model' / 'metrics.json'
FEATURE_IMPORTANCE_PATH = Path(__file__).resolve().parents[1] / 'backend' / 'model' / 'feature_importance.json'

WINE_CLASSES = {0: "Class 0 (Cultivar 1)", 1: "Class 1 (Cultivar 2)", 2: "Class 2 (Cultivar 3)"}

FEATURE_RANGES = {
    "alcohol":             (11.03, 14.83, 13.00),
    "malic_acid":          (0.74, 5.80, 2.34),
    "ash":                 (1.36, 3.23, 2.37),
    "alcalinity_of_ash":   (10.6, 30.0, 19.5),
    "magnesium":           (70.0, 162.0, 99.7),
    "total_phenols":       (0.98, 3.88, 2.30),
    "flavanoids":          (0.34, 5.08, 2.03),
    "nonflavanoid_phenols":(0.13, 0.66, 0.36),
    "proanthocyanins":     (0.41, 3.58, 1.59),
    "color_intensity":     (1.28, 13.0, 5.06),
    "hue":                 (0.48, 1.71, 0.96),
    "od280_od315":         (1.27, 4.00, 2.61),
    "proline":             (278.0, 1680.0, 746.0),
}

LOGGER = get_logger(__name__)


def send_prediction(client_input: dict) -> None:
    """Send prediction request to FastAPI backend and display the result."""
    result_container = st.empty()
    prob_container = st.empty()

    try:
        with st.spinner("Classifying wine..."):
            response = requests.post(
                f"{FASTAPI_BACKEND_ENDPOINT}/predict",
                json=client_input,
                timeout=10,
            )

        if response.status_code == 200:
            result = response.json()
            pred_class = result["class_name"]
            result_container.success(f"Predicted cultivar: **{pred_class}**")

            prob_df = pd.DataFrame(
                list(result["probabilities"].items()),
                columns=["Class", "Probability"],
            )
            prob_container.bar_chart(prob_df.set_index("Class"))
        else:
            st.toast(
                f":red[Server returned status {response.status_code}. Check backend.]",
                icon="🔴",
            )
    except requests.ConnectionError:
        st.toast(":red[Cannot reach backend. Start the FastAPI server.]", icon="🔴")
    except Exception as e:
        st.toast(":red[Prediction failed. Check logs.]", icon="🔴")
        LOGGER.error(f"Prediction error: {e}")


def run():
    st.set_page_config(
        page_title="Wine Classification Dashboard",
        page_icon="🍷",
        layout="wide",
    )

    # --- Sidebar ---
    with st.sidebar:
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT, timeout=5)
            if backend_request.status_code == 200:
                st.success("Backend online")
            else:
                st.warning("Problem connecting to backend")
        except requests.ConnectionError:
            st.error("Backend offline — start the FastAPI server")

        st.divider()
        st.subheader("Input Mode")
        input_mode = st.radio(
            "Choose how to provide features:",
            ["Sliders", "Upload JSON"],
            horizontal=True,
        )

        slider_values = {}
        test_input_data = None

        if input_mode == "Sliders":
            st.info("Adjust the 13 wine chemical measurements below")
            for feat, (lo, hi, default) in FEATURE_RANGES.items():
                step = round((hi - lo) / 100, 3) or 0.01
                slider_values[feat] = st.slider(
                    feat.replace("_", " ").title(),
                    min_value=lo,
                    max_value=hi,
                    value=default,
                    step=step,
                    format="%.2f",
                )
        else:
            uploaded = st.file_uploader("Upload test JSON file", type=["json"])
            if uploaded:
                st.write("Preview:")
                test_input_data = json.load(uploaded)
                st.json(test_input_data)
                st.session_state["uploaded_json"] = test_input_data
            elif "uploaded_json" in st.session_state:
                test_input_data = st.session_state["uploaded_json"]

        predict_button = st.button("Predict", type="primary", use_container_width=True)

    # --- Main Body ---
    st.write("# Wine Cultivar Classification 🍷")
    st.markdown(
        "This dashboard uses a **Random Forest** model trained on the "
        "[UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine) "
        "to classify wines into one of three Italian cultivar classes based on "
        "13 chemical analysis measurements."
    )

    col1, col2 = st.columns(2)

    if METRICS_PATH.is_file():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        col1.metric("Model Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("Number of Classes", "3")

    if FEATURE_IMPORTANCE_PATH.is_file():
        with open(FEATURE_IMPORTANCE_PATH) as f:
            importance = json.load(f)
        with st.expander("Feature Importance (from trained model)"):
            imp_df = pd.DataFrame(
                sorted(importance.items(), key=lambda x: x[1], reverse=True),
                columns=["Feature", "Importance"],
            )
            st.bar_chart(imp_df.set_index("Feature"))

    st.divider()

    if predict_button:
        if not FASTAPI_WINE_MODEL_LOCATION.is_file():
            st.toast(":red[Model not found. Run train.py first.]", icon="🔥")
            LOGGER.warning("wine_model.pkl not found")
        else:
            if input_mode == "Sliders":
                send_prediction(slider_values)
            elif test_input_data and "input_test" in test_input_data:
                send_prediction(test_input_data["input_test"])
            else:
                st.warning("Upload a valid JSON file before predicting.")


if __name__ == "__main__":
    run()
