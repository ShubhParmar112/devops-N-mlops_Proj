"""Streamlit front end for the Diabetes Prediction System.

Run from the project root:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

# Make `src` importable no matter where streamlit is launched from
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.metrics import METRICS_PORT, set_model_info, start_metrics_server
from src.predict import VALID_RANGES, load_metadata, load_model, predict
from src.preprocessing import FEATURES, TARGET, ZERO_AS_MISSING, load_data

st.set_page_config(page_title="Diabetes Prediction System", page_icon="🩺", layout="wide")

TEAL, RED, GREEN = "#0E7C86", "#D1495B", "#4C9F70"

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem;}
      .result-card {border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;}
      .result-pos {background: #FDECEE; border-left: 6px solid #D1495B;}
      .result-neg {background: #EAF6EF; border-left: 6px solid #4C9F70;}
      .result-card h3 {margin: 0 0 .3rem 0;}
      .disclaimer {background: #FFF8E6; border-left: 6px solid #E0A100; border-radius: 8px;
                   padding: .8rem 1rem; font-size: .92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

DISCLAIMER = (
    "<div class='disclaimer'>⚠️ <b>Educational use only.</b> This system was built for an "
    "MLOps/DevOps college practical. Its predictions come from a statistical model trained on a "
    "small public dataset and are <b>not a medical diagnosis</b>. Always consult a qualified "
    "healthcare professional about your health.</div>"
)

# (label, help text, step, default) for each input
INPUT_SPEC = {
    "Pregnancies": ("Pregnancies", "Number of times pregnant", 1, 1),
    "Glucose": ("Glucose (mg/dL)", "Plasma glucose concentration, 2-hour oral glucose tolerance test", 1, 117),
    "BloodPressure": ("Blood Pressure (mm Hg)", "Diastolic blood pressure", 1, 72),
    "SkinThickness": ("Skin Thickness (mm)", "Triceps skin-fold thickness", 1, 23),
    "Insulin": ("Insulin (μU/mL)", "2-hour serum insulin (0 = not measured)", 1, 30),
    "BMI": ("BMI (kg/m²)", "Body mass index", 0.1, 32.0),
    "DiabetesPedigreeFunction": ("Diabetes Pedigree Function", "Genetic likelihood score from family history", 0.001, 0.372),
    "Age": ("Age (years)", "Age in years", 1, 29),
}


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model():
    return load_model()


def feature_importance(model) -> pd.DataFrame | None:
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = abs(est.coef_[0])
    else:
        return None
    return pd.DataFrame({"Feature": FEATURES, "Importance": values}).sort_values("Importance")


def gauge(probability: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Estimated probability of diabetes"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": RED if probability >= 0.5 else GREEN},
            "steps": [{"range": [0, 30], "color": "#EAF6EF"},
                      {"range": [30, 60], "color": "#FFF8E6"},
                      {"range": [60, 100], "color": "#FDECEE"}],
            "threshold": {"line": {"color": "#1B2A36", "width": 3}, "value": 50},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=30, r=30))
    return fig


# ---------------------------------------------------------------- header
st.title("🩺 Diabetes Prediction System")
st.write(
    "Enter eight clinical measurements to estimate whether a person is likely to be "
    "**diabetes-positive** or **diabetes-negative**. The model was trained on the "
    "Pima Indians Diabetes Dataset as part of an end-to-end MLOps and DevOps pipeline "
    "(MLflow → GitHub Actions → Docker → Kubernetes → Prometheus/Grafana)."
)
st.markdown(DISCLAIMER, unsafe_allow_html=True)
st.write("")

try:
    model = get_model()
    metadata = load_metadata()
    metrics_running = start_metrics_server()  # Prometheus endpoint, started once per process
    set_model_info(metadata)
    df = get_data()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

tab_predict, tab_batch, tab_model, tab_data = st.tabs(
    ["🔍 Prediction", "📦 Batch prediction", "🤖 Model information", "📊 Dataset statistics"]
)

# ---------------------------------------------------------------- prediction
with tab_predict:
    with st.form("prediction_form"):
        st.subheader("Patient measurements")
        cols = st.columns(4)
        inputs = {}
        for i, feature in enumerate(FEATURES):
            label, help_text, step, default = INPUT_SPEC[feature]
            lo, hi = VALID_RANGES[feature]
            is_float = isinstance(step, float)
            with cols[i % 4]:
                inputs[feature] = st.number_input(
                    label, min_value=float(lo) if is_float else int(lo),
                    max_value=float(hi) if is_float else int(hi),
                    value=default, step=step, help=help_text,
                    format="%.3f" if feature == "DiabetesPedigreeFunction" else None,
                )
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        try:
            result = predict(inputs)
        except ValueError as e:
            st.error(f"Invalid input: {e}")
        else:
            left, right = st.columns([1, 1])
            with left:
                positive = result["prediction"] == 1
                css = "result-pos" if positive else "result-neg"
                icon = "🔴" if positive else "🟢"
                st.markdown(
                    f"<div class='result-card {css}'><h3>{icon} {result['label']}</h3>"
                    f"The model estimates a <b>{result['probability']:.1%}</b> probability of diabetes "
                    f"(decision threshold 50%).</div>",
                    unsafe_allow_html=True,
                )
                missing = [f for f in ZERO_AS_MISSING if inputs[f] == 0]
                if missing:
                    st.info(f"A value of 0 for {', '.join(missing)} is treated as *not measured* and "
                            "replaced with the typical (median) value from the training data.")
                st.caption("This is an educational estimate, not a diagnosis.")
            with right:
                st.plotly_chart(gauge(result["probability"]), use_container_width=True)

            # Compare the entered values with the dataset averages per class
            means = df.groupby(TARGET)[FEATURES].mean().T
            compare = pd.DataFrame({
                "Your input": pd.Series(inputs, dtype=float),
                "Avg. negative": means[0],
                "Avg. positive": means[1],
            })
            compare.index.name = "Feature"
            long = compare.reset_index().melt(id_vars="Feature", var_name="Group", value_name="Value")
            fig = px.bar(long, x="Feature", y="Value", color="Group", barmode="group",
                         color_discrete_map={"Your input": TEAL, "Avg. negative": GREEN, "Avg. positive": RED},
                         title="Your input compared with dataset averages")
            fig.update_layout(height=400, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- model info
with tab_model:
    m = metadata.get("metrics", {})
    st.subheader(f"Selected model: {metadata.get('model_name', type(model.named_steps['model']).__name__)}")
    st.write(
        f"Chosen by **{metadata.get('selection_metric', 'cv_f1')}** (cross-validated F1), which balances "
        "precision and recall. Accuracy alone is misleading because only ~35% of patients are positive."
    )
    c = st.columns(5)
    for col, (name, key) in zip(c, [("Accuracy", "test_accuracy"), ("Precision", "test_precision"),
                                    ("Recall", "test_recall"), ("F1-score", "test_f1"),
                                    ("ROC-AUC", "test_roc_auc")]):
        col.metric(name, f"{m.get(key, float('nan')):.3f}")
    st.caption(f"Test-set metrics on {metadata.get('test_rows', '?')} held-out patients. "
               f"Trained on {metadata.get('train_rows', '?')} patients at {metadata.get('trained_at', '?')}.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Model comparison (MLflow runs)**")
        comparison = pd.DataFrame(metadata.get("comparison", {})).T
        if not comparison.empty:
            comparison.columns = [c.replace("_", " ").replace("test", "test:").replace("cv", "CV:")
                                  for c in comparison.columns]
            st.dataframe(comparison.style.highlight_max(axis=0, color="#CDEBEE").format("{:.3f}"),
                         use_container_width=True)
        st.markdown("**Hyperparameters**")
        st.json(metadata.get("best_params", {}))
        st.markdown(
            f"**MLflow registry:** `{metadata.get('registered_model', '-')}` "
            f"version **{metadata.get('registered_version', '-')}**  \n"
            f"**Run ID:** `{metadata.get('mlflow_run_id', '-')}`"
        )
    with right:
        imp = feature_importance(model)
        if imp is not None:
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         color_discrete_sequence=[TEAL], title="Feature importance")
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)
        for fig_name in [f"confusion_matrix_{metadata.get('model_name')}.png"]:
            path = PROJECT_ROOT / "reports" / "figures" / fig_name
            if path.exists():
                st.image(str(path), caption="Confusion matrix on the test set", width=360)

# ---------------------------------------------------------------- dataset stats
with tab_data:
    c = st.columns(4)
    c[0].metric("Patients", f"{len(df)}")
    c[1].metric("Features", f"{len(FEATURES)}")
    c[2].metric("Positive cases", f"{int(df[TARGET].sum())}")
    c[3].metric("Positive rate", f"{df[TARGET].mean():.1%}")

    left, right = st.columns(2)
    with left:
        counts = df[TARGET].map({0: "Negative", 1: "Positive"}).value_counts().reset_index()
        counts.columns = ["Outcome", "Count"]
        fig = px.pie(counts, names="Outcome", values="Count", hole=0.5, title="Outcome distribution",
                     color="Outcome", color_discrete_map={"Negative": GREEN, "Positive": RED})
        st.plotly_chart(fig, use_container_width=True)
    with right:
        zeros = (df[ZERO_AS_MISSING] == 0).sum().reset_index()
        zeros.columns = ["Feature", "Zero values"]
        fig = px.bar(zeros, x="Feature", y="Zero values", color_discrete_sequence=[TEAL],
                     title="Zero values treated as missing")
        st.plotly_chart(fig, use_container_width=True)

    feature = st.selectbox("Explore a feature", FEATURES, index=1)
    fig = px.histogram(df, x=feature, color=df[TARGET].map({0: "Negative", 1: "Positive"}),
                       barmode="overlay", nbins=40, opacity=0.7,
                       color_discrete_map={"Negative": GREEN, "Positive": RED},
                       title=f"{feature} distribution by outcome")
    fig.update_layout(legend_title_text="Outcome")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Summary statistics"):
        st.dataframe(df.describe().T.round(2), use_container_width=True)

with tab_batch:
    st.subheader("Batch prediction")
    st.write(
        "Score many patients at once - either a random sample of patients from the dataset "
        "(their real outcome is shown for comparison) or your own CSV file with the eight "
        "feature columns. Every row goes through the same validated, monitored prediction path."
    )

    def run_batch(rows: pd.DataFrame, source: str) -> pd.DataFrame:
        results = []
        for _, row in rows.iterrows():
            try:
                r = predict(row[FEATURES].to_dict(), source=source)
                results.append({"Prediction": r["label"], "Probability": r["probability"], "Error": ""})
            except (ValueError, KeyError) as e:
                results.append({"Prediction": "Failed", "Probability": None, "Error": str(e)})
        return pd.concat([rows.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    left, right = st.columns(2)
    with left:
        n = st.slider("Number of random patients", 10, 200, 50, step=10)
        if st.button("Run batch on random patients", type="primary"):
            sample = df.sample(n)
            out = run_batch(sample, "batch")
            agree = ((out["Prediction"] == "Diabetes Positive").astype(int) == out[TARGET]).mean()
            st.session_state["batch_result"] = (out, f"Agreement with actual outcome: {agree:.1%}")
    with right:
        uploaded = st.file_uploader("...or upload a CSV", type="csv",
                                    help=f"Columns: {', '.join(FEATURES)}")
        if uploaded is not None and st.button("Run batch on uploaded CSV"):
            csv = pd.read_csv(uploaded)
            missing_cols = [f for f in FEATURES if f not in csv.columns]
            if missing_cols:
                st.error(f"CSV is missing columns: {missing_cols}")
            else:
                out = run_batch(csv, "csv")
                failed = int((out["Prediction"] == "Failed").sum())
                st.session_state["batch_result"] = (out, f"{len(out) - failed} rows scored, {failed} failed validation")

    if "batch_result" in st.session_state:
        out, summary = st.session_state["batch_result"]
        st.success(summary)
        counts = out["Prediction"].value_counts()
        c = st.columns(3)
        c[0].metric("Rows", len(out))
        c[1].metric("Predicted positive", int(counts.get("Diabetes Positive", 0)))
        c[2].metric("Predicted negative", int(counts.get("Diabetes Negative", 0)))
        st.dataframe(out, use_container_width=True, height=300)
        st.download_button("Download results as CSV", out.to_csv(index=False), "batch_predictions.csv", "text/csv")

st.divider()
if metrics_running:
    st.caption(f"📈 Prometheus metrics are exposed on port {METRICS_PORT} at /metrics.")
st.caption("Diabetes Prediction System · MLOps & DevOps practical · For educational purposes only.")
