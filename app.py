from pathlib import Path
import json
import re

import joblib
import pandas as pd
import streamlit as st
from scipy.sparse import hstack


PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"
MODEL_DIR = BACKEND_DIR / "models"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
RESEARCH_DIR = PROJECT_DIR / "research"


st.set_page_config(
    page_title="SentinelAI",
    page_icon="shield",
    layout="wide",
)


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_DIR / "best_model.pkl")
    word_vectorizer = joblib.load(PROCESSED_DIR / "word_vectorizer.pkl")
    char_vectorizer = joblib.load(PROCESSED_DIR / "char_vectorizer.pkl")

    with open(MODEL_DIR / "best_model_name.txt", "r", encoding="utf-8") as file:
        model_name = file.read().strip()

    with open(PROCESSED_DIR / "feature_summary.json", "r", encoding="utf-8") as file:
        feature_summary = json.load(file)

    return model, word_vectorizer, char_vectorizer, model_name, feature_summary


@st.cache_data
def load_model_metrics():
    return pd.read_csv(
        RESEARCH_DIR / "model_comparison" / "model_comparison.csv"
    )


def classify_prompt(prompt, model, word_vectorizer, char_vectorizer):
    word_features = word_vectorizer.transform([prompt])
    char_features = char_vectorizer.transform([prompt])
    features = hstack([word_features, char_features]).tocsr()

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    malicious_probability = float(probabilities[1])

    return {
        "prediction": prediction,
        "malicious_probability": malicious_probability,
        "confidence": float(max(probabilities)),
    }


def highlight_terms(prompt):
    terms = [
        "ignore",
        "reveal",
        "system",
        "prompt",
        "guidelines",
        "restrictions",
        "instructions",
        "secret",
        "bypass",
    ]
    highlighted = prompt
    for term in terms:
        highlighted = re.sub(
            rf"(?i)\b({re.escape(term)})\b",
            r'<mark>\1</mark>',
            highlighted,
        )
    return highlighted


model, word_vectorizer, char_vectorizer, model_name, feature_summary = (
    load_model_assets()
)
model_metrics = load_model_metrics()

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap');

    :html, body, [class*="css"] {
        font-family: "DM Sans", sans-serif;
    }

    .stApp {
        background: #f4f7f6;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #10231e;
    }

    [data-testid="stSidebar"] * {
        color: #eef5f2;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: "Manrope", sans-serif;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 28px;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 12px;
        background: #b9ed52;
        color: #10231e;
        font-size: 20px;
    }

    .hero {
        padding: 18px 0 8px;
    }

    .hero h1 {
        font-family: "Manrope", sans-serif;
        font-size: 44px;
        line-height: 1.05;
        letter-spacing: -2px;
        margin: 0 0 10px;
        color: #10231e;
    }

    .hero p {
        color: #6a7873;
        font-size: 16px;
        margin: 0;
    }

    .prompt-card, .metric-card, .result-card {
        background: white;
        border: 1px solid #e4ebe7;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(16, 35, 30, 0.04);
    }

    .prompt-card {
        padding: 18px;
    }

    .metric-card {
        padding: 18px 20px;
        min-height: 118px;
    }

    .metric-card .label {
        color: #7b8984;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    .metric-card .value {
        color: #10231e;
        font-family: "Manrope", sans-serif;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-card .detail {
        color: #98a49f;
        font-size: 12px;
        margin-top: 4px;
    }

    .result-card {
        padding: 22px;
    }

    .status-safe, .status-threat {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 8px 12px;
        font-size: 13px;
        font-weight: 700;
    }

    .status-safe {
        background: #edf8df;
        color: #3e6815;
    }

    .status-threat {
        background: #fff0e8;
        color: #9a4d2d;
    }

    mark {
        background: #e9f7bd;
        color: inherit;
        border-radius: 4px;
        padding: 1px 3px;
    }

    .stTextArea textarea {
        min-height: 150px;
        border: 0;
        background: #f8faf9;
        border-radius: 12px;
    }

    .stButton > button {
        width: 100%;
        border: 0;
        border-radius: 11px;
        background: #b9ed52;
        color: #10231e;
        font-weight: 800;
        height: 48px;
    }

    .stButton > button:hover {
        background: #ace63e;
        color: #10231e;
        border: 0;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4ebe7;
        border-radius: 16px;
        padding: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">S</div> SentinelAI</div>',
        unsafe_allow_html=True,
    )
    st.caption("PROMPT SECURITY CONSOLE")
    st.divider()
    st.markdown("### Model status")
    st.success(f"{model_name} active")
    st.write("Threat classification")
    st.divider()
    st.markdown("### Dataset")
    st.write("15,291 labeled prompts")
    st.write("30,000 combined features")

tab_analyze, tab_research = st.tabs(["Analyzer", "Research"])

with tab_analyze:
    st.markdown(
        """
        <div class="hero">
            <h1>Prompt risk analysis</h1>
            <p>Detect prompt-injection and jailbreak attempts before they reach your AI system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 0.85], gap="large")

    with left:
        st.markdown('<div class="prompt-card">', unsafe_allow_html=True)
        prompt = st.text_area(
            "Prompt",
            value="",
            placeholder="Paste or type a prompt to analyze...",
            label_visibility="collapsed",
        )

        analyze = st.button("Analyze prompt", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if analyze:
            if not prompt.strip():
                st.warning("Enter a prompt before running analysis.")
            else:
                result = classify_prompt(
                    prompt,
                    model,
                    word_vectorizer,
                    char_vectorizer,
                )
                st.session_state.history.insert(
                    0,
                    {
                        "prompt": prompt,
                        "label": "Malicious" if result["prediction"] == 1 else "Benign",
                        "probability": result["malicious_probability"],
                        "confidence": result["confidence"],
                    },
                )
                st.session_state.history = st.session_state.history[:8]
                st.session_state.latest_result = result

    with right:
        result = st.session_state.get("latest_result")
        if result:
            is_malicious = result["prediction"] == 1
            probability = result["malicious_probability"]
            status = "Threat detected" if is_malicious else "Likely benign"
            status_class = "status-threat" if is_malicious else "status-safe"

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(
                f'<span class="{status_class}">{"!" if is_malicious else "✓"} {status}</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<h3>Risk score</h3>", unsafe_allow_html=True)
            st.progress(probability)
            st.metric(
                "Malicious probability",
                f"{probability:.1%}",
                f"{result['confidence']:.1%} confidence",
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<span class="status-safe">✓ Ready</span>', unsafe_allow_html=True)
            st.markdown("<h3>Awaiting analysis</h3>", unsafe_allow_html=True)
            st.write("The classifier is ready to evaluate your prompt.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Recent analyses")
    if st.session_state.history:
        rows = []
        for item in st.session_state.history:
            rows.append(
                {
                    "Prompt": (
                        item["prompt"][:90] + "..."
                        if len(item["prompt"]) > 90
                        else item["prompt"]
                    ),
                    "Classification": item["label"],
                    "Malicious probability": f"{item['probability']:.1%}",
                    "Confidence": f"{item['confidence']:.1%}",
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("No prompts analyzed in this session.")

with tab_research:
    st.markdown(
        """
        <div class="hero">
            <h1>Model research</h1>
            <p>Explore dataset characteristics and benchmark performance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", f"{feature_summary['training_samples'] + feature_summary['validation_samples'] + feature_summary['testing_samples']:,}")
    c2.metric("Features", f"{feature_summary['total_features']:,}")
    c3.metric("Best model", model_name)
    c4.metric("Test accuracy", "96.1%")

    st.markdown("### Model benchmark")
    chart_data = model_metrics.set_index("Model")[[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
    ]]
    st.bar_chart(chart_data, horizontal=False)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.image(
            str(RESEARCH_DIR / "model_comparison" / "best_model_confusion_matrix.png"),
            caption="Naive Bayes confusion matrix",
            width="stretch",
        )
    with col2:
        st.image(
            str(RESEARCH_DIR / "model_comparison" / "model_performance_comparison.png"),
            caption="Model performance comparison",
            width="stretch",
        )

    col3, col4 = st.columns(2, gap="large")
    with col3:
        st.image(
            str(RESEARCH_DIR / "dataset_analysis" / "class_distribution.png"),
            caption="Benign vs malicious distribution",
            width="stretch",
        )
    with col4:
        st.image(
            str(RESEARCH_DIR / "dataset_analysis" / "text_length_distribution.png"),
            caption="Prompt length distribution",
            width="stretch",
        )
