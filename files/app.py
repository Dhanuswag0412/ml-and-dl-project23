"""
An Intelligent Environmental Data Analytics System
for Climate Pattern Analysis Using Machine Learning
-----------------------------------------------------
Streamlit application (app.py)

Pages:
  - Overview Dashboard   : KPIs + trend / distribution visuals
  - Data Explorer        : filterable table + download
  - Predict Climate Risk : live ML inference w/ probability breakdown
  - Model Insights       : feature importance, metrics, cluster patterns
  - About                : methodology & tech stack
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Climate Pattern Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "climate_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "environmental_data.csv")

# ------------------------------------------------------------------
# THEME / CUSTOM CSS  (pro dashboard look & feel)
# ------------------------------------------------------------------
PRIMARY = "#0EA5A3"
PRIMARY_DARK = "#0B7F7D"
ACCENT = "#F2994A"
BG_DARK = "#0F1720"
CARD_BG = "#111C26"
TEXT_MUTED = "#9BB0BE"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: radial-gradient(1200px 600px at 10% -10%, #12313033 0%, transparent 60%),
                    linear-gradient(180deg, {BG_DARK} 0%, #0B121A 100%);
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0C1620 0%, #0A121A 100%);
        border-right: 1px solid #1E2C38;
    }}

    h1, h2, h3, h4 {{
        color: #EAF3F3 !important;
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }}

    p, span, label, div {{
        font-family: 'Segoe UI', 'Helvetica Neue', sans-serif;
    }}

    .hero {{
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(120deg, {PRIMARY_DARK} 0%, {PRIMARY} 55%, #14464a 100%);
        box-shadow: 0 12px 30px rgba(14, 165, 163, 0.20);
        margin-bottom: 26px;
    }}
    .hero h1 {{
        color: white !important;
        font-size: 2rem;
        margin-bottom: 4px;
    }}
    .hero p {{
        color: #E9FBF9 !important;
        font-size: 0.98rem;
        margin: 0;
    }}

    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid #1E2C38;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        transition: transform 0.15s ease;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); }}
    .kpi-label {{
        color: {TEXT_MUTED};
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: #F5FBFA;
        font-size: 1.65rem;
        font-weight: 700;
    }}
    .kpi-delta-up {{ color: #F26B6B; font-size: 0.82rem; }}
    .kpi-delta-down {{ color: #4ADE80; font-size: 0.82rem; }}

    .section-card {{
        background: {CARD_BG};
        border: 1px solid #1E2C38;
        border-radius: 16px;
        padding: 22px 24px;
        margin-bottom: 20px;
    }}

    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(14,165,163,0.15);
        color: {PRIMARY};
        border: 1px solid rgba(14,165,163,0.4);
    }}

    .risk-Normal {{ color: #4ADE80; }}
    .risk-Heatwave-Risk {{ color: #F2994A; }}
    .risk-Drought-Risk {{ color: #E0B341; }}
    .risk-Flood-Risk {{ color: #4EA1F2; }}
    .risk-Extreme-Weather {{ color: #F2554A; }}

    div[data-testid="stMetric"] {{
        background: {CARD_BG};
        border: 1px solid #1E2C38;
        border-radius: 14px;
        padding: 12px 16px;
    }}

    .stButton > button {{
        background: linear-gradient(120deg, {PRIMARY_DARK}, {PRIMARY});
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        box-shadow: 0 6px 16px rgba(14,165,163,0.25);
    }}
    .stButton > button:hover {{
        filter: brightness(1.08);
    }}

    footer {{visibility: hidden;}}
    #MainMenu {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

RISK_COLORS = {
    "Normal": "#4ADE80",
    "Heatwave Risk": "#F2994A",
    "Drought Risk": "#E0B341",
    "Flood Risk": "#4EA1F2",
    "Extreme Weather": "#F2554A",
}

# ------------------------------------------------------------------
# DATA / MODEL LOADING
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model_bundle():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


bundle = load_model_bundle()
df_default = load_dataset()

model = bundle["model"]
scaler = bundle["scaler"]
label_encoder = bundle["label_encoder"]
region_encoder = bundle["region_encoder"]
kmeans = bundle["kmeans"]
FEATURE_COLS = bundle["feature_cols"]
CLASS_NAMES = bundle["class_names"]
REGIONS = bundle["regions"]
FEATURE_IMPORTANCE = bundle["feature_importance"]
METRICS = bundle["metrics"]

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.markdown(
    "<div style='text-align:center; padding: 10px 0 4px;'>"
    "<span style='font-size:2rem;'>🌍</span><br>"
    "<span style='color:#EAF3F3; font-weight:700; font-size:1.05rem;'>Climate Analytics</span><br>"
    "<span style='color:#7C93A2; font-size:0.78rem;'>ML-Powered Environmental Intelligence</span>"
    "</div><hr style='border-color:#1E2C38;'>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview Dashboard", "🔎 Data Explorer", "🤖 Predict Climate Risk",
     "🧠 Model Insights", "ℹ️ About"],
    label_visibility="collapsed",
)

st.sidebar.markdown("<hr style='border-color:#1E2C38;'>", unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader(
    "Upload your own environmental CSV", type=["csv"],
    help="Columns expected: Region, Year, Month, Temperature_C, Humidity_pct, "
         "Rainfall_mm, CO2_ppm, Wind_Speed_kmh, Pressure_hPa"
)

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {len(df):,} custom records")
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
        df = df_default
else:
    df = df_default

st.sidebar.markdown(
    f"<div style='margin-top:10px; color:#5A7280; font-size:0.75rem;'>"
    f"Model accuracy: {METRICS['accuracy']*100:.1f}% &nbsp;|&nbsp; "
    f"F1 (weighted): {METRICS['f1_weighted']*100:.1f}%</div>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# HERO HEADER
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🌍 Intelligent Environmental Data Analytics System</h1>
        <p>Climate Pattern Analysis powered by Machine Learning — real-time risk
        classification, historical trend intelligence, and unsupervised pattern discovery.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================================
# PAGE 1 — OVERVIEW DASHBOARD
# ==================================================================
if page == "📊 Overview Dashboard":

    c1, c2, c3, c4 = st.columns(4)
    kpis = [
        ("Total Records", f"{len(df):,}", c1),
        ("Avg. Temperature", f"{df['Temperature_C'].mean():.1f} °C", c2),
        ("Avg. CO₂ Level", f"{df['CO2_ppm'].mean():.0f} ppm", c3),
        ("Avg. Rainfall", f"{df['Rainfall_mm'].mean():.0f} mm", c4),
    ]
    for label, value, col in kpis:
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{value}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("#### 📈 Temperature Trend by Region")
        trend = df.groupby(["Year", "Region"], as_index=False)["Temperature_C"].mean()
        fig = px.line(
            trend, x="Year", y="Temperature_C", color="Region", markers=True,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", legend_title_text="",
            margin=dict(l=10, r=10, t=10, b=10), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### 🧭 Climate Risk Distribution")
        if "Climate_Risk_Category" in df.columns:
            risk_counts = df["Climate_Risk_Category"].value_counts().reset_index()
            risk_counts.columns = ["Category", "Count"]
            fig = px.pie(
                risk_counts, names="Category", values="Count", hole=0.55,
                color="Category", color_discrete_map=RISK_COLORS,
            )
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10), height=380,
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload data with a 'Climate_Risk_Category' column to see this chart, "
                     "or use the bundled dataset.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🌧️ Rainfall Distribution")
        fig = px.histogram(
            df, x="Rainfall_mm", nbins=40, color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=10, b=10), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### 🔗 Feature Correlation Heatmap")
        numeric_cols = [c for c in FEATURE_COLS if c in df.columns]
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="Teal", aspect="auto"
        )
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10), height=340,
        )
        st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# PAGE 2 — DATA EXPLORER
# ==================================================================
elif page == "🔎 Data Explorer":
    st.markdown("### 🔎 Explore the Environmental Dataset")

    with st.container():
        f1, f2, f3 = st.columns([1.2, 1.2, 1])
        with f1:
            region_filter = st.multiselect(
                "Region", options=sorted(df["Region"].unique()) if "Region" in df.columns else [],
                default=None,
            )
        with f2:
            if "Year" in df.columns:
                yr_min, yr_max = int(df["Year"].min()), int(df["Year"].max())
                year_range = st.slider("Year range", yr_min, yr_max, (yr_min, yr_max))
            else:
                year_range = None
        with f3:
            if "Climate_Risk_Category" in df.columns:
                risk_filter = st.multiselect(
                    "Risk category", options=sorted(df["Climate_Risk_Category"].unique())
                )
            else:
                risk_filter = []

    filtered = df.copy()
    if region_filter:
        filtered = filtered[filtered["Region"].isin(region_filter)]
    if year_range and "Year" in filtered.columns:
        filtered = filtered[filtered["Year"].between(year_range[0], year_range[1])]
    if risk_filter:
        filtered = filtered[filtered["Climate_Risk_Category"].isin(risk_filter)]

    st.markdown(
        f"<span class='badge'>{len(filtered):,} of {len(df):,} records shown</span>",
        unsafe_allow_html=True,
    )
    st.write("")
    st.dataframe(filtered, use_container_width=True, height=420)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data as CSV", data=csv_bytes,
        file_name="filtered_environmental_data.csv", mime="text/csv",
    )

    st.markdown("#### Summary Statistics")
    st.dataframe(filtered.describe().T, use_container_width=True)


# ==================================================================
# PAGE 3 — PREDICT CLIMATE RISK
# ==================================================================
elif page == "🤖 Predict Climate Risk":
    st.markdown("### 🤖 Predict Climate Risk Category")
    st.markdown(
        "<p style='color:#9BB0BE;'>Enter current environmental sensor readings to get "
        "an instant ML-based climate risk classification.</p>",
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        colA, colB, colC = st.columns(3)

        with colA:
            region_in = st.selectbox("Region", options=REGIONS)
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 27.0, 0.1)
        with colB:
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0, 0.5)
            rainfall = st.slider("Rainfall (mm)", 0.0, 400.0, 100.0, 1.0)
        with colC:
            co2 = st.slider("CO₂ Level (ppm)", 350.0, 500.0, 415.0, 0.5)
            wind = st.slider("Wind Speed (km/h)", 0.0, 60.0, 10.0, 0.5)

        pressure = st.slider("Atmospheric Pressure (hPa)", 960.0, 1045.0, 1013.0, 0.5)

        predict_clicked = st.button("🔍 Analyze Climate Risk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if predict_clicked:
        region_encoded = region_encoder.transform([region_in])[0]
        input_row = pd.DataFrame([{
            "Temperature_C": temperature,
            "Humidity_pct": humidity,
            "Rainfall_mm": rainfall,
            "CO2_ppm": co2,
            "Wind_Speed_kmh": wind,
            "Pressure_hPa": pressure,
            "Region_encoded": region_encoded,
        }])

        input_scaled = scaler.transform(input_row)
        pred_encoded = model.predict(input_scaled)[0]
        pred_proba = model.predict_proba(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]

        cluster_id = kmeans.predict(input_scaled)[0]

        st.write("")
        result_col, proba_col = st.columns([1, 1.3])

        with result_col:
            color = RISK_COLORS.get(pred_label, PRIMARY)
            st.markdown(
                f"""
                <div class="section-card" style="text-align:center;">
                    <div class="kpi-label">Predicted Climate Risk</div>
                    <div style="font-size:1.9rem; font-weight:800; color:{color}; margin:10px 0;">
                        {pred_label}
                    </div>
                    <div class="badge">Confidence: {pred_proba.max()*100:.1f}%</div>
                    <div style="margin-top:14px; color:#9BB0BE; font-size:0.85rem;">
                        Unsupervised pattern cluster: <b style="color:#EAF3F3;">Cluster {cluster_id}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            advice = {
                "Normal": "Conditions are within expected seasonal norms. Continue routine monitoring.",
                "Heatwave Risk": "Elevated temperature + low humidity detected. Recommend heat-alert protocols and hydration advisories.",
                "Drought Risk": "Low rainfall and low humidity detected. Recommend water conservation measures and irrigation planning.",
                "Flood Risk": "High rainfall and humidity detected. Recommend flood-preparedness advisories for low-lying areas.",
                "Extreme Weather": "High wind speed / low pressure detected — consistent with storm formation. Recommend issuing an extreme-weather advisory.",
            }
            st.info(advice.get(pred_label, ""))

        with proba_col:
            proba_df = pd.DataFrame({
                "Category": label_encoder.classes_,
                "Probability": pred_proba,
            }).sort_values("Probability", ascending=True)
            fig = px.bar(
                proba_df, x="Probability", y="Category", orientation="h",
                color="Category", color_discrete_map=RISK_COLORS, text="Probability",
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                xaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10), height=340,
                title="Prediction Probability Breakdown",
            )
            st.plotly_chart(fig, use_container_width=True)


# ==================================================================
# PAGE 4 — MODEL INSIGHTS
# ==================================================================
elif page == "🧠 Model Insights":
    st.markdown("### 🧠 Model Performance & Explainability")

    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "Random Forest Classifier")
    m2.metric("Accuracy", f"{METRICS['accuracy']*100:.1f}%")
    m3.metric("Weighted F1-score", f"{METRICS['f1_weighted']*100:.1f}%")

    st.write("")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### 🌟 Feature Importance")
        fi_df = pd.DataFrame({
            "Feature": list(FEATURE_IMPORTANCE.keys()),
            "Importance": list(FEATURE_IMPORTANCE.values()),
        }).sort_values("Importance", ascending=True)
        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Teal",
        )
        fig.update_layout(
            template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10), height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🧩 Unsupervised Pattern Clusters (PCA view)")
        numeric_cols = [c for c in FEATURE_COLS if c in df.columns]
        if set(numeric_cols).issubset(df.columns) and "Region" in df.columns:
            X_all = df[numeric_cols].copy()
            X_all["Region_encoded"] = region_encoder.transform(df["Region"])
            X_scaled_all = scaler.transform(X_all)
            clusters = kmeans.predict(X_scaled_all)

            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled_all)
            plot_df = pd.DataFrame({
                "PC1": coords[:, 0], "PC2": coords[:, 1],
                "Cluster": [f"Cluster {c}" for c in clusters],
            })
            fig = px.scatter(
                plot_df, x="PC1", y="PC2", color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2, opacity=0.75,
            )
            fig.update_layout(
                template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10), height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Cluster view requires the standard feature columns + Region.")

    st.markdown("#### 📋 Classification Report Summary")
    st.markdown(
        f"""
        <div class="section-card">
        The Random Forest model was trained on <b>{len(df_default):,}</b> historical
        environmental records spanning <b>{df_default['Year'].nunique()}</b> years across
        <b>{df_default['Region'].nunique()}</b> regions. It classifies conditions into five
        climate risk categories: <span class="risk-Normal">Normal</span>,
        <span class="risk-Heatwave-Risk">Heatwave Risk</span>,
        <span class="risk-Drought-Risk">Drought Risk</span>,
        <span class="risk-Flood-Risk">Flood Risk</span>, and
        <span class="risk-Extreme-Weather">Extreme Weather</span> — achieving
        <b>{METRICS['accuracy']*100:.1f}% test accuracy</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================
# PAGE 5 — ABOUT
# ==================================================================
elif page == "ℹ️ About":
    st.markdown("### ℹ️ About This Project")
    st.markdown(
        """
        <div class="section-card">
        <h4>An Intelligent Environmental Data Analytics System for Climate Pattern
        Analysis Using Machine Learning</h4>
        <p style="color:#B7C6CF;">
        This system ingests environmental sensor readings (temperature, humidity,
        rainfall, CO₂, wind speed, atmospheric pressure) and applies machine learning
        to:
        </p>
        <ul style="color:#B7C6CF;">
            <li><b>Classify</b> real-time conditions into climate risk categories using a
            supervised <b>Random Forest Classifier</b>.</li>
            <li><b>Discover</b> latent climate regimes/patterns using unsupervised
            <b>K-Means clustering</b>.</li>
            <li><b>Visualize</b> historical trends, correlations, and regional variation
            through an interactive analytics dashboard.</li>
        </ul>

        <h4>🛠 Tech Stack</h4>
        <ul style="color:#B7C6CF;">
            <li><b>Python</b> — core language</li>
            <li><b>scikit-learn</b> — RandomForestClassifier, KMeans, StandardScaler</li>
            <li><b>pandas / numpy</b> — data processing</li>
            <li><b>Plotly</b> — interactive visualizations</li>
            <li><b>Streamlit</b> — web application framework & deployment</li>
        </ul>

        <h4>📁 Project Structure</h4>
        <pre style="color:#8FE3DD; background:#0B141C; padding:14px; border-radius:10px;">
climate_ml_project/
├── app.py                     # Streamlit application (this app)
├── train_model.py             # Data generation + model training script
├── requirements.txt           # Python dependencies
├── model/
│   └── climate_model.pkl      # Trained model bundle (RF + scaler + encoders + KMeans)
└── data/
    └── environmental_data.csv # Synthetic environmental dataset
        </pre>

        <h4>🔄 Reproducing / Retraining</h4>
        <p style="color:#B7C6CF;">
        Run <code>python train_model.py</code> to regenerate the dataset and retrain the
        model bundle from scratch — or replace <code>data/environmental_data.csv</code>
        with your own real-world sensor data (same column schema) and retrain.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:#5A7280; font-size:0.8rem;'>"
        "Built with Streamlit • Machine Learning for Environmental Intelligence"
        "</p>",
        unsafe_allow_html=True,
    )
