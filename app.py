import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="World Happiness Predictor",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background-color: #F4F8F9; }
.result-box {
    background: white; border-radius: 12px;
    padding: 1.5rem; margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.resource-card {
    background: white; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.7rem;
    border-left: 4px solid #028090;
}
.metric-box {
    background: #028090; border-radius: 10px;
    padding: 1rem; text-align: center; color: white;
}
</style>
""", unsafe_allow_html=True)

FEATURES = [
    'explained_log_gdp_per_capita',
    'explained_social_support',
    'explained_healthy_life_expectancy',
    'explained_freedom',
    'explained_generosity',
    'explained_corruption',
]

FEATURE_LABELS = {
    'explained_log_gdp_per_capita': 'GDP per Capita',
    'explained_social_support': 'Social Support',
    'explained_healthy_life_expectancy': 'Healthy Life Expectancy',
    'explained_freedom': 'Freedom of Choice',
    'explained_generosity': 'Generosity',
    'explained_corruption': 'Low Corruption',
}

FEATURE_RANGES = {
    'explained_log_gdp_per_capita': (0.0, 2.5),
    'explained_social_support': (0.0, 1.5),
    'explained_healthy_life_expectancy': (0.0, 1.2),
    'explained_freedom': (0.0, 0.8),
    'explained_generosity': (0.0, 0.5),
    'explained_corruption': (0.0, 0.5),
}

@st.cache_resource
def load_model():
    improved = Path("models/improved.pkl")
    baseline = Path("models/baseline.pkl")
    if improved.exists():
        with open(improved, "rb") as f:
            return pickle.load(f), "Random Forest"
    elif baseline.exists():
        with open(baseline, "rb") as f:
            return pickle.load(f), "Linear Regression"
    return None, "Demo Mode"

@st.cache_resource
def load_data():
    path = Path("data/processed/clean.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_rag():
    try:
        from src.rag import HappinessRAG
        rag = HappinessRAG()
        rag.index()
        return rag
    except Exception:
        return None

def get_happiness_emoji(score):
    if score >= 7.0: return "😄"
    if score >= 6.0: return "🙂"
    if score >= 5.0: return "😐"
    if score >= 4.0: return "😔"
    return "😢"

def get_happiness_label(score):
    if score >= 7.0: return ("Very Happy", "#02C39A")
    if score >= 6.0: return ("Happy", "#028090")
    if score >= 5.0: return ("Moderate", "#F4A261")
    if score >= 4.0: return ("Unhappy", "#E76F51")
    return ("Very Unhappy", "#E63946")

# ── Header ──
st.markdown("<h1 style='text-align:center'>🌍 World Happiness Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888'>Predict happiness scores + RAG-powered insights from 20 years of global data</p>", unsafe_allow_html=True)
st.markdown("---")

model, model_name = load_model()
df = load_data()
rag = load_rag()

# ── Sidebar ──
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    use_rag = st.toggle("Enable RAG Insights", value=True)
    st.markdown("---")
    st.markdown(f"**Model:** `{model_name}`")
    st.markdown(f"**RAG:** {'✅ Active' if rag else '⚠️ Unavailable'}")
    if df is not None:
        st.markdown(f"**Dataset:** {len(df):,} rows · {df['country'].nunique()} countries")
    st.markdown("---")

    st.markdown("### 🌐 Explore a Country")
    if df is not None:
        countries = sorted(df['country'].unique().tolist())
        selected_country = st.selectbox("Select country:", ["— select —"] + countries)
        if selected_country != "— select —":
            country_data = df[df['country'] == selected_country].sort_values('year').iloc[-1]
            st.markdown(f"**Latest score:** {country_data['happiness_score']:.3f}")
            st.markdown(f"**Year:** {int(country_data['year'])}")
            if st.button("Load into predictor"):
                for f in FEATURES:
                    if f in country_data and not pd.isna(country_data[f]):
                        st.session_state[f] = float(country_data[f])

# ── Main ──
col1, col2 = st.columns([1.2, 0.8], gap="large")

with col1:
    st.markdown("### 🎛️ Adjust Happiness Factors")
    st.caption("Move the sliders to simulate a country's conditions")

    inputs = {}
    for feature in FEATURES:
        min_val, max_val = FEATURE_RANGES[feature]
        default = st.session_state.get(feature, (min_val + max_val) / 2)
        inputs[feature] = st.slider(
            FEATURE_LABELS[feature],
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default),
            step=0.01,
        )

    predict_btn = st.button("🔮 Predict Happiness Score", type="primary", use_container_width=True)

with col2:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app predicts a country's happiness score based on 6 key factors
    from the **World Happiness Report 2005–2025**.

    Adjust the sliders to simulate any combination of conditions and see
    the predicted happiness score instantly.

    **Pipeline:** Data Cleaning → Feature Engineering → Random Forest → RAG Insights
    """)

    if df is not None:
        st.markdown("### 🏆 Happiest Countries (Latest)")
        latest_year = df['year'].max()
        top5 = df[df['year'] == latest_year].nlargest(5, 'happiness_score')[['country', 'happiness_score']]
        for _, row in top5.iterrows():
            emoji = get_happiness_emoji(row['happiness_score'])
            st.markdown(f"{emoji} **{row['country']}** — {row['happiness_score']:.3f}")

# ── Results ──
if predict_btn:
    X = pd.DataFrame([inputs])

    if model is not None:
        score = model.predict(X)[0]
    else:
        score = sum(inputs.values()) * 1.2 + 2.5

    score = round(float(score), 3)
    emoji = get_happiness_emoji(score)
    label, color = get_happiness_label(score)

    st.markdown("---")
    st.markdown("## 🔮 Prediction Results")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""
        <div class="metric-box">
            <div style="font-size:2.5rem">{emoji}</div>
            <div style="font-size:2rem; font-weight:bold">{score}</div>
            <div style="font-size:0.9rem; opacity:0.85">Predicted Score</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="metric-box" style="background:{color}">
            <div style="font-size:2rem">🏷️</div>
            <div style="font-size:1.5rem; font-weight:bold">{label}</div>
            <div style="font-size:0.9rem; opacity:0.85">Category</div>
        </div>
        """, unsafe_allow_html=True)
    with r3:
        if df is not None:
            closest = df.iloc[(df['happiness_score'] - score).abs().argsort()[:1]]
            closest_country = closest['country'].values[0]
            closest_score = closest['happiness_score'].values[0]
            st.markdown(f"""
            <div class="metric-box" style="background:#1A2E44">
                <div style="font-size:2rem">🌐</div>
                <div style="font-size:1.3rem; font-weight:bold">{closest_country}</div>
                <div style="font-size:0.9rem; opacity:0.85">Most Similar Country ({closest_score:.3f})</div>
            </div>
            """, unsafe_allow_html=True)

    # Factor breakdown
    st.markdown("### 📊 Factor Breakdown")
    factor_df = pd.DataFrame({
        'Factor': [FEATURE_LABELS[f] for f in FEATURES],
        'Value': [inputs[f] for f in FEATURES],
    }).sort_values('Value', ascending=False)

    import plotly.express as px
    fig = px.bar(factor_df, x='Value', y='Factor', orientation='h',
                 color='Value', color_continuous_scale='Teal',
                 title="Contribution of Each Factor")
    fig.update_layout(showlegend=False, plot_bgcolor='white', height=300)
    st.plotly_chart(fig, use_container_width=True)

    # RAG
    if use_rag and rag is not None:
        st.markdown("---")
        st.markdown("## 📚 Research Insights")
        st.caption("Retrieved from World Happiness Report knowledge base")
        top_factor = FEATURE_LABELS[max(inputs, key=inputs.get)]
        query = f"What drives happiness? {top_factor} and score of {score}"
        with st.spinner("Retrieving insights..."):
            docs = rag.retrieve(query, n_results=3)
        for i, doc in enumerate(docs, 1):
            with st.expander(f"📄 {doc['metadata']['title']}", expanded=(i == 1)):
                st.markdown(f"""
                <div class="resource-card">
                    <p style="margin:0; font-size:0.88rem; color:#333">{doc['content']}</p>
                    <p style="margin:0.5rem 0 0; font-size:0.75rem; color:#888">📖 {doc['metadata']['source']}</p>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#aaa; font-size:0.75rem'>World Happiness Predictor · Capstone Project · Data: World Happiness Report</p>", unsafe_allow_html=True)
