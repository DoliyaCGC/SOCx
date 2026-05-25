import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="SOC Analysis System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

FEATURE_COLUMNS = [
    "Annual Rainfall (mm)",
    "Vegetation Index (NDVI)",
    "Soil pH",
    "Clay Content (%)",
]

INPUT_BOUNDS = {
    "Annual Rainfall (mm)": (250, 2500),
    "Vegetation Index (NDVI)": (0.05, 0.95),
    "Soil pH": (3.5, 9.5),
    "Clay Content (%)": (5, 65),
}


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(46, 125, 50, 0.14), transparent 28%),
                    radial-gradient(circle at top right, rgba(156, 204, 101, 0.15), transparent 30%),
                    linear-gradient(180deg, #f6fbf4 0%, #eef5eb 100%);
                color: #17311f;
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1f4d2a 0%, #295d34 100%);
                color: #ffffff;
            }
            [data-testid="stSidebar"] * {
                color: #ffffff !important;
            }
            .hero-card,
            .metric-card,
            .status-card,
            .recommendation-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(31, 77, 42, 0.10);
                border-radius: 18px;
                box-shadow: 0 16px 40px rgba(38, 70, 45, 0.08);
                padding: 1.2rem 1.35rem;
            }
            .hero-card h1,
            .metric-card h3,
            .status-card h3,
            .recommendation-card h3 {
                color: #16331f;
                margin-bottom: 0.35rem;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.9rem;
            }
            .status-low {
                border-left: 6px solid #c62828;
            }
            .status-optimal {
                border-left: 6px solid #2e7d32;
            }
            .status-high {
                border-left: 6px solid #ef6c00;
            }
            .small-muted {
                color: #577060;
                font-size: 0.92rem;
            }
            div[data-testid="stMetric"] {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(31, 77, 42, 0.08);
                border-radius: 16px;
                padding: 0.7rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_synthetic_soc_data(num_rows: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Generate a synthetic SOC dataset with realistic, domain-inspired relationships."""
    rng = np.random.default_rng(random_state)

    rainfall = np.clip(rng.normal(1150, 380, num_rows), 250, 2500)
    ndvi = np.clip(
        0.15 + 0.00022 * rainfall + rng.normal(0, 0.11, num_rows),
        0.05,
        0.95,
    )
    soil_ph = np.clip(
        6.4 - 0.00028 * (rainfall - rainfall.mean()) + rng.normal(0, 0.55, num_rows),
        3.5,
        9.5,
    )
    clay = np.clip(
        18
        + 0.007 * (rainfall - rainfall.mean())
        + 14 * (1 - np.abs(ndvi - 0.62))
        + rng.normal(0, 7, num_rows),
        5,
        65,
    )

    ph_penalty = np.abs(soil_ph - 6.6) * 0.18
    soc = (
        0.20
        + (rainfall / 2500) * 1.55
        + ndvi * 1.85
        + (clay / 65) * 0.65
        - ph_penalty
        + rng.normal(0, 0.18, num_rows)
    )
    soc = np.clip(soc, 0.25, 5.2)

    data = pd.DataFrame(
        {
            "Annual Rainfall (mm)": rainfall.round(2),
            "Vegetation Index (NDVI)": ndvi.round(3),
            "Soil pH": soil_ph.round(2),
            "Clay Content (%)": clay.round(2),
            "SOC (%)": soc.round(3),
        }
    )

    # Add a small amount of missing data so the cleaning pipeline is meaningful.
    for column in FEATURE_COLUMNS:
        missing_index = data.sample(frac=0.015, random_state=random_state + len(column)).index
        data.loc[missing_index, column] = np.nan

    return data


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned["Vegetation Index (NDVI)"] = cleaned["Vegetation Index (NDVI)"].clip(0.05, 0.95)
    cleaned["Soil pH"] = cleaned["Soil pH"].clip(3.5, 9.5)
    cleaned["Clay Content (%)"] = cleaned["Clay Content (%)"].clip(5, 65)
    cleaned["Annual Rainfall (mm)"] = cleaned["Annual Rainfall (mm)"].clip(250, 2500)
    cleaned["SOC (%)"] = cleaned["SOC (%)"].clip(0.1, 6.0)
    return cleaned


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model(random_state: int = 42) -> Dict[str, object]:
    data = clean_dataset(generate_synthetic_soc_data(num_rows=1600, random_state=random_state))

    X = data[FEATURE_COLUMNS]
    y = data["SOC (%)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numerical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLUMNS,
            )
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=16,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    logging.info("Model metrics -> R2 Score: %.4f | MAE: %.4f", r2, mae)
    print(f"Model metrics -> R2 Score: {r2:.4f} | MAE: {mae:.4f}")

    regressor = model.named_steps["regressor"]
    feature_importance = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": regressor.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    return {
        "model": model,
        "data": data,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "r2": r2,
        "mae": mae,
        "feature_importance": feature_importance,
    }


def validate_user_input(values: Dict[str, float]) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    for feature, value in values.items():
        lower, upper = INPUT_BOUNDS[feature]
        if not lower <= value <= upper:
            issues.append(f"{feature} must be between {lower} and {upper}.")

    ph = values["Soil pH"]
    if ph < 4.0:
        issues.append("Soil pH below 4.0 is extremely acidic and usually requires liming before cultivation.")
    elif ph > 9.0:
        issues.append("Soil pH above 9.0 is highly alkaline and may strongly limit nutrient availability.")

    return len(issues) == 0, issues


def predict_soc(model: Pipeline, values: Dict[str, float]) -> float:
    input_frame = pd.DataFrame([values])[FEATURE_COLUMNS]
    predicted_soc = float(model.predict(input_frame)[0])
    return round(max(predicted_soc, 0.1), 2)


def soc_health_status(soc_value: float) -> Tuple[str, str, str]:
    if soc_value < 1.0:
        return "Low", "status-low", "#c62828"
    if soc_value <= 3.0:
        return "Optimal", "status-optimal", "#2e7d32"
    return "High", "status-high", "#ef6c00"


def build_recommendations(predicted_soc: float, soil_ph: float) -> List[str]:
    recommendations: List[str] = []

    if predicted_soc < 1.0:
        recommendations.extend(
            [
                "Apply well-decomposed farmyard manure or compost at the start of the season to rebuild soil carbon.",
                "Introduce cover crops such as clover, vetch, or rye to increase root biomass and protect bare soil.",
                "Reduce excessive tillage and retain crop residues to slow down SOC losses.",
                "Use balanced NPK with organic amendments instead of relying only on quick-release fertilizers.",
            ]
        )
    elif predicted_soc <= 3.0:
        recommendations.extend(
            [
                "Maintain SOC with residue retention, crop rotation, and periodic compost applications.",
                "Use legumes in rotation to support microbial activity and naturally improve soil fertility.",
            ]
        )
    else:
        recommendations.extend(
            [
                "Current SOC is strong; focus on maintaining it through minimum tillage and residue recycling.",
                "Avoid over-application of nitrogen fertilizers because highly carbon-rich soils can already mineralize nutrients efficiently.",
            ]
        )

    if soil_ph < 5.5:
        recommendations.append(
            "Acidic soil detected: apply agricultural lime based on soil test recommendations and consider acid-tolerant cover crops."
        )
    elif soil_ph > 7.8:
        recommendations.append(
            "Alkaline soil detected: use gypsum or elemental sulfur where agronomically appropriate and prefer fertilizers with ammonium-based nitrogen."
        )
    else:
        recommendations.append("Soil pH is within a workable range; prioritize organic matter management and moisture conservation.")

    return recommendations


def render_synced_input(label: str, min_value: float, max_value: float, step: float, default: float) -> float:
    slider_key = f"{label}_slider"
    number_key = f"{label}_number"

    if slider_key not in st.session_state:
        st.session_state[slider_key] = default
    if number_key not in st.session_state:
        st.session_state[number_key] = default

    def slider_to_number() -> None:
        st.session_state[number_key] = st.session_state[slider_key]

    def number_to_slider() -> None:
        st.session_state[slider_key] = st.session_state[number_key]

    st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=slider_key,
        on_change=slider_to_number,
    )
    st.number_input(
        f"{label} (exact)",
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=number_key,
        on_change=number_to_slider,
    )
    return float(st.session_state[number_key])


def build_gauge_chart(predicted_soc: float) -> go.Figure:
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted_soc,
            number={"suffix": "%", "font": {"size": 38, "color": "#17311f"}},
            title={"text": "Predicted SOC", "font": {"size": 22, "color": "#17311f"}},
            gauge={
                "axis": {"range": [0, 5], "tickwidth": 1, "tickcolor": "#17311f"},
                "bar": {"color": "#2e7d32"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#dbe7d7",
                "steps": [
                    {"range": [0, 1], "color": "#ffcdd2"},
                    {"range": [1, 3], "color": "#c8e6c9"},
                    {"range": [3, 5], "color": "#ffe0b2"},
                ],
                "threshold": {
                    "line": {"color": "#bf360c", "width": 4},
                    "thickness": 0.8,
                    "value": predicted_soc,
                },
            },
        )
    )
    gauge.update_layout(height=360, margin=dict(l=20, r=20, t=60, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return gauge


def build_feature_importance_chart(feature_importance: pd.DataFrame) -> go.Figure:
    chart = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#b7d3ad", "#4f8b55", "#16331f"],
        title="Feature Importance",
    )
    chart.update_layout(
        height=360,
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Relative Importance",
        plot_bgcolor="rgba(255,255,255,0.85)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    chart.update_traces(hovertemplate="%{y}: %{x:.3f}<extra></extra>")
    return chart


def build_scatter_plot(data: pd.DataFrame) -> go.Figure:
    scatter = px.scatter(
        data.dropna(),
        x="Annual Rainfall (mm)",
        y="SOC (%)",
        color="Vegetation Index (NDVI)",
        size="Clay Content (%)",
        hover_data=["Soil pH"],
        color_continuous_scale=["#b8d9a8", "#5d9b5f", "#19422a"],
        title="Synthetic SOC Distribution",
        opacity=0.78,
    )
    scatter.update_layout(
        height=420,
        plot_bgcolor="rgba(255,255,255,0.85)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return scatter


def main() -> None:
    inject_custom_css()
    assets = train_model()
    model = assets["model"]
    feature_importance = assets["feature_importance"]
    scatter_data = assets["data"]

    st.markdown(
        """
        <div class="hero-card">
            <h1>Soil Organic Carbon (SOC) Analysis System</h1>
            <p class="small-muted">
                Predict SOC (%) from rainfall, vegetation vigor, soil pH, and clay content using a cached
                Random Forest pipeline trained on synthetic agronomic data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Input Controls")
        st.caption("Use the sliders or exact numeric boxes to simulate field conditions.")

        rainfall = render_synced_input("Annual Rainfall (mm)", 250, 2500, 10, 1100)
        ndvi = render_synced_input("Vegetation Index (NDVI)", 0.05, 0.95, 0.01, 0.58)
        soil_ph = render_synced_input("Soil pH", 3.5, 9.5, 0.1, 6.4)
        clay = render_synced_input("Clay Content (%)", 5.0, 65.0, 1.0, 28.0)

        st.markdown("---")
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    input_values = {
        "Annual Rainfall (mm)": rainfall,
        "Vegetation Index (NDVI)": ndvi,
        "Soil pH": soil_ph,
        "Clay Content (%)": clay,
    }

    is_valid, issues = validate_user_input(input_values)

    predicted_soc = None
    if predict_clicked:
        if not is_valid:
            for issue in issues:
                st.error(issue)
        else:
            predicted_soc = predict_soc(model, input_values)
            st.session_state["predicted_soc"] = predicted_soc
    elif "predicted_soc" in st.session_state:
        predicted_soc = st.session_state["predicted_soc"]

    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Model R² Score", f"{assets['r2']:.3f}")
    with metric_col2:
        st.metric("Model MAE", f"{assets['mae']:.3f}")

    st.caption("Metrics are displayed in the UI and also logged to the terminal during training.")

    top_left, top_right = st.columns([1.1, 0.9])

    with top_left:
        if predicted_soc is not None:
            st.plotly_chart(build_gauge_chart(predicted_soc), use_container_width=True)
        else:
            st.info("Set the field conditions in the sidebar and click Predict to generate the SOC forecast.")

    with top_right:
        if predicted_soc is not None:
            status_label, status_css, accent = soc_health_status(predicted_soc)
            st.markdown(
                f"""
                <div class="status-card {status_css}">
                    <h3>Soil Health Status</h3>
                    <p style="font-size:2rem; font-weight:700; color:{accent}; margin-bottom:0.4rem;">
                        {status_label}
                    </p>
                    <p class="small-muted">
                        Predicted SOC: <strong>{predicted_soc:.2f}%</strong><br/>
                        Thresholds: Low &lt; 1%, Optimal 1-3%, High &gt; 3%
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="recommendation-card"><h3>Recommendations</h3>', unsafe_allow_html=True)
            for recommendation in build_recommendations(predicted_soc, soil_ph):
                st.markdown(f"- {recommendation}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                """
                <div class="metric-card">
                    <h3>Decision Support</h3>
                    <p class="small-muted">
                        The app will translate the prediction into a soil health label and practical field
                        recommendations once a forecast is generated.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(build_feature_importance_chart(feature_importance), use_container_width=True)
    with chart_col2:
        st.plotly_chart(build_scatter_plot(scatter_data), use_container_width=True)

    with st.expander("Preview synthetic training data"):
        st.dataframe(scatter_data.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
