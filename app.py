"""
app.py

Setup:
    1. pip install -r requirements.txt
    2. streamlit run app.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium

from map_utils import (
    build_choropleth_map,
    load_geojson,
    FEATURE_LABELS,
)


st.set_page_config(
    page_title="AI Exposure Dashboard",
    layout="wide",
)


DATA_PATH = "data/features.csv"
MODEL_RESULTS_PATH = "data/model_results.csv"
RF_IMPORTANCE_PATH = "data/rf_feature_importance.csv"


@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(11)
    return df


@st.cache_data(show_spinner="Loading result file...")
def load_optional_csv(path: str):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


st.sidebar.title("Controls")

df_full = load_data(DATA_PATH)

selected_feature = st.sidebar.selectbox(
    "Map variable",
    options=list(FEATURE_LABELS.keys()),
    index=0,
    format_func=lambda x: FEATURE_LABELS[x],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter by State")

state_options = sorted(df_full["FIPS"].str[:2].unique())

selected_states = st.sidebar.multiselect(
    "State FIPS codes (blank = all)",
    options=state_options,
    default=[],
)

df = (
    df_full[df_full["FIPS"].str[:2].isin(selected_states)].copy()
    if selected_states
    else df_full.copy()
)

st.title("AI Exposure Dashboard")
st.caption(
    "Map and model results for predicting AI exposure at the census tract level."
)

tab_map, tab_models, tab_importance, tab_data = st.tabs(
    [
        "Map",
        "Model Results",
        "Feature Importance",
        "Data Preview",
    ]
)


# Map

with tab_map:
    st.subheader("Interactive Map")

    col1, col2, col3, col4 = st.columns(4)

    series = df[selected_feature].dropna()

    col1.metric("Mean", f"{series.mean():.4g}")
    col2.metric("Median", f"{series.median():.4g}")
    col3.metric("Min", f"{series.min():.4g}")
    col4.metric("Max", f"{series.max():.4g}")

    st.markdown("---")

    geojson = load_geojson()

    center = [39.5, -98.35]
    zoom = 6 if selected_states else 4

    fmap = build_choropleth_map(
        df=df,
        geojson=geojson,
        feature_col=selected_feature,
        colorscale="YlOrRd",
        center=center,
        zoom=zoom,
    )

    st_folium(
        fmap,
        width="100%",
        height=600,
        returned_objects=[],
    )


# Model Results

with tab_models:
    st.subheader("Model Performance")

    model_results = load_optional_csv(MODEL_RESULTS_PATH)

    if model_results is None:
        st.warning(
            "No model_results.csv found. Run models.ipynb first and save results to data/model_results.csv."
        )
    else:
        st.markdown(
            """
            This section compares models trained to predict AI exposure from census tract-level
            demographic, economic, occupational, commute, and digital access features.
            """
        )

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True,
        )

        if {"model", "test_rmse"}.issubset(model_results.columns):
            plot_df = model_results.sort_values("test_rmse", ascending=True)

            fig = px.bar(
                plot_df,
                x="test_rmse",
                y="model",
                orientation="h",
                title="Model Comparison by Test RMSE",
                labels={
                    "test_rmse": "Test RMSE",
                    "model": "Model",
                },
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=450,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("model_results.csv must contain columns: model, test_rmse")

        st.markdown(
            """
            **Interpretation:**  
            Test RMSE measures how well each model predicts AI exposure on held-out census tracts.
            Lower test RMSE means better predictive performance.
            """
        )

# Feature Importance

with tab_importance:
    st.subheader("Random Forest Feature Importance")

    rf_importance = load_optional_csv(RF_IMPORTANCE_PATH)

    if rf_importance is None:
        st.warning(
            "No rf_feature_importance.csv found. Run models.ipynb first and save results to data/rf_feature_importance.csv."
        )
    elif not {"feature", "importance"}.issubset(rf_importance.columns):
        st.error("rf_feature_importance.csv must contain columns: feature, importance")
    else:
        st.markdown(
            """
            This section shows which features contributed most to the Random Forest model's predictions.
            These importances are model-based, so they reflect how useful each feature was when combined
            with the rest of the feature set.
            """
        )

        k = st.slider(
            "Number of top features to show",
            min_value=5,
            max_value=min(25, len(rf_importance)),
            value=min(10, len(rf_importance)),
        )

        top_imp = (
            rf_importance.sort_values("importance", ascending=False)
            .head(k)
        )

        st.dataframe(
            top_imp,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            top_imp.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            orientation="h",
            title=f"Top {k} Random Forest Feature Importances",
            labels={
                "importance": "Feature Importance",
                "feature": "Feature",
            },
        )

        fig.update_layout(height=500)

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
            **Interpretation:**  
            Features with larger importance values were more useful for Random Forest prediction.
            This helps answer the main project question: which demographic and socioeconomic factors
            best predict AI exposure?
            """
        )

# Data Preview

with tab_data:
    st.subheader("Data Preview")

    display_cols = ["FIPS", selected_feature] + [
        c for c in FEATURE_LABELS
        if c != selected_feature and c in df.columns
    ]

    st.dataframe(
        df[display_cols].sort_values(selected_feature, ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing selected feature", int(df[selected_feature].isna().sum()))