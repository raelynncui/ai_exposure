"""
app.py

Setup:
    1. pip install -r requirements.txt
    2. streamlit run app.py
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    "Map, data analysis, and model results for predicting AI exposure at the census tract level."
)

tab_map, tab_analysis, tab_models, tab_importance, tab_data = st.tabs(
    [
        "Map",
        "Data Analysis",
        "Model Results",
        "Feature Importance",
        "Data Preview",
    ]
)

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


###########################################################################
# Tab 2: Data Analysis
###########################################################################

with tab_analysis:
    st.subheader("Exploratory Data Analysis")

    target_col = "AI Exposure Score (0-0.29)"

    numeric_cols = [
        c for c in df.columns
        if c not in ["FIPS", target_col] and pd.api.types.is_numeric_dtype(df[c])
    ]

    st.markdown(
        """
        This section summarizes the main exploratory data analysis used before modeling:
        target distribution, feature distributions, and feature correlations.
        """
    )

    # Target distribution
    st.markdown("### Distribution of AI Exposure Score")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[target_col].dropna(), bins=50, edgecolor="white", linewidth=0.3)
    ax.set_title("Distribution of AI Exposure Score")
    ax.set_xlabel("AI Exposure Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.markdown(
        """
        The target distribution shows how AI exposure varies across census tracts.
        We evaluate models using regression metrics such as RMSE.
        """
    )

    # Selected feature distribution
    st.markdown("### Distribution of Selected Feature")

    selected_analysis_feature = st.selectbox(
        "Choose a feature to inspect",
        options=numeric_cols,
        index=0,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    feature_data = df[selected_analysis_feature].dropna()
    ax.hist(feature_data, bins=50, edgecolor="white", linewidth=0.3)
    ax.axvline(
        feature_data.median(),
        linestyle="--",
        linewidth=1,
        label=f"median = {feature_data.median():.3g}",
    )
    ax.set_title(f"Distribution of {selected_analysis_feature}")
    ax.set_xlabel(selected_analysis_feature)
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    # Correlation heatmap
    st.markdown("### Correlation Heatmap")

    corr_features = st.multiselect(
        "Choose features for correlation heatmap",
        options=numeric_cols,
        default=numeric_cols[: min(12, len(numeric_cols))],
    )

    if len(corr_features) >= 2:
        corr = df[corr_features].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, aspect="auto")
        ax.set_xticks(range(len(corr_features)))
        ax.set_yticks(range(len(corr_features)))
        ax.set_xticklabels(corr_features, rotation=90)
        ax.set_yticklabels(corr_features)
        ax.set_title("Correlation Heatmap")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("Select at least two features to show the correlation heatmap.")

    # All feature distributions
    st.markdown("### Feature Distribution Grid")

    show_grid = st.checkbox("Show distribution grid for all numeric features", value=False)

    if show_grid:
        cols_to_plot = numeric_cols[:30]

        n_cols = 3
        n_rows = int(np.ceil(len(cols_to_plot) / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(14, 3.5 * n_rows),
        )

        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(cols_to_plot):
            ax = axes[i]
            data = df[col].dropna()
            ax.hist(data, bins=50, edgecolor="white", linewidth=0.3)
            ax.axvline(data.median(), linestyle="--", linewidth=1)
            ax.set_title(col, fontsize=9)
            ax.set_ylabel("Count")

        for j in range(len(cols_to_plot), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Distribution of Numeric Features", fontsize=14, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)


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

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(plot_df["model"], plot_df["test_rmse"])
            ax.set_xlabel("Test RMSE")
            ax.set_title("Model Comparison by Test RMSE")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("model_results.csv must contain columns: model, test_rmse")

        st.markdown(
            """
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

        plot_df = top_imp.sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(plot_df["feature"], plot_df["importance"])
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {k} Random Forest Feature Importances")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(
            """
            **Interpretation:**  
            Features with larger importance values were more useful for Random Forest prediction.
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