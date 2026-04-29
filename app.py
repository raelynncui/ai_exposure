"""
app.py

Setup:
    1. pip install -r requirements.txt
    2. streamlit run app.py
"""

import streamlit as st
import pandas as pd
from streamlit_folium import st_folium

from map_utils import (
    build_choropleth_map,
    load_geojson,
    FEATURE_LABELS,
)


st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)


DATA_PATH = "data/features.csv"

@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["FIPS"] = df["FIPS"].astype(str).str.zfill(11)
    return df


###########################################################################
# Sidebar
###########################################################################

st.sidebar.title("Controls")

df_full = load_data(DATA_PATH)

# Feature selector
selected_feature = st.sidebar.selectbox(
    "Map variable",
    options=list(FEATURE_LABELS.keys()),
    index=0,
    format_func=lambda x: FEATURE_LABELS[x],
)

# State filter
st.sidebar.markdown("---")
st.sidebar.subheader("Filter by State")
state_options = sorted(df_full["FIPS"].str[:2].unique())
selected_states = st.sidebar.multiselect(
    "State FIPS codes (blank = all)",
    options=state_options,
    default=[],
)

# Apply filter
df = (
    df_full[df_full["FIPS"].str[:2].isin(selected_states)].copy()
    if selected_states
    else df_full.copy()
)


###########################################################################
# Main content
###########################################################################

st.title("Dashboard")
st.caption(
    "Map of AI exposure scores and demographic features at the census tract level."
)

# Summary stats
col1, col2, col3, col4 = st.columns(4)
series = df[selected_feature].dropna()
col1.metric("Mean", f"{series.mean():.4g}")
col2.metric("Median", f"{series.median():.4g}")
col3.metric("Min", f"{series.min():.4g}")
col4.metric("Max", f"{series.max():.4g}")

st.markdown("---")

# Load GeoJSON 
geojson = load_geojson()

# Map center/zoom
if selected_states:
    center = [39.5, -98.35]
    zoom = 6
else:
    center = [39.5, -98.35]
    zoom = 4

fmap = build_choropleth_map(
    df=df,
    geojson=geojson,
    feature_col=selected_feature,
    colorscale="YlOrRd",
    center=center,
    zoom=zoom,
)

st_folium(fmap, width="100%", height=600, returned_objects=[])

st.markdown("---")

# Data preview
with st.expander("Data preview", expanded=False):
    display_cols = ["FIPS", selected_feature] + [
        c for c in FEATURE_LABELS if c != selected_feature and c in df.columns
    ]
    st.dataframe(
        df[display_cols].sort_values(selected_feature, ascending=False),
        use_container_width=True,
        hide_index=True,
    )
