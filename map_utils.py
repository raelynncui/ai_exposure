"""
map_utils.py
Map construction logic for the Streamlit app code
"""

import json
from pathlib import Path

import folium
import pandas as pd
import streamlit as st


GEOJSON_PATH = "data/tracts.geojson"

FEATURE_LABELS = {
    "AI Exposure Score (0-0.29)": "AI Exposure Score",
    "pct_mgmt_business_arts": "% Mgmt / Business / Arts",
    "pct_service": "% Service Occupations",
    "pct_sales_office": "% Sales & Office",
    "pct_natural_resources_construction": "% Natural Resources & Construction",
    "pct_bachelors_plus": "% Bachelor's Degree+",
    "pct_some_college_assoc": "% Some College / Associate's",
    "pct_hs_or_ged": "% HS Diploma / GED",
    "median_household_income": "Median Household Income ($)",
    "per_capita_income": "Per Capita Income ($)",
    "pct_worked_from_home": "% Worked From Home",
    "pct_public_transit": "% Public Transit",
    "pct_broadband": "% Broadband Access",
    "pct_has_computer": "% Has Computer",
    "pct_drove_alone": "% Drove Alone",
    "pct_long_commute": "% Long Commute (60+ min)",
    "pct_government": "% Government Workers",
    "pct_nonprofit": "% Nonprofit Workers",
    "pct_private_forprofit": "% Private For-Profit",
    "pct_self_employed_incorporated": "% Self-Employed (Incorporated)",
    "pct_industry_information": "% Information Industry",
    "pct_industry_finance": "% Finance Industry",
    "pct_industry_professional": "% Professional Services",
    "pct_industry_education_healthcare": "% Education & Healthcare",
    "pct_industry_manufacturing": "% Manufacturing",
    "pct_industry_retail": "% Retail Trade",
}



# GeoJSON loading
@st.cache_data(show_spinner="Loading tract boundaries...")
def load_geojson(path: str = GEOJSON_PATH) -> dict:
    """
    Load Census tract GeoJSON from data/tracts.geojson
    """
    p = Path(path)
    if not p.exists():
        st.error(
            f"`{path}` not found. "
        )
        st.stop()
    with open(p) as f:
        return json.load(f)



# Map builder
def build_choropleth_map(
    df: pd.DataFrame,
    geojson: dict,
    feature_col: str,
    colorscale: str = "YlOrRd",
    center: list = None,
    zoom: int = 4,
) -> folium.Map:
    if center is None:
        center = [39.5, -98.35]

    label = FEATURE_LABELS.get(feature_col, feature_col)

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron",
        control_scale=True,
    )

    choropleth = folium.Choropleth(
        geo_data=geojson,
        name="choropleth",
        data=df,
        columns=["FIPS", feature_col],
        key_on="feature.properties.GEOID",
        fill_color=colorscale,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=label,
        nan_fill_color="lightgray",
        nan_fill_opacity=0.4,
    ).add_to(m)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=["GEOID"],
            aliases=["Tract FIPS:"],
        )
    )

    folium.LayerControl().add_to(m)

    return m
