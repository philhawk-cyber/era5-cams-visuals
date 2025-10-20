import streamlit as st
import plotly.express as px
import numpy as np

st.set_page_config(page_title="ERA5 & CAMS Global Visualization", layout="wide")

st.title("🌍 ERA5 & CAMS Global Climate Visualization (Demo)")
st.markdown("""
Interactive demo version of ERA5 & CAMS visualizations.
""")

dataset = st.sidebar.selectbox("Select Dataset", ["ERA5 Temperature Anomaly", "CAMS CO₂ Concentration"])

if dataset.startswith("ERA5"):
    year = st.sidebar.slider("Year", 1991, 2024, 2020)
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-90, 90, 180)
    data = np.sin(np.radians(lat[:, None])) * np.cos(np.radians(lon[None, :])) * (year - 1990) / 34

    fig = px.imshow(
        data, x=lon, y=lat, origin="lower",
        color_continuous_scale="RdBu_r",
        labels=dict(x="Longitude", y="Latitude", color="°C Anomaly"),
        title=f"ERA5 Temperature Anomaly — {year}"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    lon = np.linspace(-180, 180, 360)
    lat = np.linspace(-90, 90, 180)
    data = 400 + 5 * np.sin(np.radians(lat[:, None]))
    fig = px.imshow(
        data, x=lon, y=lat, origin="lower",
        color_continuous_scale="Viridis",
        labels=dict(x="Longitude", y="Latitude", color="ppm"),
        title="CAMS Global CO₂ Concentration (2020)"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("© 2025 cholmin im — Demo version (no actual data files)")
