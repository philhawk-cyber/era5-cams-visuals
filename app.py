import streamlit as st
import xarray as xr
import plotly.express as px
import numpy as np

st.set_page_config(page_title="CAMS CO₂ Global Visualization", layout="wide")

st.title("🌫️ CAMS Global CO₂ Concentration Visualization (Real Data)")
st.markdown("""
This interactive viewer displays **CAMS (Copernicus Atmosphere Monitoring Service)** CO₂ concentration data.
""")

# --- Sidebar options ---
st.sidebar.header("Visualization Settings")

@st.cache_data
def load_data():
    ds = xr.open_dataset("data/fd9c5180844360480e5575ed69dc8799.nc")
    var_name = list(ds.data_vars.keys())[0]  # データ変数名を自動取得
    da = ds[var_name]
    return da

try:
    da = load_data()
except Exception as e:
    st.error(f"Error loading NetCDF file: {e}")
    st.stop()

# 軸名を推定
lat_name = [k for k in da.coords if "lat" in k.lower()][0]
lon_name = [k for k in da.coords if "lon" in k.lower()][0]

# 時間軸がある場合はスライダーを表示
if "time" in da.dims:
    t_index = st.sidebar.slider("Time index", 0, len(da["time"]) - 1, 0)
    frame = da.isel(time=t_index)
    time_label = str(da["time"].values[t_index])
else:
    frame = da
    time_label = "Static data (no time dimension)"

# カラースケール選択
color_scale = st.sidebar.selectbox("Color scale", ["Viridis", "Plasma", "Cividis", "Inferno"])

# 描画
fig = px.imshow(
    frame,
    x=frame[lon_name],
    y=frame[lat_name],
    origin="lower",
    color_continuous_scale=color_scale,
    aspect="auto",
    labels={"color": f"{da.name}"},
)
fig.update_layout(
    title=f"CAMS CO₂ Concentration — {time_label}",
    coloraxis_colorbar=dict(title="ppm")
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Data source: Copernicus Atmosphere Monitoring Service (CAMS) — ECMWF.")
