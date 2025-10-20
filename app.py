# === CAMS CO₂ 2020 Interactive Globe (Camera強制再描画版) ===
import streamlit as st
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import geopandas as gpd

st.set_page_config(page_title="CAMS CO₂ 3D Globe", layout="wide")
st.title("🌍 CAMS Global CO₂ Distribution (2020)")
st.markdown("""
**Copernicus Atmosphere Monitoring Service (CAMS)** 提供の実データをもとに、
2020年の月別 CO₂ 濃度を地球儀上で可視化します。
""")

@st.cache_data
def load_data():
    ds = xr.open_dataset("data/fd9c5180844360480e5575ed69dc8799.nc")
    for varname in ["co2", "xco2", "tcco2"]:
        if varname in ds.data_vars:
            co2 = ds[varname]
            break
    else:
        raise KeyError("❌ CO₂変数が見つかりません")

    time_key = "time" if "time" in ds.coords else "valid_time"
    lat_key = "latitude" if "latitude" in ds.coords else "lat"
    lon_key = "longitude" if "longitude" in ds.coords else "lon"
    return ds, co2, time_key, lat_key, lon_key

ds, co2, time_key, lat_key, lon_key = load_data()

times, lats, lons = ds[time_key], ds[lat_key], ds[lon_key]
lon_grid, lat_grid = np.meshgrid(lons, lats)
X = np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
Y = np.cos(np.deg2rad(lat_grid)) * np.sin(np.deg2rad(lon_grid))
Z = np.sin(np.deg2rad(lat_grid))

@st.cache_data
def load_coastline():
    url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    xs, ys = [], []
    for geom in world.geometry:
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            xs += list(x) + [None]
            ys += list(y) + [None]
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                xs += list(x) + [None]
                ys += list(y) + [None]
    xs = np.array([np.nan if x is None else x for x in xs])
    ys = np.array([np.nan if y is None else y for y in ys])
    return xs, ys

cx, cy = load_coastline()
mask = np.isfinite(cx) & np.isfinite(cy)
cx, cy = cx[mask], cy[mask]
coast_X = np.cos(np.deg2rad(cy)) * np.cos(np.deg2rad(cx))
coast_Y = np.cos(np.deg2rad(cy)) * np.sin(np.deg2rad(cx))
coast_Z = np.sin(np.deg2rad(cy))
coast_trace = go.Scatter3d(x=coast_X, y=coast_Y, z=coast_Z, mode="lines", line=dict(color="black", width=0.8), showlegend=False)

colorscale = st.sidebar.selectbox("カラースケール", ["Turbo", "Viridis", "Plasma", "RdYlGn_r"])
view_option = st.sidebar.selectbox("視点プリセット", ["Global (Default)", "Asia-Pacific", "Europe-Africa", "Americas"])

# 🎯 camera設定（zが小さいほど地球が大きく見える）
camera_presets = {
    "Global (Default)": dict(eye=dict(x=1.0, y=1.0, z=0.6)),
    "Asia-Pacific":     dict(eye=dict(x=-1.8, y=2.0, z=0.6)),
    "Europe-Africa":    dict(eye=dict(x=-1.5, y=2.0, z=0.6)),
    "Americas":         dict(eye=dict(x=2.4, y=-1.0, z=0.6)),
}
camera_eye = camera_presets[view_option]["eye"]

vmin, vmax = np.nanpercentile(co2.values, [2, 98])
month_idx = st.slider("表示月 (1–12)", 1, co2.sizes[time_key], 1, step=1) - 1

# 🎨 Surface生成
def make_surface(i):
    f = co2.isel({time_key: i}).values
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=f,
        colorscale=colorscale,
        cmin=vmin, cmax=vmax,
        showscale=True,
        lighting=dict(ambient=1.0, diffuse=0.0),
        opacity=1.0
    )

# 🚀 cameraを再生成するたびに強制再構築
fig = go.Figure(data=[make_surface(month_idx), coast_trace])
fig.update_layout(
    title=f"🌍 CAMS Global CO₂ Concentration — Month {month_idx+1}",
    width=1100, height=750,
    margin=dict(l=40, r=40, t=60, b=20),
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor="white",
        camera=dict(up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=camera_eye)
    ),
    uirevision=str(camera_eye)  # ← 再描画強制のため camera座標をrevisionキーに使用
)

fig.update_traces(
    colorbar_title="CO₂ (ppm)",
    selector=dict(type="surface"),
    colorbar_len=0.7,
    colorbar_x=1.05
)

fig.add_annotation(
    text="Data Source: Copernicus Atmosphere Monitoring Service (CAMS), ECMWF (2020)",
    xref="paper", yref="paper", x=0.5, y=-0.08,
    showarrow=False, font=dict(size=11, color="gray"), align="center"
)

# 変更後（カメラ更新を強制）
st.plotly_chart(fig, use_container_width=True, key=f"{view_option}_{month_idx}")
