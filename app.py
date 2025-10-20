# === CAMS CO₂ 2020 Interactive Globe (地域選択 + カメラ固定 + 拡大対応) ===
import streamlit as st
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import geopandas as gpd

# 🚫 キャッシュ自動クリア
st.cache_data.clear()
st.success("✅ Cache cleared successfully.")

# ---------------------------
# 🌍 ページ設定
# ---------------------------
st.set_page_config(page_title="CAMS CO₂ 3D Globe", layout="wide")

st.title("🌍 CAMS Global CO₂ Distribution (2020)")
st.markdown("""
**Copernicus Atmosphere Monitoring Service (CAMS)** 提供の実データをもとに、  
2020年の月別 CO₂ 濃度を地球儀上で可視化します。
""")

# ---------------------------
# 📦 データ読み込み
# ---------------------------
@st.cache_data
def load_data():
    file_path = "data/fd9c5180844360480e5575ed69dc8799.nc"
    ds = xr.open_dataset(file_path, engine="netcdf4")
    for var in ["co2", "xco2", "tcco2"]:
        if var in ds.data_vars:
            co2 = ds[var]
            break
    else:
        raise KeyError("❌ CO₂変数が見つかりません（co2 / xco2 / tcco2）")
    time_key = "time" if "time" in ds.coords else "valid_time"
    lat_key = "latitude" if "latitude" in ds.coords else "lat"
    lon_key = "longitude" if "longitude" in ds.coords else "lon"
    return ds, co2, time_key, lat_key, lon_key

try:
    ds, co2, time_key, lat_key, lon_key = load_data()
    st.write("📦 Variables:", list(ds.data_vars))
    st.write("🧭 Coordinates:", list(ds.coords))
    st.write(f"✅ Data shape: {co2.shape}")
except Exception as e:
    st.error(f"NetCDF読み込みエラー: {e}")
    st.stop()

# ---------------------------
# 🧭 グリッド生成
# ---------------------------
lats, lons = ds[lat_key], ds[lon_key]
lon_grid, lat_grid = np.meshgrid(lons, lats)
X = np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
Y = np.cos(np.deg2rad(lat_grid)) * np.sin(np.deg2rad(lon_grid))
Z = np.sin(np.deg2rad(lat_grid))

# 縦線除去
if not np.isclose(lons[-1], 360.0, atol=0.5):
    extra_slice = co2.isel({lon_key: 0}).copy(deep=True)
    co2 = xr.concat([co2, extra_slice], dim=lon_key)
    lons_new = np.linspace(0, 360, co2.sizes[lon_key])
    co2 = co2.assign_coords({lon_key: lons_new})
    vals = co2.values
    vals[..., -1] = (vals[..., -2] + vals[..., 0]) / 2
    co2[:] = vals

# ---------------------------
# 🌐 海岸線データ
# ---------------------------
@st.cache_data
def load_coastline():
    url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    cx, cy = [], []
    for geom in world.geometry:
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            cx += list(x) + [None]
            cy += list(y) + [None]
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                cx += list(x) + [None]
                cy += list(y) + [None]
    cx = np.array([np.nan if x is None else x for x in cx], dtype=float)
    cy = np.array([np.nan if y is None else y for y in cy], dtype=float)
    return cx, cy

cx, cy = load_coastline()
mask = np.isfinite(cx) & np.isfinite(cy)
coast_X = np.cos(np.deg2rad(cy[mask])) * np.cos(np.deg2rad(cx[mask]))
coast_Y = np.cos(np.deg2rad(cy[mask])) * np.sin(np.deg2rad(cx[mask]))
coast_Z = np.sin(np.deg2rad(cy[mask]))
coast_trace = go.Scatter3d(x=coast_X, y=coast_Y, z=coast_Z, mode="lines",
                           line=dict(color="black", width=0.8), showlegend=False)

# ---------------------------
# 🎨 カラースケール & 地域選択
# ---------------------------
vmin, vmax = np.nanpercentile(co2.values, [2, 98])
colorscale = st.sidebar.selectbox("カラースケール", ["Turbo", "Viridis", "Plasma", "RdYlGn_r"])
region = st.sidebar.selectbox("🌍 表示地域", ["Global", "Asia-Pacific", "Euro-Africa", "America"])

# カメラプリセット（強制固定）
camera_presets = {
    "Asia-Pacific": dict(x=-1.8, y=2.2, z=1.3),
    "Euro-Africa": dict(x=0.4, y=2.7, z=1.4),
    "America": dict(x=2.8, y=-1.8, z=1.3),
    "Global": dict(x=1.9, y=1.9, z=1.4)
}
camera_eye = camera_presets.get(region, camera_presets["Global"])

# ---------------------------
# 🌫️ Surface生成関数
# ---------------------------
def make_surface(idx):
    frame = co2.isel({time_key: idx}).values
    return go.Surface(x=X, y=Y, z=Z, surfacecolor=frame,
                      colorscale=colorscale, cmin=vmin, cmax=vmax,
                      showscale=True, opacity=1.0,
                      lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                                    roughness=1.0, fresnel=0.0))

# ---------------------------
# 🎚️ 月スライダー
# ---------------------------
month = st.slider("表示月 (1–12)", 1, co2.sizes[time_key], 1, step=1) - 1

fig = go.Figure(data=[make_surface(month), coast_trace])
fig.update_layout(
    title=f"🌍 CAMS CO₂ Concentration — {region} — Month {month+1}",
    width=1200, height=850,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor="white",
        camera=dict(up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=camera_eye)  # ← 強制適用
    ),
    margin=dict(l=40, r=40, t=60, b=20)
)

# カラーバーと出典
fig.update_traces(colorbar_title="CO₂ (ppm)",
                  selector=dict(type="surface"),
                  colorbar_len=0.7,
                  colorbar_x=1.05)
fig.add_annotation(
    text="Data Source: Copernicus Atmosphere Monitoring Service (CAMS), ECMWF (2020)",
    xref="paper", yref="paper", x=0.5, y=-0.08, showarrow=False,
    font=dict(size=11, color="gray"), align="center"
)

# ---------------------------
# 📺 表示
# ---------------------------
st.plotly_chart(fig, use_container_width=True)
