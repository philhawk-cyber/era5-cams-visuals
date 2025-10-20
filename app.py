# === CAMS CO₂ 2020 Interactive Globe (Streamlit Cloud 完全安定・改良版) ===
import streamlit as st
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import geopandas as gpd

# ---------------------------
# 🌍 Streamlit ページ設定
# ---------------------------
st.set_page_config(page_title="CAMS CO₂ 3D Globe", layout="wide")

st.title("🌍 CAMS Global CO₂ Distribution (2020)")
st.markdown("""
**Copernicus Atmosphere Monitoring Service (CAMS)** 提供の実データをもとに、
2020年の月別 CO₂ 濃度を地球儀上で可視化します。
""")

# ---------------------------
# 📦 データ読み込み関数（キャッシュ）
# ---------------------------
@st.cache_data
def load_data():
    file_path = "data/fd9c5180844360480e5575ed69dc8799.nc"  # 相対パスでOK
    ds = xr.open_dataset(file_path)

    # CO₂変数を検出
    for varname in ["co2", "xco2", "tcco2"]:
        if varname in ds.data_vars:
            co2 = ds[varname]
            break
    else:
        raise KeyError("❌ CO₂変数が見つかりません（co2 / xco2 / tcco2）")

    # 軸名の違いを吸収
    time_key = "time" if "time" in ds.coords else "valid_time"
    lat_key = "latitude" if "latitude" in ds.coords else "lat"
    lon_key = "longitude" if "longitude" in ds.coords else "lon"

    return ds, co2, time_key, lat_key, lon_key

try:
    ds, co2, time_key, lat_key, lon_key = load_data()
except Exception as e:
    st.error(f"NetCDF読み込みエラー: {e}")
    st.stop()

# ---------------------------
# 🧭 座標情報
# ---------------------------
times = ds[time_key]
lats = ds[lat_key]
lons = ds[lon_key]

lon_grid, lat_grid = np.meshgrid(lons, lats)
X = np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
Y = np.cos(np.deg2rad(lat_grid)) * np.sin(np.deg2rad(lon_grid))
Z = np.sin(np.deg2rad(lat_grid))

# 経度の閉鎖処理（縦線除去）
if not np.isclose(lons[-1], 360.0, atol=0.5):
    extra_slice = co2.isel({lon_key: 0}).copy(deep=True)
    co2 = xr.concat([co2, extra_slice], dim=lon_key)
    lons_new = np.linspace(0, 360, co2.sizes[lon_key])
    co2 = co2.assign_coords({lon_key: lons_new})
    co2_vals = co2.values
    co2_vals[..., -1] = (co2_vals[..., -2] + co2_vals[..., 0]) / 2
    co2[:] = co2_vals

# ---------------------------
# 🌐 海岸線データ（GeoJSON → 球面座標）
# ---------------------------
@st.cache_data
def load_coastline():
    url = "https://github.com/nvkelso/natural-earth-vector/raw/master/geojson/ne_110m_admin_0_countries.geojson"
    world = gpd.read_file(url)
    coast_x, coast_y = [], []
    for geom in world.geometry:
        if geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            coast_x += list(x) + [None]
            coast_y += list(y) + [None]
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                x, y = poly.exterior.xy
                coast_x += list(x) + [None]
                coast_y += list(y) + [None]
    # numpy変換
    coast_x = np.array([np.nan if x is None else x for x in coast_x], dtype=float)
    coast_y = np.array([np.nan if y is None else y for y in coast_y], dtype=float)
    return coast_x, coast_y

# --- 有効座標を抽出して球面変換 ---
coast_x, coast_y = load_coastline()
valid_mask = np.isfinite(coast_x) & np.isfinite(coast_y)
coast_x_valid = coast_x[valid_mask]
coast_y_valid = coast_y[valid_mask]

coast_X = np.cos(np.deg2rad(coast_y_valid)) * np.cos(np.deg2rad(coast_x_valid))
coast_Y = np.cos(np.deg2rad(coast_y_valid)) * np.sin(np.deg2rad(coast_x_valid))
coast_Z = np.sin(np.deg2rad(coast_y_valid))

coast_trace = go.Scatter3d(
    x=coast_X, y=coast_Y, z=coast_Z,
    mode="lines", line=dict(color="black", width=0.8),
    showlegend=False
)

# ---------------------------
# 🎨 カラースケール設定
# ---------------------------
vmin, vmax = np.nanpercentile(co2.values, [2, 98])
colorscale = st.sidebar.selectbox("カラースケール", ["Turbo", "Viridis", "Plasma", "RdYlGn_r"])

# ---------------------------
# 🌫️ Surface生成関数（カラーバー付き）
# ---------------------------
def make_surface(month_idx):
    co2_frame = co2.isel({time_key: month_idx}).values
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=co2_frame,
        colorscale=colorscale,
        cmin=vmin, cmax=vmax,
        showscale=True,   # ← カラーバーはON
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                      roughness=1.0, fresnel=0.0),
        opacity=1.0
    )

# ---------------------------
# 🎚️ 月スライダーで切替（再生なし）
# ---------------------------
month_idx = st.slider("表示月 (1–12)", 1, co2.sizes[time_key], 1, step=1) - 1

fig = go.Figure(data=[make_surface(month_idx), coast_trace])

fig.update_layout(
    title=f"🌍 CAMS Global CO₂ Concentration — Month {month_idx+1}",
    width=1100, height=750,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor="white",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
    ),
    margin=dict(l=40, r=40, t=60, b=20)
)

# カラーバーを統一
fig.update_traces(
    colorbar_title="CO₂ (ppm)",
    selector=dict(type="surface"),
    colorbar_len=0.7,
    colorbar_x=1.05
)

# 出典ラベル
fig.add_annotation(
    text="Data Source: Copernicus Atmosphere Monitoring Service (CAMS), ECMWF (2020)",
    xref="paper", yref="paper",
    x=0.5, y=-0.08, showarrow=False,
    font=dict(size=11, color="gray"),
    align="center"
)

# 表示
st.plotly_chart(fig, use_container_width=True)

# カラーバー全体の外観を調整
fig.update_traces(
    colorbar_title="CO₂ (ppm)",
    selector=dict(type="surface"),
    colorbar_len=0.7,
    colorbar_x=1.05
)

# ---------------------------
# 🧾 出典ラベル（中央下部）
# ---------------------------
fig.add_annotation(
    text="Data Source: Copernicus Atmosphere Monitoring Service (CAMS), ECMWF (2020)",
    xref="paper", yref="paper",
    x=0.5, y=-0.08, showarrow=False,
    font=dict(size=11, color="gray"),
    align="center"
)

# ---------------------------
# 📺 Streamlitで表示
# ---------------------------
st.plotly_chart(fig, use_container_width=True)
