# === CAMS CO₂ 2020 Interactive Globe — 視点（位置・サイズ）をスライダー変更でも保持 ===
import streamlit as st
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import geopandas as gpd

# 毎回起動時にキャッシュをクリア（開発中は便利／本番では外してOK）
st.cache_data.clear()

# ---------------------------
# ページ設定
# ---------------------------
st.set_page_config(page_title="CAMS CO₂ 3D Globe", layout="wide")
st.title("🌍 CAMS Global CO₂ Distribution (2020)")
st.caption("スライダー操作時も、手動で調整した視点（回転・ズーム）を維持します。")

# ---------------------------
# データ読み込み（キャッシュ）
# ---------------------------
@st.cache_data
def load_data():
    file_path = "data/fd9c5180844360480e5575ed69dc8799.nc"
    ds = xr.open_dataset(file_path, engine="netcdf4")

    # 変数自動検出
    for var in ["co2", "xco2", "tcco2"]:
        if var in ds.data_vars:
            co2 = ds[var]
            break
    else:
        raise KeyError("❌ CO₂変数が見つかりません（co2 / xco2 / tcco2）")

    # 軸名吸収
    time_key = "time" if "time" in ds.coords else "valid_time"
    lat_key  = "latitude" if "latitude" in ds.coords else "lat"
    lon_key  = "longitude" if "longitude" in ds.coords else "lon"
    return ds, co2, time_key, lat_key, lon_key

try:
    ds, co2, time_key, lat_key, lon_key = load_data()
except Exception as e:
    st.error(f"NetCDF読み込みエラー: {e}")
    st.stop()

# ---------------------------
# グリッド・座標
# ---------------------------
lats, lons = ds[lat_key], ds[lon_key]
lon_grid, lat_grid = np.meshgrid(lons, lats)
X = np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
Y = np.cos(np.deg2rad(lat_grid)) * np.sin(np.deg2rad(lon_grid))
Z = np.sin(np.deg2rad(lat_grid))

# 経度閉鎖（縦線除去）
if not np.isclose(lons[-1], 360.0, atol=0.5):
    extra_slice = co2.isel({lon_key: 0}).copy(deep=True)
    co2 = xr.concat([co2, extra_slice], dim=lon_key)
    lons_new = np.linspace(0, 360, co2.sizes[lon_key])
    co2 = co2.assign_coords({lon_key: lons_new})
    vals = co2.values
    vals[..., -1] = (vals[..., -2] + vals[..., 0]) / 2
    co2[:] = vals

# ---------------------------
# 海岸線（GeoJSON → 球面）
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
    mask = np.isfinite(cx) & np.isfinite(cy)
    # 球面へ
    Xc = np.cos(np.deg2rad(cy[mask])) * np.cos(np.deg2rad(cx[mask]))
    Yc = np.cos(np.deg2rad(cy[mask])) * np.sin(np.deg2rad(cx[mask]))
    Zc = np.sin(np.deg2rad(cy[mask]))
    return Xc, Yc, Zc

coast_X, coast_Y, coast_Z = load_coastline()
coast_trace = go.Scatter3d(x=coast_X, y=coast_Y, z=coast_Z, mode="lines",
                           line=dict(color="black", width=0.8), showlegend=False)

# ---------------------------
# コントロール
# ---------------------------
vmin, vmax = np.nanpercentile(co2.values, [2, 98])
colorscale = st.sidebar.selectbox("Colorscale", ["Turbo", "Viridis", "Plasma", "RdYlGn_r"], index=0)
month_idx = st.slider("表示月 (1–12)", 1, co2.sizes[time_key], 1, step=1) - 1

# ---------------------------
# Surface 生成
# ---------------------------
def surface_for(i: int) -> go.Surface:
    frame = co2.isel({time_key: i}).values
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=frame,
        colorscale=colorscale,
        cmin=vmin, cmax=vmax,
        showscale=True,
        opacity=1.0,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0),
    )

# ---------------------------
# 図の組み立て（初回だけ初期カメラ／以降は一切カメラに触れない）
# ---------------------------
fig = go.Figure(data=[surface_for(month_idx), coast_trace])

# ✅ ここが重要：
# 1) uirevision を固定値 "keep" にする → ユーザー操作（回転・ズーム）を次回以降も保持
# 2) 初回だけ“少し拡大したデフォルト視点”をセット。以降は camera を一切更新しない
scene_kwargs = dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False),
    aspectmode="data",
    bgcolor="white",
)

if "camera_initialized" not in st.session_state:
    scene_kwargs["camera"] = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.2, y=1.2, z=0.9),  # ← 近づく（ズームイン）
    )
    st.session_state.camera_initialized = True
# 2回目以降は camera を渡さない（→ ユーザー操作を完全保持）

fig.update_layout(
    title=f"🌍 CAMS CO₂ Concentration — Month {month_idx+1}",
    margin=dict(l=40, r=40, t=60, b=20),
    scene=scene_kwargs,
    uirevision="keep",        # ← これが視点保持のキモ
    width=1150, height=800,   # 初期サイズ。以降はユーザー操作を尊重
)

# カラーバー＆出典
fig.update_traces(
    selector=dict(type="surface"),
    colorbar_title="CO₂ (ppm)",
    colorbar_len=0.7,
    colorbar_x=1.05
)
fig.add_annotation(
    text="Data Source: Copernicus Atmosphere Monitoring Service (CAMS), ECMWF (2020)",
    xref="paper", yref="paper", x=0.5, y=-0.08, showarrow=False,
    font=dict(size=11, color="gray"), align="center"
)

# ---------------------------
# 表示（同じキーで出し続ける → 同じPlotlyオブジェクトとして扱われ、視点保持）
# ---------------------------
st.plotly_chart(fig, use_container_width=True, key="globe")
