# === CAMS CO₂ 2020 Interactive Globe (完全統合・安定版) ===
import streamlit as st
import numpy as np
import xarray as xr
import plotly.graph_objects as go
import geopandas as gpd

# 🚫 キャッシュを毎回クリア（再デプロイ・再起動時に古いデータを残さない）
# ※ 本番で速度を重視する場合はコメントアウトしてもOK
st.cache_data.clear()

# ---------------------------
# 🌍 Streamlit ページ設定
# ---------------------------
st.set_page_config(page_title="CAMS CO₂ 3D Globe", layout="wide")

st.title("🌍 CAMS Global CO₂ Distribution (2020)")
st.markdown("""
**Copernicus Atmosphere Monitoring Service (CAMS)** 提供のデータをもとに、  
2020年の月別 CO₂ 濃度を地球儀上でインタラクティブに可視化します。
""")

# ---------------------------
# 📦 データ読み込み関数（キャッシュ付き）
# ---------------------------
@st.cache_data
def load_data():
    file_path = "data/fd9c5180844360480e5575ed69dc8799.nc"
    ds = xr.open_dataset(file_path)

    # 変数を自動検出
    for varname in ["co2", "xco2", "tcco2"]:
        if varname in ds.data_vars:
            co2 = ds[varname]
            break
    else:
        raise KeyError("❌ CO₂変数が見つかりません（co2 / xco2 / tcco2）")

    time_key = "time" if "time" in ds.coords else "valid_time"
    lat_key = "latitude" if "latitude" in ds.coords else "lat"
    lon_key = "longitude" if "longitude" in ds.coords else "lon"
    return ds, co2, time_key, lat_key, lon_key

ds, co2, time_key, lat_key, lon_key = load_data()

# ---------------------------
# 🧭 座標変換・グリッド作成
# ---------------------------
times = ds[time_key]
lats = ds[lat_key]
lons = ds[lon_key]

lon_grid, lat_grid = np.meshgrid(lons, lats)
X = np.cos(np.deg2rad(lat_grid)) * np.cos(np.deg2rad(lon_grid))
Y = np.cos(np.deg2rad(lat_grid)) * np.sin(np.deg2rad(lon_grid))
Z = np.sin(np.deg2rad(lat_grid))

# 経度閉鎖（縦線除去）
if not np.isclose(lons[-1], 360.0, atol=0.5):
    extra = co2.isel({lon_key: 0}).copy(deep=True)
    co2 = xr.concat([co2, extra], dim=lon_key)
    co2 = co2.assign_coords({lon_key: np.linspace(0, 360, co2.sizes[lon_key])})
    co2_vals = co2.values
    co2_vals[..., -1] = (co2_vals[..., 0] + co2_vals[..., -2]) / 2
    co2[:] = co2_vals

# ---------------------------
# 🌐 海岸線データ
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
    coast_x = np.array([np.nan if x is None else x for x in coast_x], dtype=float)
    coast_y = np.array([np.nan if y is None else y for y in coast_y], dtype=float)
    return coast_x, coast_y

coast_x, coast_y = load_coastline()
mask = np.isfinite(coast_x) & np.isfinite(coast_y)
coast_X = np.cos(np.deg2rad(coast_y[mask])) * np.cos(np.deg2rad(coast_x[mask]))
coast_Y = np.cos(np.deg2rad(coast_y[mask])) * np.sin(np.deg2rad(coast_x[mask]))
coast_Z = np.sin(np.deg2rad(coast_y[mask]))

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
# 🧭 カメラプリセット設定（強制リセット対応）
# ---------------------------
view_option = st.sidebar.selectbox(
    "視点プリセット",
    ["Global (Default)", "Asia-Pacific", "Europe-Africa", "Americas"],
    key="view_option"
)

camera_presets = {
    "Global (Default)": dict(eye=dict(x=1.0, y=1.0, z=0.7)),
    "Asia-Pacific":     dict(eye=dict(x=-1.8, y=2.0, z=0.7)),
    "Europe-Africa":    dict(eye=dict(x=-1.0, y=2.0, z=0.7)),   # 西寄りに補正
    "Americas":         dict(eye=dict(x=2.8, y=-1.0, z=0.7)),   # メキシコ中央
}

if "last_view" not in st.session_state:
    st.session_state.last_view = None

if st.session_state.last_view != view_option:
    st.session_state.camera_eye = camera_presets[view_option]["eye"]
    st.session_state.last_view = view_option

camera_eye = st.session_state.camera_eye

# ---------------------------
# 🌫️ Surface生成関数
# ---------------------------
def make_surface(idx):
    co2_frame = co2.isel({time_key: idx}).values
    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=co2_frame,
        colorscale=colorscale,
        cmin=vmin, cmax=vmax,
        showscale=True,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
        opacity=1.0
    )

# ---------------------------
# 🎚️ 月スライダーで切替
# ---------------------------
month_idx = st.slider("表示月 (1–12)", 1, co2.sizes[time_key], 1, step=1) - 1
fig = go.Figure(data=[make_surface(month_idx), coast_trace])

# ---------------------------
# 📊 レイアウト設定
# ---------------------------
fig.update_layout(
    title=f"🌍 CAMS CO₂ Concentration — Month {month_idx+1}",
    width=1100, height=800,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        bgcolor="white",
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=camera_eye
        )
    ),
    margin=dict(l=40, r=40, t=60, b=20)
)

# カラーバー調整
fig.update_traces(
    selector=dict(type="surface"),
    colorbar_title="CO₂ (ppm)",
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

# ---------------------------
# 📺 Streamlit 表示（key付きで再描画）
# ---------------------------
st.plotly_chart(fig, use_container_width=True, key=f"{view_option}_{month_idx}")
