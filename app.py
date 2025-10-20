import gradio as gr
import xarray as xr
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_co2(time_index=0):
    """
    Visualize CO₂ concentration for a given month (2020)
    using CAMS reanalysis data from GitHub.
    """
    # ✅ 公開用URL（Google Drive不要）
    url = "https://raw.githubusercontent.com/philhawk-cyber/era5-cams-visuals/main/data/fd9c5180844360480e5575ed69dc8799.nc"

    # --- データ読込 ---
    ds = xr.open_dataset(url)
    co2 = ds["co2"].isel(time=int(time_index))
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    # --- Plotly 3D Surfaceで地球風描画 ---
    fig = go.Figure(go.Surface(
        z=co2.values,
        x=lon,
        y=lat,
        colorscale="Viridis",
        colorbar_title="CO₂ (ppm)"
    ))

    fig.update_layout(
        title=f"CAMS CO₂ Concentration (2020, month={int(time_index)+1})",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="CO₂"
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=700
    )

    return fig

# ✅ Gradioアプリ構成
demo = gr.Interface(
    fn=plot_co2,
    inputs=gr.Slider(0, 11, step=1, label="Month (0=Jan, 11=Dec)"),
    outputs=gr.Plotly(label="Global CO₂ Surface"),
    title="🌍 CAMS CO₂ 2020 Interactive Globe",
    description="Visualize global CO₂ concentrations using CAMS ERA5 reanalysis data (Plotly + xarray).",
    allow_flagging="never"
)

# ✅ Spaces上で自動起動
demo.launch()
