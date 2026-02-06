import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf, RegularGridInterpolator
from scipy.spatial import cKDTree
from PIL import Image
import os

st.set_page_config(layout="wide", page_title="Jeolojik 3D Risk ve Veri Modeli")

# -------------------------------------------------
# 1. DOSYA KONTROLÜ
# -------------------------------------------------
tif_path = "aaa.tif"
if not os.path.exists(tif_path):
    st.error(f"Hata: '{tif_path}' dosyası bulunamadı. Lütfen proje klasörüne ekleyin.")
    st.stop()

# -------------------------------------------------
# 2. VERİ GİRİŞİ (Sondajlar ve Loglar)
# -------------------------------------------------
boreholes = {
    "1SK": {"x": 631754.22, "y": 4290134.82, "z": 984},
    "2SK": {"x": 630597.83, "y": 4290786.54, "z": 930},
    "3SK": {"x": 629297.03, "y": 4291138.37, "z": 982},
    "4SK": {"x": 629892.93, "y": 4292514.66, "z": 936},
    "5SK": {"x": 630616.91, "y": 4291920.99, "z": 922},
    "6SK": {"x": 631675.12, "y": 4291824.45, "z": 962},
}

data_raw = {
    "1SK": [(0.2, 1.8), (14.8, 12.4), (16.0, 2.2), (18.2, 8.6), (27.5, 5.1), (31.0, 2.6), (34.0, 7.5), (36.0, 5.4), (49.0, 1.2)],
    "2SK": [(0.2, 0.8), (7.8, 0.0), (18.0, 9.4), (31.5, 1.0), (34.0, 14.6), (41.0, 1.4), (45.0, 1.2)],
    "3SK": [(0.5, 1.1), (10.0, 1.2), (20.5, 6.8), (30.5, 0.6), (40.0, 9.2)],
    "4SK": [(0.2, 0.4), (22.0, 11.2), (23.8, 7.4), (31.0, 0.6), (33.0, 4.1), (36.0, 8.2)],
    "5SK": [(0.2, 2.4), (6.0, 7.8), (10.0, 5.4), (16.0, 12.4), (23.0, 8.6), (40.0, 0.2), (42.6, 0.0)],
    "6SK": [(1.0, 1.6), (6.0, 0.4), (12.0, 3.8), (24.0, 9.4), (30.0, 4.8), (36.0, 0.1), (49.0, 0.0)],
}

# -------------------------------------------------
# 3. AYARLAR PANELİ
# -------------------------------------------------
st.sidebar.title("3D Model Kontrol Paneli")

st.sidebar.subheader("Görselleştirme")
show_volume = st.sidebar.checkbox("Risk Hacmini Göster", True)
point_size = st.sidebar.slider("Sondaj Nokta Boyutu", 3, 10, 5)
z_exag = st.sidebar.slider("Dikey Abartı", 1.0, 30.0, 10.0)

st.sidebar.subheader("Topografya Ayarları")
topo_color = st.sidebar.color_picker("Yüzey Rengi", "#C2B280")
topo_opacity = st.sidebar.slider("Yüzey Şeffaflığı", 0.0, 1.0, 0.4, step=0.1)

st.sidebar.subheader("Risk Analizi ve Enterpolasyon")
risk_min = st.sidebar.slider("Risk Eşiği (Cut-off)", 0.0, 20.0, 1.5, step=0.1)

# YENİ ÖZELLİK: Modelin ne kadar uzağa tahmin yapacağını kısıtlar
radius_limit = st.sidebar.slider("Etki Yarıçapı (Metre)", 50, 1000, 250, help="Sondaj verisinden bu kadar uzak olan bölgelerde risk hesaplanmaz.")

# -------------------------------------------------
# 4. HESAPLAMA
# -------------------------------------------------
@st.cache_data
def perform_interpolation(radius_limit_val):
    # 1. Topografya
    img = Image.open(tif_path)
    topo = np.array(img).astype(float)
    topo[topo > 3000] = np.nan
    
    x0, y0, dx, dy = 628847.89, 4293423.24, 27.73, 27.73
    xc = x0 + np.arange(topo.shape[1]) * dx
    yc = y0 - np.arange(topo.shape[0]) * dy
    
    bx = [b["x"] for b in boreholes.values()]
    by = [b["y"] for b in boreholes.values()]
    margin = 400
    
    mx = (xc >= min(bx) - margin) & (xc <= max(bx) + margin)
    my = (yc >= min(by) - margin) & (yc <= max(by) + margin)
    
    tx, ty = xc[mx], yc[my]
    tz = topo[np.ix_(my, mx)]
    
    topo_interp = RegularGridInterpolator(
        (tx, ty[::-1]), np.flipud(tz).T, 
        bounds_error=False, fill_value=None
    )
    
    # 2. Veri Hazırlığı
    rows = []
    for bh, pts in data_raw.items():
        for d, v in pts:
            rows.append({
                "BH": bh, "X": boreholes[bh]["x"], "Y": boreholes[bh]["y"],
                "Z": boreholes[bh]["z"] - d, "Val": v
            })
    df = pd.DataFrame(rows)
    
    z_top = df["Z"].max() + 5
    z_bot = min(b["z"] for b in boreholes.values()) - 55
    
    # Normalizasyon
    xn = (df["X"] - df["X"].min()) / (df["X"].max() - df["X"].min())
    yn = (df["Y"] - df["Y"].min()) / (df["Y"].max() - df["Y"].min())
    zn = (df["Z"] - z_bot) / (z_top - z_bot)
    
    # RBF ENTERPOLASYON (DÜZELTİLDİ)
    # 'multiquadric' yerine 'linear' kullanıldı. Multiquadric boşluklarda sonsuza gider.
    # Linear ise veri noktaları arasında daha sıkı durur.
    rbf = Rbf(xn, yn, zn, df["Val"], epsilon=0.5, function='linear')
    
    # Grid
    res = 45 
    xi = yi = zi = np.linspace(0, 1, res)
    XG, YG, ZG = np.meshgrid(xi, yi, zi, indexing="ij")
    
    values_norm = rbf(XG.flatten(), YG.flatten(), ZG.flatten())
    
    # Grid Koordinatlarını Geri Çevirme
    rx = XG.flatten() * (df["X"].max() - df["X"].min()) + df["X"].min()
    ry = YG.flatten() * (df["Y"].max() - df["Y"].min()) + df["Y"].min()
    rz = ZG.flatten() * (z_top - z_bot) + z_bot
    
    # --- YENİ BÖLÜM: MESAFE MASKELEME (CLAMPING) ---
    # Her grid noktasının en yakın gerçek veriye olan uzaklığını bul
    # KDTree bu işlem için çok hızlıdır.
    real_points = np.column_stack([df["X"], df["Y"], df["Z"]])
    grid_points = np.column_stack([rx, ry, rz])
    
    tree = cKDTree(real_points)
    distances, _ = tree.query(grid_points)
    
    # Belirlenen yarıçaptan uzaktaki noktaları maskele (-999 yaparak çizimden atıyoruz)
    values_norm[distances > radius_limit_val] = -999
    
    # Topografya kesimi
    surf_vals = topo_interp(np.column_stack([rx, ry]))
    
    return tx, ty, tz, rx, ry, rz, values_norm, df, zmin_plot, zmax_plot, surf_vals

zmin_plot = min(b["z"] for b in boreholes.values()) - 60
zmax_plot = max(b["z"] for b in boreholes.values()) + 50

# Hesaplamayı çağır (Cache için radius da parametre olarak gidiyor)
tx, ty, tz, rx, ry, rz, raw_values, df_raw, z_min, z_max, surf_vals = perform_interpolation(radius_limit)

# -------------------------------------------------
# 5. GÖRSEL İŞLEME
# -------------------------------------------------
plot_values = raw_values.copy()
mask_air = rz > (surf_vals)
plot_values[mask_air] = -999

st.sidebar.markdown("---")
st.sidebar.info(f"Maksimum Veri Değeri: {df_raw['Val'].max():.2f}")

# -------------------------------------------------
# 6. ÇİZİM
# -------------------------------------------------
fig = go.Figure()

# Topografya
fig.add_trace(go.Surface(
    x=tx, y=ty, z=tz,
    opacity=topo_opacity,
    colorscale=[[0, topo_color], [1, topo_color]],
    showscale=False,
    name="Topografya",
    hoverinfo='none'
))

# Sondajlar
for bh, b in boreholes.items():
    fig.add_trace(go.Scatter3d(
        x=[b["x"], b["x"]],
        y=[b["y"], b["y"]],
        z=[b["z"], b["z"] - 50],
        mode="lines",
        line=dict(color="black", width=5),
        text=bh,
        showlegend=False
    ))

# Numuneler
fig.add_trace(go.Scatter3d(
    x=df_raw["X"],
    y=df_raw["Y"],
    z=df_raw["Z"],
    mode="markers",
    marker=dict(
        size=point_size,
        color=df_raw["Val"],
        colorscale="Jet",
        cmin=0, cmax=df_raw["Val"].max(),
        line=dict(color="black", width=0.5)
    ),
    name="Numuneler",
    hovertemplate="Değer: %{marker.color:.2f}"
))

# Risk Hacmi
if show_volume:
    fig.add_trace(go.Volume(
        x=rx, y=ry, z=rz,
        value=plot_values,
        isomin=risk_min,
        isomax=df_raw["Val"].max(),
        opacity=0.6,
        surface_count=25,
        colorscale="Reds",
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        name="Risk Hacmi"
    ))

# KUZEY OKU (SABİT)
arrow_x = 630500 
arrow_y = df_raw["Y"].max() + 200
arrow_z = 1100
arrow_len = 250 

fig.add_trace(go.Scatter3d(
    x=[arrow_x, arrow_x],
    y=[arrow_y, arrow_y + arrow_len],
    z=[arrow_z, arrow_z],
    mode="lines+text",
    line=dict(color="red", width=15),
    text=["", "N"],
    textposition="top center",
    textfont=dict(size=40, color="black", family="Arial Black"),
    showlegend=False
))

fig.add_trace(go.Cone(
    x=[arrow_x],
    y=[arrow_y + arrow_len],
    z=[arrow_z],
    u=[0], v=[1], w=[0],
    sizemode="absolute",
    sizeref=80,
    anchor="tail",
    colorscale=[[0, "red"], [1, "red"]],
    showscale=False,
    hoverinfo='none'
))

fig.update_layout(
    scene=dict(
        xaxis_title="Doğu (X)",
        yaxis_title="Kuzey (Y)",
        zaxis_title="Kot (Z)",
        zaxis=dict(range=[z_min, 1200]),
        aspectratio=dict(x=1, y=1, z=z_exag * 0.15),
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
    ),
    margin=dict(r=0, l=0, b=0, t=0),
    height=800,
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)
