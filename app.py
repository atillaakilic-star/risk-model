import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf, RegularGridInterpolator
from scipy.spatial import cKDTree
from PIL import Image
import os
import io

st.set_page_config(layout="wide", page_title="Jeolojik 3D Mineral Risk Modeli")

# -------------------------------------------------
# 1. VERİ YÜKLEME VE TEMİZLEME
# -------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Dosya adını senin yüklediğin dosya adına göre güncelledim
    csv_path = "sondaj faz yüzleri son.xlsx - Sheet2.csv"
    
    # Veriyi oku (ilk 2 satır başlık ve açıklama olduğu için onları atlıyoruz)
    df = pd.read_csv(csv_path, skiprows=1)
    
    # Sütun isimlerini temizle
    df.columns = [c.strip() for c in df.columns]
    
    # Sondaj ve Derinlik bilgisini ayır (Örn: "1SK_1 (0,2 m)" -> "1SK" ve 0.2)
    def parse_well(val):
        try:
            well_name = val.split('_')[0]
            depth_str = val.split('(')[1].split(' m')[0].replace(',', '.')
            return well_name, float(depth_str)
        except:
            return None, None

    # İlk sütun (isimsiz olan) üzerinden parse et
    df['Sondaj'], df['Derinlik'] = zip(*df.iloc[:, 0].apply(parse_well))
    df = df.dropna(subset=['Sondaj'])
    
    # Sayısal sütunlardaki virgülleri noktaya çevir ve float yap
    for col in df.columns:
        if col not in ['Sondaj', 'Derinlik'] and df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)').astype(float)
            
    return df

# -------------------------------------------------
# 2. AYARLAR VE FORM
# -------------------------------------------------
st.sidebar.title("🔬 3D Mineral Analizi")

try:
    df_main = load_and_clean_data()
    available_minerals = [c for c in df_main.columns if c not in ['Sondaj', 'Derinlik'] and not c.startswith('Unnamed')]
    
    with st.sidebar.form("model_ayarlari"):
        target_mineral = st.selectbox("Modellenecek Minerali Seçin", available_minerals, index=available_minerals.index('Erionit') if 'Erionit' in available_minerals else 0)
        
        st.subheader("Görsel Parametreler")
        risk_cutoff = st.slider(f"{target_mineral} Eşik Değeri (%)", 0.0, 20.0, 1.0)
        radius_limit = st.slider("Etki Alanı (Metre)", 50, 1000, 300)
        z_exag = st.slider("Dikey Abartı (Z)", 1, 30, 10)
        
        st.subheader("Görünüm")
        topo_opacity = st.slider("Yüzey Şeffaflığı", 0.0, 1.0, 0.3)
        point_size = st.slider("Örnek Boyutu", 2, 10, 4)
        
        submitted = st.form_submit_button("🌋 MODELİ OLUŞTUR / GÜNCELLE")

except Exception as e:
    st.error(f"Veri yüklenirken hata oluştu: {e}")
    st.stop()

# -------------------------------------------------
# 3. 3D HESAPLAMA MOTORU
# -------------------------------------------------
@st.cache_data
def run_3d_engine(mineral_name, radius_val):
    # Sondaj koordinatları (Statik - Bunları bir dosyadan da okuyabiliriz)
    borehole_coords = {
        "1SK": {"x": 631754.22, "y": 4290134.82, "z": 984},
        "2SK": {"x": 630597.83, "y": 4290786.54, "z": 930},
        "3SK": {"x": 629297.03, "y": 4291138.37, "z": 982},
        "4SK": {"x": 629892.93, "y": 4292514.66, "z": 936},
        "5SK": {"x": 630616.91, "y": 4291920.99, "z": 922},
        "6SK": {"x": 631675.12, "y": 4291824.45, "z": 962},
    }

    # Model için veri noktalarını hazırla
    plot_data = []
    for _, row in df_main.iterrows():
        well = row['Sondaj']
        if well in borehole_coords:
            plot_data.append({
                'x': borehole_coords[well]['x'],
                'y': borehole_coords[well]['y'],
                'z': borehole_coords[well]['z'] - row['Derinlik'],
                'val': row[mineral_name] if not np.isnan(row[mineral_name]) else 0
            })
    
    df_plot = pd.DataFrame(plot_data)
    
    # RBF İnterpolasyon
    z_min, z_max = df_plot['z'].min() - 20, df_plot['z'].max() + 10
    res = 40
    xi = np.linspace(df_plot['x'].min()-200, df_plot['x'].max()+200, res)
    yi = np.linspace(df_plot['y'].min()-200, df_plot['y'].max()+200, res)
    zi = np.linspace(z_min, z_max, res)
    XG, YG, ZG = np.meshgrid(xi, yi, zi, indexing='ij')
    
    # Normalizasyon ve Tahmin
    rbf = Rbf(df_plot['x'], df_plot['y'], df_plot['z'], df_plot['val'], function='linear', epsilon=1)
    vals = rbf(XG.flatten(), YG.flatten(), ZG.flatten())
    
    # Mesafe Maskeleme
    tree = cKDTree(np.column_stack([df_plot['x'], df_plot['y'], df_plot['z']]))
    dist, _ = tree.query(np.column_stack([XG.flatten(), YG.flatten(), ZG.flatten()]))
    vals[dist > radius_val] = -1
    
    return xi, yi, zi, XG, YG, ZG, vals, df_plot, borehole_coords

# -------------------------------------------------
# 4. GÖRSELLEŞTİRME
# -------------------------------------------------
if submitted or 'initialized' not in st.session_state:
    st.session_state.initialized = True
    with st.spinner(f"{target_mineral} dağılımı hesaplanıyor..."):
        xi, yi, zi, XG, YG, ZG, vals, df_points, bh_coords = run_3d_engine(target_mineral, radius_limit)
        
        fig = go.Figure()

        # 1. Sondaj Kuyuları
        for name, c in bh_coords.items():
            fig.add_trace(go.Scatter3d(
                x=[c['x'], c['x']], y=[c['y'], c['y']], z=[c['z'], c['z']-60],
                mode='lines', line=dict(color='black', width=4), name=name
            ))

        # 2. Örnek Noktaları
        fig.add_trace(go.Scatter3d(
            x=df_points['x'], y=df_points['y'], z=df_points['z'],
            mode='markers',
            marker=dict(size=point_size, color=df_points['val'], colorscale='Viridis', showscale=True, colorbar=dict(title=f"{target_mineral} %")),
            name="Analiz Noktaları"
        ))

        # 3. Risk Hacmi (Volume)
        fig.add_trace(go.Volume(
            x=XG.flatten(), y=YG.flatten(), z=ZG.flatten(),
            value=vals,
            isomin=risk_cutoff, isomax=df_points['val'].max(),
            opacity=0.5, surface_count=15,
            colorscale='Reds', showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        fig.update_layout(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=z_exag * 0.1),
                xaxis_title="Doğu (X)", yaxis_title="Kuzey (Y)", zaxis_title="Kot (Z)"
            ),
            margin=dict(l=0, r=0, b=0, t=0), height=850
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Gösterilen Mineral: **{target_mineral}** | Maksimum Ölçülen Değer: %{df_points['val'].max():.2f}")
