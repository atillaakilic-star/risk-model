import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree

st.set_page_config(layout="wide", page_title="Jeolojik 3D Mineral Risk Modeli")

# -------------------------------------------------
# 1. VERİ GİRİŞİ (KOD İÇİNE GÖMÜLÜ - HARDCODED)
# -------------------------------------------------
raw_mineral_data = [
  {"Sondaj": "1SK", "Derinlik": 0.2, "Kalsit": 31.4, "Dolomit": 0.06, "Aragonit": 6.24, "Kuvars": 8.85, "Kristobalit": 0.37, "Albit": 1.46, "Sanidin": 2.09, "Klinoptilolit": 0.25, "Mordenit": 0.75, "Erionit": 1.25, "Şabazit": 0.08, "Analsim": 0.15, "Hornblend": 0.05, "Aktinolit": 0.04, "Kil Grubu": 1.78, "Volkanik Cam": 45.15},
  {"Sondaj": "1SK", "Derinlik": 14.8, "Kalsit": 11.33, "Dolomit": 0.14, "Aragonit": 15.32, "Kuvars": 13.96, "Kristobalit": 1.22, "Albit": 10.24, "Sanidin": 10.08, "Klinoptilolit": 0.88, "Mordenit": 1.68, "Erionit": 2.85, "Şabazit": 0.25, "Analsim": 0.42, "Hornblend": 0.12, "Aktinolit": 0.13, "Kil Grubu": 2.9, "Volkanik Cam": 28.45},
  {"Sondaj": "1SK", "Derinlik": 16.0, "Kalsit": 65.02, "Dolomit": 0.14, "Aragonit": 4.1, "Kuvars": 8.52, "Kristobalit": 0.76, "Albit": 1.88, "Sanidin": 2.2, "Klinoptilolit": 0.38, "Mordenit": 0.84, "Erionit": 1.55, "Şabazit": 0.1, "Analsim": 0.16, "Hornblend": 0.05, "Aktinolit": 0.07, "Kil Grubu": 1.86, "Volkanik Cam": 12.33},
  {"Sondaj": "1SK", "Derinlik": 18.2, "Kalsit": 5.41, "Dolomit": 0.27, "Aragonit": 12.75, "Kuvars": 12.49, "Kristobalit": 1.41, "Albit": 23.01, "Sanidin": 9.06, "Klinoptilolit": 0.64, "Mordenit": 1.59, "Erionit": 2.78, "Şabazit": 0.19, "Analsim": 0.39, "Hornblend": 0.72, "Aktinolit": 6.93, "Kil Grubu": 2.37, "Volkanik Cam": 19.85},
  {"Sondaj": "1SK", "Derinlik": 27.5, "Kalsit": 1.16, "Dolomit": 0.29, "Aragonit": 14.04, "Kuvars": 18.15, "Kristobalit": 1.41, "Albit": 12.53, "Sanidin": 10.29, "Klinoptilolit": 0.67, "Mordenit": 1.6, "Erionit": 2.88, "Şabazit": 0.2, "Analsim": 0.52, "Hornblend": 0.78, "Aktinolit": 8.32, "Kil Grubu": 2.47, "Volkanik Cam": 24.6},
  {"Sondaj": "1SK", "Derinlik": 31.0, "Kalsit": 19.57, "Dolomit": 11.74, "Aragonit": 5.16, "Kuvars": 17.95, "Kristobalit": 0.75, "Albit": 4.04, "Sanidin": 3.34, "Klinoptilolit": 0.35, "Mordenit": 1.14, "Erionit": 2.67, "Şabazit": 0.05, "Analsim": 0.09, "Hornblend": 0.14, "Aktinolit": 2.19, "Kil Grubu": 1.74, "Volkanik Cam": 29.05},
  {"Sondaj": "1SK", "Derinlik": 34.0, "Kalsit": 6.83, "Dolomit": 0.3, "Aragonit": 24.34, "Kuvars": 10.92, "Kristobalit": 1.66, "Albit": 5.73, "Sanidin": 15.89, "Klinoptilolit": 0.78, "Mordenit": 1.79, "Erionit": 2.91, "Şabazit": 0.19, "Analsim": 0.46, "Hornblend": 0.55, "Aktinolit": 3.38, "Kil Grubu": 2.73, "Volkanik Cam": 21.5},
  {"Sondaj": "1SK", "Derinlik": 36.0, "Kalsit": 3.4, "Dolomit": 0.1, "Aragonit": 13.66, "Kuvars": 17.31, "Kristobalit": 1.59, "Albit": 13.31, "Sanidin": 10.37, "Klinoptilolit": 0.92, "Mordenit": 1.94, "Erionit": 2.93, "Şabazit": 0.02, "Analsim": 0.34, "Hornblend": 0.6, "Aktinolit": 3.61, "Kil Grubu": 3.03, "Volkanik Cam": 26.8},
  {"Sondaj": "1SK", "Derinlik": 49.0, "Kalsit": 2.85, "Dolomit": 0.07, "Aragonit": 3.65, "Kuvars": 12.43, "Kristobalit": 1.57, "Albit": 12.42, "Sanidin": 5.39, "Klinoptilolit": 0.72, "Mordenit": 1.73, "Erionit": 2.49, "Şabazit": 0.03, "Analsim": 0.23, "Hornblend": 0.42, "Aktinolit": 3.69, "Kil Grubu": 2.58, "Volkanik Cam": 49.7},
  {"Sondaj": "2SK", "Derinlik": 0.2, "Kalsit": 16.71, "Dolomit": 0.2, "Aragonit": 1.84, "Kuvars": 11.13, "Kristobalit": 1.34, "Albit": 20.06, "Sanidin": 6.02, "Klinoptilolit": 0.6, "Mordenit": 2.01, "Erionit": 3.18, "Şabazit": 0.07, "Analsim": 0.33, "Hornblend": 0.17, "Aktinolit": 2.34, "Kil Grubu": 3.34, "Volkanik Cam": 30.63},
  {"Sondaj": "2SK", "Derinlik": 7.8, "Kalsit": 4.45, "Dolomit": 0.35, "Aragonit": 2.2, "Kuvars": 14.11, "Kristobalit": 1.88, "Albit": 29.42, "Sanidin": 8.12, "Klinoptilolit": 0.74, "Mordenit": 2.85, "Erionit": 3.56, "Şabazit": 0.18, "Analsim": 0.65, "Hornblend": 0.55, "Aktinolit": 4.26, "Kil Grubu": 4.0, "Volkanik Cam": 22.65},
  {"Sondaj": "2SK", "Derinlik": 18.0, "Kalsit": 1.97, "Dolomit": 0.13, "Aragonit": 1.04, "Kuvars": 22.94, "Kristobalit": 2.21, "Albit": 9.72, "Sanidin": 10.18, "Klinoptilolit": 0.85, "Mordenit": 3.55, "Erionit": 4.68, "Şabazit": 0.07, "Analsim": 0.43, "Hornblend": 0.33, "Aktinolit": 4.94, "Kil Grubu": 4.59, "Volkanik Cam": 32.35},
  {"Sondaj": "2SK", "Derinlik": 31.5, "Kalsit": 25.18, "Dolomit": 0.42, "Aragonit": 3.15, "Kuvars": 13.85, "Kristobalit": 1.92, "Albit": 12.42, "Sanidin": 6.12, "Klinoptilolit": 0.74, "Mordenit": 2.85, "Erionit": 3.95, "Şabazit": 0.28, "Analsim": 0.65, "Hornblend": 0.58, "Aktinolit": 3.54, "Kil Grubu": 4.39, "Volkanik Cam": 19.84},
  {"Sondaj": "2SK", "Derinlik": 34.0, "Kalsit": 14.24, "Dolomit": 0.03, "Aragonit": 1.14, "Kuvars": 18.52, "Kristobalit": 1.95, "Albit": 10.12, "Sanidin": 5.88, "Klinoptilolit": 0.85, "Mordenit": 2.96, "Erionit": 4.32, "Şabazit": 0.01, "Analsim": 0.34, "Hornblend": 0.08, "Aktinolit": 3.94, "Kil Grubu": 4.14, "Volkanik Cam": 31.45},
  {"Sondaj": "2SK", "Derinlik": 41.0, "Kalsit": 34.25, "Dolomit": 0.85, "Aragonit": 3.96, "Kuvars": 10.12, "Kristobalit": 1.45, "Albit": 8.85, "Sanidin": 4.32, "Klinoptilolit": 0.55, "Mordenit": 2.45, "Erionit": 3.15, "Şabazit": 0.08, "Analsim": 0.42, "Hornblend": 0.28, "Aktinolit": 3.04, "Kil Grubu": 3.56, "Volkanik Cam": 22.64},
  {"Sondaj": "3SK", "Derinlik": 0.5, "Kalsit": 4.68, "Dolomit": 0.01, "Aragonit": 2.1, "Kuvars": 4.12, "Kristobalit": 0.85, "Albit": 14.24, "Sanidin": 3.55, "Klinoptilolit": 0.55, "Mordenit": 1.68, "Erionit": 3.12, "Şabazit": 0.01, "Analsim": 0.18, "Hornblend": 0.04, "Aktinolit": 1.96, "Kil Grubu": 3.27, "Volkanik Cam": 21.04},
  {"Sondaj": "3SK", "Derinlik": 10.0, "Kalsit": 27.05, "Dolomit": 29.12, "Aragonit": 1.96, "Kuvars": 7.12, "Kristobalit": 0.88, "Albit": 5.85, "Sanidin": 3.24, "Klinoptilolit": 0.42, "Mordenit": 1.68, "Erionit": 3.15, "Şabazit": 0.01, "Analsim": 0.14, "Hornblend": 0.02, "Aktinolit": 2.09, "Kil Grubu": 2.77, "Volkanik Cam": 14.45},
  {"Sondaj": "3SK", "Derinlik": 20.5, "Kalsit": 19.62, "Dolomit": 31.45, "Aragonit": 1.42, "Kuvars": 7.15, "Kristobalit": 0.95, "Albit": 6.12, "Sanidin": 3.55, "Klinoptilolit": 0.45, "Mordenit": 2.12, "Erionit": 4.08, "Şabazit": 0.01, "Analsim": 0.14, "Hornblend": 0.03, "Aktinolit": 2.73, "Kil Grubu": 3.28, "Volkanik Cam": 16.84},
  {"Sondaj": "3SK", "Derinlik": 40.0, "Kalsit": 3.95, "Dolomit": 15.22, "Aragonit": 11.15, "Kuvars": 12.48, "Kristobalit": 1.22, "Albit": 8.12, "Sanidin": 9.42, "Klinoptilolit": 0.55, "Mordenit": 2.45, "Erionit": 4.62, "Şabazit": 0.01, "Analsim": 0.12, "Hornblend": 0.04, "Aktinolit": 2.84, "Kil Grubu": 3.44, "Volkanik Cam": 24.35},
  {"Sondaj": "4SK", "Derinlik": 0.2, "Kalsit": 2.12, "Dolomit": 0.02, "Aragonit": 0.74, "Kuvars": 7.62, "Kristobalit": 1.15, "Albit": 6.15, "Sanidin": 3.58, "Klinoptilolit": 0.45, "Mordenit": 2.95, "Erionit": 5.32, "Şabazit": 0.01, "Analsim": 0.14, "Hornblend": 0.06, "Aktinolit": 2.8, "Kil Grubu": 4.88, "Volkanik Cam": 61.95},
  {"Sondaj": "4SK", "Derinlik": 23.8, "Kalsit": 2.12, "Dolomit": 0.01, "Aragonit": 7.85, "Kuvars": 13.22, "Kristobalit": 1.55, "Albit": 8.14, "Sanidin": 6.12, "Klinoptilolit": 18.66, "Mordenit": 3.95, "Erionit": 5.88, "Şabazit": 1.96, "Analsim": 0.25, "Hornblend": 0.0, "Aktinolit": 3.77, "Kil Grubu": 4.35, "Volkanik Cam": 22.14},
  {"Sondaj": "4SK", "Derinlik": 31.0, "Kalsit": 1.91, "Dolomit": 0.01, "Aragonit": 14.21, "Kuvars": 10.85, "Kristobalit": 1.32, "Albit": 8.41, "Sanidin": 4.92, "Klinoptilolit": 28.45, "Mordenit": 3.56, "Erionit": 6.15, "Şabazit": 2.85, "Analsim": 0.55, "Hornblend": 0.18, "Aktinolit": 3.6, "Kil Grubu": 0.85, "Volkanik Cam": 12.18},
  {"Sondaj": "4SK", "Derinlik": 36.0, "Kalsit": 1.91, "Dolomit": 0.93, "Aragonit": 8.58, "Kuvars": 9.56, "Kristobalit": 1.61, "Albit": 13.42, "Sanidin": 4.67, "Klinoptilolit": 24.34, "Mordenit": 2.76, "Erionit": 5.52, "Şabazit": 0.51, "Analsim": 0.64, "Hornblend": 0.13, "Aktinolit": 4.46, "Kil Grubu": 2.68, "Volkanik Cam": 18.22},
  {"Sondaj": "5SK", "Derinlik": 0.2, "Kalsit": 1.12, "Dolomit": 0.0, "Aragonit": 2.85, "Kuvars": 9.65, "Kristobalit": 0.94, "Albit": 14.12, "Sanidin": 4.88, "Klinoptilolit": 31.45, "Mordenit": 2.42, "Erionit": 6.24, "Şabazit": 2.15, "Analsim": 0.42, "Hornblend": 0.04, "Aktinolit": 4.07, "Kil Grubu": 1.79, "Volkanik Cam": 17.82},
  {"Sondaj": "5SK", "Derinlik": 10.0, "Kalsit": 0.18, "Dolomit": 10.24, "Aragonit": 8.85, "Kuvars": 11.28, "Kristobalit": 0.45, "Albit": 9.22, "Sanidin": 4.45, "Klinoptilolit": 20.82, "Mordenit": 2.12, "Erionit": 6.22, "Şabazit": 1.68, "Analsim": 0.05, "Hornblend": 0.02, "Aktinolit": 4.12, "Kil Grubu": 1.12, "Volkanik Cam": 19.15},
  {"Sondaj": "5SK", "Derinlik": 16.0, "Kalsit": 0.35, "Dolomit": 0.03, "Aragonit": 9.62, "Kuvars": 5.15, "Kristobalit": 0.24, "Albit": 17.14, "Sanidin": 2.85, "Klinoptilolit": 38.14, "Mordenit": 2.42, "Erionit": 6.48, "Şabazit": 1.92, "Analsim": 0.08, "Hornblend": 0.01, "Aktinolit": 4.4, "Kil Grubu": 0.9, "Volkanik Cam": 10.25},
  {"Sondaj": "5SK", "Derinlik": 23.0, "Kalsit": 0.58, "Dolomit": 0.04, "Aragonit": 4.96, "Kuvars": 8.65, "Kristobalit": 0.38, "Albit": 25.78, "Sanidin": 4.05, "Klinoptilolit": 31.64, "Mordenit": 1.88, "Erionit": 6.72, "Şabazit": 1.22, "Analsim": 0.14, "Hornblend": 0.02, "Aktinolit": 4.09, "Kil Grubu": 0.5, "Volkanik Cam": 9.32},
  {"Sondaj": "5SK", "Derinlik": 40.0, "Kalsit": 0.12, "Dolomit": 0.04, "Aragonit": 4.62, "Kuvars": 10.95, "Kristobalit": 0.15, "Albit": 11.18, "Sanidin": 0.88, "Klinoptilolit": 54.12, "Mordenit": 1.65, "Erionit": 6.42, "Şabazit": 0.35, "Analsim": 0.05, "Hornblend": 0.01, "Aktinolit": 3.37, "Kil Grubu": 0.2, "Volkanik Cam": 5.85},
  {"Sondaj": "6SK", "Derinlik": 1.0, "Kalsit": 20.42, "Dolomit": 25.18, "Aragonit": 12.18, "Kuvars": 14.25, "Kristobalit": 0.34, "Albit": 3.56, "Sanidin": 0.88, "Klinoptilolit": 0.42, "Mordenit": 0.0, "Erionit": 1.15, "Şabazit": 0.0, "Analsim": 2.12, "Hornblend": 0.04, "Aktinolit": 0.87, "Kil Grubu": 1.85, "Volkanik Cam": 6.12},
  {"Sondaj": "6SK", "Derinlik": 12.0, "Kalsit": 5.37, "Dolomit": 15.85, "Aragonit": 1.92, "Kuvars": 19.28, "Kristobalit": 0.94, "Albit": 10.12, "Sanidin": 3.24, "Klinoptilolit": 2.94, "Mordenit": 1.45, "Erionit": 5.85, "Şabazit": 0.0, "Analsim": 0.35, "Hornblend": 0.12, "Aktinolit": 3.5, "Kil Grubu": 4.47, "Volkanik Cam": 23.88},
  {"Sondaj": "6SK", "Derinlik": 24.0, "Kalsit": 4.48, "Dolomit": 6.43, "Aragonit": 17.16, "Kuvars": 13.36, "Kristobalit": 0.97, "Albit": 15.19, "Sanidin": 11.03, "Klinoptilolit": 3.7, "Mordenit": 1.75, "Erionit": 5.26, "Şabazit": 0.31, "Analsim": 0.19, "Hornblend": 0.08, "Aktinolit": 3.31, "Kil Grubu": 2.53, "Volkanik Cam": 14.19},
  {"Sondaj": "6SK", "Derinlik": 30.0, "Kalsit": 2.88, "Dolomit": 6.28, "Aragonit": 14.74, "Kuvars": 13.34, "Kristobalit": 1.16, "Albit": 4.88, "Sanidin": 10.41, "Klinoptilolit": 3.16, "Mordenit": 2.14, "Erionit": 7.21, "Şabazit": 0.46, "Analsim": 0.23, "Hornblend": 0.09, "Aktinolit": 4.56, "Kil Grubu": 6.65, "Volkanik Cam": 21.62},
  {"Sondaj": "6SK", "Derinlik": 36.0, "Kalsit": 6.37, "Dolomit": 8.53, "Aragonit": 10.59, "Kuvars": 12.5, "Kristobalit": 0.88, "Albit": 9.07, "Sanidin": 3.19, "Klinoptilolit": 4.17, "Mordenit": 1.72, "Erionit": 8.09, "Şabazit": 0.39, "Analsim": 0.25, "Hornblend": 0.1, "Aktinolit": 4.91, "Kil Grubu": 6.23, "Volkanik Cam": 22.94},
  {"Sondaj": "6SK", "Derinlik": 49.0, "Kalsit": 61.52, "Dolomit": 2.67, "Aragonit": 3.9, "Kuvars": 11.07, "Kristobalit": 0.08, "Albit": 4.47, "Sanidin": 0.41, "Klinoptilolit": 0.49, "Mordenit": 0.21, "Erionit": 0.9, "Şabazit": 0.04, "Analsim": 0.04, "Hornblend": 0.02, "Aktinolit": 0.74, "Kil Grubu": 2.01, "Volkanik Cam": 11.42}
]

df_main = pd.DataFrame(raw_mineral_data)

# -------------------------------------------------
# 2. AYARLAR VE FORM
# -------------------------------------------------
st.sidebar.title("🔬 3D Mineral Analizi")

available_minerals = [c for c in df_main.columns if c not in ['Sondaj', 'Derinlik']]

with st.sidebar.form("model_ayarlari"):
    target_mineral = st.selectbox("Modellenecek Minerali Seçin", available_minerals, index=available_minerals.index('Erionit') if 'Erionit' in available_minerals else 0)
    
    st.subheader("Görsel Parametreler")
    risk_cutoff = st.slider(f"{target_mineral} Gösterim Eşiği (%)", 0.0, 20.0, 1.0)
    radius_limit = st.slider("2D Yatay Etki Alanı (Metre)", 50, 1500, 600, help="Değer büyüdükçe kütleler birbirine daha çok kaynaşır.")
    z_exag = st.slider("Dikey Abartı (Z)", 1, 30, 8)
    
    st.subheader("Görünüm")
    topo_color = st.color_picker("Topografya Rengi", "#C2B280") 
    topo_opacity = st.slider("Yüzey Şeffaflığı", 0.0, 1.0, 0.4)
    point_size = st.slider("Örnek Boyutu", 2, 10, 5)
    
    submitted = st.form_submit_button("🌋 MODELİ OLUŞTUR / GÜNCELLE")

# -------------------------------------------------
# 3. YÜZEY (TOPOGRAFYA) OLUŞTURUCU (HATA DÜZELTİLDİ)
# -------------------------------------------------
def generate_topography(bh_coords, padding=200):
    """Sondajların tepe kotlarından doğrudan RBF ile topografya haritası üretir."""
    collar_x = [c['x'] for c in bh_coords.values()]
    collar_y = [c['y'] for c in bh_coords.values()]
    collar_z = [c['z'] for c in bh_coords.values()]
    
    tx = np.linspace(min(collar_x) - padding, max(collar_x) + padding, 50)
    ty = np.linspace(min(collar_y) - padding, max(collar_y) + padding, 50)
    TX, TY = np.meshgrid(tx, ty)
    
    # YENİ: griddata yerine RBF (Radial Basis Function) kullandık
    # Bu sayede sadece 6 sondaj olsa bile haritanın tamamını pürüzsüzce doldurur (NaN hatası vermez).
    rbf_topo = Rbf(collar_x, collar_y, collar_z, function='thin_plate')
    TZ = rbf_topo(TX, TY)
    
    return tx, ty, TZ, collar_x, collar_y, collar_z

# -------------------------------------------------
# 4. HESAPLAMA MOTORU (GEOMETRİ VE HACİM)
# -------------------------------------------------
@st.cache_data
def run_3d_engine(mineral_name, radius_val):
    borehole_coords = {
        "1SK": {"x": 631754.22, "y": 4290134.82, "z": 984},
        "2SK": {"x": 630597.83, "y": 4290786.54, "z": 930},
        "3SK": {"x": 629297.03, "y": 4291138.37, "z": 982},
        "4SK": {"x": 629892.93, "y": 4292514.66, "z": 936},
        "5SK": {"x": 630616.91, "y": 4291920.99, "z": 922},
        "6SK": {"x": 631675.12, "y": 4291824.45, "z": 962},
    }

    # Gerçek veri noktalarını hazırla
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
    
    # KOORDİNAT NORMALİZASYONU (Sürekli Hacim İçin)
    x_min, x_max = df_plot['x'].min(), df_plot['x'].max()
    y_min, y_max = df_plot['y'].min(), df_plot['y'].max()
    z_min, z_max = df_plot['z'].min() - 15, df_plot['z'].max() + 5
    
    xn = (df_plot['x'] - x_min) / (x_max - x_min)
    yn = (df_plot['y'] - y_min) / (y_max - y_min)
    zn = (df_plot['z'] - z_min) / (z_max - z_min)

    # İşlem Alanı (Grid)
    res = 40
    margin_xy = 250
    xi = np.linspace(x_min - margin_xy, x_max + margin_xy, res)
    yi = np.linspace(y_min - margin_xy, y_max + margin_xy, res)
    zi = np.linspace(z_min, z_max, res)
    XG, YG, ZG = np.meshgrid(xi, yi, zi, indexing='ij')
    
    XGn = (XG - x_min) / (x_max - x_min)
    YGn = (YG - y_min) / (y_max - y_min)
    ZGn = (ZG - z_min) / (z_max - z_min)
    
    # RBF İnterpolasyon
    rbf = Rbf(xn, yn, zn, df_plot['val'], function='thin_plate')
    vals = rbf(XGn.flatten(), YGn.flatten(), ZGn.flatten())
    vals = np.clip(vals, 0, df_plot['val'].max())

    # Topografyayı Çek
    tx, ty, TZ, collar_x, collar_y, collar_z = generate_topography(borehole_coords, padding=margin_xy)
    
    # TOPOGRAFYA TIRAŞLAMASI (Hacimleri topografyaya uydur - RBF ile güncellendi)
    rbf_topo_vol = Rbf(collar_x, collar_y, collar_z, function='thin_plate')
    TZ_vol = rbf_topo_vol(XG.flatten(), YG.flatten())
    vals[ZG.flatten() > TZ_vol] = -1

    # YATAY (2D) MESAFE MASKESİ
    tree_2d = cKDTree(np.column_stack([df_plot['x'], df_plot['y']]))
    dist_2d, _ = tree_2d.query(np.column_stack([XG.flatten(), YG.flatten()]))
    vals[dist_2d > radius_val] = -1
    
    return tx, ty, TZ, XG, YG, ZG, vals, df_plot, borehole_coords

# -------------------------------------------------
# 5. GÖRSELLEŞTİRME VE ÇİZİM
# -------------------------------------------------
if submitted or 'initialized' not in st.session_state:
    st.session_state.initialized = True
    with st.spinner(f"Kapsamlı Blok Model Oluşturuluyor... Lütfen Bekleyin."):
        tx, ty, TZ, XG, YG, ZG, vals, df_points, bh_coords = run_3d_engine(target_mineral, radius_limit)
        
        fig = go.Figure()

        # 1. Topografya 
        fig.add_trace(go.Surface(
            x=tx, y=ty, z=TZ, 
            opacity=topo_opacity, 
            colorscale=[[0, topo_color], [1, topo_color]],
            showscale=False, 
            name="Yer Yüzeyi"
        ))

        # 2. Sondaj Kuyuları
        for name, c in bh_coords.items():
            fig.add_trace(go.Scatter3d(
                x=[c['x'], c['x']], y=[c['y'], c['y']], z=[c['z'], c['z']-60],
                mode='lines+text', line=dict(color='black', width=5), 
                text=["", name], textposition="top center", name=name, showlegend=False
            ))

        # 3. Örnek Noktaları
        fig.add_trace(go.Scatter3d(
            x=df_points['x'], y=df_points['y'], z=df_points['z'],
            mode='markers',
            marker=dict(size=point_size, color=df_points['val'], colorscale='Viridis', showscale=True, colorbar=dict(title=f"%")),
            name="Sondaj Numuneleri",
            hovertemplate="Değer: %{marker.color:.2f}%<extra></extra>"
        ))

        # 4. Katı Mineral Hacmi (Doldurulmuş Blok Model)
        fig.add_trace(go.Volume(
            x=XG.flatten(), y=YG.flatten(), z=ZG.flatten(),
            value=vals,
            isomin=risk_cutoff, isomax=df_points['val'].max(),
            opacity=0.9,               
            surface_count=45,           
            colorscale='Reds', 
            showscale=False,
            caps=dict(x_show=True, y_show=True, z_show=True), 
            name="Mineral Bloğu"
        ))

        fig.update_layout(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=z_exag * 0.1),
                xaxis_title="Doğu (X)", yaxis_title="Kuzey (Y)", zaxis_title="Kot (Z)",
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
            ),
            margin=dict(l=0, r=0, b=0, t=0), height=850
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"Geçerli Görünüm: **{target_mineral}**. Seçili mineralin ölçülen en yüksek değeri: **%{df_points['val'].max():.2f}**")
