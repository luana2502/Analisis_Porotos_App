# ===========================================
# UI de Streamlit para análisis de porotos
# ===========================================

import os, math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from skimage.measure import label, regionprops

# ===========================================
# Config general
# ===========================================
st.set_page_config(
    page_title="Análisis morfológico de porotos",
    layout="wide"
)

# ===========================================
# LOGOS
# ===========================================
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.image("data/logo_utec.png", use_container_width=True)
with col2:
    st.image("data/logo_aria.png", use_container_width=True)
with col3:
    st.image("data/logo_gasma.jpg", use_container_width=True)

st.title("🌱 Análisis morfológico de porotos")

# ===========================================
# Rutas demo
# ===========================================
DEMO_OVERLAY_PATH = "data/demo_overlay.jpg"
DEMO_RESULTS_PATH = "data/demo_results.csv"

# ===========================================
# ================= FUNCIONES =================
# (NO TOCADAS)
# ===========================================
def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def regiones_validas_ordenadas(mask_bin, area_min=1000, excluir_borde=True, margen_borde_px=20):
    mb = (mask_bin > 0)
    lab = label(mb)
    regs0 = regionprops(lab)
    h, w = lab.shape
    regs = []
    for r in regs0:
        if r.area < area_min:
            continue
        minr, minc, maxr, maxc = r.bbox
        if excluir_borde and margen_borde_px > 0:
            if (minr < margen_borde_px or minc < margen_borde_px or
                maxr > (h - 1 - margen_borde_px) or maxc > (w - 1 - margen_borde_px)):
                continue
        regs.append(r)
    regs.sort(key=lambda r: (r.centroid[1], r.centroid[0]))
    return lab, regs

def _place_text_inside(ax, bbox, texto, img_w, img_h, label_w_px=160, label_h_px=78, margin=5):
    minr, minc, maxr, maxc = bbox
    x, ha = minc + margin, 'left'
    if (minc + label_w_px + margin) > img_w:
        x, ha = maxc - margin, 'right'
    y, va = minr + margin, 'top'
    if (minr + label_h_px + margin) > img_h:
        y, va = maxr - margin, 'bottom'
    ax.text(x, y, texto, color='yellow', fontsize=11, fontweight='bold',
            ha=ha, va=va,
            bbox=dict(facecolor=(0,0,0,0.45), edgecolor='none', pad=2),
            clip_on=True)

def segmentar_porotos(img_bgr, azul_bajo, azul_alto, kernel_size, close_iters, open_iters):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, np.array(azul_bajo), np.array(azul_alto))
    mask = cv2.bitwise_not(mask_bg)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iters)
    return mask

def medir_porotos(img_bgr, mask_bin, dpi, area_min, descartar_borde, margen_borde_px,
                   calcular_color_promedio, calcular_kmeans, k, kmeans_sample_max, random_state):
    px_to_mm = 25.4 / dpi
    px2_to_mm2 = px_to_mm ** 2
    labels, regs = regiones_validas_ordenadas(mask_bin, area_min, descartar_borde, margen_borde_px)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bgr2rgb(img_bgr)); ax.axis("off")

    if not regs:
        return pd.DataFrame(), fig, None

    filas, filas_km = [], []
    mask_bool = mask_bin > 0
    img_h, img_w = img_bgr.shape[:2]

    for idx, r in enumerate(regs, start=1):
        minr, minc, maxr, maxc = r.bbox
        area_px = r.area
        per_px = r.perimeter
        maj_px = max(r.major_axis_length, 1e-6)
        min_px = max(r.minor_axis_length, 1e-6)

        area_mm2 = area_px * px2_to_mm2
        per_mm = per_px * px_to_mm
        eje_mayor_mm = maj_px * px_to_mm
        eje_menor_mm = min_px * px_to_mm
        circularidad = 4 * math.pi * area_px / (per_px**2 + 1e-12)

        row = {
            "id_poroto": idx,
            "area_mm2": area_mm2,
            "perimetro_mm": per_mm,
            "eje_mayor_mm": eje_mayor_mm,
            "eje_menor_mm": eje_menor_mm,
            "circularidad": circularidad
        }

        if calcular_color_promedio:
            roi = img_bgr[minr:maxr, minc:maxc]
            m = mask_bool[minr:maxr, minc:maxc]
            if m.sum() > 0:
                b,g,r = [float(np.mean(roi[:,:,i][m])) for i in range(3)]
                hsv = cv2.cvtColor(np.uint8([[[b,g,r]]]), cv2.COLOR_BGR2HSV)[0,0]
                row.update({"R":r,"G":g,"B":b,"H":int(hsv[0]),"S":int(hsv[1]),"V":int(hsv[2])})

        filas.append(row)

        ax.add_patch(plt.Rectangle((minc,minr),maxc-minc,maxr-minr,
                                   edgecolor="lime",linewidth=2,fill=False))
        txt = f"{idx}\nA:{area_mm2:.1f}\nP:{per_mm:.1f}\nL:{eje_mayor_mm:.1f}\nW:{eje_menor_mm:.1f}"
        _place_text_inside(ax,(minr,minc,maxr,maxc),txt,img_w,img_h)

    return pd.DataFrame(filas), fig, None

# ===========================================
# SIDEBAR — Parámetros
# ===========================================
st.sidebar.header("Parámetros")
dpi = st.sidebar.number_input("DPI", 100, 2400, 800, 50)
area_min = st.sidebar.number_input("Área mínima (px²)", 10, 1_000_000, 1000)
descartar_borde = st.sidebar.checkbox("Excluir borde", True)
margen_borde_px = st.sidebar.slider("Margen borde (px)", 0, 100, 20)

st.sidebar.subheader("HSV fondo azul")
h_low = st.sidebar.slider("H min", 0, 179, 90)
s_low = st.sidebar.slider("S min", 0, 255, 50)
v_low = st.sidebar.slider("V min", 0, 255, 50)
h_high = st.sidebar.slider("H max", 0, 179, 140)
s_high = st.sidebar.slider("S max", 0, 255, 255)
v_high = st.sidebar.slider("V max", 0, 255, 255)
azul_bajo, azul_alto = (h_low,s_low,v_low),(h_high,s_high,v_high)

kernel_size = st.sidebar.slider("Kernel", 1, 15, 5, 2)
close_iters = st.sidebar.slider("Cierre", 0, 5, 2)
open_iters = st.sidebar.slider("Apertura", 0, 5, 1)

do_color = st.sidebar.checkbox("Color promedio", True)

# ===========================================
# TABS PRINCIPALES
# ===========================================
tab_about, tab_demo, tab_user = st.tabs([
    "ℹ️ ¿Qué hace la herramienta?",
    "📊 Ejemplo de resultado",
    "👆 Subí tu propia imagen"
])

# -------- TAB 1 --------
with tab_about:
    st.markdown("""
    ### ¿Qué hace esta herramienta?
    Esta aplicación realiza **análisis morfológico y cromático de porotos**
    a partir de imágenes escaneadas con fondo azul.

    **Flujo principal:**
    1. Segmentación automática por HSV  
    2. Identificación y filtrado de porotos válidos  
    3. Cálculo de métricas morfológicas en mm  
    4. Análisis de color promedio y opcional K-Means  

    El diseño separa explícitamente:
    - `df_med`: métricas cuantitativas  
    - `fig_overlay`: visualización interpretativa  
    - `df_km`: análisis cromático avanzado  

    Esto asegura **reproducibilidad, trazabilidad y uso académico**.
    """)

# -------- TAB 2 --------
with tab_demo:
    if os.path.exists(DEMO_OVERLAY_PATH):
        st.image(DEMO_OVERLAY_PATH, use_container_width=True)
    if os.path.exists(DEMO_RESULTS_PATH):
        df_demo = pd.read_csv(DEMO_RESULTS_PATH)
        st.dataframe(df_demo.round(2), use_container_width=True)

# -------- TAB 3 --------
with tab_user:
    up = st.file_uploader("Subí tu imagen", ["jpg","png","tif","tiff"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.read(),np.uint8),1)
        mask = segmentar_porotos(img, azul_bajo, azul_alto, kernel_size, close_iters, open_iters)
        df_med, fig, _ = medir_porotos(
            img, mask, dpi, area_min, descartar_borde,
            margen_borde_px, do_color, False, 3, 15000, 0
        )
        st.pyplot(fig, use_container_width=True)
        st.dataframe(df_med.round(2), use_container_width=True)

