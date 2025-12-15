# app.py
# ===========================================
# Análisis morfológico de porotos
# Herramienta desarrollada en el marco de una
# Beca de Iniciación a la Investigación – UTEC
# ===========================================

import os, math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from skimage.measure import label, regionprops

# ===========================================
# CONFIGURACIÓN GENERAL
# ===========================================
st.set_page_config(
    page_title="Análisis morfológico de porotos",
    layout="wide"
)

# ===========================================
# LOGOS (GitHub friendly)
# ===========================================
col_logo1, col_logo2, col_logo3 = st.columns([1, 1, 1])
with col_logo1:
    st.image("data/logo_utec.png", use_container_width=True)
with col_logo2:
    st.image("data/logo_aria.png", use_container_width=True)
with col_logo3:
    st.image("data/logo_gasma.jpg", use_container_width=True)

st.title("🌱 Análisis morfológico de porotos")

# ===========================================
# TABS PRINCIPALES
# ===========================================
tab_demo, tab_upload, tab_about = st.tabs([
    "📊 Ejemplo de resultado obtenido",
    "👆 Subir mi propia imagen",
    "ℹ️ Acerca de la herramienta"
])

# ===========================================
# RUTAS DEMO
# ===========================================
DEMO_OVERLAY_PATH = "data/demo_overlay.jpg"
DEMO_RESULTS_PATH = "data/demo_results.csv"

# ===========================================
# UTILIDADES
# ===========================================
def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def regiones_validas_ordenadas(mask_bin, area_min=1000, excluir_borde=True, margen_borde_px=20):
    lab = label(mask_bin > 0)
    regs0 = regionprops(lab)
    h, w = lab.shape
    regs = []
    for r in regs0:
        if r.area < area_min:
            continue
        minr, minc, maxr, maxc = r.bbox
        if excluir_borde:
            if minr < margen_borde_px or minc < margen_borde_px or maxr > h-margen_borde_px or maxc > w-margen_borde_px:
                continue
        regs.append(r)
    regs.sort(key=lambda r: (r.centroid[1], r.centroid[0]))
    return lab, regs


# ===========================================
# SEGMENTACIÓN HSV (FONDO AZUL)
# ===========================================
def segmentar_porotos(img_bgr, azul_bajo, azul_alto, kernel_size, close_iters, open_iters):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, np.array(azul_bajo), np.array(azul_alto))
    mask = cv2.bitwise_not(mask_bg)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iters)
    return mask


# ===========================================
# MEDICIÓN
# ===========================================
def medir_porotos(img_bgr, mask_bin, dpi, area_min, descartar_borde, margen_borde_px):
    px_to_mm = 25.4 / dpi
    px2_to_mm2 = px_to_mm ** 2
    labels, regs = regiones_validas_ordenadas(mask_bin, area_min, descartar_borde, margen_borde_px)

    filas = []
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bgr2rgb(img_bgr)); ax.axis("off")

    for idx, r in enumerate(regs, start=1):
        minr, minc, maxr, maxc = r.bbox
        area_mm2 = r.area * px2_to_mm2
        per_mm = r.perimeter * px_to_mm
        eje_mayor_mm = r.major_axis_length * px_to_mm
        eje_menor_mm = r.minor_axis_length * px_to_mm
        circularidad = (4 * math.pi * r.area) / (r.perimeter ** 2 + 1e-12)

        filas.append({
            "id_poroto": idx,
            "area_mm2": area_mm2,
            "perimetro_mm": per_mm,
            "eje_mayor_mm": eje_mayor_mm,
            "eje_menor_mm": eje_menor_mm,
            "circularidad": circularidad
        })

        ax.add_patch(plt.Rectangle((minc, minr), maxc-minc, maxr-minr, fill=False, edgecolor='lime', linewidth=2))
        ax.text(minc, minr, str(idx), color='yellow', fontsize=12, weight='bold')

    return pd.DataFrame(filas), fig


# ===========================================
# SIDEBAR – PARÁMETROS
# ===========================================
st.sidebar.header("Parámetros")

dpi = st.sidebar.number_input("DPI del escáner", 300, 2400, 800, step=50)
area_min = st.sidebar.number_input("Área mínima (px²)", 100, 1000000, 1000)

descartar_borde = st.sidebar.checkbox("Excluir objetos cerca del borde", True)
margen_borde_px = st.sidebar.slider("Margen de borde (px)", 0, 100, 20)

st.sidebar.subheader("Segmentación HSV (fondo azul)")
h_low = st.sidebar.slider("H min", 0, 179, 90)
s_low = st.sidebar.slider("S min", 0, 255, 50)
v_low = st.sidebar.slider("V min", 0, 255, 50)
h_high = st.sidebar.slider("H max", 0, 179, 140)
s_high = st.sidebar.slider("S max", 0, 255, 255)
v_high = st.sidebar.slider("V max", 0, 255, 255)

azul_bajo = (h_low, s_low, v_low)
azul_alto = (h_high, s_high, v_high)

kernel_size = st.sidebar.slider("Kernel morfológico (px)", 1, 15, 5, step=2)
close_iters = st.sidebar.slider("Cierre", 0, 5, 2)
open_iters = st.sidebar.slider("Apertura", 0, 5, 1)


# ===========================================
# TAB 1 – RESULTADOS DEMO
# ===========================================
with tab_demo:
    st.subheader("Resultados obtenidos en la investigación")

    if os.path.exists(DEMO_OVERLAY_PATH):
        st.image(DEMO_OVERLAY_PATH, caption="Imagen demo: segmentación y medición automática", use_container_width=True)
    else:
        st.warning("No se encontró la imagen demo.")

    if os.path.exists(DEMO_RESULTS_PATH):
        df_demo = pd.read_csv(DEMO_RESULTS_PATH)
        st.dataframe(df_demo, use_container_width=True)
        st.download_button(
            "⬇️ Descargar CSV demo",
            data=df_demo.to_csv(index=False).encode("utf-8"),
            file_name="resultados_demo.csv",
            mime="text/csv"
        )
    else:
        st.warning("No se encontró el archivo CSV demo.")


# ===========================================
# TAB 2 – SUBIR IMAGEN
# ===========================================
with tab_upload:
    st.markdown("**⚠️ Esta herramienta funciona únicamente con imágenes capturadas con el escáner Epson Perfection V850 PRO y fondo azul uniforme.**")

    up = st.file_uploader("Subí tu imagen", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if up is None:
        st.info("Esperando imagen para iniciar el análisis.")
        st.stop()

    file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    mask = segmentar_porotos(img_bgr, azul_bajo, azul_alto, kernel_size, close_iters, open_iters)

    colA, colB = st.columns(2)
    with colA:
        st.image(bgr2rgb(img_bgr), caption="Imagen original", use_container_width=True)
    with colB:
        st.image(mask, caption="Máscara segmentada", use_container_width=True)

    df_med, fig = medir_porotos(img_bgr, mask, dpi, area_min, descartar_borde, margen_borde_px)

    if df_med.empty:
        st.warning("No se detectaron porotos válidos.")
        st.stop()

    st.pyplot(fig, use_container_width=True)
    st.dataframe(df_med.round(2), use_container_width=True)


# ===========================================
# TAB 3 – ACERCA DE
# ===========================================
with tab_about:
    st.markdown("""
### ¿Qué hace esta herramienta?
Esta aplicación permite la **segmentación automática, medición morfológica y análisis básico de color** de porotos a partir de imágenes escaneadas, proporcionando métricas objetivas y reproducibles para estudios de semillas.

### Marco institucional
Esta herramienta fue desarrollada en el marco de una **Beca de Iniciación a la Investigación**, financiada por la **Dirección de Investigación y Desarrollo de la Universidad Tecnológica del Uruguay (UTEC)**.

### Tutoría
- Nelcy Atehortua  
- Daniel Bueno  
- Natalia De Almeida

### Grupos de investigación
- **Grupo de Agroecología y Medio Ambiente (GASMA)**  
- **Grupo de investigación en Aplicaciones en Inteligencia Artificial (ARIA)**

### Instructivo de uso
1. Escanear los porotos con **Epson Perfection V850 PRO**, resolución conocida (ej. 800 DPI).
2. Utilizar **fondo azul uniforme**.
3. Subir la imagen en la pestaña correspondiente.
4. Ajustar parámetros de segmentación y morfología en la barra lateral.
5. Descargar los resultados en formato CSV.

### Parámetros (barra lateral)
- **DPI**: convierte píxeles a milímetros.
- **Área mínima**: elimina ruido.
- **Margen de borde**: descarta objetos tocando el marco.
- **HSV**: define el rango del fondo azul.
- **Cierre / Apertura**: corrige huecos o ruido.

---
Desarrollado con fines académicos y de investigación.
""")
