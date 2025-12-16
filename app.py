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
# TABS – ORDEN CORRECTO
# ===========================================
tab_about, tab_demo, tab_upload = st.tabs([
    "ℹ️ ¿Qué hace la herramienta?",
    "📊 Ejemplo de resultado obtenido",
    "👆 Subí tu propia imagen"
])

# ===========================================
# RUTAS DEMO
# ===========================================
DEMO_OVERLAY_PATH = "data/demo_overlay.jpg"
DEMO_RESULTS_PATH = "data/demo_results.csv"

# -----------------------------
# Utilidades
# -----------------------------
def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def regiones_validas_ordenadas(
    mask_bin: np.ndarray,
    area_min: int = 1000,
    excluir_borde: bool = True,
    margen_borde_px: int = 20
):
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
            if (
                minr < margen_borde_px
                or minc < margen_borde_px
                or maxr > (h - 1 - margen_borde_px)
                or maxc > (w - 1 - margen_borde_px)
            ):
                continue
        regs.append(r)

    regs.sort(key=lambda r: (r.centroid[1], r.centroid[0]))
    return lab, regs


def _place_text_inside(
    ax, bbox, texto, img_w, img_h,
    label_w_px=160, label_h_px=78, margin=5
):
    minr, minc, maxr, maxc = bbox

    x = minc + margin
    ha = 'left'
    if (minc + label_w_px + margin) > img_w:
        x = maxc - margin
        ha = 'right'

    y = minr + margin
    va = 'top'
    if (minr + label_h_px + margin) > img_h:
        y = maxr - margin
        va = 'bottom'

    ax.text(
        x, y, texto,
        color='yellow',
        fontsize=11,
        fontweight='bold',
        ha=ha,
        va=va,
        bbox=dict(facecolor=(0,0,0,0.45), edgecolor='none', pad=2.0),
        clip_on=True
    )

# -----------------------------
# Segmentación HSV (fondo azul)
# -----------------------------
def segmentar_porotos(
    img_bgr,
    azul_bajo=(90, 50, 50),
    azul_alto=(140, 255, 255),
    kernel_size=5,
    close_iters=2,
    open_iters=1
):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(azul_bajo, dtype=np.uint8)
    high = np.array(azul_alto, dtype=np.uint8)

    mask_bg = cv2.inRange(hsv, low, high)
    mask = cv2.bitwise_not(mask_bg)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iters)

    return mask

# -----------------------------
# Medición + forma + color
# -----------------------------
def medir_porotos(
    img_bgr,
    mask_bin,
    dpi=800,
    area_min=1000,
    descartar_borde=True,
    margen_borde_px=20,
    calcular_color_promedio=True,
    calcular_kmeans=False,
    k=3,
    kmeans_sample_max=15000,
    random_state=0
):
    px_to_mm = 25.4 / float(dpi)
    px2_to_mm2 = px_to_mm ** 2

    labels, regs = regiones_validas_ordenadas(
        mask_bin, area_min, descartar_borde, margen_borde_px
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bgr2rgb(img_bgr))
    ax.axis("off")

    if not regs:
        ax.set_title("Sin porotos válidos")
        return pd.DataFrame(), fig, None

    ax.set_title(f"Porotos detectados: {len(regs)}")

    img_h, img_w = img_bgr.shape[:2]
    filas = []
    filas_km = []
    mask_bool = (mask_bin > 0)

    for idx, r in enumerate(regs, start=1):
        minr, minc, maxr, maxc = r.bbox

        area_px = int(r.area)
        per_px = float(r.perimeter)
        maj_px = max(float(r.major_axis_length), 1e-6)
        min_px = max(float(r.minor_axis_length), 1e-6)

        area_mm2 = area_px * px2_to_mm2
        per_mm = per_px * px_to_mm
        eje_mayor_mm = maj_px * px_to_mm
        eje_menor_mm = min_px * px_to_mm

        circularidad = (4.0 * math.pi * area_px) / (per_px**2 + 1e-12)

        row = {
            "id_poroto": idx,
            "area_mm2": area_mm2,
            "perimetro_mm": per_mm,
            "eje_mayor_mm": eje_mayor_mm,
            "eje_menor_mm": eje_menor_mm,
            "circularidad": circularidad
        }

        filas.append(row)

        ax.add_patch(
            plt.Rectangle(
                (minc, minr),
                (maxc - minc),
                (maxr - minr),
                edgecolor="lime",
                linewidth=2.0,
                fill=False
            )
        )

        txt = (
            f"{idx}\n"
            f"A:{area_mm2:.1f} mm²\n"
            f"P:{per_mm:.1f} mm\n"
            f"L:{eje_mayor_mm:.1f} mm\n"
            f"W:{eje_menor_mm:.1f} mm"
        )

        _place_text_inside(ax, (minr, minc, maxr, maxc), txt, img_w, img_h)

    df_med = pd.DataFrame(filas)
    df_km = None

    fig.tight_layout()
    return df_med, fig, df_km

# ===========================================
# SIDEBAR – PARÁMETROS
# ===========================================
st.sidebar.header("Parámetros")

dpi = st.sidebar.number_input("DPI del escáner", 300, 2400, 800, step=50)
area_min = st.sidebar.number_input("Área mínima (px²)", 100, 1000000, 1000)
descartar_borde = st.sidebar.checkbox("Excluir borde", True)
margen_borde_px = st.sidebar.slider("Margen borde (px)", 0, 100, 20)

# ===========================================
# TAB 3 – SUBIR IMAGEN
# ===========================================
with tab_upload:
    up = st.file_uploader("Subí tu imagen", type=["jpg", "png", "tif", "tiff"])
    if up is None:
        st.info("Esperando imagen.")
        st.stop()

    file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)

    mask = segmentar_porotos(img_bgr)

    df_med, fig, _ = medir_porotos(
        img_bgr,
        mask,
        dpi,
        area_min,
        descartar_borde,
        margen_borde_px
    )

    st.pyplot(fig, use_container_width=True)
    st.dataframe(df_med.round(2), use_container_width=True)
