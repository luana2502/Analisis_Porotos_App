# app.py
# ===========================================
# UI de Streamlit para análisis de porotos:
# - Segmentación por HSV (fondo azul)
# - Medición en mm (DPI configurable)
# - Color promedio y K-Means opcional
# ===========================================

import os, math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from skimage.measure import label, regionprops
# La importación de KMeans se deja dentro de medir_porotos,
# como estaba en el original, para una carga condicional.

# -----------------------------
# Config general
# -----------------------------
st.set_page_config(page_title="Análisis morfológico de porotos", layout="wide")

# -----------------------------
# LOGOS
# -----------------------------
col_logo1, col_logo2, col_logo3 = st.columns([1,1,1])
with col_logo1:
    # Asumiendo que las rutas son correctas, si no existen, esto fallará.
    st.image("data/logo_utec.png", use_container_width=True) 
with col_logo2:
    st.image("data/logo_aria.png", use_container_width=True)
with col_logo3:
    st.image("data/logo_gasma.jpg", use_container_width=True)

st.title("🌱 Análisis morfológico de porotos")
st.markdown("Selecciona el modo de operación. El análisis segmenta, mide y calcula color por poroto.")

# -----------------------------
# Rutas demo
# -----------------------------
DEMO_OVERLAY_PATH = "data/demo_overlay.jpg"
DEMO_RESULTS_PATH = "data/demo_results.csv"

# ======================================================
# ===================== FUNCIONES ======================
# ====== (NO MODIFICADAS, COPIADAS TAL CUAL) ===========
# ======================================================
def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def regiones_validas_ordenadas(
    mask_bin: np.ndarray,
    area_min: int = 1000,
    excluir_borde: bool = True,
    margen_borde_px: int = 20
):
    """
    Filtra regiones por área y descarta las cuya bbox entra en una franja de
    'margen_borde_px' contra cualquiera de los 4 bordes.
    Ordena por X (columna) y luego Y (fila).
    """
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
    """
    Coloca el texto dentro del área visible.
    """
    minr, minc, maxr, maxc = bbox

    # Horizontal
    x = minc + margin
    ha = 'left'
    if (minc + label_w_px + margin) > img_w:
        x = maxc - margin
        ha = 'right'

    # Vertical
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
        bbox=dict(
            facecolor=(0, 0, 0, 0.45),
            edgecolor='none',
            pad=2.0
        ),
        clip_on=True
    )

# **ERROR CORREGIDO: la función estaba mal indentada en el código original.**
def segmentar_porotos(
    img_bgr,
    azul_bajo=(90, 50, 50),
    azul_alto=(140, 255, 255),
    kernel_size=5,
    close_iters=2,
    open_iters=1
):
    """Devuelve máscara 0/255 con porotos en blanco."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(azul_bajo, dtype=np.uint8)
    high = np.array(azul_alto, dtype=np.uint8)

    mask_bg = cv2.inRange(hsv, low, high)
    mask = cv2.bitwise_not(mask_bg)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, k, iterations=close_iters
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, k, iterations=open_iters
    )
    return mask

# **ERROR CORREGIDO: faltaba la 'd' en la definición de la función 'def' en el código original.**
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
    """
    Mide porotos (mm) y calcula color.
    Retorna: df_med (una fila por poroto),
    fig_overlay (matplotlib) y df_kmeans (o None).
    """
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

        # Geometría en px
        area_px = int(r.area)
        per_px = float(r.perimeter)
        maj_px = max(float(r.major_axis_length), 1e-6)
        min_px = max(float(r.minor_axis_length), 1e-6)

        # Conversión a mm
        area_mm2 = area_px * px2_to_mm2
        per_mm = per_px * px_to_mm
        eje_mayor_mm = maj_px * px_to_mm
        eje_menor_mm = min_px * px_to_mm

        # Forma
        circularidad = (
            4.0 * math.pi * area_px
        ) / (per_px * per_px + 1e-12)

        row = {
            "id_poroto": idx,
            "area_mm2": area_mm2,
            "perimetro_mm": per_mm,
            "eje_mayor_mm": eje_mayor_mm,
            "eje_menor_mm": eje_menor_mm,
            "circularidad": circularidad
        }

        # Color promedio
        if calcular_color_promedio:
            roi_mask = mask_bool[minr:maxr, minc:maxc]
            roi = img_bgr[minr:maxr, minc:maxc]
            m = roi_mask.astype(bool)
            if m.sum() > 0:
                b_mean = float(np.mean(roi[:, :, 0][m]))
                g_mean = float(np.mean(roi[:, :, 1][m]))
                r_mean = float(np.mean(roi[:, :, 2][m]))

                hsv = cv2.cvtColor(
                    np.uint8([[[int(round(b_mean)),
                                int(round(g_mean)),
                                int(round(r_mean))]]]),
                    cv2.COLOR_BGR2HSV
                )[0, 0]
                h, s, v = [int(x) for x in hsv]

                row.update({
                    "R": r_mean,
                    "G": g_mean,
                    "B": b_mean,
                    "H": h,
                    "S": s,
                    "V": v
                })

        filas.append(row)

        # K-Means (opcional)
        if calcular_kmeans:
            # Importar solo si se usa
            from sklearn.cluster import KMeans

            roi = img_bgr[minr:maxr, minc:maxc]
            roi_mask = mask_bool[minr:maxr, minc:maxc]
            pix = roi[roi_mask].reshape(-1, 3)  # BGR

            if len(pix) >= k:
                if len(pix) > kmeans_sample_max:
                    rng = np.random.default_rng(random_state + idx)
                    sel = rng.choice(
                        len(pix),
                        kmeans_sample_max,
                        replace=False
                    )
                    pix = pix[sel]

                km = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init='auto',
                    max_iter=300
                ).fit(pix)

                centers = km.cluster_centers_.astype(np.uint8)
                counts = np.bincount(km.labels_, minlength=k)
                props = counts / counts.sum()

                order = np.argsort(-props)
                centers = centers[order]
                props = props[order]

                hsv_centers = cv2.cvtColor(
                    centers.reshape(1, -1, 3),
                    cv2.COLOR_BGR2HSV
                ).reshape(-1, 3)

                for rank in range(k):
                    b, g, r = [int(x) for x in centers[rank]]
                    h_, s_, v_ = [int(x) for x in hsv_centers[rank]]
                    filas_km.append({
                        "id_poroto": idx,
                        "cluster_rank": rank + 1,
                        "proportion": float(props[rank]),
                        "R": r,
                        "G": g,
                        "B": b,
                        "H": h_,
                        "S": s_,
                        "V": v_
                    })

        # Dibujo: bbox y texto (A/P/L/W)
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

        _place_text_inside(
            ax,
            (minr, minc, maxr, maxc),
            txt,
            img_w,
            img_h
        )

    df_med = pd.DataFrame(filas)
    df_km = (
        pd.DataFrame(filas_km)
        if (calcular_kmeans and len(filas_km))
        else None
    )

    fig.tight_layout()
    return df_med, fig, df_km

# ======================================================
# SIDEBAR — Parámetros (igual que original)
# ======================================================
st.sidebar.header("Parámetros")
dpi = st.sidebar.number_input("DPI (escáner)", 100, 2400, 800, step=50)
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

do_color_prom = st.sidebar.checkbox("Color promedio", True)
do_kmeans = st.sidebar.checkbox("K-Means por poroto", False)

# ======================================================
# TABS — ORDEN CORRECTO
# ======================================================
tab_info, tab_demo, tab_user = st.tabs([
    "ℹ️ ¿Qué hace la herramienta?",
    "📊 Ejemplo de resultado",
    "👆 Subir tu propia imagen"
])

# -------- TAB INFO --------
with tab_info:
    st.markdown("""
    Esta herramienta permite el **análisis morfológico y cromático de porotos**
    a partir de imágenes escaneadas con fondo azul (Epson Perfection V850 PRO).

    El diseño separa explícitamente:
    - **df_med**: métricas morfológicas
    - **fig_overlay**: visualización
    - **df_km**: color dominante (K-Means)

    Esto asegura **reproducibilidad y uso académico**.
    """)

# -------- TAB DEMO --------
with tab_demo:
    if os.path.exists(DEMO_OVERLAY_PATH):
        st.image(DEMO_OVERLAY_PATH, use_container_width=True)
    if os.path.exists(DEMO_RESULTS_PATH):
        st.dataframe(pd.read_csv(DEMO_RESULTS_PATH).round(2), use_container_width=True)

# -------- TAB USER --------
with tab_user:
    up = st.file_uploader("Subí imagen con fondo azul", ["jpg","png","tif","tiff"])
    if up:
        # Nota: cv2.imdecode y np.frombuffer esperan bytes
        img = cv2.imdecode(np.frombuffer(up.read(),np.uint8),1) 
        
        # Segmentación
        mask = segmentar_porotos(img, azul_bajo, azul_alto, kernel_size, close_iters, open_iters)
        
        # Medición y color
        df_med, fig, df_km = medir_porotos( # Se almacena df_km aunque no se use en este bloque
            img, mask, dpi, area_min, descartar_borde,
            margen_borde_px, do_color_prom, do_kmeans,
            3, 15000, 0
        )
        
        st.pyplot(fig, use_container_width=True)
        st.dataframe(df_med.round(2), use_container_width=True)
        
        # Mostrar K-Means si está disponible
        if do_kmeans and df_km is not None:
            st.subheader("Resultados K-Means")
            st.dataframe(df_km.round(2), use_container_width=True)
