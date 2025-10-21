# app.py
# ==========================
# Plataforma interactiva (Streamlit) para análisis morfológico y color de porotos
# Basado en tu pipeline de Colab: segmentación HSV, medición (regionprops), features de forma
# y extracción de color (promedio y K-Means opcional).

import os
import io
import cv2
import ast
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

# -----------------------------
# Configuración general UI
# -----------------------------
st.set_page_config(page_title="Análisis morfológico de porotos", layout="wide")
st.title("🌱 Análisis morfológico y de color de porotos")

st.markdown(
    "Subí una imagen (fondo azul) y ajustá los parámetros. "
    "El pipeline segmenta, mide, calcula forma y (opcional) color."
)

# -----------------------------
# Utilidades
# -----------------------------
def bgr2rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb2bgr_tuple(rgb):
    r, g, b = rgb
    return (b, g, r)

def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def color_patch_rgb(rgb_tuple, w=80, h=40):
    """Pequeño parche en RGB para mostrar un chip de color en Streamlit."""
    if rgb_tuple is None:
        return np.full((h, w, 3), 200, dtype=np.uint8)
    r, g, b = rgb_tuple
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[:, :] = [r, g, b]
    return patch

def clear_border(mask_bin: np.ndarray) -> np.ndarray:
    """
    Elimina componentes conectados que tocan el borde.
    mask_bin: 0/255 o bool. Devuelve 0/255.
    """
    mb = mask_bin.astype(bool)
    lab = label(mb)
    h, w = lab.shape
    border_labels = set(np.unique(lab[0, :])) | set(np.unique(lab[-1, :])) \
                  | set(np.unique(lab[:, 0])) | set(np.unique(lab[:, -1]))
    out = mb.copy()
    for lbl in border_labels:
        if lbl == 0:
            continue
        out[lab == lbl] = False
    return (out.astype(np.uint8) * 255)

def regiones_validas_ordenadas(mask_bin: np.ndarray, area_min=1000, descartar_borde=True):
    """
    Filtra por áreas válidas y descarta borde si se pide. Devuelve (labels, regiones ordenadas).
    Orden: por X (columna del centroide), luego Y.
    """
    mb = (mask_bin > 0).astype(np.uint8)
    if descartar_borde:
        mb = clear_border(mb)

    lab = label(mb.astype(bool))
    regs = [r for r in regionprops(lab) if r.area >= area_min]
    regs.sort(key=lambda r: (r.centroid[1], r.centroid[0]))
    return lab, regs

def clasificar_forma_reglas(AR, circularidad, solidez, rectangularidad, asim_LR=None, asim_UD=None, sharp_izq=None, sharp_der=None):
    """
    Reglas simples para categorizar forma. Ajustá umbrales según tu colección.
    """
    LR = 0.0 if asim_LR is None else asim_LR
    UD = 0.0 if asim_UD is None else asim_UD
    S_l = 2.0 if sharp_izq is None else sharp_izq
    S_r = 2.0 if sharp_der is None else sharp_der
    Smax = max(S_l, S_r)

    # 1) Redonda
    if AR < 1.2 and circularidad > 0.80 and LR < 0.05 and UD < 0.05:
        return "redonda"

    # 2) Oval
    if 1.2 <= AR < 1.8 and circularidad > 0.60 and Smax < 1.8:
        return "oval"

    # 3) Riñón (asimet./concavidad)
    if solidez < 0.92 and (LR > 0.08 or UD > 0.08):
        return "riñón"

    # 4) Adelgazada en extremos
    if AR >= 1.8 and S_l > 1.9 and S_r > 1.9:
        return "adelgazada_extremos"

    # 5) Truncada (algún extremo aplanado)
    if AR >= 1.4 and (S_l < 1.4 or S_r < 1.4):
        return "truncada"

    # 6) Cúbica (rectangularidad alta y poca circularidad)
    if rectangularidad > 0.85 and circularidad < 0.65:
        return "cúbica"

    return "otra"

def end_cap_sharpness(mask_bool, frac=0.15):
    """
    Índice de 'puntiagudez' en extremos del eje mayor.
    Aproximado: per^2/(4π area) en tiras izquierda/derecha.
    """
    h, w = mask_bool.shape
    t = max(1, int(round(w * frac)))
    left_strip  = mask_bool[:, :t]
    right_strip = mask_bool[:, w - t:]

    def sharp(strip):
        if strip.sum() == 0:
            return 0.0
        cnts_info = cv2.findContours(strip.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
        if not cnts:
            return 0.0
        per = sum(cv2.arcLength(c, True) for c in cnts)
        area = float(strip.sum())
        return float((per * per) / (4.0 * math.pi * area)) if area > 0 else 0.0

    return sharp(left_strip), sharp(right_strip)

# -----------------------------
# Segmentación HSV
# -----------------------------
def segmentar_porotos(img_bgr,
                      azul_bajo=(90, 50, 50),
                      azul_alto=(140, 255, 255),
                      kernel_size=5,
                      close_iters=2,
                      open_iters=1):
    """
    Segmenta fondo azul con HSV -> porotos en blanco (255), fondo 0.
    Incluye cierre y apertura para limpieza.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(azul_bajo, dtype=np.uint8)
    high = np.array(azul_alto, dtype=np.uint8)
    mask_bg = cv2.inRange(hsv, low, high)
    mask = cv2.bitwise_not(mask_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=open_iters)
    return mask

# -----------------------------
# Medición principal
# -----------------------------
def medir_porotos(img_bgr,
                  mask_bin,
                  dpi=800,
                  area_min=1000,
                  descartar_borde=True,
                  calcular_color_promedio=True,
                  calcular_kmeans=False,
                  k=2,
                  kmeans_sample_max=15000,
                  random_state=0):
    """
    Mide porotos (mm) y calcula features de forma. Opcional: color promedio y K-Means.
    Retorna:
      - df_med (una fila por poroto, IDs consecutivos 1..N)
      - fig_overlay (matplotlib con bounding boxes y texto)
      - df_kmeans (opcional, una fila por cluster) o None
    """
    px_to_mm = 25.4 / float(dpi)
    px2_to_mm2 = px_to_mm ** 2

    labels, regs = regiones_validas_ordenadas(mask_bin, area_min=area_min, descartar_borde=descartar_borde)
    if not regs:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(bgr2rgb(img_bgr)); ax.set_title("Sin porotos válidos"); ax.axis("off")
        return pd.DataFrame(), fig, None

    # Overlay
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bgr2rgb(img_bgr))
    ax.set_title(f"Porotos detectados: {len(regs)}")
    ax.axis('off')

    filas = []
    filas_km = []

    # Preparación para cálculos adicionales de forma (simetrías/sharpness)
    mask_bool_global = (mask_bin > 0)

    for idx, region in enumerate(regs, start=1):
        minr, minc, maxr, maxc = region.bbox

        # Geometría en px
        area_px = int(region.area)
        per_px  = float(region.perimeter)
        maj_px  = max(float(region.major_axis_length), 1e-6)
        min_px  = max(float(region.minor_axis_length), 1e-6)
        cy_px, cx_px = region.centroid

        # Conversión a mm
        area_mm2     = area_px * px2_to_mm2
        per_mm       = per_px * px_to_mm
        eje_mayor_mm = maj_px * px_to_mm
        eje_menor_mm = min_px * px_to_mm

        # Features de forma
        AR  = maj_px / min_px
        circ = (4.0 * math.pi * area_px) / (per_px * per_px + 1e-12)
        # solidez/rectangularidad en ROI
        roi_mask = mask_bool_global[minr:maxr, minc:maxc]
        hull = convex_hull_image(roi_mask).astype(np.uint8)
        hull_area = int(hull.sum())
        solidez = float(area_px) / (hull_area + 1e-12)
        rectangularidad = float(area_px) / (maj_px * min_px + 1e-12)

        # Simetrías + puntas (en ROI alineado aprox: usamos tiras en extremos sin rotar para costo bajo)
        sharp_izq, sharp_der = end_cap_sharpness(roi_mask, frac=0.15)
        # (Opcional) podrías calcular asimetrías LR/UD alineando por eje mayor. Aquí lo dejamos simple.
        asim_LR = np.nan
        asim_UD = np.nan

        clase_forma = clasificar_forma_reglas(AR, circ, solidez, rectangularidad, asim_LR, asim_UD, sharp_izq, sharp_der)

        row = {
            "id_poroto": idx,
            "area_mm2": area_mm2,
            "perimetro_mm": per_mm,
            "eje_mayor_mm": eje_mayor_mm,
            "eje_menor_mm": eje_menor_mm,
            "cx_px": cx_px,
            "cy_px": cy_px,
            "AR": AR,
            "circularidad": circ,
            "solidez": solidez,
            "rectangularidad": rectangularidad,
            "sharp_izq": sharp_izq,
            "sharp_der": sharp_der,
            "clase_forma_reglas": clase_forma
        }

        # ----- COLOR PROMEDIO -----
        if calcular_color_promedio:
            roi_color = img_bgr[minr:maxr, minc:maxc].copy()
            local_mask = roi_mask.astype(bool)
            if local_mask.sum() > 0:
                b_mean = float(np.mean(roi_color[:, :, 0][local_mask]))
                g_mean = float(np.mean(roi_color[:, :, 1][local_mask]))
                r_mean = float(np.mean(roi_color[:, :, 2][local_mask]))
                # HSV (OpenCV: H in [0,179], S,V in [0,255])
                hsv_u8 = cv2.cvtColor(
                    np.uint8([[[int(round(b_mean)), int(round(g_mean)), int(round(r_mean))]]]),
                    cv2.COLOR_BGR2HSV
                )[0, 0]
                h, s, v = [int(x) for x in hsv_u8]
                row.update({
                    "R": r_mean, "G": g_mean, "B": b_mean,
                    "H": h, "S": s, "V": v
                })
            else:
                row.update({"R": np.nan, "G": np.nan, "B": np.nan, "H": np.nan, "S": np.nan, "V": np.nan})

        filas.append(row)

        # ----- K-MEANS (opcional) -----
        if calcular_kmeans:
            from sklearn.cluster import KMeans
            roi = img_bgr[minr:maxr, minc:maxc]
            mask_local = roi_mask
            pix = roi[mask_local].reshape(-1, 3)  # BGR
            if len(pix) >= k:
                # muestrar si hay demasiados píxeles (rendimiento)
                if len(pix) > kmeans_sample_max:
                    rng = np.random.default_rng(random_state + idx)
                    sel = rng.choice(len(pix), size=kmeans_sample_max, replace=False)
                    pix_sub = pix[sel]
                else:
                    pix_sub = pix

                km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
                km.fit(pix_sub)
                centers = km.cluster_centers_.astype(np.uint8)
                # aproximar proporciones a partir de pix_sub
                counts = np.bincount(km.labels_, minlength=k)
                props = counts / counts.sum()

                # ordenar por proporción desc
                order = np.argsort(-props)
                centers = centers[order]
                props = props[order]

                # convertir a HSV cada centro
                hsv_centers = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

                for rank in range(k):
                    b, g, r = [int(x) for x in centers[rank]]
                    h_, s_, v_ = [int(x) for x in hsv_centers[rank]]
                    filas_km.append({
                        "id_poroto": idx,
                        "cluster_rank": rank + 1,
                        "proportion": float(props[rank]),
                        "R": r, "G": g, "B": b,
                        "H": h_, "S": s_, "V": v_
                    })

        # ----- Dibujo overlay -----
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='lime', linewidth=2.0, fill=False)
        ax.add_patch(rect)
        ax.text(minc + 5, minr + 5,
                f"{idx}\nA:{area_mm2:.1f} mm²\nL:{eje_mayor_mm:.1f} mm",
                color='yellow', fontsize=10, fontweight='bold',
                va='top', ha='left',
                bbox=dict(facecolor=(0, 0, 0, 0.45), edgecolor='none', pad=1.5))

    df_med = pd.DataFrame(filas)
    df_km = pd.DataFrame(filas_km) if calcular_kmeans and len(filas_km) else None
    fig.tight_layout()
    return df_med, fig, df_km


# =============================
# Sidebar – Parámetros
# =============================
st.sidebar.header("Parámetros")

dpi = st.sidebar.number_input("DPI (resolución del escáner)", min_value=100, max_value=2400, value=800, step=50)
area_min = st.sidebar.number_input("Área mínima (px²) para aceptar poroto", min_value=10, max_value=1_000_000, value=1000, step=50)
descartar_borde = st.sidebar.checkbox("Excluir objetos que tocan el borde", value=True)

st.sidebar.subheader("Segmentación (HSV fondo azul)")
h_low = st.sidebar.slider("H min", 0, 179, 90)
s_low = st.sidebar.slider("S min", 0, 255, 50)
v_low = st.sidebar.slider("V min", 0, 255, 50)
h_high = st.sidebar.slider("H max", 0, 179, 140)
s_high = st.sidebar.slider("S max", 0, 255, 255)
v_high = st.sidebar.slider("V max", 0, 255, 255)
azul_bajo = (h_low, s_low, v_low)
azul_alto = (h_high, s_high, v_high)

st.sidebar.subheader("Morfología")
kernel_size = st.sidebar.slider("Kernel (px)", 1, 15, 5, step=2)
close_iters = st.sidebar.slider("Cierre (iteraciones)", 0, 5, 2)
open_iters  = st.sidebar.slider("Apertura (iteraciones)", 0, 5, 1)

st.sidebar.subheader("Color (opcional)")
do_color_prom = st.sidebar.checkbox("Calcular color promedio por poroto (RGB+HSV)", value=True)
do_kmeans = st.sidebar.checkbox("Calcular K-Means de color por poroto", value=False)
k_clusters = st.sidebar.slider("k (clusters de color)", 2, 5, 3)
k_sample_max = st.sidebar.number_input("Muestreo máx. de pixeles por poroto (K-Means)", min_value=1000, max_value=200000, value=15000, step=1000)

# =============================
# Carga de imagen
# =============================
st.subheader("1) Cargar imagen")
up = st.file_uploader("Subí una imagen con fondo azul (JPG/PNG/TIF)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if up is None:
    st.info("Esperando imagen…")
    st.stop()

file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
if img_bgr is None:
    st.error("No se pudo leer la imagen. Verificá el formato.")
    st.stop()

# =============================
# Segmentación
# =============================
st.subheader("2) Segmentación")
mask = segmentar_porotos(
    img_bgr,
    azul_bajo=azul_bajo,
    azul_alto=azul_alto,
    kernel_size=kernel_size,
    close_iters=close_iters,
    open_iters=open_iters
)

colA, colB = st.columns(2)
with colA:
    st.image(bgr2rgb(img_bgr), caption="Imagen original", use_column_width=True)
with colB:
    st.image(mask, caption="Máscara (porotos en blanco)", use_column_width=True, clamp=True, channels="GRAY")

# =============================
# Medición + Forma + (Color)
# =============================
st.subheader("3) Medición y forma (y color opcional)")
df_med, fig_overlay, df_km = medir_porotos(
    img_bgr,
    mask,
    dpi=dpi,
    area_min=area_min,
    descartar_borde=descartar_borde,
    calcular_color_promedio=do_color_prom,
    calcular_kmeans=do_kmeans,
    k=k_clusters,
    kmeans_sample_max=k_sample_max,
    random_state=0
)

if df_med.empty:
    st.warning("No se detectaron porotos válidos con los parámetros actuales.")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    st.pyplot(fig_overlay, use_container_width=True)

with col2:
    st.markdown("**Tabla de mediciones (por poroto)**")
    # Mostrar chips de color si están disponibles
    show_cols = ["id_poroto", "area_mm2", "perimetro_mm", "eje_mayor_mm", "eje_menor_mm",
                 "AR", "circularidad", "solidez", "rectangularidad", "clase_forma_reglas"]
    if {"R","G","B","H","S","V"}.issubset(df_med.columns):
        show_cols += ["R","G","B","H","S","V"]
    st.dataframe(df_med[show_cols], use_container_width=True)

    # Descarga CSV de mediciones
    csv_med = df_med.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar mediciones (CSV)",
        data=csv_med,
        file_name=f"mediciones_{os.path.splitext(up.name)[0]}.csv",
        mime="text/csv"
    )

# =============================
# Visualizaciones de tamaño
# =============================
st.subheader("4) Visualizaciones de tamaño")
has_area = "area_mm2" in df_med.columns
has_axes = {"eje_mayor_mm","eje_menor_mm"}.issubset(df_med.columns)

col3, col4 = st.columns(2)
if has_area:
    with col3:
        fig = plt.figure(figsize=(5, 3.2))
        plt.hist(df_med["area_mm2"], bins=20)
        plt.xlabel("Área (mm²)"); plt.ylabel("Frecuencia"); plt.title("Histograma de área")
        st.pyplot(fig, use_container_width=True)

if has_axes:
    with col4:
        fig2 = plt.figure(figsize=(5, 3.2))
        plt.scatter(df_med["eje_mayor_mm"], df_med["eje_menor_mm"], s=12)
        plt.xlabel("Eje mayor (mm)"); plt.ylabel("Eje menor (mm)"); plt.title("Dispersión de ejes")
        plt.grid(True, linestyle="--", alpha=0.3)
        st.pyplot(fig2, use_container_width=True)

# =============================
# Color – KMeans (si se pidió)
# =============================
if df_km is not None and not df_km.empty:
    st.subheader("5) Colores dominantes (K-Means)")
    st.caption("cluster_rank=1 es el más predominante en cada poroto (sobre una muestra de píxeles).")

    # Mostrar poroto a poroto el parche de colores concatenado por proporción
    for pid in sorted(df_km["id_poroto"].unique()):
        sub = df_km[df_km["id_poroto"] == pid].sort_values("cluster_rank")
        # Construir franja proporcional de colores (BGR → RGB)
        width = 300
        patch = np.zeros((40, width, 3), dtype=np.uint8)
        start = 0
        for _, row in sub.iterrows():
            w = int(width * float(row["proportion"]))
            rgb = (int(row["R"]), int(row["G"]), int(row["B"]))
            patch[:, start:start+w, :] = np.array(rgb, dtype=np.uint8)
            start += w
        st.image(patch, caption=f"Poroto {pid} — mezcla por proporción", use_column_width=False)

    # Descargar CSV KMeans
    csv_km = df_km.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar K-Means (CSV)",
        data=csv_km,
        file_name=f"kmeans_{os.path.splitext(up.name)[0]}.csv",
        mime="text/csv"
    )

# =============================
# Notas técnicas
# =============================
st.markdown("""
---
**Notas:**
- HSV de OpenCV: `H∈[0,179]`, `S,V∈[0,255]`.
- DPI: controla la conversión px→mm (`px_to_mm = 25.4 / DPI`).
- *Área mínima* y *Excluir borde* impactan fuertemente en el conteo/medición.
- **IDs**: solo se asignan tras filtrar y ordenar las regiones válidas (consistentes con color).
- K-Means se ejecuta **por poroto** y puede tardar; usa muestreo para acelerar.
""")
