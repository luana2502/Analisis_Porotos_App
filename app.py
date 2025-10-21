# app.py
# ===========================================
# UI de Streamlit para análisis de porotos:
# - Segmentación por HSV (fondo azul)
# - Medición en mm (DPI configurable)
# - Tabla: id_poroto, área, perímetro, ejes, circularidad
# - Color promedio (RGB/HSV) y K-Means opcional por poroto
# - Filtro "margen de borde" para evitar falsos positivos pegados al marco
# ===========================================

import os, math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

# -----------------------------
# Config general de la página
# -----------------------------
st.set_page_config(page_title="Análisis morfológico de porotos", layout="wide")
st.title("🌱 Análisis morfológico y de color de porotos")
st.markdown("Subí una imagen con **fondo azul** y ajustá los parámetros. El pipeline segmenta, mide y calcula color por poroto.")

# -----------------------------
# Utilidades
# -----------------------------
def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def regiones_validas_ordenadas(mask_bin: np.ndarray,
                               area_min: int = 1000,
                               excluir_borde: bool = True,
                               margen_borde_px: int = 20):
    """
    Filtra regiones por área y descarta las cuya bbox entra en una franja
    de 'margen_borde_px' contra cualquiera de los 4 bordes.
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
            if (minr < margen_borde_px or minc < margen_borde_px or
                maxr > (h - 1 - margen_borde_px) or
                maxc > (w - 1 - margen_borde_px)):
                continue
        regs.append(r)

    regs.sort(key=lambda r: (r.centroid[1], r.centroid[0]))
    return lab, regs

def _place_text_inside(ax, bbox, texto, img_w, img_h,
                       label_w_px=160, label_h_px=78, margin=5):
    """
    Coloca el texto dentro del área visible:
    - si no entra a la derecha, lo alinea a la derecha del bbox
    - si no entra abajo, lo alinea al borde inferior del bbox
    Usa clip_on=True para no dibujar fuera de los ejes.
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
        color='yellow', fontsize=11, fontweight='bold',
        ha=ha, va=va,
        bbox=dict(facecolor=(0,0,0,0.45), edgecolor='none', pad=2.0),
        clip_on=True
    )

# -----------------------------
# Segmentación HSV (fondo azul)
# -----------------------------
def segmentar_porotos(img_bgr,
                      azul_bajo=(90, 50, 50),
                      azul_alto=(140, 255, 255),
                      kernel_size=5, close_iters=2, open_iters=1):
    """Devuelve máscara 0/255 con porotos en blanco."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    low  = np.array(azul_bajo, dtype=np.uint8)
    high = np.array(azul_alto, dtype=np.uint8)
    mask_bg = cv2.inRange(hsv, low, high)
    mask = cv2.bitwise_not(mask_bg)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=open_iters)
    return mask

# -----------------------------
# Medición + forma + color
# -----------------------------
def medir_porotos(img_bgr, mask_bin,
                  dpi=800, area_min=1000,
                  descartar_borde=True, margen_borde_px=20,
                  calcular_color_promedio=True,
                  calcular_kmeans=False, k=3,
                  kmeans_sample_max=15000, random_state=0):
    """
    Mide porotos (mm) y calcula color.
    Retorna: df_med (una fila por poroto), fig_overlay (matplotlib) y df_kmeans (o None).
    """
    px_to_mm   = 25.4 / float(dpi)
    px2_to_mm2 = px_to_mm ** 2

    labels, regs = regiones_validas_ordenadas(mask_bin, area_min, descartar_borde, margen_borde_px)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bgr2rgb(img_bgr)); ax.axis("off")

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
        per_px  = float(r.perimeter)
        maj_px  = max(float(r.major_axis_length), 1e-6)
        min_px  = max(float(r.minor_axis_length), 1e-6)
        cy_px, cx_px = r.centroid

        # Conversión a mm
        area_mm2     = area_px * px2_to_mm2
        per_mm       = per_px  * px_to_mm
        eje_mayor_mm = maj_px  * px_to_mm
        eje_menor_mm = min_px  * px_to_mm

        # Forma
        circularidad = (4.0 * math.pi * area_px) / (per_px * per_px + 1e-12)

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
                    np.uint8([[[int(round(b_mean)), int(round(g_mean)), int(round(r_mean))]]]),
                    cv2.COLOR_BGR2HSV
                )[0,0]
                h, s, v = [int(x) for x in hsv]
                row.update({"R": r_mean, "G": g_mean, "B": b_mean, "H": h, "S": s, "V": v})

        filas.append(row)

        # K-Means (opcional)
        if calcular_kmeans:
            from sklearn.cluster import KMeans
            roi = img_bgr[minr:maxr, minc:maxc]
            roi_mask = mask_bool[minr:maxr, minc:maxc]
            pix = roi[roi_mask].reshape(-1, 3)  # BGR
            if len(pix) >= k:
                if len(pix) > kmeans_sample_max:
                    rng = np.random.default_rng(random_state + idx)
                    sel = rng.choice(len(pix), kmeans_sample_max, replace=False)
                    pix = pix[sel]
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300).fit(pix)
                centers = km.cluster_centers_.astype(np.uint8)
                counts  = np.bincount(km.labels_, minlength=k)
                props   = counts / counts.sum()
                order   = np.argsort(-props)
                centers = centers[order]; props = props[order]
                hsv_centers = cv2.cvtColor(centers.reshape(1,-1,3), cv2.COLOR_BGR2HSV).reshape(-1,3)
                for rank in range(k):
                    b,g,r = [int(x) for x in centers[rank]]
                    h_,s_,v_ = [int(x) for x in hsv_centers[rank]]
                    filas_km.append({
                        "id_poroto": idx,
                        "cluster_rank": rank+1,
                        "proportion": float(props[rank]),
                        "R": r, "G": g, "B": b, "H": h_, "S": s_, "V": v_
                    })

        # Dibujo: bbox y texto (A/P/L/W)
        ax.add_patch(plt.Rectangle((minc, minr), (maxc - minc), (maxr - minr),
                                   edgecolor="lime", linewidth=2.0, fill=False))
        txt = (f"{idx}\n"
               f"A:{area_mm2:.1f} mm²\n"
               f"P:{per_mm:.1f} mm\n"
               f"L:{eje_mayor_mm:.1f} mm\n"
               f"W:{eje_menor_mm:.1f} mm")
        _place_text_inside(ax, (minr, minc, maxr, maxc), txt, img_w, img_h)

    df_med = pd.DataFrame(filas)
    df_km  = pd.DataFrame(filas_km) if (calcular_kmeans and len(filas_km)) else None
    fig.tight_layout()
    return df_med, fig, df_km

# ===========================================
# Sidebar — Parámetros
# ===========================================
st.sidebar.header("Parámetros")
dpi = st.sidebar.number_input("DPI (escáner)", 100, 2400, 800, step=50)
area_min = st.sidebar.number_input("Área mínima (px²)", 10, 1_000_000, 1000, step=50)

st.sidebar.subheader("Borde")
descartar_borde = st.sidebar.checkbox("Excluir objetos cerca del borde", True)
margen_borde_px = st.sidebar.slider("Margen de borde a ignorar (px)", 0, 100, 20)

st.sidebar.subheader("Segmentación (HSV, fondo azul)")
h_low = st.sidebar.slider("H min", 0, 179, 90)
s_low = st.sidebar.slider("S min", 0, 255, 50)
v_low = st.sidebar.slider("V min", 0, 255, 50)
h_high = st.sidebar.slider("H max", 0, 179, 140)
s_high = st.sidebar.slider("S max", 0, 255, 255)
v_high = st.sidebar.slider("V max", 0, 255, 255)
azul_bajo, azul_alto = (h_low, s_low, v_low), (h_high, s_high, v_high)

st.sidebar.subheader("Morfología")
kernel_size = st.sidebar.slider("Kernel (px)", 1, 15, 5, step=2)
close_iters = st.sidebar.slider("Cierre (iteraciones)", 0, 5, 2)
open_iters  = st.sidebar.slider("Apertura (iteraciones)", 0, 5, 1)

st.sidebar.subheader("Color")
do_color_prom = st.sidebar.checkbox("Color promedio (RGB/HSV)", True)
do_kmeans     = st.sidebar.checkbox("K-Means por poroto", False)
k_clusters    = st.sidebar.slider("k clusters", 2, 5, 3)
k_sample_max  = st.sidebar.number_input("Muestreo máx. (pix por poroto)", 1000, 200000, 15000, step=1000)

# ===========================================
# 1) Cargar imagen
# ===========================================
st.subheader("1) Cargar imagen")
up = st.file_uploader("Subí una imagen con fondo azul (JPG/PNG/TIF)", type=["jpg","jpeg","png","tif","tiff"])
if up is None:
    st.info("Esperando imagen…")
    st.stop()

file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)
if img_bgr is None:
    st.error("No se pudo leer la imagen.")
    st.stop()

# ===========================================
# 2) Segmentación
# ===========================================
st.subheader("2) Segmentación")
mask = segmentar_porotos(img_bgr, azul_bajo, azul_alto, kernel_size, close_iters, open_iters)

colA, colB = st.columns(2)
with colA:
    st.image(bgr2rgb(img_bgr), caption="Imagen original", use_container_width=True)
with colB:
    st.image(mask, caption="Máscara (porotos = blanco)", use_container_width=True, channels="GRAY")

# ===========================================
# 3) Medición
# ===========================================
st.subheader("3) Medición")
df_med, fig_overlay, df_km = medir_porotos(
    img_bgr, mask,
    dpi=dpi, area_min=area_min,
    descartar_borde=descartar_borde, margen_borde_px=margen_borde_px,
    calcular_color_promedio=do_color_prom,
    calcular_kmeans=do_kmeans, k=k_clusters,
    kmeans_sample_max=k_sample_max, random_state=0
)

if df_med.empty:
    st.warning("No se detectaron porotos válidos con los parámetros actuales.")
    st.stop()

col1, col2 = st.columns([1, 1])
with col1:
    st.pyplot(fig_overlay, use_container_width=True)

with col2:
    st.markdown("**Tabla de medidas básicas por poroto**")
    cols_basic = ["id_poroto","area_mm2","perimetro_mm","eje_mayor_mm","eje_menor_mm","circularidad"]
    st.dataframe(df_med[cols_basic].round(2), use_container_width=True)

    csv_med = df_med[cols_basic].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar tabla (CSV)",
                       data=csv_med,
                       file_name=f"medidas_{os.path.splitext(up.name)[0]}.csv",
                       mime="text/csv")

# ===========================================
# 4) Color
# ===========================================
st.subheader("4) Color")
if do_color_prom and {"R","G","B","H","S","V"}.issubset(df_med.columns):
    st.markdown("**Color promedio por poroto**")
    df_color = df_med[["id_poroto","R","G","B","H","S","V"]].copy()
    df_color[["R","G","B"]] = df_color[["R","G","B"]].round(0)
    st.dataframe(df_color, use_container_width=True)

    csv_color = df_color.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar color promedio (CSV)",
                       data=csv_color,
                       file_name=f"color_promedio_{os.path.splitext(up.name)[0]}.csv",
                       mime="text/csv")
else:
    st.caption("Activa “Color promedio (RGB/HSV)” en la barra lateral para ver esta tabla.")

if df_km is not None and not df_km.empty:
    st.markdown("**Colores dominantes por poroto (K-Means)** — `cluster_rank=1` es el más predominante.")
    st.dataframe(df_km[["id_poroto","cluster_rank","proportion","R","G","B","H","S","V"]].round(3),
                 use_container_width=True)

    # Parches proporcionales por poroto
    st.markdown("**Parches proporcionales**")
    for pid in sorted(df_km["id_poroto"].unique()):
        sub = df_km[df_km["id_poroto"] == pid].sort_values("cluster_rank")
        width = 320
        patch = np.zeros((40, width, 3), dtype=np.uint8)
        start = 0
        for _, row in sub.iterrows():
            w = int(width * float(row["proportion"]))
            patch[:, start:start+w, :] = [int(row["R"]), int(row["G"]), int(row["B"])]
            start += w
        st.image(patch, caption=f"Poroto {pid} — mezcla por proporción", use_container_width=False)

    csv_km = df_km.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar K-Means (CSV)",
                       data=csv_km,
                       file_name=f"kmeans_{os.path.splitext(up.name)[0]}.csv",
                       mime="text/csv")
else:
    st.caption("Podés habilitar K-Means en la barra lateral (puede tardar en imágenes grandes).")

# ===========================================
# Descripción de columnas
# ===========================================
st.markdown("""
---
### ¿Qué significa cada columna?

- **id_poroto**: identificador consecutivo (orden espacial IZQ→DER y ARR→ABAJO) tras filtrar.
- **area_mm2 (A)**: área proyectada (mm²). Conversión px→mm usa el **DPI** configurado.
- **perimetro_mm (P)**: perímetro del contorno (mm).
- **eje_mayor_mm (L)**: longitud del eje mayor del elipse equivalente (mm).
- **eje_menor_mm (W)**: longitud del eje menor del elipse equivalente (mm).
- **circularidad**: \\( 4\\pi A / P^2 \\). Cercano a 1 indica formas más redondeadas.
- **R,G,B**: color promedio del poroto en RGB (0–255).
- **H,S,V**: color promedio en HSV (OpenCV: **H∈[0,179]**, **S,V∈[0,255]**).
- **cluster_rank** (K-Means): 1 = color más predominante del poroto.
- **proportion** (K-Means): fracción del poroto representada por cada cluster.

**Notas**
- El **margen de borde (px)** descarta componentes pegados al marco (útil para brillos/ruido).
- Aumentar **Área mínima** y/o la **Apertura** morfológica reduce puntitos de ruido.
""")
