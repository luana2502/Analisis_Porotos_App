# app.py
# ===========================================
# UI de Streamlit para análisis de porotos:
# - Segmentación por HSV (fondo azul)
# - Medición en mm, con DPI configurable
# - Tabla de métricas básicas (A, P, L, W, circularidad)
# - Color promedio por poroto (RGB/HSV)
# - (Opcional) K-Means por poroto con parches de color
# ===========================================

import os, math, io
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import streamlit as st
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

# -----------------------------
# Utilidades
# -----------------------------
def bgr2rgb(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def clear_border(mask_bin: np.ndarray) -> np.ndarray:
    """Elimina componentes conectados que tocan el borde (devuelve 0/255)."""
    mb = mask_bin.astype(bool)
    lab = label(mb)
    h, w = lab.shape
    border = set(np.unique(lab[0, :])) | set(np.unique(lab[-1, :])) \
           | set(np.unique(lab[:, 0])) | set(np.unique(lab[:, -1]))
    out = mb.copy()
    for lbl in border:
        if lbl != 0:
            out[lab == lbl] = False
    return (out.astype(np.uint8) * 255)

def regiones_validas_ordenadas(mask_bin: np.ndarray, area_min=1000, descartar_borde=True):
    """Filtra por area y descarta borde; ordena IZQ→DER luego ARR→ABAJO."""
    mb = (mask_bin > 0).astype(np.uint8)
    if descartar_borde:
        mb = clear_border(mb)
    lab = label(mb.astype(bool))
    regs = [r for r in regionprops(lab) if r.area >= area_min]
    regs.sort(key=lambda r: (r.centroid[1], r.centroid[0]))
    return lab, regs

# -----------------------------
# Segmentación HSV (fondo azul)
# -----------------------------
def segmentar_porotos(img_bgr,
                      azul_bajo=(90,50,50),
                      azul_alto=(140,255,255),
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
                  dpi=800, area_min=1000, descartar_borde=True,
                  calcular_color_promedio=True,
                  calcular_kmeans=False, k=3,
                  kmeans_sample_max=15000, random_state=0):
    """Mide (mm), calcula forma y color. Retorna df_med, fig_overlay y df_kmeans (o None)."""
    px_to_mm = 25.4 / float(dpi)
    px2_to_mm2 = px_to_mm ** 2

    labels, regs = regiones_validas_ordenadas(mask_bin, area_min, descartar_borde)
    if not regs:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(bgr2rgb(img_bgr)); ax.set_title("Sin porotos válidos"); ax.axis("off")
        return pd.DataFrame(), fig, None

    # Overlay
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bgr2rgb(img_bgr)); ax.axis("off")
    ax.set_title(f"Porotos detectados: {len(regs)}")

    filas = []
    filas_km = []
    mask_bool = (mask_bin > 0)

    for idx, r in enumerate(regs, start=1):
        minr, minc, maxr, maxc = r.bbox
        area_px = int(r.area)
        per_px  = float(r.perimeter)
        maj_px  = max(float(r.major_axis_length), 1e-6)
        min_px  = max(float(r.minor_axis_length), 1e-6)
        cy_px, cx_px = r.centroid

        # mm
        area_mm2     = area_px * px2_to_mm2
        per_mm       = per_px  * px_to_mm
        eje_mayor_mm = maj_px  * px_to_mm
        eje_menor_mm = min_px  * px_to_mm

        # forma
        AR   = maj_px / min_px
        circ = (4.0 * math.pi * area_px) / (per_px * per_px + 1e-12)

        # solidez y rectangularidad (en ROI)
        roi_mask = mask_bool[minr:maxr, minc:maxc]
        hull = convex_hull_image(roi_mask).astype(np.uint8)
        hull_area = int(hull.sum())
        solidez = float(area_px) / (hull_area + 1e-12)
        rectangularidad = float(area_px) / (maj_px * min_px + 1e-12)

        row = {
            "id_poroto": idx,
            "area_mm2": area_mm2,
            "perimetro_mm": per_mm,
            "eje_mayor_mm": eje_mayor_mm,
            "eje_menor_mm": eje_menor_mm,
            "circularidad": circ,
            "solidez": solidez,
            "rectangularidad": rectangularidad,
            "cx_px": cx_px, "cy_px": cy_px
        }

        # color promedio
        if calcular_color_promedio:
            roi = img_bgr[minr:maxr, minc:maxc]
            m   = roi_mask.astype(bool)
            if m.sum() > 0:
                b_mean = float(np.mean(roi[:, :, 0][m]))
                g_mean = float(np.mean(roi[:, :, 1][m]))
                r_mean = float(np.mean(roi[:, :, 2][m]))
                hsv = cv2.cvtColor(
                    np.uint8([[[int(round(b_mean)), int(round(g_mean)), int(round(r_mean))]]]),
                    cv2.COLOR_BGR2HSV
                )[0,0]
                h, s, v = [int(v) for v in hsv]
                row.update({"R": r_mean, "G": g_mean, "B": b_mean, "H": h, "S": s, "V": v})

        filas.append(row)

        # K-Means opcional
        if calcular_kmeans:
            from sklearn.cluster import KMeans
            roi = img_bgr[minr:maxr, minc:maxc]
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

        # Dibujo overlay (A/P/L/W)
        ax.add_patch(plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                   edgecolor="lime", linewidth=2.0, fill=False))
        ax.text(minc + 5, minr + 5,
                f"{idx}\n"
                f"A:{area_mm2:.1f} mm²\n"
                f"P:{per_mm:.1f} mm\n"
                f"L:{eje_mayor_mm:.1f} mm\n"
                f"W:{eje_menor_mm:.1f} mm",
                color="yellow", fontsize=11, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor=(0,0,0,0.45), edgecolor="none", pad=2.0))

    df_med = pd.DataFrame(filas)
    df_km  = pd.DataFrame(filas_km) if (calcular_kmeans and len(filas_km)) else None
    fig.tight_layout()
    return df_med, fig, df_km

# ===========================================
# UI
# ===========================================
st.set_page_config(layout="wide", page_title="Análisis de porotos")
st.markdown("—")

# Sidebar (parámetros)
st.sidebar.header("Parámetros")
dpi = st.sidebar.number_input("DPI (escáner)", 100, 2400, 800, step=50)
area_min = st.sidebar.number_input("Área mínima (px²)", 10, 1_000_000, 1000, step=50)
descartar_borde = st.sidebar.checkbox("Excluir objetos que tocan el borde", True)

st.sidebar.subheader("Segmentación (HSV fondo azul)")
h_low = st.sidebar.slider("H min", 0, 179, 90)
s_low = st.sidebar.slider("S min", 0, 255, 50)
v_low = st.sidebar.slider("V min", 0, 255, 50)
h_high = st.sidebar.slider("H max", 0, 179, 140)
s_high = st.sidebar.slider("S max", 0, 255, 255)
v_high = st.sidebar.slider("V max", 0, 255, 255)
azul_bajo, azul_alto = (h_low, s_low, v_low), (h_high, s_high, v_high)

st.sidebar.subheader("Morfología")
kernel_size = st.sidebar.slider("Kernel", 1, 15, 5, step=2)
close_iters = st.sidebar.slider("Cierre", 0, 5, 2)
open_iters  = st.sidebar.slider("Apertura", 0, 5, 1)

st.sidebar.subheader("Color")
do_color_prom = st.sidebar.checkbox("Color promedio (RGB/HSV)", True)
do_kmeans     = st.sidebar.checkbox("K-Means por poroto", False)
k_clusters    = st.sidebar.slider("k clusters", 2, 5, 3)
k_sample_max  = st.sidebar.number_input("Muestreo máx. (pix por poroto)", 1000, 200000, 15000, step=1000)

# Carga de imagen
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

# Segmentación
st.subheader("2) Segmentación")
mask = segmentar_porotos(img_bgr, azul_bajo, azul_alto, kernel_size, close_iters, open_iters)
colA, colB = st.columns(2)
with colA: st.image(bgr2rgb(img_bgr), caption="Imagen original", use_column_width=True)
with colB: st.image(mask, caption="Máscara (porotos=blanco)", use_column_width=True, channels="GRAY")

# Medición
st.subheader("3) Medición")
df_med, fig_overlay, df_km = medir_porotos(
    img_bgr, mask,
    dpi=dpi, area_min=area_min, descartar_borde=descartar_borde,
    calcular_color_promedio=do_color_prom,
    calcular_kmeans=do_kmeans, k=k_clusters,
    kmeans_sample_max=k_sample_max, random_state=0
)

if df_med.empty:
    st.warning("No se detectaron porotos válidos con los parámetros actuales.")
    st.stop()

# Vista principal que pediste: overlay + tabla básica
col1, col2 = st.columns([1, 1])
with col1:
    st.pyplot(fig_overlay, use_container_width=True)

with col2:
    st.markdown("**Tabla de medidas básicas por poroto**")
    cols_basic = ["id_poroto","area_mm2","perimetro_mm","eje_mayor_mm","eje_menor_mm","circularidad"]
    df_basic = df_med[cols_basic].copy().round(2)
    st.dataframe(df_basic, use_container_width=True)

    csv_med = df_basic.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar tabla (CSV)",
                       data=csv_med,
                       file_name=f"medidas_{os.path.splitext(up.name)[0]}.csv",
                       mime="text/csv")

# COLOR
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
    st.caption("Activa “Color promedio (RGB/HSV)” en el panel lateral para ver esta tabla.")

if df_km is not None and not df_km.empty:
    st.markdown("**Colores dominantes por poroto (K-Means)** — `cluster_rank=1` es el más predominante.")
    # Tabla resumida (todas las filas de clusters)
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
        st.image(patch, caption=f"Poroto {pid} — mezcla por proporción", use_column_width=False)

    csv_km = df_km.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar K-Means (CSV)",
                       data=csv_km,
                       file_name=f"kmeans_{os.path.splitext(up.name)[0]}.csv",
                       mime="text/csv")
else:
    st.caption("Podés habilitar K-Means en la barra lateral (puede tardar en imágenes grandes).")

# Descripciones
st.markdown("""
---
### ¿Qué significan las columnas?

**id_poroto**: identificador consecutivo asignado tras filtrar (orden por posición: izquierda→derecha y arriba→abajo).  
**area_mm2 (A)**: área proyectada del poroto en milímetros cuadrados (usa tu DPI para convertir px→mm).  
**perimetro_mm (P)**: perímetro del contorno del poroto en milímetros.  
**eje_mayor_mm (L)**: longitud del eje mayor del elipse equivalente (mm).  
**eje_menor_mm (W)**: longitud del eje menor del elipse equivalente (mm).  
**circularidad**: \\( 4\\pi A / P^2 \\) — cercano a 1 indica formas más redondeadas.  
**R,G,B**: color promedio del poroto (0–255) en espacio RGB.  
**H,S,V**: color promedio en HSV de OpenCV (H∈[0,179], S,V∈[0,255]).  
**cluster_rank**: orden de dominancia de color (1 = más predominante).  
**proportion**: proporción (0–1) del cluster dentro del poroto en el muestreo.

**Notas**:  
- Cambiar **DPI** cambia las unidades en mm (usa el del escáner realmente usado).  
- **Área mínima** y **Excluir borde** influyen en qué porotos se aceptan.  
- El rango **HSV** es para el **fondo azul**; si ves fugas, ajustalo.  
""")

