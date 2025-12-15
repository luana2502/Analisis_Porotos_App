# app.py
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

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(page_title="Análisis morfológico de porotos", layout="wide")

# -----------------------------
# Logos institucionales (carpeta data/)
# -----------------------------
logo_utec  = "data/06-isologotipo-para-fondo-blanco.png"
logo_aria  = "data/Color.png"
logo_gasma = "data/Imagen de WhatsApp 2024-06-05 a las 20.20.39.png"

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    if os.path.exists(logo_utec):
        st.image(logo_utec, use_container_width=True)
with c2:
    if os.path.exists(logo_aria):
        st.image(logo_aria, use_container_width=True)
with c3:
    if os.path.exists(logo_gasma):
        st.image(logo_gasma, use_container_width=True)

# -----------------------------
# Título
# -----------------------------
st.title("🌱 Análisis morfológico y cromático de porotos")

# -----------------------------
# ¿Qué hace la herramienta?
# -----------------------------
st.markdown("""
---
## 📌 ¿Qué hace esta herramienta?

Esta aplicación permite realizar el **análisis morfológico y cromático de porotos (Phaseolus spp.)**
a partir de imágenes digitales, utilizando técnicas de **visión por computadora**.

La herramienta realiza automáticamente:

- Segmentación de porotos sobre fondo azul  
- Medición morfológica en milímetros (área, perímetro, ejes mayor y menor, circularidad)  
- Cálculo de color promedio (RGB y HSV)  
- Identificación opcional de colores dominantes mediante **K-Means**  
- Exportación de resultados en formato **CSV**

Está diseñada como apoyo a tareas de **caracterización fenotípica**, análisis de semillas
y estudios no destructivos.
""")

# -----------------------------
# Marco institucional
# -----------------------------
st.markdown("""
---
## 🎓 Marco institucional y financiamiento

Esta herramienta fue desarrollada en el marco de una **Beca de Iniciación a la Investigación**,
financiada por la **Dirección de Investigación y Desarrollo de la Universidad Tecnológica del Uruguay (UTEC)**.

### Tutoría académica
- **Nelcy Atehortua**
- **Daniel Bueno**
- **Natalia De Almeida**

### Grupos de investigación involucrados
- **Grupo de Agroecología y Medio Ambiente (GASMA)**
- **Grupo de Investigación en Aplicaciones en Inteligencia Artificial (ARIA)**
""")

# -----------------------------
# Requisitos de las imágenes
# -----------------------------
st.markdown("""
---
## ⚠️ Requisitos obligatorios de las imágenes

⚠️ **La herramienta funciona únicamente con imágenes que cumplan TODAS las siguientes condiciones:**

- 📸 Capturadas con **escáner Epson Perfection V850 PRO**
- 🔵 Fondo **azul uniforme**
- 🖼️ Porotos separados entre sí (sin superposición)
- 📁 Formatos admitidos: JPG, PNG, TIF / TIFF
- 🖨️ Resolución coherente con el **DPI configurado**

El uso de imágenes que no cumplan estos requisitos puede generar errores en la segmentación
y en las mediciones morfológicas.
""")

# -----------------------------
# Instructivo de uso
# -----------------------------
st.markdown("""
---
## 🧭 Instructivo de uso

### Paso 1 – Subir imagen
Ir a la pestaña **“Subir mi propia imagen”** y cargar una imagen con fondo azul.

### Paso 2 – Ajustar parámetros
Configurar los parámetros desde la **barra lateral derecha** según la imagen.

### Paso 3 – Segmentación
La aplicación separa automáticamente los porotos del fondo azul utilizando el espacio de color HSV.

### Paso 4 – Medición
Se calculan las variables morfológicas en milímetros usando el DPI indicado.

### Paso 5 – Resultados
- Visualización de porotos detectados
- Tablas descargables en CSV
- Análisis de color promedio y colores dominantes (opcional)
""")

# -----------------------------
# Explicación de parámetros
# -----------------------------
st.markdown("""
---
## ⚙️ Explicación de los parámetros

### 🖨️ DPI (escáner)
Define la resolución de la imagen y permite convertir píxeles a milímetros.

### 📏 Área mínima (px²)
Elimina objetos pequeños que no corresponden a porotos (ruido).

### 🧱 Borde
- **Excluir objetos cerca del borde**: descarta porotos cortados por el marco.
- **Margen de borde**: distancia en píxeles desde el borde que será ignorada.

### 🎨 Segmentación HSV (fondo azul)
Define el rango de color azul que será identificado como fondo.
- **H**: tono
- **S**: saturación
- **V**: brillo

### 🧩 Morfología
- **Kernel**: tamaño del elemento estructurante.
- **Cierre**: rellena huecos internos.
- **Apertura**: elimina ruido pequeño.

### 🌈 Color
- **Color promedio**: calcula RGB y HSV por poroto.
- **K-Means**: identifica colores dominantes dentro de cada poroto.
""")

# ======================================================
# A PARTIR DE AQUÍ: TU PIPELINE ORIGINAL (SIN CAMBIOS)
# ======================================================

DEMO_OVERLAY_PATH = "data/demo_overlay.jpg"
DEMO_RESULTS_PATH = "data/demo_results.csv"

def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def segmentar_porotos(img_bgr, azul_bajo, azul_alto, kernel_size, close_iters, open_iters):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, np.array(azul_bajo), np.array(azul_alto))
    mask = cv2.bitwise_not(mask_bg)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iters)
    return mask

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Parámetros")
dpi = st.sidebar.number_input("DPI (escáner)", 100, 2400, 800, step=50)
area_min = st.sidebar.number_input("Área mínima (px²)", 10, 1_000_000, 1000, step=50)

st.sidebar.subheader("Borde")
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

st.sidebar.subheader("Morfología")
kernel_size = st.sidebar.slider("Kernel (px)", 1, 15, 5, step=2)
close_iters = st.sidebar.slider("Cierre", 0, 5, 2)
open_iters  = st.sidebar.slider("Apertura", 0, 5, 1)

st.sidebar.subheader("Color")
do_color_prom = st.sidebar.checkbox("Color promedio (RGB/HSV)", True)
do_kmeans = st.sidebar.checkbox("K-Means por poroto", False)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["📊 Resultados demo", "👆 Subir mi propia imagen"])

with tab2:
    up = st.file_uploader(
        "Subí una imagen capturada con escáner Epson Perfection V850 PRO y fondo azul",
        type=["jpg", "jpeg", "png", "tif", "tiff"]
    )

if up is None:
    st.info("Esperando imagen para iniciar el análisis.")
    st.stop()

file_bytes = np.asarray(bytearray(up.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, 1)

st.subheader("Segmentación")
mask = segmentar_porotos(img_bgr, azul_bajo, azul_alto, kernel_size, close_iters, open_iters)

colA, colB = st.columns(2)
with colA:
    st.image(bgr2rgb(img_bgr), caption="Imagen original", use_container_width=True)
with colB:
    st.image(mask, caption="Máscara segmentada", use_container_width=True)
