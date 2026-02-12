import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from pathlib import Path

st.set_page_config(
    page_title='Análisis VOP - GIBIO',
    page_icon='📈', # This is an emoji shortcode. Could be a URL too.
)

st.title = "Análisis VOP - GIBIO"
'''
# Análisis VOP - GIBIO
'''

st.divider()

# Extraer datos de JSON
with open('data/vop_data.json') as f:
    data = json.load(f)

# Seleccionar sexo y edad
left_column, right_column, blank = st.columns([1, 1, 3])
with left_column:
    st.radio('Sexo', ("Masculino", "Femenino"), key="sexo")
with right_column:
    st.number_input("VOP", min_value=0.0, max_value=30.0, step=0.1, value=7.0, key="vop")

sexo = st.session_state.sexo
vop = st.session_state.vop

if not vop:
    st.warning("Selecconar una VOP válida")
    st.stop()

# Configuración según sexo
titulo = f"Curvas VOP - {sexo}"
linea = 'dashed' if sexo == 'Masculino' else 'solid'
letra = 'h' if sexo == 'Masculino' else 'm'
curva_default = f"{letra}50"
curvas_config = [
    (f"{letra}95", "orchid", "95"),
    (f"{letra}75", "deepskyblue", "75"),
    (f"{letra}25", "peru", "25"),
    (f"{letra}5",  "lightcoral", "5"),
]

# Crear gráfico
fig1, ax1 = plt.subplots(figsize=(6, 2))
fig, ax = plt.subplots(figsize=(6, 2))
curvas = []

def generar_curva(ax, datos_x, datos_y, color, nombre, linea):
    curva, = ax.plot(datos_x, datos_y, color=color, linestyle=linea, label=nombre)
    return curva

# Dibujar curva por defecto
curva_def = generar_curva(ax1, data[curva_default]["x"], data[curva_default]["y"], "limegreen", "50", linea)

# Dibujar curvas restantes
for key, color, label in curvas_config:
    curva = generar_curva(ax, data[key]["x"], data[key]["y"], color, label, linea)
    curvas.append(curva)

ax.set_xlabel("Edad")
ax.set_ylabel("VOP")
ax.set_title(titulo)
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.grid(True)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.subplots_adjust(right=0.78)

# Interpolación
def edad_por_vop(curva, vop):
    return np.interp(vop, curva.get_ydata(), curva.get_xdata())

# Función para armar el Dataframe de las curvas
def armar_dataframe(curvas, vop):
    rows = []
    for curva in curvas:
        edad_estimada = edad_por_vop(curva, vop)
        rows.append({
            "Curva": curva.get_label(),
            "Edad": round(edad_estimada, 2)
        })
    return pd.DataFrame(rows)

# Función marcar edad estimada en el gráfico
def marcar_edad_estimada(ax, edad, vop):
    ax.plot(edad, vop, marker="x", color="red")

# Calcular edad estimada y marcar en gráfico
edad_estimada = edad_por_vop(curva_def, vop)
marcar_edad_estimada(ax1, edad_estimada, vop)
df_resultado = armar_dataframe([curva_def], vop)

for curva in curvas:
    edad_estimada = edad_por_vop(curva, vop)
    marcar_edad_estimada(ax, edad_estimada, vop)
df_resultados = armar_dataframe(curvas, vop) 

# Mostrar gráfico y tabla
st.pyplot(fig1)
st.dataframe(df_resultado, width=400)

st.pyplot(fig)
st.dataframe(df_resultados, width=400)
