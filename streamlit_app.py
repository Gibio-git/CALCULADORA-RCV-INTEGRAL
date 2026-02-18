import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from pathlib import Path


st.set_page_config(
    page_title='CALCULADORA-RCV-INTEGRAL',
    page_icon='📈', # This is an emoji shortcode. Could be a URL too.
)

st.title = "CALCULADORA-RCV-INTEGRAL"
'''
# CALCULADORA-RCV-INTEGRAL
'''

# st.divider()

#########################################

col1, col2, col3 = st.columns([1, 1.5, 1])

with col2:
    sexo = {1:"Masculino", 0:"Femenino"}
    sexo = st.segmented_control(
        "Sexo",
        options=sexo.keys(),
        format_func=lambda option: sexo[option],
        key="sexo"
    )  

    edad = st.number_input("Edad", min_value=0, max_value=120, step=1, key="edad")

    tabaco = {0:"No", 1:"Sí"}
    tabaco = st.segmented_control(
        "¿Fuma tabaco?",
        options=tabaco.keys(),
        format_func=lambda option: tabaco[option],
        key="tabaco"
    )

    cintura = st.number_input("Perímetro de cintura (cm)", min_value=0.0, max_value=200.0, step=0.1, key="cintura")

    vop = st.number_input("VOP", min_value=0.0, max_value=30.0, step=0.1, key="vop")

    pas = st.number_input("Presión Arterial Sistólica (mmHg)", min_value=0.0, max_value=300.0, step=0.1, key="pas")

    pad = st.number_input("Presión Arterial Diastólica (mmHg)", min_value=0.0, max_value=200.0, step=0.1, key="pad")

    fc = st.number_input("Frecuencia Cardíaca (lpm)", min_value=0.0, max_value=200.0, step=0.1, key="fc")
    
# Validaciión de inputs
warnings = []

def validar_input_bool(valor, nombre):
    if valor is None:
        warnings.append(f"⚠️ Seleccione {nombre}")

def validar_input_num(x, nombre):
    if x is None or x <= 0:
        warnings.append(f"⚠️ {nombre} debe ser mayor a 0")

fields = {
    "Sexo": (sexo, "bool"),
    "si fuma Tabaco": (tabaco, "bool"),
    "Edad": (edad, "num"),
    "Cintura": (cintura, "num"),
    "VOP": (vop, "num"),
    "PAS": (pas, "num"),
    "PAD": (pad, "num"),
    "FC": (fc, "num")
}

for nombre, (valor, tipo) in fields.items():
    if tipo == "bool":
        validar_input_bool(valor, nombre)
    elif tipo == "num":
        validar_input_num(valor, nombre)

if warnings:
    for w in warnings:
        st.warning(w)
    st.stop()

# RIESGO CARDIOVASCULAR
#Cálculo del Riesgo Cardiovascular SCORE2
def RCV_Score2(VOP, EDAD, FC, SEXO, PAS, PAD, TABACO):
    """
    Determinación del RCV NO LAB (SCORE-2 ESC 2021, European Heart Journal.)
    Parámetros: EDAD (40–89), PAS (mmHg), TABACO (1 = SI, 0 = NO), SEXO (1 para hombre, 0 para mujer)
    Retorna:
        riesgo estimado en %
    """
    #Transformaciones estándar del SCORE2
    cage = (EDAD - 60) / 5
    csbp = (PAS - 120) / 20
    #Descriminación por SEXO ()
    if SEXO == 1: #SEXO=1 MASCULINO
       beta_age = 0.3742
       beta_sbp = 0.2777
       beta_smoke = 0.6012
       beta_age_smoke = -0.0755
       beta_age_sbp = -0.0255
       #Supervivencia basal a 10 años (hombres)
       S0 = 0.9605
    else: #SEXO=0 FEMENINO
       beta_age = 0.4648
       beta_sbp = 0.3131
       beta_smoke = 0.7744
       beta_age_smoke = -0.1088
       beta_age_sbp = -0.0277
       #Supervivencia basal a 10 años (mujeres)
       S0 = 0.9776
    #Linear predictor (LP)
    LP = (
       beta_age * cage +
       beta_sbp * csbp +
       beta_smoke * TABACO +
       beta_age_smoke * (cage * TABACO) +
       beta_age_sbp * (cage * csbp)
    )
    #Conversión a RCV
    risk = (1 - (S0 ** math.exp(LP)))*100
    #Clasifica el riesgo SCORE2 según las categorías recomendadas (EU 2021):
    if risk < 2.5:
        clasif= "Bajo"
    elif risk < 7.5:
        clasif= "Moderado"
    elif risk < 10:
        clasif= "Alto"
    else:
        clasif= "Muy alto"
    return max(0, min(risk, 30)), clasif  # limitar entre 0% y 30% (rango normal SCORE2)

#Cálculo del Riesgo Cardiovascular SCORE2 (OPCIONAL A EVALUAR)
def score2_base_risk_OP(age, sex, smoker, sbp):
    """
    Cálculo aproximado del SCORE2 (riesgo a 10 años de eventos cardiovasculares fatales y no fatales).
    La fórmula es una simplificación linealizada basada en el patrón de las tablas europeas.

    Parámetros:
        age : edad en años (40–69)
        sex : "M" o "F"
        smoker : True / False
        sbp : presión arterial sistólica (mmHg)

    Retorna:
        riesgo estimado en %
    """
    # Coeficientes aproximados que reproducen el comportamiento de SCORE2
    base = -8.5                           # intercepto
    beta_age = 0.085                      # sensibilidad a la edad
    beta_sbp = 0.015                      # sensibilidad a la PAS
    beta_smoke = 0.8 if smoker else 0     # penalización por tabaquismo
    beta_sex = 0.5 if sex == "M" else 0   # mayor riesgo masculino

    logit = base + beta_age*age + beta_sbp*sbp + beta_smoke + beta_sex
    risk = 100 * (1 / (1 + np.exp(-logit))) # convertir logit a probabilidad %
    return max(0, min(risk, 30))            #Limitar entre 0% y 30% (rango normal SCORE2)

#RECLASIFICACIÓN Del Riesgo Cardiovascular POR PWV
def RCV_reclasif_VOP(categoria, pwv):
    """
    Reclasificación basada en PWV según guías europeas:
    - PWV ≥ 10 m/s → aumenta 1 categoría de riesgo
    """
    categorias = ["Bajo", "Moderado", "Alto", "Muy alto"]
    idx = categorias.index(categoria)
    if pwv >= 10:
        idx = min(idx + 1, len(categorias) - 1)
    return categorias[idx]

#RECLASIFICACIÓN Del Riesgo Cardiovascular POR PERÍMETRO DE CINTURA
def RCV_reclasif_PC(categoria, sexo, cintura_cm):
    """
    Reclasifica el riesgo cardiovascular SCORE2 incorporando
    obesidad abdominal (perímetro de cintura).
    """
    # Umbral de cintura por sexo
    if sexo == 1:
        cintura_alta = cintura_cm > 102
    elif sexo == 0:
        cintura_alta = cintura_cm > 88
    else:
        raise ValueError("sexo debe ser 'hombre' o 'mujer'")

    # Escala ordinal
    escala = ["Bajo", "Moderado", "Alto", "Muy Alto"]
    idx = escala.index(categoria)

    if cintura_alta and idx < len(escala) - 1:
        riesgo_final = escala[idx + 1]
    else:
        riesgo_final = categoria

    return riesgo_final


# ANÁLISIS INTEGRAL
def Analizar_RCV (VOP, EDAD, FC, SEXO, PAS, PAD, TABACO, PC):
   #Determinación de la eVOP (estimated PWV, complementaria, Greve SV, et al. “Estimated Pulse Wave Velocity Calculated from Age and Mean Arterial Blood Pressure")
   PAM=PAD+0.4*(PAS-PAD)
   eVOP=9.587-0.402*EDAD+4.560e-3*EDAD**2-2.621e-5*EDAD**2*PAM+3.176e-3*EDAD*PAM-1.832e-2*PAM

   #Ajuste de la VOP por Presión Arterial Sistólica y Frecuencia Cardíaca (Importante para la EDAD VASCULAR)
   #Se utiliza la fórmula de ajuste poblacional VOP_norm=VOP_med-beta_PAS*(PAS-PAS_ref)-beta_FC*(FC-FC_ref) - beta_SEXO*(Sexo-Sexo_ref) (McEniery et al. Normal Vascular Ageing 2005)
   #Se ajusta por SEXO dado que las curvas son poblacionales (no separadas)
   VOP_norm = VOP - 0.03*(PAS-120) - 0.01*(FC-75) - 0.25*(SEXO-0)

   #Cálculo RCV
   RCV_10a,categoria = RCV_Score2(VOP, EDAD, FC, SEXO, PAS, PAD, TABACO)
   #Ajuste del RCV por VOP
   categoria_VOP=RCV_reclasif_VOP(categoria, VOP_norm)
   #Ajuste del RCV por PERÍMETRO DE CINTURA
   categoria_PC=RCV_reclasif_PC(categoria, SEXO, PC)

   #Armado del dataframe
   datos_individuales = {
      "DATOS DEMOGRÁFICOS": [""],
      "Edad Cronológica (años)": [f"{EDAD:.1f}"],
      "Sexo (0=Femeninio, 1=Masculino)": [f"{SEXO}"],
      "Presión Arterial Braquial Sistólica (mmHg)": [f"{PAS:.1f}"],
      "Presión Arterial Braquial Diastólica (mmHg)": [f"{PAD:.1f}"],
      "Frecuencia Cardíaca (lpm)": [f"{FC:.1f}"],
      "Consumo de Tabaco (1=SI, 0=NO)": [f"{TABACO}"],
      "Perímetro de Cintura (cm)": [f"{PC}"],
      "-": [""],
      "VELOCIDAD DE LA ONDA DEL PULSO ARTERIAL (INDICADOR DE RIGIDEZ AÓRTICA)": [""],
      "Velocidad de la Onda del Pulso CF medida (m/s)": [f"{VOP:.1f}"],
      "Velocidad de la Onda del Pulso CF ajustada 120/@75 (m/s)": [f"{VOP_norm:.1f}"],
      "Velocidad de la Onda del Pulso CF estimada (clínica) (m/s)": [f"{eVOP:.1f}"],
      "--": [""],
      "RIESGO CARDIOVASCULAR": [""],
      "Riesgo Cardiovascular (No Lab) (%)": [f"{RCV_10a:.1f} ({categoria})"],
      "Riesgo Cardiovascular reclasificado por VOP": [categoria_VOP],
      "Riesgo Cardiovascular reclasificado por PERÍMETRO DE CINTURA": [categoria_PC],
      "---": [""],
   }
   # Crear DataFrame vertical: índice = etiquetas, columna única de valores
   df_Eval_Individuo = pd.DataFrame.from_dict(datos_individuales, orient='index', columns=['Valor'])
#    df_Eval_Individuo = df_Eval_Individuo.style.set_table_styles(
#     [{'selector': 'th.row_heading',
#       'props': [('text-align', 'left')]}]
#    ).set_properties(
#     subset=pd.IndexSlice[:, :],
#     **{'text-align': 'left'}
#    )

   return df_Eval_Individuo, VOP_norm


# EVALUACIÓN
#Parámetros de Evaluación
# Datos_Indiv={
#     "EDAD": edad,
#     "PAS":    pas,
#     "PAD":    pad,
#     "FC":     fc,
#     "VOP":    vop,
#     "SEXO":   sexo,    #0=Femenino, 1=Masculino
#     "TABACO": tabaco,    #0=NO, 1=SI
#     "CINTURA": cintura
# }

#Análisis
Eval, VOP_norm=Analizar_RCV(vop, edad, fc, sexo, pas, pad, tabaco, cintura)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.colheader_justify', 'left')

#Visualización de los resultados obtenidos
df_resultados = Eval.reset_index()
df_resultados.columns = ["Campo", "Valor"]

st.dataframe(
    df_resultados,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Campo": st.column_config.TextColumn(
            "Campo",
            width="large"
        ),
        "Valor": st.column_config.TextColumn(
            "Valor",
            width="small"
        )
    }
)

'''
## EDAD-ESTIMADA-SEGÚN-VOP
'''
# Extraer datos de JSON
with open('data/vop_data.json') as f:
    data = json.load(f)

# Configuración según sexo
titulo = f"Curvas VOP - {'Masculino' if sexo == 1 else 'Femenino'}"
linea = 'dashed' if sexo == 1 else 'solid'
letra = 'm' if sexo == 1 else 'f'
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
ax1.set_xlabel("Edad Estimada (años)")
ax1.set_ylabel("VOP")
ax1.set_title(titulo + " - Mediana")

# Dibujar curvas restantes
for key, color, label in curvas_config:
    curva = generar_curva(ax, data[key]["x"], data[key]["y"], color, label, linea)
    curvas.append(curva)

ax.set_xlabel("Edad Estimada (años)")
ax.set_ylabel("VOP")
ax.set_title(titulo)
ax.xaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.set_major_locator(plt.MultipleLocator(5))
ax.grid(True)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
plt.subplots_adjust(right=0.78)

# Interpolación
def edad_por_vop(curva, vop):
    return round(np.interp(vop, curva.get_ydata(), curva.get_xdata()))

# Función para armar el Dataframe de las curvas
def armar_dataframe(curvas, vop):
    rows = []
    for curva in curvas:
        edad_estimada = edad_por_vop(curva, vop)
        rows.append({
            "Curva": curva.get_label(),
            "Edad Estimada": round(edad_estimada, 2)
        })
    return pd.DataFrame(rows)

# Función marcar edad estimada en el gráfico
def marcar_edad_estimada(ax, edad, vop):
    ax.plot(edad, vop, marker="x", color="red")

# Calcular edad estimada y marcar en gráfico
edad_estimada_def = edad_por_vop(curva_def, vop)
marcar_edad_estimada(ax1, edad_estimada_def, vop)
df_resultado = armar_dataframe([curva_def], vop)

for curva in curvas:
    edad_estimada = edad_por_vop(curva, vop)
    marcar_edad_estimada(ax, edad_estimada, vop)
df_resultados = armar_dataframe(curvas, vop) 

# Mostrar gráfico y tabla
st.pyplot(fig1)
st.dataframe(df_resultado, width=400, hide_index=True)
st.write("Edad Real:", edad, " vs  Edad Estimada: ", edad_estimada_def)


st.pyplot(fig)
st.dataframe(df_resultados, width=400, hide_index=True) 