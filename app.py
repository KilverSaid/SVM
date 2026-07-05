import streamlit as St
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuración de la página
st.set_page_config(page_title="Riesgo Actuarial - IS-701", layout="wide")
sns.set_theme(style='whitegrid', context='notebook')

st.title("📊 Riesgo Actuarial con K-means y SVM")
st.markdown("### Asignatura: IS-701-Inteligencia Artificial - Campus Comayagua")
st.write("Visualización y predicción interactiva utilizando los modelos y datos preprocesados.")

St.markdown("---")

# 1. Carga de Datos Preprocesados
st.header("1. Datos de Seguros con Segmentación (Clusters)")
data_file = "insurance_con_clusters.csv"
kernels_file = "svm_resultados_kernels.csv"

if os.path.exists(data_file):
    Df = pd.read_csv(data_file)
    St.success(f"Archivo '{data_file}' cargado con éxito.")
    
    with st.expander("Ver registros del dataset segmentado"):
        St.dataframe(df.head())
        St.write(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas.")
else:
    St.error(f"No se encontró el archivo '{data_file}' en el directorio actual. Verifica tu espacio de trabajo.")

# Mostrar tabla de rendimiento de Kernels si existe
if os.path.exists(kernels_file):
    With st.expander("Ver Comparativa de Rendimiento de Kernels SVM"):
        Df_kernels = pd.read_csv(kernels_file)
        St.dataframe(df_kernels)

St.markdown("---")

# 2. Análisis Exploratorio Basado en los Clusters Actuariales
if 'df' in locals():
    St.header("2. Análisis de Clústeres Actuariales")
    St.write("Distribuciones y relaciones clave segmentadas por el modelo K-means entrenado.")
    
    # Determinar qué columna representa al cluster (por ejemplo 'cluster' o 'cluster_riesgo')
    Cluster_col = 'cluster' if 'cluster' in df.columns else [c for c in df.columns if 'cluster' in c.lower()][0] if any('cluster' in c.lower() for c in df.columns) else None
    
    If cluster_col:
        Fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Gráfico 1: Distribución de cargos por Cluster
        Sns.histplot(data=df, x='charges', hue=cluster_col, kde=True, bins=35, palette='tab10', multiple='stack', ax=axes[0,0])
        Axes[0,0].set_title("Distribución de Cargos por Cluster")

        # Gráfico 2: Dispersión Edad vs Cargos por Cluster
        Sns.scatterplot(data=df, x='age', y='charges', hue=cluster_col, palette='tab10', alpha=.7, ax=axes[0,1])
        Axes[0,1].set_title("Edad vs Cargos (Segmentado por Cluster)")

        # Gráfico 3: Boxplot de Cargos según Condición de Fumador y Cluster
        Sns.boxplot(data=df, x='smoker', y='charges', hue=cluster_col, palette='tab10', ax=axes[1,0])
        Axes[1,0].set_title("Impacto del Tabaquismo por Cluster")

        # Gráfico 4: Distribución de Masa Corporal (BMI) vs Cargos por Cluster
        Sns.scatterplot(data=df, x='bmi', y='charges', hue=cluster_col, palette='tab10', alpha=.7, ax=axes[1,1])
        Axes[1,1].set_title("BMI vs Cargos (Segmentado por Cluster)")

        Plt.tight_layout()
        St.pyplot(fig)
    Else:
        St.warning("No se detectó explícitamente una columna de clúster en el CSV. Asegúrate de mapear la columna correspondiente.")

St.markdown("---")

# 3. Módulo de Inferencia (Predicción)
st.header("3. Simulación de Predicción de Riesgo")
st.write("Ingresa los datos del cliente para evaluar su segmento mediante los archivos `.pkl` cargados.")

# Formulario de entrada de usuario
with st.form("prediction_form"):
    C1, c2, c3 = st.columns(3)
    Age = c1.number_input("Edad", min_value=18, max_value=100, value=30)
    Sex = c2.selectbox("Sexo", options=["female", "male"])
    Bmi = c3.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=60.0, value=25.0)
    
    C4, c5, c6 = st.columns(3)
    Children = c4.number_input("Hijos", min_value=0, max_value=5, value=0)
    Smoker = c5.selectbox("¿Fuma?", options=["yes", "no"])
    Region = c6.selectbox("Región", options=["southwest", "southeast", "northwest", "northeast"])
    
    Submit = st.form_submit_button("Calcular Riesgo Actuarial")

If submit:
    # Generar el dataframe con la estructura de entrada
    Input_data = pd.DataFrame([{
        'age': age, 'sex': sex, 'bmi': bmi, 
        'children': children, 'smoker': smoker, 'region': region
    }])
    
    St.write("Datos ingresados para evaluación:")
    St.dataframe(input_data)
    
    # Nombres exactos de tus archivos según la captura de pantalla
    Kmeans_path = 'kmeans_riesgo_actuarial.pkl'
    Svm_path = 'svm_riesgo_actuarial.pkl'
    
    If os.path.exists(kmeans_path) and os.path.exists(svm_path):
        Try:
            Kmeans_model = joblib.load(kmeans_path)
            Svm_model = joblib.load(svm_path)
            
            St.success("¡Modelos `.pkl` cargados con éxito de manera local!")
            
            # NOTA: Descomenta e integra según cómo proceses tus variables categóricas (OneHot/LabelEncoder) en tu pipeline:
            # cluster_pred = kmeans_model.predict(input_data)
            # riesgo_pred = svm_model.predict(input_data)
            # st.metric(label="Clúster Asignado", value=f"Cluster {cluster_pred[0]}")
            
        Except Exception as e:
            St.error(f"Error al ejecutar la inferencia con los archivos .pkl: {e}")
    Else:
        St.error("Asegúrate de que 'kmeans_riesgo_actuarial.pkl' y 'svm_riesgo_actuarial.pkl' estén en el mismo directorio que app.py.")
