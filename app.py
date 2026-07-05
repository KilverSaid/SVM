import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Configuración de la página
st.set_page_config(page_title="Riesgo Actuarial - IS-701", layout="wide")
sns.set_theme(style='whitegrid', context='notebook')

st.title("Riesgo Actuarial con K-means y SVM")
st.markdown("### Asignatura: IS-701-Inteligencia Artificial - Campus Comayagua")
st.write("Visualización y predicción interactiva utilizando los modelos y datos preprocesados.")

st.markdown("---")

# 1. Carga de Datos Preprocesados
st.header("1. Datos de Seguros con Segmentación (Clusters)")
data_file = "insurance_con_clusters.csv"
kernels_file = "svm_resultados_kernels.csv"

if os.path.exists(data_file):
    df = pd.read_csv(data_file)
    st.success(f"Archivo '{data_file}' cargado con éxito.")
    
    with st.expander("Ver registros del dataset segmentado"):
        st.dataframe(df.head())
        st.write(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas.")
else:
    st.error(f"No se encontró el archivo '{data_file}' en el directorio actual. Verifica tu espacio de trabajo.")

# Mostrar tabla de rendimiento de Kernels si existe
if os.path.exists(kernels_file):
    with st.expander("Ver Comparativa de Rendimiento de Kernels SVM"):
        df_kernels = pd.read_csv(kernels_file)
        st.dataframe(df_kernels)

st.markdown("---")

# 2. Análisis Exploratorio Basado en los Clusters Actuariales
if 'df' in locals():
    st.header("2. Análisis de Clústeres Actuariales")
    st.write("Distribuciones y relaciones clave segmentadas por el modelo K-means entrenado.")
    
    cluster_col = 'cluster' if 'cluster' in df.columns else [c for c in df.columns if 'cluster' in c.lower()][0] if any('cluster' in c.lower() for c in df.columns) else None
    
    if cluster_col:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Gráfico 1: Distribución de cargos por Cluster
        sns.histplot(data=df, x='charges', hue=cluster_col, kde=True, bins=35, palette='tab10', multiple='stack', ax=axes[0,0])
        axes[0,0].set_title("Distribución de Cargos por Cluster")

        # Gráfico 2: Dispersión Edad vs Cargos por Cluster
        sns.scatterplot(data=df, x='age', y='charges', hue=cluster_col, palette='tab10', alpha=.7, ax=axes[0,1])
        axes[0,1].set_title("Edad vs Cargos (Segmentado por Cluster)")

        # Gráfico 3: Boxplot de Cargos según Condición de Fumador y Cluster
        sns.boxplot(data=df, x='smoker', y='charges', hue=cluster_col, palette='tab10', ax=axes[1,0])
        axes[1,0].set_title("Impacto del Tabaquismo por Cluster")

        # Gráfico 4: Distribución de Masa Corporal (BMI) vs Cargos por Cluster
        sns.scatterplot(data=df, x='bmi', y='charges', hue=cluster_col, palette='tab10', alpha=.7, ax=axes[1,1])
        axes[1,1].set_title("BMI vs Cargos (Segmentado por Cluster)")

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No se detectó explícitamente una columna de clúster en el CSV. Asegúrate de mapear la columna correspondiente.")

st.markdown("---")

# 3. Módulo de Inferencia (Predicción adaptado a image_6fb152.png)
st.header("3. Simulación de Predicción de Riesgo")
st.write("Ingresa los datos del cliente para evaluar su segmento según el formato del modelo entrenado.")

# Formulario de entrada de usuario
with st.form("prediction_form"):
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Edad (age)", min_value=18, max_value=100, value=45)
    sex = c2.selectbox("Sexo (sex)", options=["male", "female"])
    bmi = c3.number_input("Índice de Masa Corporal (bmi)", min_value=10.0, max_value=60.0, value=31.2)
    children = c4.number_input("Hijos (children)", min_value=0, max_value=5, value=2)
    
    c5, c6, c7 = st.columns(3)
    smoker = c5.selectbox("¿Fuma? (smoker)", options=["yes", "no"])
    region = c6.selectbox("Región (region)", options=["southeast", "southwest", "northwest", "northeast"])
    charges = c7.number_input("Cargos Médicos Anuales (charges)", min_value=0, max_value=100000, value=28000)
    
    submit = st.form_submit_button("Calcular Riesgo Actuarial")

if submit:
    # Generar el DataFrame con las 7 columnas exactas requeridas por el modelo
    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])
    
    st.write("Estructura del DataFrame enviado al modelo:")
    st.dataframe(cliente)
    
    kmeans_path = 'kmeans_riesgo_actuarial.pkl'
    svm_path = 'svm_riesgo_actuarial.pkl'
    
    if os.path.exists(kmeans_path):
        try:
            # Carga del modelo tal como muestra el código base
            modelo = joblib.load(kmeans_path)
            
            # Ejecución idéntica a la guía de la imagen
            cluster = modelo.predict(cliente)[0]
            
            # Mostrar resultado estilizado
            st.success("¡Predicción completada exitosamente!")
            
            metric_c1, metric_c2 = st.columns(2)
            metric_c1.metric(label="Clúster Predicho (K-means)", value=f"Cluster {cluster}")
            
            # Intentar clasificar con SVM opcionalmente si el archivo existe
            if os.path.exists(svm_path):
                modelo_svm = joblib.load(svm_path)
                # Nota: Si el SVM también procesa el mismo DF:
                # prediccion_svm = modelo_svm.predict(cliente)[0]
                # metric_c2.metric(label="Clasificación de Riesgo (SVM)", value=f"{prediccion_svm}")
                pass
                
        except Exception as e:
            st.error(f"Error al ejecutar la predicción con el modelo: {e}")
            st.info("Consejo: Si salta un error de tipos, asegúrate de que el modelo guardado contenga un Pipeline con encoders para las columnas categóricas ('sex', 'smoker', 'region').")
    else:
        st.error(f"No se encontró el archivo '{kmeans_path}' en el directorio. Por favor ponlo junto al app.py.")
