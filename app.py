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

st.title("📊 Riesgo Actuarial con K-means y SVM")
st.markdown("### Asignatura: IS-701-Inteligencia Artificial - Campus Comayagua")
st.write("Propósito didáctico: practicar segmentación no supervisada con K-means y clasificación con Máquinas de Vectores de Soporte.")

---

# 1. Carga de Datos
st.header("1. Carga del Dataset")
uploaded_file = st.file_name = "insurance.csv" # Por defecto busca el archivo local

if os.path.exists(uploaded_file):
    df = pd.read_csv(uploaded_file)
    st.success("Dataset 'insurance.csv' cargado con éxito.")
    
    with st.expander("Ver datos originales (Primeros 5 registros)"):
        st.dataframe(df.head())
        st.write(f"Dimensiones originales: {df.shape}")
else:
    st.warning("No se encontró el archivo 'insurance.csv' en el directorio actual. Por favor, súbelo aquí:")
    uploaded_file = st.file_uploader("Elige un archivo CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

# Si el dataset está disponible, procedemos
if 'df' in locals():
    
    # 2. Limpieza Básica
    st.header("2. Limpieza Básica de Datos")
    df_model = df.copy()
    
    # Normalizar textos
    for col in ['sex', 'smoker', 'region']:
        if col in df_model.columns:
            df_model[col] = df_model[col].astype(str).str.strip().str.lower()
            
    # Eliminar duplicados
    duplicados = df_model.duplicated().sum()
    df_model = df_model.drop_duplicates()
    
    col1, col2 = st.columns(2)
    col1.metric("Duplicados eliminados", duplicados)
    col2.metric("Dimensiones finales", f"{df_model.shape[0]} filas, {df_model.shape[1]} col.")
    
    with st.expander("Ver estadísticas descriptivas"):
        st.dataframe(df_model.describe())

    ---

    # 3. Análisis Exploratorio (Seaborn)
    st.header("3. Análisis Exploratorio de Datos")
    st.write("Visualización de las distribuciones y relaciones clave con respecto a los cargos (`charges`).")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Gráfico 1: Dispersión
    sns.scatterplot(data=df_model, x='age', y='charges', hue='smoker', size="bmi", alpha=.65, ax=axes[0,1])
    axes[0,1].set_title("Edad, cargos, fumador y BMI")

    # Gráfico 2: Histograma
    sns.histplot(data=df_model, x='charges', hue='smoker', kde=True, bins=35, ax=axes[0,0])
    axes[0,0].set_title("Distribución de cargos por condición")

    # Gráfico 3: Caja de pivote (Boxplot)
    sns.boxplot(data=df_model, x='smoker', y='charges', hue='sex', ax=axes[1,0])
    axes[1,0].set_title("Cargos por fumador y sexo")

    # Gráfico 4: Hojas/Violín
    sns.violinplot(data=df_model, x='region', y='charges', hue='smoker', ax=axes[1,1])
    axes[1,1].set_title("Cargos por región y fumador")

    plt.tight_layout()
    st.pyplot(fig)

    ---

    # 4. Predicción / Modelos (Estructura de ejecución)
    st.header("4. Simulación de Predicción de Riesgo")
    st.write("Ingresa los datos del cliente para evaluar su segmento de riesgo.")
    
    # Formulario de entrada de usuario
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Edad", min_value=18, max_value=100, value=30)
        sex = c2.selectbox("Sexo", options=["female", "male"])
        bmi = c3.number_input("Índice de Masa Corporal (BMI)", min_value=10.0, max_value=60.0, value=25.0)
        
        c4, c5, c6 = st.columns(3)
        children = c4.number_input("Hijos", min_value=0, max_value=5, value=0)
        smoker = c5.selectbox("¿Fuma?", options=["yes", "no"])
        region = c6.selectbox("Región", options=["southwest", "southeast", "northwest", "northeast"])
        
        submit = st.form_submit_button("Calcular Riesgo Actuarial")
    
    if submit:
        # Crear DataFrame con la entrada del usuario
        input_data = pd.DataFrame([{
            'age': age, 'sex': sex, 'bmi': bmi, 
            'children': children, 'smoker': smoker, 'region': region
        }])
        
        st.write("Datos ingresados para el modelo:")
        st.dataframe(input_data)
        
        # Rutas de los modelos indicados en tu notebook
        kmeans_path = 'models/kmeasn_riesgo_actuarial.pk1'
        svm_path = 'models/svm_riesgo_actuarial.pk1'
        
        # Validación de la existencia de los archivos de modelo
        if os.path.exists(kmeans_path) and os.path.exists(svm_path):
            try:
                kmeans_model = joblib.load(kmeans_path)
                svm_model = joblib.load(svm_path)
                
                # Ejemplo de flujo de predicción (requiere que los .pk1 incluyan el pipeline de preprocesamiento)
                # cluster = kmeans_model.predict(input_data)
                # riesgo = svm_model.predict(input_data)
                
                st.success("¡Modelos cargados correctamente! (Aplica aquí tus funciones de inferencia)")
            except Exception as e:
                st.error(f"Error al procesar los modelos: {e}")
        else:
            st.info("💡 **Nota:** Para generar las predicciones en vivo, asegúrate de haber ejecutado el entrenamiento en tu notebook y guardado los archivos correspondientes en la carpeta `./models/`.")
