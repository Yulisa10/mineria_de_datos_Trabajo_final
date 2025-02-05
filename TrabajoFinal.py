
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Análisis de Detección de Ocupación")

# Crear una tabla de contenido en la barra lateral
seccion = st.sidebar.radio("Tabla de Contenidos", 
                           ["Vista previa de los datos", 
                            "Información del dataset", 
                            "Mapa de calor de correlaciones", 
                            "Distribución de la variable objetivo", 
                            "Relación entre CO2 y Humedad"])

# Cargar los datos
def load_data():
    df_train = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatrain.csv")
    df_test = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatest.csv")
    df = pd.concat([df_train, df_test], axis=0)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# Mostrar contenido basado en la selección
if seccion == "Vista previa de los datos":
    st.subheader("Vista previa de los datos")
    st.write(df.head())

elif seccion == "Información del dataset":
    st.subheader("Información del dataset")
    if st.checkbox("Mostrar información del dataset"):
        st.write(df.info())

elif seccion == "Mapa de calor de correlaciones":
    st.subheader("Mapa de calor de correlaciones")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)

elif seccion == "Distribución de la variable objetivo":
    st.subheader("Distribución de la variable objetivo")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Occupancy"], ax=ax)
    st.pyplot(fig)

elif seccion == "Relación entre CO2 y Humedad":
    st.subheader("Relación entre CO2 y Humedad")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="CO2", y="Humidity", hue="Occupancy", ax=ax)
    st.pyplot(fig)

st.sidebar.write("Este es un análisis inicial, se pueden agregar modelos predictivos y más visualizaciones interactivas.")

