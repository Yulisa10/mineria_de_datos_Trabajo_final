
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("Análisis de Detección de Ocupación")

# Crear una tabla de contenido en la barra lateral
seccion = st.sidebar.radio("Tabla de Contenidos", 
                           ["Vista previa de los datos", 
                            "Información del dataset", 
                            "Análisis Descriptivo", 
                            "Mapa de calor de correlaciones", 
                            "Distribución de la variable objetivo", 
                            "Relación entre CO2 y Humedad", 
                            "Entrenamiento del Modelo MLP", 
                            "Hacer una Predicción"])

# Cargar los datos
def load_data():
    df_train = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatrain.csv")
    df_test = pd.read_csv("https://raw.githubusercontent.com/JuanPablo9999/Mineria_de_datos_streamlit/main/datatest.csv")
    df = pd.concat([df_train, df_test], axis=0)
    df.drop(columns=["id", "date"], inplace=True, errors='ignore')
    return df

df = load_data()

# Preprocesamiento
def preprocess_data(df):
    X = df.drop(columns=["Occupancy"], errors='ignore')
    y = df["Occupancy"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

X, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo MLP
def train_mlp():
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=500, verbose=0)
    return model

# Mostrar contenido basado en la selección
if seccion == "Vista previa de los datos":
    st.subheader("Vista previa de los datos")
    st.write(df.head())

elif seccion == "Información del dataset":
    st.subheader("Información del dataset")
    if st.checkbox("Mostrar información del dataset"):
        st.write(df.info())
        st.write("La base de datos seleccionada para el desarrollo de la aplicación corresponde a un estudio diseñado para optimizar actividades de clasificación binaria para determinar sí una habitación está ocupada o no. Dentro de sus características, se recopilan mediciones ambientales tales como la temperatura, la humedad del ambiente, la luz o nivel de luminosidad, y niveles de CO2, donde, con base a estas se determina sí la habitación está ocupada. La información de ocupación se obtuvo mediante la obtención de imágenes capturadas por minuto, garantizando etiquetas precisas para la clasificación. Este conjunto de datos resulta muy importante y útil para la investigación basada en la detección ambiental y el diseño de sistemas de edificios inteligentes según sea el interés del usuario.")
        st.write("La base cuenta con un total de 17.895 datos con un total de 8 variables, sin embargo, se utilizará una cantidad reducida de variables debido a que aquellas como “ID” y “Fecha” no aportan información relevante para la aplicación de los temas anteriormente tratados.")
        st.write("El conjunto de datos fue obtenido del repositorio público Kaggle, ampliamente utilizado en investigaciones relacionadas con sistemas inteligentes y monitoreo ambiental. La fuente original corresponde al trabajo disponible en el siguiente enlace: https://www.kaggle.com/datasets/pooriamst/occupancy-detection.")

elif seccion == "Análisis Descriptivo":
    st.subheader("Resumen de los datos")
    st.write(df.describe())
    st.subheader("Histograma de Temperature")
    # Temperatura
    # Crear el histograma
    plt.figure(figsize=(8, 6))
    plt.hist(x="Temperature", bins=30, color='blue', edgecolor='black', alpha=0.7)
    # Etiquetas y título
    plt.xlabel("Temperatura")
    plt.ylabel('Frecuencia')
    plt.title('Histograma de Temperature')
    plt.show()










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

elif seccion == "Entrenamiento del Modelo MLP":
    st.subheader("Entrenamiento del Modelo MLP")
    if st.button("Entrenar Modelo"):
        model = train_mlp()
        st.success("Modelo entrenado con éxito")
        st.session_state["mlp_model"] = model

elif seccion == "Hacer una Predicción":
    st.subheader("Hacer una Predicción")
    def user_input():
        features = {}
        for col in df.drop(columns=["Occupancy"], errors='ignore').columns:
            features[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        return pd.DataFrame([features])
    
    if "mlp_model" in st.session_state:
        input_data = user_input()
        input_scaled = scaler.transform(input_data)
        prediction = st.session_state["mlp_model"].predict(input_scaled)
        occupancy = "Ocupado" if prediction[0][0] > 0.5 else "No Ocupado"
        st.write(f"Predicción: {occupancy}")

st.sidebar.write("Este es un análisis inicial, se pueden agregar modelos predictivos y más visualizaciones interactivas.")







