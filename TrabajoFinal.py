
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.title("Análisis de Detección de Ocupación con MLP")

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
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# Preprocesamiento
def preprocess_data(df):
    X = df.drop(columns=["date", "Occupancy"])
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

elif seccion == "Análisis Descriptivo":
    st.subheader("Análisis Descriptivo")
    st.write(df.describe())

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
    st.sidebar.subheader("Entrenamiento del Modelo MLP")
    if st.sidebar.button("Entrenar Modelo"):
        model = train_mlp()
        st.sidebar.success("Modelo entrenado con éxito")
        st.session_state["mlp_model"] = model

elif seccion == "Hacer una Predicción":
    st.sidebar.subheader("Hacer una Predicción")
    def user_input():
        features = {}
        for col in df.drop(columns=["date", "Occupancy"]).columns:
            features[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        return pd.DataFrame([features])
    
    if "mlp_model" in st.session_state:
        input_data = user_input()
        input_scaled = scaler.transform(input_data)
        prediction = st.session_state["mlp_model"].predict(input_scaled)
        occupancy = "Ocupado" if prediction[0][0] > 0.5 else "No Ocupado"
        st.sidebar.write(f"Predicción: {occupancy}")

st.sidebar.write("Este es un análisis inicial, se pueden agregar modelos predictivos y más visualizaciones interactivas.")



