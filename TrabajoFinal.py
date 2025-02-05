
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.image("image1.jpg", use_container_width=True)
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
    st.write(df.info())
    st.write("La base de datos seleccionada para el desarrollo de la aplicación corresponde a un estudio diseñado para optimizar actividades de clasificación binaria para determinar sí una habitación está ocupada o no. Dentro de sus características, se recopilan mediciones ambientales tales como la temperatura, la humedad del ambiente, la luz o nivel de luminosidad, y niveles de CO2, donde, con base a estas se determina sí la habitación está ocupada. La información de ocupación se obtuvo mediante la obtención de imágenes capturadas por minuto, garantizando etiquetas precisas para la clasificación. Este conjunto de datos resulta muy importante y útil para la investigación basada en la detección ambiental y el diseño de sistemas de edificios inteligentes según sea el interés del usuario.")
    st.write("La base cuenta con un total de 17.895 datos con un total de 8 variables, sin embargo, se utilizará una cantidad reducida de variables debido a que aquellas como “ID” y “Fecha” no aportan información relevante para la aplicación de los temas anteriormente tratados.")
    st.write("El conjunto de datos fue obtenido del repositorio público Kaggle, ampliamente utilizado en investigaciones relacionadas con sistemas inteligentes y monitoreo ambiental. La fuente original corresponde al trabajo disponible en el siguiente enlace: https://www.kaggle.com/datasets/pooriamst/occupancy-detection.")

elif seccion == "Análisis Descriptivo":
    st.subheader("Resumen de los datos")
    st.write(df.describe())
    st.subheader("Histograma de Temperature")
    # Temperatura
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["Temperature"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de Temperature')
    st.pyplot(fig)
    st.write("Del histograma anterior, se denota que la mayoría de imágenes tomadas de la habitación captaron una temperatura de entre 20°C y 21°C, siendo una temperatura ambiente la que más predomina en el conjunto de datos. Además, se observa que la temperatura mínima registrada es de 19°C y la máxima es un poco superior a 24°C. Por tanto, en la habitación no hay presencia de temperaturas que se consideren bajas o altas.")
    #  Humidity
    st.subheader("Histograma de Humidity")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["Humidity"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Humidity')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de Humidity')
    st.pyplot(fig)
    st.write("De la variable “Humidity”, se observa que la humedad se encuentra entre aproximadamente un 16% y un 40%. Para su interpretación en este caso, se debe conocer cuáles son los valores de humedad normales en una habitación, para ello, la empresa Philips (sin fecha) en su publicación “¿Cómo medir la humedad recomendada en casa?” afirma que la humedad ideal debe encontrarse entre 30% y 60% para la conservación de los materiales de las paredes y el piso; por otra parte, en el blog Siber. (n.d.) mencionan que el ser humano puede estar en espacios con una humedad de 20% a 75%. Teniendo en cuenta lo anterior, se puede afirmar que la humedad en la mayoría de los datos es adecuada para las personas, para los casos cuyo valor de humedad es menor a 20% no resulta ideal pero no debería ser un inconveniente significativo.")
    # HumidityRatio
    st.subheader("Histograma de HumidityRatio")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["HumidityRatio"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('HumidityRatio')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de HumidityRatio')
    st.pyplot(fig)
    st.write("Este histograma corresponde a la cantidad derivada de la temperatura y la humedad relativa dada en kilogramo de vapor de agua por kilogramo de aire, los valores se encuentran entre 0.002 kg vapor de agua/kg de aire hasta 0.0065 kg vapor de agua/ kg de aire aproximadamente. Según la explicación de la variable anterior, los resultados de la relación se encuentran en un rango adecuado.")
    # Light
    st.subheader("Histograma de Light")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["Light"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Light')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de Light')
    st.pyplot(fig)
    st.write("De la variable Light, se observa que en la gran mayoría de los datos no hubo presencia de luz, no obstante, se denota el incremento en los valores cercanos a 500lux, esto indica que en estos casos sí se hizo uso de la luz eléctrica en la habitación debido al flujo luminoso provocado por el bombillo. Este podría ser un factor importante en la determinación de sí la habitación está ocupada o no, pero esto se confirmará más adelante en los resultados.")
    # CO2
    st.subheader("Histograma de CO2")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(x=df["CO2"], bins=30, color='blue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('CO2')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de CO2')
    st.pyplot(fig)
    st.write("Para la variable de CO2, se observa que los niveles de CO2 dados en ppm (partículas por millón) de aproximadamente 400 a 700pm son los más presentes en el conjunto de datos. Se registran más casos donde los niveles de CO2 son mucho mayores a los recurrentes, llegando hasta los 2000ppm. Para comprender la tolerancia de una persona hacia el CO2, la empresa Enectiva (2017) en su publicación “Efectos de la concentración de CO₂ para la salud humana” expone que las habitaciones deben tener niveles de CO2 máximo recomendado en 1200-1500ppm, a partir de este valor pueden presentarse efectos secundarios sobre las personas, como la fatiga y la pérdida de concentración; a niveles mayores a los presentes en el histograma puede provocar aumento del ritmo cardíaco, dificultades respiratorias, náuseas, e inclusive la pérdida de la consciencia. Los niveles de CO2 pueden ser un indicativo clave para determinar sí la habitación está ocupada o no debido a la naturaleza del ser humano de expulsar dióxido de carbono “CO2” en su exhalación, aunque debe tenerse en cuenta que un nivel elevado de CO2 puede deberse a razones diferentes del proceso de respiración de la persona.")
    
elif seccion == "Distribución de la variable objetivo":
    st.subheader("Distribución de la variable objetivo")
    fig, ax = plt.subplots()
    sns.countplot(x=df["Occupancy"], ax=ax)
    st.pyplot(fig)
    st.write("De la variable respuesta “Occupancy”, se obtiene que en su mayoría de casos se tiene como resultado que la habitación no se encuentra ocupada, denotada con el valor de cero y por el valor 1 en el caso contrario. Se obtuvo que en el 78.9% de los casos la habitación está vacía, y en el 21.1% se encuentra ocupada.")

elif seccion == "Mapa de calor de correlaciones":
    st.subheader("Mapa de calor de correlaciones")
    st.write("Se plantea la matriz de correlación de las variables mencionadas para verificar qué tan relacionadas se encuentran con la variable respuesta de “Occupancy” y así observar cuáles tendrían mayor incidencia en la toma de decisión:")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), vmin=-1, vmax=1, cmap="coolwarm", annot=True, ax=ax)
    st.pyplot(fig)
    st.write("Según la matriz, la variable que más se correlaciona con la variable respuesta es la luz (“Light”), pues es una determinante importante en la ocupación de una habitación; seguido de ésta, se denotan las variables de temperatura y CO2, cuyas características se encuentran estrechamente relacionadas con la presencia de personas en un espacio. Por último, debe mencionarse que las variables relacionadas con la humedad presentan una muy baja correlación con la ocupación de una habitación, esto debe tenerse en cuenta en la formulación del modelo para la aplicación y considerar sí se eliminan estas variables dependiendo de los resultados que se obtengan.")

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







