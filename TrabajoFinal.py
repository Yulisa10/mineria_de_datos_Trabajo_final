import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# Mostrar la imagen solo en la página de inicio
st.title("Análisis de Detección de Ocupación")
st.write("Grupo: Yulisa Ortiz Giraldo y Juan Pablo Noreña Londoño")
if "image_displayed" not in st.session_state:
    st.image("image1.jpg", use_container_width=True)
    st.session_state["image_displayed"] = True  # Marcar que la imagen ya se mostró

# Crear una tabla de contenido en la barra lateral
seccion = st.sidebar.radio("Tabla de Contenidos", 
                           ["Vista previa de los datos", 
                            "Información del dataset", 
                            "Análisis Descriptivo", 
                            "Mapa de calor de correlaciones", 
                            "Distribución de la variable objetivo", 
                            "Boxplots", 
                            "Conclusión: Selección del Mejor Modelo",  # Nueva ubicación
                            "Modelo XGBoost",  # Nueva sección
                            "Entrenamiento del Modelo MLP", 
                            "Hacer una Predicción",
                           "Modelo de redes neuronales"])

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

elif seccion == "Boxplots":
    st.subheader("Conjunto de boxplots")
    st.image("Boxplots.jpg", use_container_width=True)
    st.write("""
    ### Análisis de Variables

    #### CO2 (Dióxido de carbono):
    - **Habitación vacía (rojo):** Niveles considerablemente más bajos, con una mediana en torno a 500ppm.
    - **Habitación ocupada (verde):** Niveles mucho más altos, con una mediana cerca de 1000ppm.
    - El nivel de CO2 aumenta notablemente con la ocupación, posiblemente debido a la respiración de las personas.

    #### Humidity (Humedad):
    - **Habitación vacía (rojo):** Mediana ligeramente por encima de 25, con dispersión moderada.
    - **Habitación ocupada (verde):** Mediana cerca de 30, con valores más altos.
    - La ocupación no parece variar mucho la humedad, en línea con la matriz de correlaciones.

    #### HumidityRatio (Proporción de humedad):
    - **Habitación vacía (rojo):** Valores concentrados alrededor de 0.0035.
    - **Habitación ocupada (verde):** Valores ligeramente más altos, alrededor de 0.004.
    - Aunque las diferencias no son grandes, la ocupación está asociada con un pequeño incremento en la proporción de humedad.

    #### Light (Luz):
    - **Habitación vacía (rojo):** Gran dispersión, con valores extremos muy altos.
    - **Habitación ocupada (verde):** Valores más bajos y concentrados.
    - La ocupación 0 (habitación vacía) está asociada con niveles de luz más altos y variables, posiblemente por la ausencia de personas que reduzcan el uso de iluminación artificial.

    #### Temperature (Temperatura):
    - **Habitación vacía (rojo):** Mediana cerca de 20°C, con dispersión moderada.
    - **Habitación ocupada (verde):** Mediana ligeramente más alta, alrededor de 22°C.
    - La temperatura es más alta con ocupación, posiblemente por el calor generado por las personas o el uso de calefacción.

     #### Conclusión:
    La ocupación tiene un impacto claro en **CO2, Light (luz) y Temperature (temperatura)**, aumentando sus valores en comparación con la falta de ocupación. En particular, la luz tiende a ser más alta y variable cuando hay ocupación. Otras variables como la humedad presentan cambios menores, pero no son significativos.
    """)

# Nueva sección: Conclusión sobre la selección del mejor modelo
elif seccion == "Conclusión: Selección del Mejor Modelo":
    st.subheader("Conclusión: Selección del Mejor Modelo (XGBoost)")
    st.markdown("""
    Después de evaluar varios modelos de machine learning para la tarea de predecir la ocupación de habitaciones, se determinó que el **XGBoost Classifier** es el modelo más adecuado para este problema. A continuación, se detallan las razones por las que se seleccionó este modelo y por qué los otros no fueron la mejor opción:

    #### Razones para elegir XGBoost:
    1. **Alto Rendimiento en Precisión y F1-Score**:
       - XGBoost demostró un rendimiento superior en términos de precisión y F1-Score, lo que indica que es capaz de predecir correctamente tanto las habitaciones ocupadas como las desocupadas. Esto es especialmente importante en problemas de clasificación donde el equilibrio entre precisión y recall es crucial.

    2. **Manejo de Desequilibrio de Clases**:
       - En problemas donde las clases están desequilibradas (por ejemplo, más datos de habitaciones desocupadas que ocupadas), XGBoost es conocido por su capacidad para manejar este desequilibrio de manera efectiva, lo que lo hace más robusto y confiable.

    3. **Interpretabilidad de las Características**:
       - XGBoost proporciona una clara interpretación de la importancia de las características, lo que permite identificar qué variables (como el nivel de CO2, la luz o la humedad) son más relevantes para la predicción. Esto es invaluable para entender el problema y tomar decisiones basadas en datos.

    4. **Eficiencia y Escalabilidad**:
       - XGBoost es un modelo altamente eficiente y escalable, lo que lo hace adecuado para conjuntos de datos más grandes y complejos. Aunque en este caso el conjunto de datos no es extremadamente grande, su eficiencia asegura un entrenamiento rápido y un rendimiento óptimo.

    5. **Robustez ante Overfitting**:
       - Gracias a sus técnicas de regularización, XGBoost es menos propenso al sobreajuste (overfitting) en comparación con otros modelos, lo que garantiza que el modelo generalice bien a nuevos datos.

    #### Razones por las que otros modelos no fueron seleccionados:
    - **Random Forest**: Aunque es un modelo potente, tiende a ser más lento y menos eficiente en términos de memoria en comparación con XGBoost. Además, XGBoost suele superar a Random Forest en términos de precisión y F1-Score en muchos casos.
    
    - **Decision Tree**: Es un modelo más simple y propenso al overfitting, especialmente en conjuntos de datos más complejos. No tiene la capacidad de regularización que tiene XGBoost, lo que lo hace menos confiable para generalizar.

    - **K-Nearest Neighbors (KNN)**: Aunque es un modelo intuitivo, KNN es computacionalmente costoso y no maneja bien el desequilibrio de clases. Además, no proporciona una interpretación clara de la importancia de las características, lo que limita su utilidad en este contexto.

    - **Red Neuronal**: Aunque las redes neuronales pueden ser muy poderosas, requieren una gran cantidad de datos y ajustes hiperparamétricos para alcanzar su máximo potencial. En este caso, el modelo secuencial utilizado es relativamente simple y no supera a XGBoost en términos de precisión o F1-Score.

    ### Conclusión Final:
    El **XGBoost Classifier** fue seleccionado como el mejor modelo debido a su alto rendimiento, capacidad para manejar el desequilibrio de clases, interpretabilidad de las características, eficiencia y robustez ante el overfitting. Estos factores lo convierten en la opción más adecuada para la tarea de predecir la ocupación de habitaciones, superando a otros modelos como Random Forest, Decision Tree, KNN y la red neuronal en este contexto específico.
    """)


elif seccion == "Modelo XGBoost":
    st.subheader("Modelo planteado con XGBoost")
    def load_model():
        filename = 'xgb_model.pkl.gz'
        with gzip.open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
     model = pickle.load(f)
    # Predicciones y evaluación del modelo
    
    accuracy = accuracy_score(model)
    f1 = f1_score(model)
    recall = recall_score(model)
    precision = precision_score(model)

    # Mostrar métricas de evaluación
    st.write(f'**Accuracy del modelo en datos de prueba:** {round(accuracy * 100, 2)}%')
    st.write(f'**F1-Score del modelo:** {round(f1, 2)}')
    st.write(f'**Recall del modelo:** {round(recall, 2)}')
    st.write(f'**Precision del modelo:** {round(precision, 2)}')

    # Gráfico de importancia de características
    st.subheader("Importancia de las características")
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_importances.plot(kind='barh', ax=ax)
    ax.set_title('Importancia de las características')
    ax.set_xlabel('Importancia')
    ax.set_ylabel('Características')
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

elif seccion == "Modelo de redes neuronales":
    st.subheader("Modelo planteado con redes neuronales")

    def load_model():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""
    filename = 'best_model.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model2 = pickle.load(f)
    return model
    model2 = pickle.load(f)
    model2.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    
    # Obtención del historial de entrenamiento
    accuracy = clf.history['accuracy']
    loss = clf.history['loss']

    # Gráficos de Accuracy y Loss
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.lineplot(y=accuracy, x=range(1, len(accuracy) + 1), marker='o', ax=axes[0])
    sns.lineplot(y=loss, x=range(1, len(loss) + 1), marker='o', ax=axes[1])
    axes[0].set_title('Accuracy')
    axes[1].set_title('Loss')

    # Mostrar gráficos en Streamlit
    st.pyplot(fig)

    # Evaluación del modelo
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    st.write(f'**Accuracy del modelo en datos de prueba:** {round(test_accuracy * 100, 2)}%')



