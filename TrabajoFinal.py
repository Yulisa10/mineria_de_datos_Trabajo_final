
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')

# Cargar los datos
@st.cache
def load_data():
    df_train = pd.read_csv('/kaggle/input/occupancy-detection/datatrain.csv')
    df_test = pd.read_csv('/kaggle/input/occupancy-detection/datatest.csv')
    df = pd.concat([df_train, df_test], axis=0)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df = df.drop(columns=['id'])
    return df

df = load_data()

st.title('Análisis de Ocupación de Habitaciones')

# Mostrar los datos
if st.checkbox('Mostrar datos'):
    st.write(df)

# Seleccionar una columna para analizar
column = st.selectbox('Selecciona una columna para analizar', df.columns)

# Mostrar un histograma de la columna seleccionada
st.write(f'Histograma de {column}')
fig, ax = plt.subplots()
ax.hist(df[column].dropna(), bins=20)
st.pyplot(fig)

# Mostrar la matriz de correlación
if st.checkbox('Mostrar matriz de correlación'):
    st.write('Matriz de correlación')
    corr = df.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, ax=ax)
    st.pyplot(fig)

# Gráfico de dispersión
if st.checkbox('Mostrar gráfico de dispersión'):
    x_axis = st.selectbox('Selecciona el eje X', df.columns)
    y_axis = st.selectbox('Selecciona el eje Y', df.columns)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='Occupancy', ax=ax)
    st.pyplot(fig)
st.title('Modelos de Machine Learning')

# Preparar los datos para el modelo
X = df.drop(['date', 'Occupancy'], axis=1)
y = df['Occupancy']

# Escalar los datos
sc = MinMaxScaler()
X_scaled = sc.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Seleccionar el modelo
model_option = st.selectbox('Selecciona un modelo', ['XGBoost', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors'])

if model_option == 'XGBoost':
    model = XGBClassifier(enable_categorical=True)
elif model_option == 'Random Forest':
    model = RandomForestClassifier(n_estimators=300, random_state=42)
elif model_option == 'Decision Tree':
    model = DecisionTreeClassifier(max_depth=None, random_state=0)
elif model_option == 'K-Nearest Neighbors':
    n_neighbors = st.slider('Número de vecinos', 1, 15, 7)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# Entrenar el modelo
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Mostrar métricas
st.write('Accuracy:', accuracy_score(y_test, y_pred))
st.write('F1 score:', f1_score(y_test, y_pred))
st.write('Recall:', recall_score(y_test, y_pred))
st.write('Precision:', precision_score(y_test, y_pred))

# Mostrar importancia de las características
if hasattr(model, 'feature_importances_'):
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=True)
    fig, ax = plt.subplots()
    feat_importances.plot(kind='barh', ax=ax)
    plt.title('Importancia de las características')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
