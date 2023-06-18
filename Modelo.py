# El grado de eco-Puntuacion va desde 0 hasta 4, donde 0 es un bajo impacto ecologico
# y 4 es un alto impacto ecologico, se debe hacer un modelo que clasifique correctamente cada producto
# de Danone en una de estas 4 categorias.
import json
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical

# Leer los datos preprocesados desde el archivo JSON
with open('preprocessed_train_data.json', 'r') as file:
    data = json.load(file)

# Convertir los datos en un arreglo numpy
X_train = np.array(data)

# Agregar una dimensión adicional a los datos de entrenamiento
X_train = np.expand_dims(X_train, axis=1)

# Obtener el número de timesteps (longitud de la secuencia) a partir de los datos
timesteps = X_train.shape[1]

# Leer las etiquetas de entrenamiento desde el archivo JSON
with open('etiquetas_entrenamiento.json', 'r') as file:
    labels = json.load(file)

# Convertir las etiquetas en un arreglo numpy
y_train = np.array(labels)

# Definir el modelo
model = Sequential()
model.add(LSTM(64, input_shape=(timesteps,15)))  # Capa LSTM en lugar de Dense
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=2000, batch_size=64)

# Guardar el modelo entrenado
model.save('trained_model.h5')