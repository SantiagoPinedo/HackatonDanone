import json
import numpy as np
from keras.models import load_model

# Cargar el modelo entrenado
model = load_model('trained_model.h5')

# Leer los datos preprocesados de prueba desde el archivo JSON
with open('preprocessed_test_data.json', 'r') as file:
    test_data = np.array(json.load(file))

# Convertir los datos de prueba en un arreglo numpy
X_test = np.array(test_data)

# Agregar una dimensión adicional a los datos de prueba
X_test = np.expand_dims(X_test, axis=1)

# Realizar predicciones en los datos de prueba
predictions = model.predict(X_test)

# Escalar las predicciones al rango 0-4
predictions = predictions * 4

# Obtener el número de instancias en los datos de prueba
num_instances = test_data.shape[0]

# Crear una lista para almacenar las predicciones con características y target
output = {}

# Recorrer las predicciones y las características correspondientes
n = 0

for i in range(num_instances):
    instance = {
        str(n): int(predictions[i].item())
    }
    n = n + 1
    output.update(instance)

# Guardar las predicciones en un archivo JSON
with open('predictions.json', 'w') as file:
    json.dump({"target": output}, file)

print("Predicciones guardadas en el archivo 'predictions.json'.")
