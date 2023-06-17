# El grado de eco-Puntuacion va desde 0 hasta 4, donde 0 es un bajo impacto ecologico
# y 4 es un alto impacto ecologico, se debe hacer un modelo que clasifique correctamente cada producto
# de Danone en una de estas 4 categorias.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
# To Do 
    #Preprocesamiento de los datos.
#CArgar los datos de entrenamiento y prueba
train_data = pd.read_json('train_products.json')
test_data = pd.read_json('test_products.json')

#Mostrar la estructura de los datos
print(train_data.head())

# Manejar valores faltantes o inconsistentes (ejemplo: rellenar NaN con 0)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)


#Extraccion de caracteristicas relevantes
#Entrenamiento de la red neuronal 
#Probar el model con datos de test.