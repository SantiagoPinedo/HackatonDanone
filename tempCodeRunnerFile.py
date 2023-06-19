# Definir el modelo
model = Sequential()
model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu", input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(Flatten())
model.add(Dense(128, activation="tanh"))
model.add(Dense(5, activation="sigmoid"))