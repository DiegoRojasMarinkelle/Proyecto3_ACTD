# Importar librerías
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import mlflow
import mlflow.sklearn

# Leer datos
X_over_train = pd.read_csv("X_over_train.csv")
X_over_val = pd.read_csv("X_over_val.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_over_train.csv")
y_val = pd.read_csv("y_over_val.csv")
y_test = pd.read_csv("y_test.csv")

# Codificar variables objetivo
y_over_train_encoded = to_categorical(y_train)
y_over_val_encoded = to_categorical(y_val)
y_test_encoded = to_categorical(y_test)

# Fijar la semilla para reproducibilidad
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Definición del modelo
# Definición de la función para crear el modelo
def crear_modelo(depth=50, p=0.5, alpha=0.001, input_shape=None):
    # Capa de entrada
    input_layer = Input(shape=(input_shape,))

    # Capas densas
    dense1 = Dense(depth*4, activation='relu')(input_layer)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(p)(dense1)

    dense2 = Dense(depth*4, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(p)(dense2)

    dense3 = Dense(depth*3, activation='relu')(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(p)(dense3)

    dense4 = Dense(depth*2, activation='relu')(dense3)
    dense4 = BatchNormalization()(dense4)
    dense4 = Dropout(p)(dense4)

    # Capa de salida
    prediction = Dense(3, activation='softmax')(dense4)

    # Definición del modelo
    optimizer = Adam(learning_rate=alpha)
    model = Model(inputs=input_layer, outputs=prediction)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Definir el servidor para el registro de modelos y artefactos MLflow
mlflow.set_tracking_uri('http://18.212.105.57:8050')
experiment = mlflow.set_experiment("sklearn-diab")

# Iniciar ejecución del experimento
with mlflow.start_run(experiment_id=experiment.experiment_id):
    Depth = 50
    P = 0.5
    Alpha = 0.003
    Epocas = 10

    modelo = crear_modelo(input_shape=X_over_train.shape[1], depth=Depth, p=P, alpha=Alpha)
    history_base = modelo.fit(X_over_train, y_over_train_encoded, validation_data=(X_over_val, y_over_val_encoded), batch_size=16, epochs=Epocas)
    
    # Obtener métricas de entrenamiento
    history_dict = history_base.history
    accuracy = history_dict['accuracy']
    val_accuracy = history_dict['val_accuracy']
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']

    # Evaluar modelo en el conjunto de test
    test = modelo.evaluate(X_test, y_test_encoded)
    
    # Registrar parámetros y métricas en MLflow
    mlflow.log_param("Neuronas", Depth)
    mlflow.log_param("Dropout", P)
    mlflow.log_param("Learning rate", Alpha)
    mlflow.log_param("Accuracy", accuracy)
    mlflow.log_param("Val_Accuracy", val_accuracy)
    mlflow.log_param("Loss", loss_values)
    mlflow.log_param("Val_Loss", val_loss_values)
    mlflow.log_param("Epocas", Epocas)
    
    mlflow.sklearn.log_model(modelo, "red-neuronal")
    
    mlflow.log_metric("Loss_test", test[0])
    mlflow.log_metric("Accuracy_test", test[1])
    print(test[0])
    print(test[1])
