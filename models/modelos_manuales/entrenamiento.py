import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
import tensorflowjs as tfjs

# Crear directorios para los modelos
def crear_directorios_modelo(nombre_base):
    # Directorios para diferentes formatos del modelo
    directorios = {
        'h5': f'{nombre_base}',
        'saved_model': f'{nombre_base}_saved_model',
        'tfjs': f'{nombre_base}_tfjs'
    }
    
    for directorio in directorios.values():
        os.makedirs(directorio, exist_ok=True)
        
    return directorios

# Función para guardar modelo en múltiples formatos
def guardar_modelo(modelo, nombre_base):
    # Crear directorios
    directorios = crear_directorios_modelo(nombre_base)
    
    # 1. Guardar en formato H5 (compatibilidad)
    h5_path = os.path.join(directorios['h5'], f"{nombre_base}.h5")
    modelo.save(h5_path)
    print(f"Modelo guardado en formato H5: {h5_path}")
    
    # 2. Guardar en formato SavedModel
    saved_model_path = os.path.join(directorios['saved_model'], nombre_base)
    tf.saved_model.save(modelo, saved_model_path)
    print(f"Modelo guardado en formato SavedModel: {saved_model_path}")
    
    # 3. Convertir a TensorFlow.js
    tfjs_path = os.path.join(directorios['tfjs'], nombre_base)
    try:
        # Limpiar directorio si existe
        if os.path.exists(tfjs_path):
            shutil.rmtree(tfjs_path)
        os.makedirs(tfjs_path, exist_ok=True)
        
        # Convertir usando API de TensorFlow.js
        tfjs.converters.save_keras_model(modelo, tfjs_path)
        print(f"Modelo convertido a TensorFlow.js: {tfjs_path}")
    except Exception as e:
        print(f"Error al convertir a TensorFlow.js: {e}")
    
    return directorios

# Crear directorios base para los modelos
for modelo_dir in ['modelodensoAD', 'modelocnnAD', 'modelocnn2AD']:
    crear_directorios_modelo(modelo_dir)

# Cargamos el dataset de flores
dataset, metadata = tfds.load('tf_flowers', as_supervised=True, with_info=True)

# Configuramos la figura
# plt.figure(figsize=(10, 10))
TAMANO_IMG = 100
for i, (imagen, etiqueta) in enumerate(dataset['train'].take(25)):
    # Redimensionar la imagen
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#     plt.subplot(5, 5, i+1)
#     plt.imshow(imagen, cmap='gray')
#     plt.axis('off')  # opcional, para no mostrar ejes

# plt.tight_layout()  # ajusta el espacio entre subplots
# plt.show()

datos_entrenamiento = []

for i, (imagen, etiqueta) in enumerate(dataset['train']):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1) # Camabiamos a 100x100x1
    datos_entrenamiento.append([imagen, etiqueta])

# print(datos_entrenamiento[0]) # imagen y etiqueta

# print(metadata.features['label'].names) # nombres de las etiquetas

# print(len(datos_entrenamiento)) # cantidad de datos

X = [] # imagenes
y = [] # etiquetas (0, 1, 2, 3, 4)

for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen)
    y.append(etiqueta)

#Normalizamos las imagenes
X = np.array(X).astype('float') / 255.0
y = np.array(y)
# print(X.shape)

#1er modelo de entrenamiento

modeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax'),
])
#2do modelo de entrenamiento
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

#3er modelo de entrenamiento
modeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

#Compilamos el modelo
modeloDenso.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

modeloCNN.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

modeloCNN2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#Entrenamiento del primero modelo
# #Usar tensorboard
# tensorboardDenso = TensorBoard(log_dir='logs/denso')

# #Entrenamos el modelo
# modeloDenso.fit(X, y, batch_size=32, 
#                 epochs=100, 
#                 validation_split=0.15, 
#                 callbacks=[tensorboardDenso])

#Entrenamiento del segundo modelo
#Usar tensorboard
# tensorboardCNN = TensorBoard(log_dir='logs/cnn')

# #Entrenamos el modelo
# modeloCNN.fit(X, y, batch_size=32, 
#                 epochs=100, 
#                 validation_split=0.15, 
#                 callbacks=[tensorboardCNN])

#Entrenamiento del tercer modelo
#Usar tensorboard
# tensorboardCNN2 = TensorBoard(log_dir='logs/cnn2')

#Entrenamos el modelo
# modeloCNN2.fit(X, y, batch_size=32, 
#                 epochs=100, 
#                 validation_split=0.15, 
#                 callbacks=[tensorboardCNN2])

#Aumento de datos usando ImageDataGenerator con comentarios sobre cada parámetro
datagen = ImageDataGenerator(
    rotation_range=30,            # Se mantiene la rotación en ±30°
    width_shift_range=0.08,       # Se reduce el desplazamiento horizontal de 0.1 a 0.08
    height_shift_range=0.08,      # Se reduce el desplazamiento vertical de 0.1 a 0.08
    shear_range=0.15,             # Se disminuye el shear de 0.2 a 0.15
    zoom_range=0.2,               # Se mantiene el zoom
    horizontal_flip=True,         # Se mantiene el volteo horizontal
    fill_mode='nearest',          # Se mantiene el método de relleno
    validation_split=0.15         # Se reserva el 15% para validación
)

datagen.fit(X)

# Crea iteradores para entrenamiento y validación
train_generator = datagen.flow(X, y, batch_size=32, subset='training')
val_generator   = datagen.flow(X, y, batch_size=32, subset='validation')

#Mostrar imagenes generadas
# plt.figure(figsize=(10, 10))

# for imagen, etiqueta in datagen.flow(X, y, batch_size=32, shuffle = True):
#     for i in range(10):
#         plt.subplot(2, 5, i+1)
#         plt.imshow(imagen[i].reshape(TAMANO_IMG, TAMANO_IMG), cmap='gray')
#         plt.axis('off')
#     plt.imshow(imagen[1].reshape(TAMANO_IMG, TAMANO_IMG), cmap='gray')
#     break
# plt.tight_layout()
# plt.show()

#Generacion de diferentes modelos

modeloDensoAD = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax'),
])

modeloCNNAD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

modeloCNN2AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(TAMANO_IMG, TAMANO_IMG, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

#Compilamos el modelo
modeloDensoAD.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

modeloCNNAD.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

modeloCNN2AD.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

#Entrenamiento del primero modelo (DensoAD)
# Usar tensorboard (comentado)
# tensorboardDensoAD = TensorBoard(log_dir='logs/densoAD')
#
# Entrenamos el modelo
# modeloDensoAD.fit(
#     train_generator,
#     epochs=100,
#     validation_data=val_generator,
#     callbacks=[tensorboardDensoAD]
# )
# Guardar el modelo en múltiples formatos
# guardar_modelo(modeloDensoAD, 'modelodensoAD')

#Entrenamiento del segundo modelo (CNNAD)
# Usar tensorboard
tensorboardCNNAD = TensorBoard(log_dir='logs/cnnAD')

#Entrenamos el modelo
modeloCNNAD.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[tensorboardCNNAD]
)
#Guardar el modelo en múltiples formatos
guardar_modelo(modeloCNNAD, 'modelocnnAD')

#Entrenamiento del tercer modelo (CNN2AD)
# Usar tensorboard
# tensorboardCNN2AD = TensorBoard(log_dir='logs/cnn2AD')

# # Entrenamos el modelo
# modeloCNN2AD.fit(
#     train_generator,
#     epochs=100,
#     validation_data=val_generator,
#     callbacks=[tensorboardCNN2AD]
# )
# # Guardar el modelo en múltiples formatos
# guardar_modelo(modeloCNN2AD, 'modelocnn2AD')