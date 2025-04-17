import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

os.makedirs('modelodensoAD', exist_ok=True)
os.makedirs('modelocnnAD', exist_ok=True)
os.makedirs('modelocnn2AD', exist_ok=True)

# Cargamos el dataset de flores
(ds_train, ds_test), ds_info = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True,
)

class_names = ds_info.features['label'].names
n_classes = len(class_names)
print(f"Clases: {class_names}")
print(f"Número de clases: {n_classes}")

# Preprocesamiento de los datos
def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image, label

def prepare_for_training(ds, batch_size=32, cache=True, shuffle_buffer_size=1000):
    # Resize and rescale all datasets
    ds = ds.map(resize_and_rescale)
    
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
            
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    
    # Repeat forever
    ds = ds.repeat()
    
    ds = ds.batch(batch_size)
    
    # `prefetch` lets the dataset fetch batches in the background while the model is training
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return ds

batch_size = 32
train_ds = prepare_for_training(ds_train, batch_size=batch_size)
test_ds = prepare_for_training(ds_test, batch_size=batch_size)

# Aumento de datos
data_augmentation = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Definimos el modelo denso
modelo_denso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

# Definimos el modelo CNN
modelo_cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

# Definimos el modelo CNN con dropout
modelo_cnn2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),  # Añadimos dropout
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

# Configuramos los modelos
modelo_denso.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo_cnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo_cnn2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Definimos callbacks para TensorBoard
tensorboard_callback_denso = TensorBoard(
    log_dir='logs/densoAD',
    histogram_freq=1
)

tensorboard_callback_cnn = TensorBoard(
    log_dir='logs/cnnAD',
    histogram_freq=1
)

tensorboard_callback_cnn2 = TensorBoard(
    log_dir='logs/cnn2AD',
    histogram_freq=1
)

# Entrenamos los modelos
epochs = 25
steps_per_epoch = tf.math.ceil(ds_info.splits['train'].num_examples * 0.8 / batch_size).numpy()
validation_steps = tf.math.ceil(ds_info.splits['train'].num_examples * 0.2 / batch_size).numpy()

print("Entrenando modelo denso...")
history_denso = modelo_denso.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback_denso]
)

print("Entrenando modelo CNN...")
history_cnn = modelo_cnn.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback_cnn]
)

print("Entrenando modelo CNN con dropout...")
history_cnn2 = modelo_cnn2.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback_cnn2]
)

# Guardamos los modelos
modelo_denso.save('modelodensoAD/modelodensoAD.h5')
modelo_cnn.save('modelocnnAD/modelocnnAD.h5')
modelo_cnn2.save('modelocnn2AD/modelocnn2AD.h5')

print("Modelos guardados correctamente.")