import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
import os

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

# Definimos el modelo para Keras Tuner
def model_builder(hp):
    # Define el modelo con parámetros ajustables
    model = tf.keras.Sequential()
    
    # Capas convolucionales
    for i in range(hp.Int('conv_blocks', 1, 3, default=2)):
        filters = hp.Int(f'filters_{i}', 32, 128, step=32, default=64)
        model.add(tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3) if i == 0 else None))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(tf.keras.layers.Flatten())
    
    # Regularización con dropout
    dropout_rate = hp.Float('dropout', 0, 0.5, step=0.1, default=0.2)
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Capas densas
    dense_units = hp.Int('dense_units', 64, 512, step=64, default=128)
    model.add(tf.keras.layers.Dense(dense_units, activation='relu'))
    
    # Capa de salida
    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    
    # Compilación del modelo
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Configuramos el directorio para los logs
log_dir = os.path.join('keras_tuner_experiment', 'logs', 'search')
os.makedirs(log_dir, exist_ok=True)

# Creamos el buscador de hiperparámetros
tuner = kt.RandomSearch(
    model_builder,
    objective='val_accuracy',
    max_trials=5,
    directory='keras_tuner_experiment',
    project_name='tuner_results',
    overwrite=True
)

print(tuner.search_space_summary())

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

# Buscamos los mejores hiperparámetros
steps_per_epoch = tf.math.ceil(ds_info.splits['train'].num_examples * 0.8 / batch_size).numpy()
validation_steps = tf.math.ceil(ds_info.splits['train'].num_examples * 0.2 / batch_size).numpy()

tuner.search(
    train_ds,
    epochs=15,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=[early_stop, tensorboard_callback]
)

# Obtenemos el mejor modelo
best_hps = tuner.get_best_hyperparameters(1)[0]
print(f"\nMejores hiperparámetros encontrados: {best_hps.values}")

# Construimos el modelo final con los mejores hiperparámetros
model = tuner.hypermodel.build(best_hps)

# Entrenamos el modelo final
epochs = 30
history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback]
)

# Evaluamos el modelo
evaluation = model.evaluate(test_ds, steps=validation_steps)
print(f"\nEvaluación del modelo final: {evaluation}")

# Guardamos el modelo final
model.save('keras_tuner_experiment/best_model.h5')
print("Modelo guardado correctamente.")