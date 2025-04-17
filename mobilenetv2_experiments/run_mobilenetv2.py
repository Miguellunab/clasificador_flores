import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import datetime
import os

# Creamos directorios para guardar resultados
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_dir = os.path.join('mobilenetv2_experiments', f'mobilenetv2_{date_time}')
os.makedirs(os.path.join(experiment_dir, 'logs', 'train'), exist_ok=True)
os.makedirs(os.path.join(experiment_dir, 'logs', 'validation'), exist_ok=True)
os.makedirs(os.path.join(experiment_dir, 'model'), exist_ok=True)

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

# Cargamos el modelo base MobileNetV2 pre-entrenado
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Congelamos las capas del modelo base
base_model.trainable = False

# Creamos el modelo completo con una capa densa para clasificación
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3), name='input_layer_1'),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

# Compilamos el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Mostramos el resumen del modelo
model.summary()

# Callbacks para TensorBoard
train_log_dir = os.path.join(experiment_dir, 'logs', 'train')
val_log_dir = os.path.join(experiment_dir, 'logs', 'validation')
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=train_log_dir,
    histogram_freq=1
)

# Callback para guardar el mejor modelo
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(experiment_dir, 'model', 'mobilenetv2_flowers.h5'),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# Entrenamos el modelo
epochs = 30
steps_per_epoch = tf.math.ceil(ds_info.splits['train'].num_examples * 0.8 / batch_size).numpy()
validation_steps = tf.math.ceil(ds_info.splits['train'].num_examples * 0.2 / batch_size).numpy()

history = model.fit(
    train_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=test_ds,
    validation_steps=validation_steps,
    callbacks=[tensorboard_callback, checkpoint_callback]
)

# Evaluamos el modelo final
print("\nEvaluando el modelo final...")
evaluation = model.evaluate(test_ds, steps=validation_steps)
print(f"Pérdida: {evaluation[0]}")
print(f"Precisión: {evaluation[1]}")

# Guardamos el modelo final (por si acaso el mejor no se guardó)
model.save(os.path.join(experiment_dir, 'model', 'mobilenetv2_flowers_final.h5'))
print("\nModelo guardado correctamente.")

# Visualizamos la precisión y pérdida durante el entrenamiento
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

plt.savefig(os.path.join(experiment_dir, 'training_history.png'))
plt.show()