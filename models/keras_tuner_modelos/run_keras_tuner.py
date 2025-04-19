import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
from sklearn.model_selection import train_test_split # Importar train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
import os
import datetime

# --- Carga y Preprocesamiento de Datos ---
TAMANO_IMG = 100

# Cargar dataset
dataset, metadata = tfds.load('tf_flowers', as_supervised=True, with_info=True)
num_classes = metadata.features['label'].num_classes # Obtener número de clases
print(f"Número de clases detectado: {num_classes}")

# Preparar datos (Mantener Color)
datos_entrenamiento = []
for i, (imagen, etiqueta) in enumerate(dataset['train']):
    # Redimensionar imagen, mantener canales de color
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    # Sin conversión a escala de grises: imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # No se necesita reshape aquí si se mantiene el color, la forma será (100, 100, 3)
    datos_entrenamiento.append([imagen, etiqueta])

X = [] # imágenes
y = [] # etiquetas

for imagen, etiqueta in datos_entrenamiento:
    X.append(imagen)
    y.append(etiqueta)

# Convertir a arrays de numpy
X = np.array(X)
y = np.array(y)

# --- División de Datos (70% Entrenamiento, 15% Validación, 15% Prueba) ---
# Primera división: 85% para entrenamiento/validación, 15% para prueba
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y # Usar stratify para divisiones balanceadas
)

# Segunda división: Dividir el 85% en 70% entrenamiento y 15% validación (relativo al original)
# Tamaño de validación relativo a X_train_val: 0.15 / (1 - 0.15) = 0.15 / 0.85
val_split_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_split_ratio, random_state=42, stratify=y_train_val
)

# Normalizar imágenes DESPUÉS de dividir
X_train = X_train.astype('float') / 255.0
X_val = X_val.astype('float') / 255.0
X_test = X_test.astype('float') / 255.0

print(f"Formas Entrenamiento: X={X_train.shape}, y={y_train.shape}")
print(f"Formas Validación: X={X_val.shape}, y={y_val.shape}")
print(f"Formas Prueba: X={X_test.shape}, y={y_test.shape}")

# --- Aumento de Datos (Comentado para prueba) ---
# Generador para datos de entrenamiento (con aumento)
# train_datagen = ImageDataGenerator(
#     rotation_range=30,
#     width_shift_range=0.08,
#     height_shift_range=0.08,
#     shear_range=0.15,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# Generador para datos de validación y prueba (típicamente sin aumento, solo reescalado si es necesario)
# La normalización ya está hecha, así que este generador no hace nada extra.
# Aún lo usamos por consistencia al crear los iteradores flow.
val_test_datagen = ImageDataGenerator() # Sin aumento para validación/prueba

# Crear generadores flow
batch_size = 32
# Usar el generador sin aumento también para entrenamiento en esta prueba
train_generator = val_test_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator   = val_test_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False) # Sin shuffle para validación
test_generator  = val_test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False) # Sin shuffle para prueba

# --- Función de Construcción de Modelo Keras Tuner ---
def build_model(hp):
    model = tf.keras.models.Sequential()

    # Capa de Entrada - Actualizar input_shape a (TAMANO_IMG, TAMANO_IMG, 3) para color
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=64, step=32), # Step cambiado a 32
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(TAMANO_IMG, TAMANO_IMG, 3) # <-- Actualizado para color
    ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Segundo Bloque Conv
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=64, max_value=128, step=32),
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    # Tercer Bloque Conv
    model.add(tf.keras.layers.Conv2D(
        filters=hp.Int('conv_3_filter', min_value=128, max_value=256, step=64),
        kernel_size=(3, 3),
        activation='relu'
    ))

    # Dropout antes de Flatten (Fijo en 0.2)
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dropout(
    #     rate=hp.Float('dropout_1', min_value=0.2, max_value=0.5, step=0.1)
    # ))

    model.add(tf.keras.layers.Flatten())

    # Capa Densa
    model.add(tf.keras.layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=192, step=32),
        activation='relu'
    ))

    # Capa de Salida
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) # Usar num_classes dinámico

    # Compilar el modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-3, 1e-4, 5e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Instanciar el Tuner ---
# Usando RandomSearch, pero también puedes probar Hyperband: kt.Hyperband(...)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy', # El objetivo es maximizar la precisión de validación
    max_trials=5, # Número de combinaciones de hiperparámetros a probar (reducido a 5)
    executions_per_trial=1, # Cuántas veces entrenar cada combinación (mayor para estabilidad)
    directory='keras_tuner_experiment/tuner_results', # Directorio para guardar resultados del tuner
    project_name='flower_cnn_tuning_color_no_aug_fixed_dropout' # Nuevo nombre para esta ejecución
)

# --- Definir Callbacks ---
# Early Stopping: Detener el entrenamiento si val_loss no mejora durante 10 épocas
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# TensorBoard: Registrar el proceso de entrenamiento de la búsqueda
search_log_dir = os.path.join("keras_tuner_experiment", "logs", "search", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=search_log_dir, histogram_freq=1)

print("--- Iniciando Búsqueda de Hiperparámetros ---")
# --- Ejecutar la Búsqueda de Hiperparámetros ---
# Calcular pasos por época si se usan generadores
steps_per_epoch_train = len(X_train) // batch_size
steps_per_epoch_val = len(X_val) // batch_size

tuner.search(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=30, # Máximo de épocas por prueba (ajustar según sea necesario)
    validation_data=val_generator,
    validation_steps=steps_per_epoch_val,
    callbacks=[early_stopping, tensorboard_callback] # Usar el callback de búsqueda
)
print("--- Búsqueda de Hiperparámetros Finalizada ---")

# --- Obtener los Mejores Hiperparámetros ---
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
Número óptimo de filtros en la primera capa conv: {best_hps.get('conv_1_filter')}
Número óptimo de filtros en la segunda capa conv: {best_hps.get('conv_2_filter')}
Número óptimo de filtros en la tercera capa conv: {best_hps.get('conv_3_filter')}
# Tasa óptima de dropout: {best_hps.get('dropout_1'):.2f} # Comentado porque dropout es fijo
Número óptimo de unidades en la capa densa: {best_hps.get('dense_units')}
Tasa de aprendizaje óptima: {best_hps.get('learning_rate')}
""")

# --- Construir el Mejor Modelo ---
best_model = tuner.hypermodel.build(best_hps)

# --- (Opcional) Entrenar Más el Mejor Modelo ---
# Podrías querer entrenar el modelo final con los mejores hiperparámetros por más épocas
print("--- Entrenando el Mejor Modelo ---")
# Crear un nuevo callback de TensorBoard para la ejecución final de entrenamiento
final_log_dir = os.path.join("keras_tuner_experiment", "logs", "final_training", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
final_tensorboard_callback = TensorBoard(log_dir=final_log_dir, histogram_freq=1)
# Reusar early stopping o crear uno nuevo
final_early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


history = best_model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=100, # Entrenar por más épocas (ajustar según sea necesario)
    validation_data=val_generator,
    validation_steps=steps_per_epoch_val,
    callbacks=[final_early_stopping, final_tensorboard_callback] # Usar el callback final
)

# --- Guardar el Mejor Modelo ---
model_save_dir = 'keras_tuner_experiment/model'
os.makedirs(model_save_dir, exist_ok=True)
best_model.save(os.path.join(model_save_dir, 'flower_cnn_best_color.h5'))
print(f"--- Mejor Modelo Ajustado Guardado en {model_save_dir} ---")

# --- Evaluar el Mejor Modelo en el Conjunto de Validación ---
print("--- Evaluando en el Conjunto de Validación ---")
val_loss, val_accuracy = best_model.evaluate(val_generator, steps=steps_per_epoch_val)
print(f"Pérdida Final de Validación: {val_loss:.4f}")
print(f"Precisión Final de Validación: {val_accuracy:.4f}")

# --- Evaluar el Mejor Modelo en el Conjunto de Prueba ---
print("--- Evaluando en el Conjunto de Prueba ---")
steps_per_epoch_test = len(X_test) // batch_size
test_loss, test_accuracy = best_model.evaluate(test_generator, steps=steps_per_epoch_test)
print(f"Pérdida Final de Prueba: {test_loss:.4f}")
print(f"Precisión Final de Prueba: {test_accuracy:.4f}")