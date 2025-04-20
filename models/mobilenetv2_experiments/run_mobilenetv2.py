import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import os
import datetime
import tensorflowjs as tfjs
import shutil

# --- Configuración ---
IMG_SIZE = 224
BATCH_SIZE = 32
EXPERIMENT_NAME = f"mobilenetv2_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
RESULTS_DIR = os.path.join("Resultados")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
MODEL_DIR = os.path.join(RESULTS_DIR, "model")
SAVED_MODEL_DIR = os.path.join(RESULTS_DIR, "saved_model")  # Nueva ruta para SavedModel
TFJS_MODEL_DIR = os.path.join(RESULTS_DIR, "tfjs_model")  # Nueva ruta para TensorFlow.js

# Crear directorios
for dir_path in [LOG_DIR, MODEL_DIR, SAVED_MODEL_DIR, TFJS_MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- Carga y Preprocesamiento de Datos ---
dataset, metadata = tfds.load('tf_flowers', as_supervised=True, with_info=True)
num_classes = metadata.features['label'].num_classes
print(f"Número de clases detectado: {num_classes}")

datos_entrenamiento = []
for i, (imagen, etiqueta) in enumerate(dataset['train']):
    # Redimensionar a 224x224 (puede estirar la imagen)
    imagen = cv2.resize(imagen.numpy(), (IMG_SIZE, IMG_SIZE))
    datos_entrenamiento.append([imagen, etiqueta])

X = np.array([img for img, _ in datos_entrenamiento])
y = np.array([lbl for _, lbl in datos_entrenamiento])

# División de datos (70% train, 15% val, 15% test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
val_split_ratio = 0.15 / 0.85
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_split_ratio, random_state=42, stratify=y_train_val
)

# Normalizar imágenes
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(f"Formas Entrenamiento: X={X_train.shape}, y={y_train.shape}")
print(f"Formas Validación: X={X_val.shape}, y={y_val.shape}")
print(f"Formas Prueba: X={X_test.shape}, y={y_test.shape}")

# --- Generadores ---
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# --- Modelo MobileNetV2 ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Transfer learning: solo entrenar la parte superior

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
tensorboard_callback = TensorBoard(log_dir=LOG_DIR)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# --- Entrenamiento ---
history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=val_dataset,
    callbacks=[tensorboard_callback, early_stopping]
)

# --- Guardar modelo en diferentes formatos ---
# 1. Guardar en formato H5 (mantener para compatibilidad)
h5_path = os.path.join(MODEL_DIR, 'mobilenetv2_flowers.h5')
model.save(h5_path)
print(f"Modelo guardado en formato H5: {h5_path}")

# 2. Guardar en formato SavedModel (recomendado para TensorFlow 2.x)
saved_model_path = os.path.join(SAVED_MODEL_DIR, 'mobilenetv2_flowers')
tf.saved_model.save(model, saved_model_path)
print(f"Modelo guardado en formato SavedModel: {saved_model_path}")

# 3. Convertir directamente a TensorFlow.js
tfjs_path = os.path.join(TFJS_MODEL_DIR, 'mobilenetv2_flowers')
try:
    # Limpiar directorio anterior si existe
    if os.path.exists(tfjs_path):
        shutil.rmtree(tfjs_path)
    os.makedirs(tfjs_path, exist_ok=True)
    
    # Convertir usando la API de TensorFlow.js
    tfjs.converters.save_keras_model(model, tfjs_path)
    print(f"Modelo convertido a TensorFlow.js: {tfjs_path}")
    
    # Verificar archivos generados
    print("Archivos TensorFlow.js generados:")
    for root, dirs, files in os.walk(tfjs_path):
        for file in files:
            print(f"  - {os.path.join(root, file)}")
            
    print("\nPara usar este modelo en tu aplicación web:")
    print(f"1. Copia los archivos de {tfjs_path} a tu aplicación web")
    print("2. Actualiza la ruta del modelo en index.html")
    print("3. Asegúrate de preprocesar las imágenes a tamaño 224x224 y normalizarlas a [0,1]")
except Exception as e:
    print(f"Error al convertir a TensorFlow.js: {e}")

# --- Evaluación ---
val_loss, val_acc = model.evaluate(val_dataset)
print(f"Precisión de validación: {val_acc:.4f}")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Precisión de prueba: {test_acc:.4f}")
