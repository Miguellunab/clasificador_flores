"""
Script mejorado para convertir un modelo Keras directamente a formato TensorFlow.js
utilizando SavedModel como formato intermedio para mejor compatibilidad.
"""

import os
import shutil
import tensorflow as tf
import tensorflowjs as tfjs

def convertir_modelo(ruta_modelo_h5, nombre_base=None):
    """
    Convierte un modelo .h5 a formato SavedModel y TensorFlow.js
    
    Args:
        ruta_modelo_h5: Ruta al archivo del modelo .h5
        nombre_base: Nombre base para los directorios de salida (opcional)
    
    Returns:
        Diccionario con las rutas de los modelos generados
    """
    # Verificar si el modelo existe
    if not os.path.exists(ruta_modelo_h5):
        raise FileNotFoundError(f"No se encontró el modelo en: {ruta_modelo_h5}")
    
    # Extraer el nombre base si no se proporcionó
    if nombre_base is None:
        nombre_base = os.path.basename(ruta_modelo_h5).replace('.h5', '')
    
    # Determinar directorios de salida
    dir_base = os.path.dirname(ruta_modelo_h5)
    saved_model_dir = os.path.join(dir_base, f"{nombre_base}_saved_model")
    tfjs_model_dir = os.path.join(dir_base, f"{nombre_base}_tfjs")
    
    # Crear directorios si no existen
    for directorio in [saved_model_dir, tfjs_model_dir]:
        os.makedirs(directorio, exist_ok=True)
    
    resultados = {
        'h5': ruta_modelo_h5,
        'saved_model': saved_model_dir,
        'tfjs': tfjs_model_dir
    }
    
    try:
        # 1. Cargar el modelo .h5
        print(f"Cargando modelo desde: {ruta_modelo_h5}")
        modelo = tf.keras.models.load_model(ruta_modelo_h5)
        print("Modelo cargado exitosamente")
        
        # 2. Guardar en formato SavedModel
        print(f"Guardando modelo en formato SavedModel: {saved_model_dir}")
        tf.saved_model.save(modelo, saved_model_dir)
        print("Modelo SavedModel guardado exitosamente")
        
        # 3. Convertir a TensorFlow.js
        print(f"Convirtiendo modelo a TensorFlow.js: {tfjs_model_dir}")
        # Limpiar directorio si existe
        if os.path.exists(tfjs_model_dir):
            shutil.rmtree(tfjs_model_dir)
        os.makedirs(tfjs_model_dir, exist_ok=True)
        
        # Convertir usando la API de tensorflowjs
        tfjs.converters.save_keras_model(modelo, tfjs_model_dir)
        print("Modelo TensorFlow.js generado exitosamente")
        
        # Verificar archivos generados en TensorFlow.js
        print("\nArchivos generados en TensorFlow.js:")
        for file in os.listdir(tfjs_model_dir):
            print(f"  - {os.path.join(tfjs_model_dir, file)}")
            
    except Exception as e:
        print(f"Error durante la conversión: {e}")
        raise
    
    return resultados

def main():
    """Función principal para ejecutar el script desde línea de comandos"""
    import argparse
    parser = argparse.ArgumentParser(description='Convertir modelos .h5 a SavedModel y TensorFlow.js')
    parser.add_argument('ruta_modelo', help='Ruta al archivo del modelo .h5')
    parser.add_argument('--nombre', help='Nombre base para los directorios de salida (opcional)', default=None)
    
    args = parser.parse_args()
    
    try:
        resultados = convertir_modelo(args.ruta_modelo, args.nombre)
        print("\n=== Conversión completada con éxito ===")
        print(f"Modelo original (.h5): {resultados['h5']}")
        print(f"Modelo SavedModel: {resultados['saved_model']}")
        print(f"Modelo TensorFlow.js: {resultados['tfjs']}")
        print("\nPara usar el modelo en tu aplicación web:")
        print(f"1. Copia los archivos de {resultados['tfjs']} a tu aplicación web")
        print("2. Actualiza la ruta del modelo en index.html a: './ruta/model.json'")
        print("3. Asegúrate de preprocesar las imágenes correctamente (tamaño y normalización)")
    except Exception as e:
        print(f"La conversión falló: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())