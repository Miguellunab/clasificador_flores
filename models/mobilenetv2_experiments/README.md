# Información del Modelo MobileNetV2

Este directorio contiene el código y modelos entrenados para clasificación de flores usando MobileNetV2.

## Estructura de directorios
- `run_mobilenetv2.py` - Script principal para entrenar el modelo MobileNetV2
- `Resultados/` - Carpeta con los resultados del mejor experimento
  - `model/` - Contiene el modelo entrenado
  - `logs/` - Contiene los logs de TensorBoard para análisis

## Nota importante
Los archivos originales del experimento se encuentran en:
`mobilenetv2_20250417-170634/`

Por motivos de organización, lo ideal sería mover manualmente los siguientes archivos:
1. Copiar `mobilenetv2_20250417-170634/model/mobilenetv2_flowers.h5` a `Resultados/model/`
2. Copiar los archivos de TensorBoard de `mobilenetv2_20250417-170634/logs/train/` a `Resultados/logs/train/`
3. Copiar los archivos de TensorBoard de `mobilenetv2_20250417-170634/logs/validation/` a `Resultados/logs/validation/`

## Instrucciones para entrenar un nuevo modelo
Para entrenar un nuevo modelo MobileNetV2, ejecuta el script `run_mobilenetv2.py`. Los resultados se guardarán en la carpeta `Resultados`.

## Resultados de entrenamiento
El modelo MobileNetV2 alcanzó aproximadamente un 80% de precisión en el conjunto de validación después de 30 épocas de entrenamiento.