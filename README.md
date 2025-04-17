# Clasificador de Flores

Este proyecto implementa varios modelos de deep learning para clasificar imágenes de flores utilizando el dataset `tf_flowers` de TensorFlow.

## Estructura del Proyecto

El proyecto está organizado en tres carpetas principales:

### 1. Modelos Manuales
Contiene implementaciones de modelos CNN y densos creados manualmente:
- `entrenamiento.py`: Script principal para entrenar los modelos
- Carpetas con modelos entrenados (modelocnn2AD, modelocnnAD, modelodensoAD)
- Logs de TensorBoard para visualizar el entrenamiento

### 2. Keras Tuner
Contiene experimentos de optimización de hiperparámetros usando Keras Tuner:
- `run_keras_tuner.py`: Script para ejecutar la búsqueda de hiperparámetros
- Resultados de diferentes experimentos de optimización

### 3. MobileNetV2
Implementación de transfer learning usando el modelo pre-entrenado MobileNetV2:
- `run_mobilenetv2.py`: Script para entrenar el modelo MobileNetV2
- Resultados del entrenamiento

## Modelos Implementados

1. **Modelo Denso**: Red neuronal fully-connected simple
2. **Modelo CNN**: Red neuronal convolucional básica
3. **Modelo CNN2**: Red neuronal convolucional mejorada con dropout
4. **MobileNetV2**: Modelo de transfer learning usando MobileNetV2 pre-entrenado

## Requisitos

```
tensorflow>=2.5.0
tensorflow-datasets
matplotlib
opencv-python
keras-tuner
```

## Uso

Para entrenar los modelos manuales:
```
python modelos_manuales/entrenamiento.py
```

Para ejecutar la optimización de hiperparámetros:
```
python keras_tuner_modelos/run_keras_tuner.py
```

Para entrenar el modelo MobileNetV2:
```
python mobilenetv2_experiments/run_mobilenetv2.py
```

## Resultados

Los resultados de entrenamiento se guardan en las respectivas carpetas de modelos y se pueden visualizar usando TensorBoard:

```
tensorboard --logdir=logs/
```

## Licencia

Este proyecto está bajo la licencia MIT.