# Protocolo experimental pre-registrado

## Objetivo y fuentes
**Problema:** clasificación de fallas **presentes** en registros de maquinaria sintética, no predicción prospectiva en tiempo real. Fuente: UCI AI4I 2020, DOI 10.24432/C5HS5C; snapshot siete columnas de la Actividad 4. Licencia CC BY 4.0.

## Integridad y preprocesamiento
- Conservar `datos/datos_ini/ai4i2020.csv` sin modificar. Crear copia validada y versionada en `datos/datos_limp/` con `indice_original`.
- Comprobar valores nulos, duplicados, clase minoritaria, rangos, tipos y anomalías. Documentar resultados reales; no eliminar observaciones arbitrariamente.
- Imputación numérica por mediana, categórica por categoría frecuente, One-Hot Encoding y estandarización de numéricas **solo dentro del Pipeline** en cada fold.
- Semilla global 42 y particiones fijas previas en `datos/splits.csv`: 70% entrenamiento, 15% validación y 15% prueba.

## Entrenamiento
- **LogisticRegression** con rejilla sobre `C` y `class_weight`.
- **RandomForestClassifier** con rejilla de profundidad, número de árboles, tamaño mínimo de hoja y ponderación de clase.
- `StratifiedKFold(n_splits=5,shuffle=True,random_state=42)`; `GridSearchCV(scoring='average_precision',refit=True)` usando exclusivamente train.
- Baseline `DummyClassifier` para contextualizar desbalance.

## Selección, prueba y métricas
- En validación: elegir umbral buscando F2 máximo con precisión >= 0.50 (sujeto a factibilidad), y ordenar modelos por F2; desempate recall y AP.
- Congelar modelo y umbral **antes** de calcular resultados de test. Solo después evaluar test una vez con accuracy, balanced accuracy, precision, recall, F1, F2, ROC-AUC, Average Precision y matriz de confusión.
- Comparar rendimiento de clasificación y coste computacional (tiempos de Grid Search e inferencia).
- El bootstrap pareado de diferencias de F1 es descriptivo; el test contiene pocas fallas y no es prueba de validez industrial.

## MLflow
- Tracking local al servidor del usuario `http://127.0.0.1:5000`, SQLite; nombre de experimento `PredictiveMaintenance_Actividad5`.
- Runs padre por modelo y runs hijos por combinación de Grid Search con `cv_mean_average_precision`, `cv_std_average_precision` y parámetros; modelo serializado y artefactos, métricas de validación/test, notas de provenance.
- Para Colab, no usar `localhost` de Windows: se habilita un backend SQLite **dentro de Colab** y se exportan evidencias; panel Windows y métricas locales se tomarán del entrenamiento local.

## Limitaciones y deuda técnica
- Conjunto sintético y no temporal, ausencia de validación entre máquinas y monitorización de drift, prevalencia de fallas minoritaria.
- Modelo y selección académicos, no despliegue en planta; verificar datos reales y costos antes de pilotaje.
- No subir bases SQLite ni artefactos voluminosos sin necesidad; entregar runs exportados y capturas reales del panel.
