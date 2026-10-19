# Resultados observados de la Actividad 5

Estos archivos **ya fueron generados mediante un entrenamiento real en Windows 11**, Python 3.13.5 y MLflow 3.17.1, y no contienen métricas inventadas.

- `calidad_datos.json`, `datos_manifest.json`: calidad, splits y SHA256.
- `comparacion_modelos.csv`: métricas de CV, validación y test.
- `gridsearch_cv.csv`, `gridsearch_LogisticRegression.csv`, `gridsearch_RandomForest.csv`: búsqueda de 14 configuraciones con cinco folds.
- `seleccion_congelada.json`: decisión de RandomForest y umbral 0.20 sobre validación.
- `umbrales_validacion.csv`: rejilla completa de umbrales.
- `predicciones_test.csv`: probabilidades, predicciones y etiquetas para 1,500 registros.
- `bootstrap_pareado.csv`: 2,000 remuestreos, diferencia F1 e intervalo de confianza.
- `baseline_dummy.json`: precisión/recall/F1 del clasificador mayoritario.
- `resumen_experimento.json`: modelo seleccionado, versiones, IDs de runs y métricas.
- `figuras/`: ocho gráficos PNG auténticos con desbalance, calidad, comparación, curvas, matrices de confusión y bootstrap.

**Lectura de resultados:** RandomForest (recall 76.47%, F1 63.41%) detecta 39 de 51 fallas, frente a 17 con regresión logística, pero produce 33 falsas alarmas, frente a 15. La comparación debe leerse con las limitaciones del dataset sintético. Véanse el [README](../README.md) y el [PDF técnico](../docs/Reporte_Tecnico_Actividad5.pdf).
