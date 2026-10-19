# Evidencias reales del panel MLflow — Actividad 5

El experimento `PredictiveMaintenance_Actividad5` contiene **17 runs reales**: 1 baseline DummyClassifier, 2 runs principales y 14 combinaciones de Grid Search. Como no caben todos en una sola pantalla, el listado se documenta mediante **dos o tres capturas consecutivas**, sin manipular su contenido.

## Capturas del listado de 17 runs (misma pantalla y experimento)

- `01_experimentos.png` — parte **superior**; nombre del experimento, contador de 17 runs y grupo RandomForest.
- `01_experimentos_02.png` — continuación central; mismas columnas y runs adicionales al desplazarse.
- `01_experimentos_03.png` — parte **inferior**; ejecuciones de regresión logística y DummyClassifier.

Las dos continuaciones son opcionales para el generador PDF, pero **se recomiendan ambas** para demostrar el listado completo. Mantener igual el filtro, el experimento y las columnas; desplazar únicamente la tabla verticalmente. El reporte añade automáticamente las capturas adicionales que existan.

## Otras evidencias obligatorias

- `02_gridsearch_parametros.png` — una ejecución hija de Grid Search mostrando hiperparámetros y métricas CV.
- `03_comparacion_runs.png` — comparación de los modelos principales, con columnas de F1, recall y AP.
- `04_modelos_artefactos.png` — vista de los modelos registrados o de los artefactos guardados en uno de los runs.

Guardar imágenes **PNG auténticas**, sin fabricar pantallas ni alterar métricas. Las capturas solo complementan `runs_exportados.csv` y `runs_manifest.json`, donde se conservan los registros estructurados de MLflow.

El panel está disponible en **http://127.0.0.1:5000** únicamente en el equipo del estudiante; el profesor no podrá acceder a esa dirección desde GitHub. No incluir contraseñas, tokens, bases de datos privadas ni artefactos de ejecución innecesariamente voluminosos.
