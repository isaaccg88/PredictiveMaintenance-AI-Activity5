# Registro de cambios — Actividad 5

## 2026-10-18 — Inicialización
- Repositorio independiente para la Actividad 5; mismo caso de uso académico que la Actividad 4.
- Snapshot original de trabajo del dataset AI4I 2020 (siete columnas de variables seleccionadas; no el archivo bruto completo de UCI).
- Particiones trazables de entrenamiento (7,000), validación (1,500) y prueba (1,500) reutilizadas del experimento previo sin inspeccionar prueba para ajuste.
- Arquitectura prevista: regresión logística vs. random forest, GridSearchCV estratificado, tracking MLflow 3.17.1, visualizaciones y reporte PDF.

> Se incorporarán aquí ejecuciones, cambios de datos, versiones y resultados en commits posteriores. No se reportan métricas de modelos de la Actividad 5 antes del entrenamiento real.
