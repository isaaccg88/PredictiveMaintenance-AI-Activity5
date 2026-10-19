# Registro de cambios — Actividad 5

## 2026-10-18 — Entrenamiento ejecutado y evidencia final (versión de entrega)
- Se entrenaron en Windows 11, Python 3.13.5, **regresión logística** y **Random Forest** con GridSearchCV de cinco folds estratificados; se añadió DummyClassifier como baseline.
- Se guardaron 17 ejecuciones reales en MLflow 3.17.1 (14 candidatos, 2 principales y 1 baseline), con modelos registrados, artefactos y tabla CSV exportada.
- Se conservó la separación train/validación/test (7,000/1,500/1,500) de la Actividad 4. Los datos preparados mantienen índice original y manifiestos SHA256.
- Modelo ganador seleccionado antes de test: **RandomForest** (umbral 0.20, F2 validación 0.7143). En test F1 0.6341, recall 0.7647, AP 0.7518, 39 verdaderos positivos, 12 falsos negativos y 33 falsos positivos.
- Se calculó bootstrap pareado: delta F1 respecto de regresión logística = 0.2245, IC95 percentil [0.0897, 0.3668] con 2,000 remuestreos.
- Se incorporaron ocho figuras descriptivas/comparativas y seis capturas auténticas del panel de MLflow, publicadas junto con métricas y predicciones.
- Se generó y verificó el [**reporte técnico PDF**](docs/Reporte_Tecnico_Actividad5.pdf): 11 páginas y 14 imágenes, auditadas mediante GitHub Actions.
- Se documentaron análisis cuantitativos, desbalance, riesgos de sobreajuste, deuda técnica, MLOps y declaración de uso de ChatGPT.

## 2026-10-18 — Inicialización y protocolo
- Repositorio independiente para la Actividad 5; mismo caso de uso académico de la Actividad 4.
- Snapshot del dataset AI4I 2020 (siete columnas derivadas de UCI) y particiones originales incorporados al control de versiones.
- Configuración de scripts de limpieza, evaluación, entrenamiento y MLflow; notebook Jupyter/Colab, documentación y validaciones automáticas.
- Se probó el pipeline completo con GitHub Actions antes de ejecutarlo en Windows; se corrigió una incompatibilidad de serialización MLflow con scikit-learn/NumPy.

> El dataset es sintético y clasifica falla presente. Los experimentos no validan anticipación temporal de fallas ni despliegue industrial.
