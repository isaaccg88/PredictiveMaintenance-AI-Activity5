# Predictive Maintenance AI — Actividad 5

**Asignatura:** Gestión de Proyectos de Inteligencia Artificial · Módulo 2, semana 6  
**Entorno principal:** Windows, Python 3.13.5 y MLflow 3.17.1

> **Estado:** infraestructura, datos y código preparados. Los experimentos deben ejecutarse en el Windows del estudiante. No se publican métricas de modelos que todavía no se hayan medido; el PDF final se generará tras el entrenamiento.

## 1. Resumen y problema
Clasificación binaria de **fallas presentes** en máquinas sintéticas mediante Regresión Logística y Random Forest. Se realiza limpieza trazable, GridSearchCV, validación cruzada estratificada, MLflow, comparación estadística y análisis de deuda MLOps. No se afirma capacidad de anticipar fallas futuras ni validez en producción.

## 2. Datos y versión
Snapshot académico de siete columnas derivado de [AI4I 2020, UCI, DOI 10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), licencia CC BY 4.0; conservado de la [Actividad 4](https://github.com/isaaccg88/PredictiveMaintenance-AI-Activity4). **No es el archivo completo original de UCI**. Está publicado en [datos/datos_ini/ai4i2020.csv](datos/datos_ini/ai4i2020.csv), congelado, junto con [datos/splits.csv](datos/splits.csv).

| Comprobación inicial | Valor confirmado |
|---|---:|
| Registros | 10,000 |
| Fallas | 339 (3.39%) |
| Duplicados exactos | 0 |
| Filas con campos vacíos o categorías inválidas | 0 |
| Train / validación / test | 7,000 / 1,500 / 1,500 |
| Errores entre splits y etiquetas | 0 |

El pipeline produce un dataset preparado, hashes SHA256, reporte de calidad e histogramas. La imputación, One-Hot Encoding y normalización se ajustan **dentro de cada fold**, mediante scikit-learn Pipeline, para prevenir contaminación de test.

## 3. Arquitectura experimental

| Dimensión | Configuración |
|---|---|
| Baseline | DummyClassifier, clase mayoritaria |
| Modelos | LogisticRegression y RandomForestClassifier |
| Ajuste | GridSearchCV, validación cruzada estratificada de 5 folds |
| Semilla | 42 |
| Métrica primaria de Grid Search | Average Precision, por desbalance |
| Rejilla logística | C = 0.1, 1, 10; class_weight = None, balanced |
| Rejilla Random Forest | 120 / 240 árboles; profundidad 8 / libre; min_samples_leaf 1 / 5 |
| Total | 14 combinaciones, 70 ajustes CV y refit final |
| Selección | Umbral y modelo basados **solo en validación**, F2 con precision mínima de 0.50 si es posible |
| Métricas test | Accuracy, balanced accuracy, precision, recall, F1, F2, AP, ROC-AUC y latencia |
| Estadística | Bootstrap pareado de diferencia F1, 2,000 remuestreos |
| MLflow | Un run padre por modelo y un run hijo por candidato de Grid Search; modelos y artefactos registrados |

## 4. Organización de la entrega

| Carpeta o archivo | Finalidad |
|---|---|
| [datos/datos_ini/](datos/datos_ini/) | Snapshot inalterado del dataset académico |
| [datos/datos_limp/](datos/datos_limp/) | Dataset preparado, se genera al ejecutar |
| [fuentes/datos_prep.py](fuentes/datos_prep.py) | Limpieza, calidad, hashes y EDA |
| [fuentes/train.py](fuentes/train.py) | Modelos, Grid Search, selección y MLflow |
| [fuentes/evaluation.py](fuentes/evaluation.py) | Métricas y umbrales |
| [fuentes/entrena.ipynb](fuentes/entrena.ipynb) | Notebook Jupyter / Colab |
| [fuentes/export_evidencias.py](fuentes/export_evidencias.py) | Runs MLflow en CSV/JSON |
| [fuentes/generar_reporte.py](fuentes/generar_reporte.py) | Crea el reporte PDF con cifras reales |
| [resultados/](resultados/) | CSV, figuras, calidad y bootstrap |
| [evidencias/mlflow/](evidencias/mlflow/) | Capturas **reales** y runs exportados |
| [docs/](docs/) | Protocolo y reporte PDF académico |
| [CHANGELOG.md](CHANGELOG.md) | Registro de cambios y decisiones |
| [tests/](tests/) | Pruebas automáticas |

## 5. Ejecución local recomendada (Windows)

Mantener el servidor MLflow iniciado en la ventana PowerShell existente, desde la carpeta personal MLflow-Actividad5, con este comando:

```powershell
.\.venv\Scripts\mlflow.exe server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Abre el repositorio con **GitHub Desktop > Repository > Show in Explorer**. En el Explorador de Windows, escribe **powershell** en la barra de direcciones para abrir una **segunda ventana** ubicada en la raíz del repositorio. Ejecuta los comandos siguientes, en orden:

**5.1. Instalar dependencias del proyecto:**

```powershell
& "$HOME\MLflow-Actividad5\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

**5.2. Ejecutar pruebas de la lógica de evaluación:**

```powershell
& "$HOME\MLflow-Actividad5\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

**5.3. Entrenar los modelos y registrar los runs MLflow** (puede tardar varios minutos):

```powershell
& "$HOME\MLflow-Actividad5\.venv\Scripts\python.exe" -m fuentes.train --tracking-uri http://127.0.0.1:5000
```

**5.4. Exportar el historial reproducible de MLflow:**

```powershell
& "$HOME\MLflow-Actividad5\.venv\Scripts\python.exe" -m fuentes.export_evidencias --tracking-uri http://127.0.0.1:5000
```

**5.5. Capturar el panel MLflow.** Abre http://127.0.0.1:5000, selecciona **Model training** y guarda cuatro capturas reales según [evidencias/mlflow/README.md](evidencias/mlflow/README.md). El profesor no puede abrir el localhost de tu equipo.

**5.6. Crear el PDF con los resultados y capturas obligatorias:**

```powershell
& "$HOME\MLflow-Actividad5\.venv\Scripts\python.exe" -m fuentes.generar_reporte --require-screenshots
```

El archivo final será **docs/Reporte_Tecnico_Actividad5.pdf**. Revisarlo visualmente, añadir interpretación profunda al README/CHANGELOG y publicar el commit final a GitHub. Los CSV y figuras se generan desde el código; **nunca se insertan cifras inventadas**.

### Notebook alternativo en Colab
[**Abrir notebook de Actividad 5**](https://colab.research.google.com/github/isaaccg88/PredictiveMaintenance-AI-Activity5/blob/main/fuentes/entrena.ipynb).

En Colab se crea un SQLite en su propio runtime. **El localhost de Colab no es el localhost de tu Windows**; por ello las capturas pedidas por el profesor se obtendrán más fácilmente con la ejecución local.

## 6. MLflow, trazabilidad y evidencias
Se publicarán IDs de experimentos y runs, métricas y parámetros de cada combinación Grid Search, mejores modelos y artefactos de las corridas, resumen de validación y test, capturas reales del panel y auditoría de hashes. Los scripts no suben la base local SQLite ni secretos a GitHub.

## 7. Riesgos y acciones MLOps
La alta desproporción de clases exige reportar precision, recall, AP y falsos negativos, no solo accuracy. Limitaciones: dataset sintético, sin horizonte temporal ni validación prospectiva por máquina. Futuro: monitoreo de drift, nuevas particiones temporales, calibración, registro y promoción controlada de modelos, revisión humana, políticas de reversión y auditoría de alertas.

## 8. Formato de entrega
La entrega acordada es **la URL del repositorio GitHub**, que contendrá código, notebook, datos, métricas, **reporte PDF** y evidencias de MLflow. No se entregará ZIP. **Pendiente:** entrenamiento local real, capturas, PDF, interpretación de métricas y auditoría final.

## 9. Declaración de uso de IA

| Herramienta | Actividades | Responsabilidad |
|---|---|---|
| ChatGPT (OpenAI) | Asistencia en diseño experimental, programación, depuración, documentación y revisión de GitHub | El estudiante ejecuta los experimentos, verifica los artefactos y decide la interpretación final; las métricas se derivan de código reproducible. |

## 10. Referencias
- UCI AI4I 2020: https://doi.org/10.24432/C5HS5C
- Scikit-learn: https://scikit-learn.org/stable/
- MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking/
- Protocolo: [docs/protocolo_experimental.md](docs/protocolo_experimental.md)
