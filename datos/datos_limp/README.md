# Dataset preparado y versionado

`ai4i2020_limpio.csv` **ya está publicado**; se generó mediante `fuentes/datos_prep.py` a partir del snapshot original de siete columnas de UCI AI4I 2020. Se conservaron los 10,000 registros.

Se verificaron: 339 fallas, cero duplicados exactos, cero nulos, tipos válidos, coincidencia con las etiquetas y particiones fijas de `datos/splits.csv`. La columna `indice_original` enlaza cada registro con el origen.

No se eliminan valores extremos IQR sin justificación: se observaron 418 en velocidad y 69 en torque. Los imputadores, escaladores y One-Hot Encoding se ajustan dentro de cada fold en `scikit-learn Pipeline`, evitando ajustar parámetros de preprocesamiento sobre validación o prueba.

Hashes SHA256, conteos y política de tratamiento están publicados en `resultados/datos_manifest.json` y `resultados/calidad_datos.json`.
