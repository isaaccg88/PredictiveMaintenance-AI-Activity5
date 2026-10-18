# Datos preparados

El script `fuentes/datos_prep.py` generará aquí `ai4i2020_limpio.csv` tras validar tipos, duplicados, faltantes y rangos. La columna `indice_original` conecta cada fila con el snapshot inicial y con `datos/splits.csv`.

**No hay resultados inventados en este directorio.** El archivo preparado aparecerá cuando se ejecute el pipeline. El escalado, imputación y codificación se entrenan **dentro** del Pipeline de scikit-learn para cada fold de validación cruzada, no antes del split.
