# Datos iniciales — AI4I 2020 (original y snapshot)

**Fuente original oficial:** [AI4I 2020 Predictive Maintenance Dataset, UCI Repository, ID 601](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset), DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C), licencia CC BY 4.0. Es un dataset **sintético**, no observaciones industriales verificadas.

| Archivo | Filas × columnas | Finalidad |
|---|---:|---|
| [`uci_ai4i2020_original.csv`](uci_ai4i2020_original.csv) | 10,000 × 14 | Descarga íntegra del CSV oficial UCI; no se utiliza para reentrenar |
| [`ai4i2020.csv`](ai4i2020.csv) | 10,000 × 7 | Snapshot académico de la Actividad 4 utilizado realmente para entrenar |
| [`../splits.csv`](../splits.csv) | 10,000 × 3 | Particiones congeladas de entrenamiento, validación y prueba |

**Correspondencia demostrada:** [`fuentes/verificar_fuente_uci.py`](../../fuentes/verificar_fuente_uci.py) cotejó las siete variables **fila por fila** (0 diferencias). Consulta el [manifiesto verificable con hashes](../../docs/uci_source_manifest.json).

Las columnas `UDI` y `Product ID` se excluyeron por tratarse de identificadores; `TWF`, `HDF`, `PWF`, `OSF`, `RNF` se excluyeron por representar modos de falla y filtrarían información del objetivo. El subconjunto conserva Type, las cinco variables numéricas y Machine failure. El snapshot no sufrió modificaciones al añadir el origen: por tanto, los resultados de MLflow siguen siendo los originales.
