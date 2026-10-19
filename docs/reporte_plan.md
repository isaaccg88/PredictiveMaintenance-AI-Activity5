# Plan y control del reporte técnico — Actividad 5

**Estado: ejecutado.** La versión final fue generada el 18 de octubre de 2026 usando los resultados reales de `resultados/` y las seis capturas del panel MLflow.

**Entregable:** [Reporte_Tecnico_Actividad5.pdf](Reporte_Tecnico_Actividad5.pdf) — 11 páginas, 14 imágenes y comprobación automática de integridad y renderizado.

Secciones incluidas:
1. Portada, resumen ejecutivo, alcance y objetivos.
2. Fuente y diccionario de datos, archivo oficial UCI de 14 columnas, verificación fila a fila del snapshot de 7 columnas, limpieza y hashes SHA256.
3. Diagnóstico de datos desbalanceados, completitud, valores extremos, figuras 1–4 reordenadas e interpretación individual.
4. Validación cruzada, Grid Search, selección de hiperparámetros.
5. Evidencias de MLflow: runs, panel, modelos y artefactos.
6. Comparación numérica y visual de modelos, matrices de confusión, PR y ROC.
7. Intervalos bootstrap e interpretación de incertidumbre.
8. Costos computacionales, deuda técnica y propuestas MLOps.
9. Conclusiones, referencias, declaración de uso de ChatGPT y reproducibilidad.

Se conserva el generador reproducible en `fuentes/generar_reporte.py`; al volver a ejecutarlo, corresponde actualizar resultados, capturas y reporte juntos.

**Evidencia adicional:** [notebook auditor ejecutado](../fuentes/Evidencia_Ejecutada_Actividad5.ipynb), con seis celdas de código y resultados reales verificados sin reentrenamiento. La plantilla de entrenamiento principal se conserva en `fuentes/entrena.ipynb`.
