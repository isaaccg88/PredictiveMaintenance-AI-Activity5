"""Genera PDF académico desde métricas REALES previamente calculadas.

Ejecutar después de train.py y export_evidencias.py:
  python -m fuentes.generar_reporte --require-screenshots

No construye tablas ficticias ni sustituye capturas reales del panel.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle
)
from fuentes.datos_prep import ROOT

RESULT = ROOT / "resultados"
FIG = RESULT / "figuras"
EVIDENCE = ROOT / "evidencias" / "mlflow"
OUTPUT = ROOT / "docs" / "Reporte_Tecnico_Actividad5.pdf"
REQUIRED_CAPTURES = [
    "01_experimentos.png", "02_gridsearch_parametros.png",
    "03_comparacion_runs.png", "04_modelos_artefactos.png"
]
# Capturas adicionales opcionales del listado de 17 runs, en orden de scroll.
# Se incorporan automáticamente al PDF cuando están presentes.
EXTRA_RUNS_CAPTURES = [
    "01_experimentos_02.png",
    "01_experimentos_03.png",
]
TEAL = colors.HexColor("#147A87")
DARK = colors.HexColor("#193240")
MUTED = colors.HexColor("#5E717A")


def txt(value):
    return html.escape(str(value))


def p(text, style):
    return Paragraph(text, style)


def document(require_screenshots: bool = False) -> Path:
    required = [
        RESULT / "comparacion_modelos.csv", RESULT / "calidad_datos.json",
        RESULT / "seleccion_congelada.json", RESULT / "bootstrap_pareado.csv",
        EVIDENCE / "runs_exportados.csv", EVIDENCE / "runs_manifest.json"
    ]
    absent = [str(f.relative_to(ROOT)) for f in required if not f.exists()]
    if absent:
        raise FileNotFoundError("Primero ejecutar train.py y export_evidencias.py. "
                                "Faltan: " + ", ".join(absent))
    missing_captures = [x for x in REQUIRED_CAPTURES if not (EVIDENCE / x).exists()]
    if require_screenshots and missing_captures:
        raise FileNotFoundError("Capturas MLflow ausentes: " + ", ".join(missing_captures))
    quality = json.loads((RESULT / "calidad_datos.json").read_text(encoding="utf-8"))
    decisions = json.loads((RESULT / "seleccion_congelada.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(RESULT / "comparacion_modelos.csv")
    bootstrap = pd.read_csv(RESULT / "bootstrap_pareado.csv").iloc[0]
    runs = pd.read_csv(EVIDENCE / "runs_exportados.csv")
    run_manifest = json.loads((EVIDENCE / "runs_manifest.json").read_text(encoding="utf-8"))
    if len(comparison) != 2:
        raise ValueError("Se esperaban dos modelos principales: LR y RF.")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CoverTitle", fontName="Helvetica-Bold", fontSize=21, leading=26,
        textColor=DARK, alignment=TA_CENTER, spaceAfter=16
    ))
    styles.add(ParagraphStyle(
        name="SectionA5", fontName="Helvetica-Bold", fontSize=12.5, leading=16,
        textColor=TEAL, spaceBefore=16, spaceAfter=9, keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name="NormalA5", fontName="Helvetica", fontSize=9.3, leading=14,
        textColor=DARK, spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name="CaptionA5", fontName="Helvetica-Oblique", fontSize=8,
        leading=11, textColor=MUTED, spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="SubA5", fontName="Helvetica-Bold", fontSize=10,
        leading=14, textColor=DARK, spaceBefore=10, spaceAfter=5
    ))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=2.15*cm,
                            leftMargin=2.15*cm, topMargin=2.2*cm,
                            bottomMargin=2.0*cm, title="Actividad 5 - Predictive Maintenance AI")
    story = []

    def para(t):
        story.append(p(t, styles["NormalA5"]))

    def header(t):
        story.append(p(t, styles["SectionA5"]))

    def caption(t):
        story.append(p(t, styles["CaptionA5"]))

    def table(headers, rows, widths=None):
        values = [[p(txt(c), styles["CaptionA5"]) for c in headers]]
        values += [[p(txt(c), styles["NormalA5"]) for c in row] for row in rows]
        tb = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
        tb.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDECEF")),
            ("GRID", (0, 0), (-1, -1), .35, colors.HexColor("#CDDADF")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(tb)
        story.append(Spacer(1, .25*cm))

    def figure(name, title, width=15.4*cm):
        file = FIG / name
        if not file.exists():
            raise FileNotFoundError(f"Figura obligatoria inexistente: {file}")
        from reportlab.lib.utils import ImageReader
        width_px, height_px = ImageReader(str(file)).getSize()
        h = width * height_px / width_px
        if h > 11*cm:
            h = 11*cm
            width = h * width_px / height_px
        story.append(Image(str(file), width=width, height=h))
        caption(title)

    story += [Spacer(1, 2.1*cm),
              p("Predictive Maintenance AI", styles["CoverTitle"]),
              p("Actividad 5 - Gestión de proyectos de inteligencia artificial",
                styles["SubA5"]),
              Spacer(1, .35*cm)]
    para("Módulo 2 / Semana 6 | Entrenamiento, búsqueda de hiperparámetros y trazabilidad con MLflow")
    para("Universidad Tecmilenio | Dataset AI4I 2020, UCI | Entorno: Windows / Python 3.13.5 / MLflow 3.17.1")
    para(f"Modelo seleccionado antes del test: <b>{txt(decisions['modelo_seleccionado'])}</b>; "
         f"umbral validado: <b>{decisions['umbral_congelado']:.2f}</b>.")
    para("Documento generado a partir de resultados y evidencias reales. "
         "Las limitaciones del conjunto sintético se describen explícitamente.")
    story.append(PageBreak())

    header("1. Resumen ejecutivo y problema")
    para("La manufactura requiere identificar condiciones operativas relacionadas con "
         "fallas sin sobreinterpretar un dataset sintético como predicción futura. "
         "Esta actividad compara regresión logística y random forest entrenados sobre "
         "la misma partición, con GridSearchCV y cinco folds estratificados. "
         "MLflow guarda parámetros, métricas y modelos para reproducibilidad.")

    para("RESULTADOS PRINCIPALES: Random Forest fue seleccionado exclusivamente en "
         "validación (F2 0.7143, umbral 0.20). En test logró F1 0.6341, "
         "recall 0.7647 y Average Precision 0.7518. La regresión logística "
         "obtuvo F1 0.4096, recall 0.3333 y Average Precision 0.4322. "
         "La mejora operativa se acompaña de más falsas alarmas y mayor "
         "costo de entrenamiento, discutidos en las secciones 6 y 8.")
    para("Reproducibilidad: código Python y notebook publicados en GitHub, "
         "semilla 42, splits congelados, SHA256 y entorno Windows 11 "
         "con Python 3.13.5, scikit-learn 1.9.1 y MLflow 3.17.1.")
    header("2. Procedencia y diccionario del dataset")
    para("Fuente original: UCI AI4I 2020 Predictive Maintenance Dataset "
         "(DOI 10.24432/C5HS5C, licencia CC BY 4.0). Se parte del snapshot académico "
         "de siete columnas usado previamente en la Actividad 4, por lo que no se "
         "afirma que sea el archivo bruto completo de UCI.")
    table(["Variable", "Tipo y significado"], [
        ["Type", "Categoría de tipo de producto: L, M o H."],
        ["Air temperature", "Temperatura ambiente (K)."],
        ["Process temperature", "Temperatura del proceso (K)."],
        ["Rotational speed", "Velocidad de rotación (rpm)."],
        ["Torque", "Par o torque de operación (Nm)."],
        ["Tool wear", "Desgaste de herramienta (min)."],
        ["Machine failure", "Variable objetivo: 1=falla, 0=sin falla."],
    ], widths=[4.8*cm, 11.5*cm])
    header("3. Gestión, limpieza y versionamiento")
    para(f"Snapshot: {quality['filas']:,} registros; "
         f"clases {txt(quality['clases'])}. "
         f"Duplicados exactos: {quality['duplicados_exactos']}. "
         f"Nulos totales: {sum(quality['nulos_por_variable'].values())}. "
         f"Etiquetas de split: entrenamiento 7,000, validación 1,500, prueba 1,500. "
         "Si hubiera duplicados, el pipeline los detectaría y detendría la ejecución "
         "para revisar las particiones, en vez de introducir contaminación. "
         "La imputación, codificación y escala se ajustan dentro del pipeline de CV.")
    para(f"SHA256 snapshot inicial: {quality['sha256_origen']}.")
    para(f"SHA256 snapshot limpio: {quality['sha256_preparado']}.")
    para("Los valores extremos se identificaron mediante el método IQR; no se "
         "eliminaron observaciones extremas válidas sin criterio industrial documentado.")
    table(["Variable", "Posibles valores extremos (IQR)"], [
        [x, quality["iqr_outliers_por_variable"][x]]
        for x in quality["iqr_outliers_por_variable"]
    ], widths=[9*cm, 7.3*cm])
    figure("01_balance_clases.png",
           "Figura 1. Distribución de la clase objetivo: enfatiza la necesidad de métricas no dominadas por la clase mayoritaria.")
    figure("03_distribuciones.png",
           "Figura 3. Variables por clase: patrones descriptivos que no implican causalidad.")

    figure("02_nulos.png",
           "Figura 2. Auditoría de faltantes: ningún campo presenta nulos; no fue necesaria imputación en esta muestra, aunque el pipeline la soporta.")
    figure("04_tipos_falla.png",
           "Figura 4. Tasa de fallo según tipo de producto: diferencias descriptivas; no demuestran causalidad y deben contrastarse con datos reales.")
    para("ANÁLISIS DE CALIDAD: 339 de 10,000 registros son positivos (3.39%). "
         "La clase mayoritaria alcanza 96.61%, por lo que accuracy aislada "
         "es engañosa. No existen valores nulos ni duplicados exactos. "
         "El método IQR señala 418 posibles valores extremos en velocidad "
         "y 69 en torque. Se conservaron: un valor extremo operativo "
         "puede ser precisamente una señal de falla. Las figuras 1 a 4 "
         "documentan desbalance, completitud y patrones; las variables "
         "numéricas fueron tratadas dentro del Pipeline para evitar fuga.")
    header("4. Protocolo de entrenamiento y búsqueda")
    para("Cada algoritmo se entrenó solo con train, usando "
         "StratifiedKFold(n_splits=5, shuffle=True, random_state=42). "
         "GridSearchCV maximizó Average Precision. La regresión logística varió "
         "regularización C y class_weight; Random Forest varió árboles, profundidad "
         "y min_samples_leaf. Los hiperparámetros se determinaron con validación "
         "cruzada; los umbrales se eligieron con validación usando F2 y una "
         "restricción de precision >= 0.50 cuando resultara factible.")
    table(["Modelo", "AP de CV", "F2 validación", "Umbral"],
          [[row["Modelo"], f"{row['cv_average_precision']:.4f}",
            f"{row['validation_f2']:.4f}", f"{row['validation_threshold']:.2f}"]
           for _, row in comparison.iterrows()],
          widths=[5*cm, 3.8*cm, 4*cm, 3.5*cm])


    para("JUSTIFICACIÓN: con 3.39% de fallas, se optimizó Average Precision "
         "en la validación cruzada, en lugar de accuracy; posteriormente "
         "se seleccionó el umbral operativo maximizando F2 con precision "
         "mínima de 0.50 si existía configuración factible. "
         "Esto penaliza especialmente fallas no detectadas sin "
         "ocultar el incremento de alertas falsas.")
    table(["Modelo", "Hiperparámetros elegidos por Grid Search"], [
        [row["Modelo"], str(json.loads(row["best_params_json"]))]
        for _, row in comparison.iterrows()
    ], widths=[4.8*cm, 11.5*cm])
    para("La búsqueda evaluó 6 combinaciones de regresión logística "
         "(3 valores de C y 2 ponderaciones) y 8 combinaciones de "
         "Random Forest (2 árboles, 2 profundidades, 2 tamaños de hoja): "
         "14 candidatos por cinco folds, 70 ajustes de CV. "
         "El pipeline reentrenó cada mejor configuración sobre train; "
         "la validación externa ajustó el umbral sin utilizar test.")
    header("5. Experimentos reproducibles con MLflow")
    para(f"Se registraron {run_manifest['numero_runs']} runs "
         f"({run_manifest['runs_hijos']} combinaciones hijas) en el experimento "
         f"{txt(run_manifest['experimento'])}. Las ejecuciones padre incluyen "
         "los mejores modelos completos y sus resultados de validación y prueba; "
         "los hijos preservan parámetros y promedios de validación cruzada. "
         "Las tablas exportadas son consultables en evidencias/mlflow/.")

    para("Para la trazabilidad, cada candidato tiene run hijo con "
         "parámetros, media/desviación del Average Precision de CV "
         "y ranking; cada run padre conserva el pipeline completo, "
         "tablas, figuras y el resultado de prueba. El experimento "
         "se ejecutó en el servidor MLflow local; los 17 runs "
         "también se publican como CSV porque localhost es inaccesible "
         "para el evaluador. La evidencia visual se documenta abajo.")
    if missing_captures:
        para("<b>Advertencia:</b> faltan capturas de la interfaz real de MLflow: "
             + txt(", ".join(missing_captures)) + ". Este PDF es preliminar.")
    # Mostrar el listado completo primero, seguido de GridSearch, comparación y modelos.
    capture_sequence = [
        ("01_experimentos.png", "Evidencia MLflow 1a. Listado de runs: parte superior."),
        ("01_experimentos_02.png", "Evidencia MLflow 1b. Listado de runs: continuación."),
        ("01_experimentos_03.png", "Evidencia MLflow 1c. Listado de runs: tramo inferior."),
        ("02_gridsearch_parametros.png", "Evidencia MLflow 2. Hiperparámetros de Grid Search."),
        ("03_comparacion_runs.png", "Evidencia MLflow 3. Comparación entre modelos."),
        ("04_modelos_artefactos.png", "Evidencia MLflow 4. Modelos registrados y artefactos.")
    ]
    for filename, description in capture_sequence:
        file = EVIDENCE / filename
        if file.exists():
            from reportlab.lib.utils import ImageReader
            wp, hp = ImageReader(str(file)).getSize()
            width = 15.4*cm
            height = width * hp / wp
            if height > 10*cm:
                height = 10*cm
                width = height * wp / hp
            story.append(KeepTogether([
                Image(str(file), width=width, height=height),
                p(description, styles["CaptionA5"])
            ]))
    header("6. Desempeño y contraste de modelos sobre prueba")
    table(["Modelo", "Precisión", "Recall", "F1", "F2", "AP", "ROC-AUC"], [
        [row["Modelo"], f"{row['test_precision']:.3f}",
         f"{row['test_recall']:.3f}", f"{row['test_f1']:.3f}",
         f"{row['test_f2']:.3f}", f"{row['test_average_precision']:.3f}",
         f"{row['test_roc_auc']:.3f}"]
        for _, row in comparison.iterrows()
    ], widths=[4.6*cm, 2.0*cm, 2.0*cm, 1.7*cm, 1.7*cm, 2.1*cm, 2.2*cm])

    para("LECTURA CUANTITATIVA DE TEST: Random Forest detectó "
         "39 de 51 fallas; regresión logística detectó 17. "
         "Se redujeron los falsos negativos de 34 a 12 "
         "(22 fallas adicionales identificadas; 64.7% menos omisiones). "
         "En contraparte, las falsas alertas subieron de 15 a 33 "
         "(18 inspecciones potenciales adicionales). "
         "La precisión fue semejante (54.17% frente a 53.12%), "
         "mientras que el recall pasó de 33.33% a 76.47%. "
         "En test la mejora absoluta de F1 fue 22.45 puntos porcentuales.")
    table(["Modelo", "TP", "FN", "FP", "TN", "Balanced acc."], [
        [row["Modelo"], int(row["test_tp"]), int(row["test_fn"]),
         int(row["test_fp"]), int(row["test_tn"]),
         f"{row['test_balanced_accuracy']:.4f}"]
        for _, row in comparison.iterrows()
    ], widths=[4.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 3.8*cm])
    para("BASELINE: DummyClassifier alcanzó 96.6% de accuracy en test "
         "prediciendo siempre 'sin falla', pero recall = 0, F1 = 0 "
         "y Balanced Accuracy = 0.50. Por tanto, una accuracy "
         "superior a 96% no prueba utilidad del clasificador.")
    figure("05_06_comparacion_modelos.png",
           "Figuras 5 y 6. Comparación de validación y prueba; inspeccionar la posible sensibilidad al umbral.")
    figure("07_08_curvas_pr_roc.png",
           "Figuras 7 y 8. Precision-Recall y ROC, ambas necesarias por el desbalance del target.")
    figure("09_matrices_confusion_test.png",
           "Figura 9. Matrices de confusión para examinar falsos negativos y falsas alarmas.")

    header("7. Análisis estadístico e incertidumbre")
    lower, upper = float(bootstrap["IC95_inferior"]), float(bootstrap["IC95_superior"])
    para(f"Bootstrap pareado con 2,000 remuestreos sobre los mismos registros "
         f"test; delta F1 = RandomForest - LogisticRegression = "
         f"{bootstrap['delta_observada']:.4f}. "
         f"IC 95% percentil = [{lower:.4f}, {upper:.4f}]. "
         + ("El intervalo incluye cero: no se afirma ventaja estadística concluyente."
            if lower <= 0 <= upper else
            "El intervalo no incluye cero bajo este procedimiento descriptivo; "
            "no constituye validación industrial ni confirmación fuera de la muestra."))
    figure("10_bootstrap_f1.png", "Figura 10. Variación de la diferencia F1 por bootstrap pareado.")


    para("INCERTIDUMBRE: la diferencia F1 a favor de Random Forest "
         "fue 0.2245 (22.45 puntos); el bootstrap pareado percentil "
         "de 2,000 remuestreos produjo IC95 [0.0897, 0.3668]. "
         "El intervalo excluye cero bajo este procedimiento, "
         "pero los casos positivos son únicamente 51, pertenecen "
         "a un dataset sintético y no existe prueba temporal "
         "ni industrial; no se infiere superioridad universal.")
    header("8. Recursos, deuda técnica y enfoque MLOps")
    table(["Modelo", "Tiempo GridSearch (s)", "Latencia por registro (ms)"], [
        [row["Modelo"], f"{row['cv_fit_seconds']:.2f}",
         f"{row['test_ms_per_row']:.3f}"] for _, row in comparison.iterrows()
    ], widths=[6.2*cm, 5*cm, 5.1*cm])
    para("Deuda técnica: snapshots académicos sin datos temporales de equipos, "
         "ausencia de monitoreo de distribución (drift), degradación por máquina, "
         "validación prospectiva y cuantificación real de costo por falso negativo. "
         "Acciones MLOps: versionar datos/pipelines, registrar nuevos datasets "
         "y modelos en MLflow, definir reglas de promoción, monitorear métricas "
         "por segmento, revisar drift, auditar alertas y mantener reversión humana.")

    para("LECTURA DE EFICIENCIA Y GENERALIZACIÓN: Random Forest necesitó "
         "25.90 s para Grid Search frente a 4.48 s de regresión "
         "logística en el hardware observado (~5.8 veces más), "
         "y 0.0366 frente a 0.0067 ms por registro de prueba. "
         "La menor latencia de regresión logística no compensó "
         "la omisión de 34 fallas. La configuración ganadora "
         "de Random Forest obtuvo AP de entrenamiento cercano a 1.0, "
         "frente a 0.7347 de CV: señal de posible sobreajuste "
         "que exige validación externa antes de un despliegue.")
    para("SENSIBILIDAD ECONÓMICA ILUSTRATIVA (sin equivalencia en pesos): "
         "si una falsa alarma cuesta 1 unidad y una falla omitida "
         "10 unidades, el costo observado sería RF = "
         "33 + 10x12 = 153 y logística = 15 + 10x34 = 355. "
         "Los costos de una planta real se desconocen; "
         "estos números no justifican por sí solos invertir.")
    para("PROPUESTA MLOps POR ETAPAS: (1) publicar manifests SHA256 "
         "y modelo por versión, (2) automatizar pruebas y validación "
         "de schema, (3) monitorear drift de velocidad, torque "
         "y prevalencia de fallas, (4) medir recall/precision "
         "por máquina y periodo, (5) definir rollback y aprobación "
         "humana antes de cualquier acción sobre maquinaria.")
    header("9. Conclusión crítica")
    para(f"Con el protocolo preestablecido se seleccionó {txt(decisions['modelo_seleccionado'])} "
         "sobre validación, antes de evaluar test. El modelo seleccionado debe "
         "entenderse como una línea base académica sobre datos sintéticos, no un "
         "sistema listo para anticipar fallas en una planta. Los resultados, sus "
         "intervalos, y los runs MLflow permiten evaluar reproducibilidad y "
         "sostener decisiones sin exagerar su generalización.")

    header("10. Declaración de uso de inteligencia artificial")
    table(["Herramienta", "Uso declarado"], [
        ["ChatGPT (OpenAI)", "Diseño experimental, asistencia en scripts Python, "
         "documentación, depuración y control de versiones; "
         "las métricas proceden de ejecuciones reales verificables."],
        ["Responsabilidad humana", "Configuración, ejecución y revisión de resultados, "
         "interpretación académica y publicación final."],
    ], widths=[4.5*cm, 11.8*cm])

    header("11. Evidencias, reproducibilidad y límites")
    para("Repositorio: datos/datos_ini, datos/datos_limp, fuentes/train.py, "
         "fuentes/entrena.ipynb, resultados/, evidencias/mlflow/ "
         "y docs/Reporte_Tecnico_Actividad5.pdf. "
         "Para reproducir: instalar requirements.txt, "
         "iniciar MLflow en 127.0.0.1:5000, ejecutar "
         "python -m fuentes.train y después "
         "python -m fuentes.export_evidencias. "
         "Los umbrales quedan congelados antes de calcular prueba.")
    para("La actividad cubre un flujo académico de MLOps; NO se valida "
         "tiempo hasta fallo, sensores reales, desgaste por activo, "
         "causalidad, deriva futura, eficiencia bajo carga productiva "
         "ni costos reales de mantenimiento. "
         "No se almacenan tokens ni el SQLite privado en GitHub.")

    header("12. Bibliografía y recursos")
    para("UCI AI4I 2020, DOI https://doi.org/10.24432/C5HS5C. "
         "Scikit-learn: https://scikit-learn.org/. "
         "MLflow Tracking: https://mlflow.org/docs/latest/ml/tracking/. "
         "Repositorio y código: https://github.com/isaaccg88/PredictiveMaintenance-AI-Activity5")

    def decorate(canvas, document):
        canvas.saveState()
        w, h = A4
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(.7)
        canvas.line(2.15*cm, h-1.65*cm, w-2.15*cm, h-1.65*cm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(2.15*cm, 1.3*cm, "Predictive Maintenance AI | Actividad 5")
        canvas.drawRightString(w-2.15*cm, 1.3*cm, f"Página {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print("Reporte PDF generado:", OUTPUT)
    if missing_captures:
        print("ADVERTENCIA: PDF preliminar hasta incorporar capturas reales:", missing_captures)
    return OUTPUT


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-screenshots", action="store_true",
                        help="No generar el PDF final si falta alguna captura obligatoria.")
    args = parser.parse_args()
    document(require_screenshots=args.require_screenshots)
