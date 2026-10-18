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
           "Figura 2. Variables por clase: patrones descriptivos que no implican causalidad.")
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

    header("5. Experimentos reproducibles con MLflow")
    para(f"Se registraron {run_manifest['numero_runs']} runs "
         f"({run_manifest['runs_hijos']} combinaciones hijas) en el experimento "
         f"{txt(run_manifest['experimento'])}. Las ejecuciones padre incluyen "
         "los mejores modelos completos y sus resultados de validación y prueba; "
         "los hijos preservan parámetros y promedios de validación cruzada. "
         "Las tablas exportadas son consultables en evidencias/mlflow/.")
    if missing_captures:
        para("<b>Advertencia:</b> faltan capturas de la interfaz real de MLflow: "
             + txt(", ".join(missing_captures)) + ". Este PDF es preliminar.")
    for i, filename in enumerate(REQUIRED_CAPTURES, start=1):
        file = EVIDENCE / filename
        if file.exists():
            from reportlab.lib.utils import ImageReader
            wp, hp = ImageReader(str(file)).getSize()
            w = 15.4*cm
            story.append(Image(str(file), width=w, height=min(10*cm, w*hp/wp)))
            caption(f"Evidencia MLflow {i}. Captura real del seguimiento experimental.")
    header("6. Desempeño y contraste de modelos sobre prueba")
    table(["Modelo", "Precisión", "Recall", "F1", "F2", "AP", "ROC-AUC"], [
        [row["Modelo"], f"{row['test_precision']:.3f}",
         f"{row['test_recall']:.3f}", f"{row['test_f1']:.3f}",
         f"{row['test_f2']:.3f}", f"{row['test_average_precision']:.3f}",
         f"{row['test_roc_auc']:.3f}"]
        for _, row in comparison.iterrows()
    ], widths=[4.6*cm, 2.0*cm, 2.0*cm, 1.7*cm, 1.7*cm, 2.1*cm, 2.2*cm])
    figure("05_06_comparacion_modelos.png",
           "Figura 3. Comparación de validación y prueba; inspeccionar la posible sensibilidad al umbral.")
    figure("07_08_curvas_pr_roc.png",
           "Figura 4. Precision-Recall y ROC, ambas necesarias por el desbalance del target.")
    figure("09_matrices_confusion_test.png",
           "Figura 5. Matrices de confusión para examinar falsos negativos y falsas alarmas.")

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
    figure("10_bootstrap_f1.png", "Figura 6. Variación de la diferencia F1 por bootstrap pareado.")

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
    header("11. Bibliografía y recursos")
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
