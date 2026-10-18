"""Exporta metadatos de experimentos MLflow a CSV y JSON auditables.

Las capturas del panel son capturadas manualmente por el estudiante.
No se exporta la base SQLite ni se cargan modelos binarios en GitHub.
"""
from __future__ import annotations
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient
from fuentes.train import EXPERIMENT_NAME
from fuentes.datos_prep import ROOT

OUT = ROOT / "evidencias" / "mlflow"


def export_runs(tracking_uri: str = "http://127.0.0.1:5000") -> pd.DataFrame:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("No existe el experimento: ejecuta primero fuentes.train")
    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    rows = []
    for run in runs:
        data = {
            "run_id": run.info.run_id,
            "experiment_id": experiment.experiment_id,
            "estado": run.info.status,
            "nombre": run.data.tags.get("mlflow.runName", ""),
            "parent_run_id": run.data.tags.get("mlflow.parentRunId", ""),
            "algoritmo": run.data.tags.get("algoritmo", ""),
            "es_mejor": run.data.tags.get("is_best", ""),
            "started_ms": run.info.start_time,
            "ended_ms": run.info.end_time,
        }
        data.update({"param_" + k: str(v) for k, v in run.data.params.items()})
        data.update({"metric_" + k: float(v) for k, v in run.data.metrics.items()})
        rows.append(data)
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "runs_exportados.csv", index=False)
    manifest = {
        "experimento": EXPERIMENT_NAME,
        "id_experimento": experiment.experiment_id,
        "fecha_exportacion": datetime.now(timezone.utc).isoformat(),
        "numero_runs": len(rows),
        "runs_hijos": int((frame["parent_run_id"].fillna("") != "").sum())
        if not frame.empty else 0,
        "campos": frame.columns.tolist(),
        "nota": "Metadatos MLflow auditables; capturas del panel pendientes de incorporación manual.",
    }
    (OUT / "runs_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Experimento:", EXPERIMENT_NAME)
    print("Runs exportados:", len(frame))
    print("Archivo:", OUT / "runs_exportados.csv")
    print("Capturas reales pendientes: consultar evidencias/mlflow/README.md")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri",
                        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    args = parser.parse_args()
    export_runs(args.tracking_uri)
