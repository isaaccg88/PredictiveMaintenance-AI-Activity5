"""Descarga y verifica una copia del CSV ORIGINAL (14 columnas) desde UCI dataset 601.

No sobrescribe el snapshot académico ni realiza reentrenamiento.
Uso:
    python -m fuentes.verificar_fuente_uci

Fuente oficial: https://doi.org/10.24432/C5HS5C (CC BY 4.0).
"""
from __future__ import annotations
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fuentes.datos_prep import ROOT, RAW, TARGET, NUMERIC, sha256

UCI_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
UCI_CSV = ROOT / "datos/datos_ini/uci_ai4i2020_original.csv"
MANIFEST = ROOT / "docs/uci_source_manifest.json"

UCI_NAMES = {
    "Type": "Type",
    "Air temperature": "Air temperature [K]",
    "Process temperature": "Process temperature [K]",
    "Rotational speed": "Rotational speed [rpm]",
    "Torque": "Torque [Nm]",
    "Tool wear": "Tool wear [min]",
    "Machine failure": "Machine failure",
}
IDENTIFIERS = ["UDI", "Product ID"]
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def fetch_and_verify() -> dict:
    if UCI_CSV.exists():
        payload = UCI_CSV.read_bytes()
        origin = "CSV original ya versionado en datos/datos_ini/"
    else:
        with urllib.request.urlopen(UCI_URL, timeout=90) as response:
            bundle = response.read()
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            csv_names = [n for n in archive.namelist()
                         if Path(n).name.lower() == "ai4i2020.csv"]
            if len(csv_names) != 1:
                raise RuntimeError(f"CSV esperado no encontrado: {archive.namelist()}")
            payload = archive.read(csv_names[0])
        origin = "descarga oficial UCI 601"
    raw_uci = pd.read_csv(io.BytesIO(payload))
    snapshot = pd.read_csv(RAW)
    if raw_uci.shape != (10_000, 14):
        raise ValueError(f"Esquema UCI inesperado: {raw_uci.shape}")
    if not all(k in raw_uci.columns for k in
               [*UCI_NAMES.values(), *IDENTIFIERS, *FAILURE_MODES]):
        raise ValueError(f"Columnas oficiales no encontradas: {raw_uci.columns.tolist()}")
    if len(snapshot) != 10_000 or list(snapshot.columns) != list(UCI_NAMES):
        raise ValueError("Snapshot de la Actividad 4 fuera del esquema esperado")
    errors = {}
    for academic, official in UCI_NAMES.items():
        a = snapshot[academic]
        b = raw_uci[official]
        if academic == "Type":
            match = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        else:
            match = np.isclose(pd.to_numeric(a).to_numpy(dtype=float),
                               pd.to_numeric(b).to_numpy(dtype=float),
                               atol=1e-9, rtol=0)
        errors[academic] = int((~match).sum())
    if any(errors.values()):
        raise ValueError(f"El snapshot no coincide FILA A FILA con UCI: {errors}")
    # Se excluyen los modos de falla para evitar fuga de etiqueta;
    # UDI/Product ID son identificadores y no predictores operativos.
    sha = hashlib.sha256(payload).hexdigest()
    UCI_CSV.parent.mkdir(parents=True, exist_ok=True)
    UCI_CSV.write_bytes(payload)
    manifest = {
        "fuente": "UCI AI4I 2020 Predictive Maintenance Dataset",
        "doi": "10.24432/C5HS5C",
        "licencia": "CC BY 4.0",
        "url_descarga": UCI_URL,
        "metodo": origin,
        "fecha_verificacion_utc": datetime.now(timezone.utc).isoformat(),
        "archivo_original_uci": str(UCI_CSV.relative_to(ROOT)),
        "original_filas": int(raw_uci.shape[0]),
        "original_columnas": int(raw_uci.shape[1]),
        "sha256_original_uci": sha,
        "sha256_snapshot_7_cols": sha256(RAW),
        "comparacion_fila_a_fila": "APROBADA",
        "diferencias_por_columna": errors,
        "mapa_columnas_seleccionadas": UCI_NAMES,
        "excluidas_por_identificador": IDENTIFIERS,
        "excluidas_por_fuga_de_objetivo": FAILURE_MODES,
        "nota": "El dataset original es sintético; conservar el snapshot/particiones/experimentos existentes sin cambios.",
    }
    MANIFEST.parent.mkdir(exist_ok=True, parents=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "filas":manifest["original_filas"],"columnas":manifest["original_columnas"],
        "comparacion":manifest["comparacion_fila_a_fila"],
        "diferencias":errors,"sha256":sha,
    },indent=2))
    return manifest


if __name__ == "__main__":
    fetch_and_verify()
