"""Validación, preparación trazable y EDA; no ajusta transformers sobre test."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datos" / "datos_ini" / "ai4i2020.csv"
SPLITS = ROOT / "datos" / "splits.csv"
CLEAN = ROOT / "datos" / "datos_limp" / "ai4i2020_limpio.csv"
OUT = ROOT / "resultados"
FIG = OUT / "figuras"
NUMERIC = [
    "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear"
]
FEATURES = ["Type", *NUMERIC]
TARGET = "Machine failure"
EXPECTED = ["Type", *NUMERIC, TARGET]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare_data(make_figures: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    CLEAN.parent.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(RAW)
    if list(source.columns) != EXPECTED:
        raise ValueError(f"Esquema diferente del esperado: {source.columns.tolist()}")
    split = pd.read_csv(SPLITS)
    if set(split.columns) != {"indice_original", "conjunto", "machine_failure"}:
        raise ValueError("Esquema de splits no válido")
    if source.shape[0] != 10_000 or len(split) != 10_000:
        raise ValueError("Se esperaba el snapshot de 10,000 filas del protocolo.")
    if split["indice_original"].isna().any() or split["indice_original"].duplicated().any():
        raise ValueError("Índices repetidos o faltantes en las particiones.")
    if set(split["indice_original"]) != set(range(len(source))):
        raise ValueError("Particiones incompletas.")
    if not set(split["conjunto"]).issubset({"train", "validation", "test"}):
        raise ValueError("Etiquetas de partición no válidas.")
    data = source.copy()
    data.insert(0, "indice_original", np.arange(len(data), dtype=int))
    data["Type"] = data["Type"].astype("string").str.strip().str.upper()
    invalid_categories = int((~data["Type"].isin(["L", "M", "H"])).fillna(False).sum())
    data.loc[~data["Type"].isin(["L", "M", "H"]).fillna(False), "Type"] = pd.NA
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # El objetivo no se imputa; valores inválidos impiden un experimento correcto.
    target = pd.to_numeric(data[TARGET], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all():
        raise ValueError("Variable objetivo inválida; revisar origen.")
    data[TARGET] = target.astype(int)
    duplicates = int(data.drop(columns=["indice_original"]).duplicated().sum())
    # Conservar índices inmutables por reproducibilidad de splits.
    # Si aparecieran duplicados, detener el proceso y resolver explícitamente
    # antes de generar particiones nuevas para evitar data leakage.
    if duplicates:
        raise ValueError(f"Se detectaron {duplicates} duplicados exactos. "
                         "Revisar y versionar nuevo split antes de entrenar.")
    if not np.array_equal(
        data.set_index("indice_original").loc[split["indice_original"], TARGET].to_numpy(),
        split["machine_failure"].to_numpy(),
    ):
        raise ValueError("Fuga/errores de etiquetas: splits no corresponden al origen.")
    counts = split["conjunto"].value_counts().to_dict()
    if counts != {"train": 7000, "validation": 1500, "test": 1500}:
        raise ValueError(f"Proporciones de particiones inesperadas: {counts}")
    outliers = {}
    for column in NUMERIC:
        q1, q3 = data[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[column] = int(((data[column] < q1 - 1.5 * iqr)
                                | (data[column] > q3 + 1.5 * iqr)).sum())
    missing = {name: int(value) for name, value in data.isna().sum().items()}
    quality = {
        "dataset": "AI4I 2020, snapshot academico de siete columnas",
        "origen": "Actividad 4, data/ai4i2020.csv (derivado de UCI 601)",
        "licencia": "CC BY 4.0",
        "filas": int(len(data)),
        "columnas_predictoras": FEATURES,
        "clases": {str(k): int(v) for k, v in data[TARGET].value_counts().items()},
        "duplicados_exactos": duplicates,
        "categoria_tipo_invalida": invalid_categories,
        "nulos_por_variable": missing,
        "iqr_outliers_por_variable": outliers,
        "split": counts,
        "politica": "sin borrar registros validos; imputation, onehot y scaler solo en pipeline por fold",
        "sha256_origen": sha256(RAW),
        "sha256_splits": sha256(SPLITS),
    }
    # Evita convertir a string por accidente todos los tipos en pandas.
    data.to_csv(CLEAN, index=False, float_format="%.10g")
    quality["sha256_preparado"] = sha256(CLEAN)
    (OUT / "calidad_datos.json").write_text(
        json.dumps(quality, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "datos_manifest.json").write_text(
        json.dumps({k: quality[k] for k in (
            "dataset", "filas", "sha256_origen", "sha256_splits", "sha256_preparado", "politica")},
            indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if make_figures:
        _plots(data, missing)
    return data, split, quality


def _plots(data: pd.DataFrame, missing: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = data[TARGET].value_counts().sort_index()
    ax.bar(["Sin falla", "Con falla"], counts.values)
    for i, count in enumerate(counts):
        ax.text(i, count, f"{count:,}", ha="center", va="bottom")
    ax.set_title("Figura 1. Desbalance del target (AI4I 2020)")
    ax.set_ylabel("Número de registros")
    fig.tight_layout()
    fig.savefig(FIG / "01_balance_clases.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    var = [x for x in FEATURES if x in missing]
    ax.bar(range(len(var)), [missing[x] for x in var])
    ax.set_xticks(range(len(var)), var, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Valores faltantes")
    ax.set_title("Figura 2. Diagnóstico de valores faltantes")
    fig.tight_layout()
    fig.savefig(FIG / "02_nulos.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, col in zip(axes.flat, NUMERIC):
        for label, group in data.groupby(TARGET):
            ax.hist(group[col].dropna(), bins=25, histtype="step",
                    density=True, label=f"Falla={label}")
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("Figura 3. Distribuciones de variables por clase")
    fig.tight_layout()
    fig.savefig(FIG / "03_distribuciones.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    type_table = pd.crosstab(data["Type"], data[TARGET], normalize="index")
    type_table.plot(kind="bar", ax=ax, rot=0)
    ax.set_title("Figura 4. Tasa de falla por tipo de producto")
    ax.set_ylabel("Proporción por tipo")
    fig.tight_layout()
    fig.savefig(FIG / "04_tipos_falla.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    _, _, quality = prepare_data()
    print(json.dumps(quality, ensure_ascii=False, indent=2))
