"""Actividad 5 — ML clásico + GridSearchCV + MLflow Tracking 3.

Desde la raíz:
    python -m fuentes.train
    python -m fuentes.train --tracking-uri http://127.0.0.1:5000

NO consulta el conjunto test hasta concluir GridSearchCV y selección
de modelos/umbrales con train y validation. No abre el servidor MLflow;
el usuario debe mantenerlo encendido en otra ventana.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fuentes.datos_prep import (
    FEATURES, FIG, NUMERIC, OUT, ROOT, TARGET, prepare_data,
)
from fuentes.evaluation import scores, select_threshold

SEED = 42
EXPERIMENT_NAME = "PredictiveMaintenance_Actividad5"
THRESHOLD_PRECISION_MIN = 0.50


def pipeline_and_grid(model: str):
    # Toda transformación aprendida ocurre dentro de cada fold del pipeline.
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, ["Type"])],
        remainder="drop",
    )
    if model == "LogisticRegression":
        estimator = LogisticRegression(
            random_state=SEED, solver="liblinear", max_iter=2000
        )
        param_grid = {
            "est__C": [0.1, 1.0, 10.0],
            "est__class_weight": [None, "balanced"],
        }
    elif model == "RandomForest":
        estimator = RandomForestClassifier(
            random_state=SEED, n_jobs=1, class_weight="balanced_subsample"
        )
        param_grid = {
            "est__n_estimators": [120, 240],
            "est__max_depth": [8, None],
            "est__min_samples_leaf": [1, 5],
        }
    else:
        raise ValueError(model)
    return Pipeline([("prep", preprocess), ("est", estimator)]), param_grid


def save_visualizations(results: pd.DataFrame, predictions: dict,
                        y_test: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    results.set_index("Modelo")[["validation_f2", "test_f2", "test_f1"]].plot(
        kind="bar", ax=axes[0], rot=0)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Puntuación")
    axes[0].set_title("Figura 5. Desempeño validación y test")
    results.set_index("Modelo")[["test_recall", "test_precision"]].plot(
        kind="bar", ax=axes[1], rot=0)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Figura 6. Trade-off recall/precision")
    fig.tight_layout()
    fig.savefig(FIG / "05_06_comparacion_modelos.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, d in predictions.items():
        PrecisionRecallDisplay.from_predictions(
            y_test, d["probabilities"], ax=axes[0], name=name
        )
        RocCurveDisplay.from_predictions(
            y_test, d["probabilities"], ax=axes[1], name=name
        )
    axes[0].set_title("Figura 7. Curvas precision-recall test")
    axes[1].set_title("Figura 8. Curvas ROC test")
    fig.tight_layout()
    fig.savefig(FIG / "07_08_curvas_pr_roc.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4))
    for ax, (name, d) in zip(np.atleast_1d(axes), predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test, d["labels"], labels=[0, 1], ax=ax,
            colorbar=False, values_format="d",
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(FIG / "09_matrices_confusion_test.png", dpi=170)
    plt.close(fig)


def _metadata(quality):
    return {
        "dataset_sha256": quality["sha256_origen"],
        "splits_sha256": quality["sha256_splits"],
        "prepared_sha256": quality["sha256_preparado"],
        "seed": SEED,
        "cv_folds": 5,
        "scoring_cv": "average_precision",
        "data_source": "UCI AI4I 2020; subset 7 columnas Actividad4",
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "mlflow_version": mlflow.__version__,
        "platform": platform.platform(),
    }


def run(tracking_uri: str = "http://127.0.0.1:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    # Comprobar conectividad antes de consumir CPU durante el Grid Search.
    try:
        MlflowClient().search_experiments(max_results=1)
    except Exception as e:
        raise RuntimeError(
            f"MLflow no responde en {tracking_uri}. "
            "Inicia el servidor local en otra ventana PowerShell o usa "
            "--tracking-uri sqlite:///ruta/a/mlflow.db en Colab."
        ) from e
    mlflow.set_experiment(EXPERIMENT_NAME)

    source_data, split, quality = prepare_data(make_figures=True)
    by_id = source_data.set_index("indice_original")
    # La selección por particiones conserva exclusivamente los IDs asignados.
    ids = {k: split.loc[split["conjunto"] == k, "indice_original"].to_numpy()
           for k in ("train", "validation", "test")}
    train_data = by_id.loc[ids["train"]]
    val_data = by_id.loc[ids["validation"]]
    test_data = by_id.loc[ids["test"]]
    X_train, y_train = train_data[FEATURES], train_data[TARGET]
    X_val, y_val = val_data[FEATURES], val_data[TARGET]
    X_test, y_test = test_data[FEATURES], test_data[TARGET]

    print("Fuente:", quality["dataset"], "| Registros", quality["filas"])
    print("Train/validation/test:", len(X_train), len(X_val), len(X_test))
    print("MLflow:", tracking_uri, "| Experimento:", EXPERIMENT_NAME)

    metadata = _metadata(quality)
    results = []
    grids = []
    thresholds = []
    fitted = {}
    run_ids = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Baseline: tasa de clase mayoritaria, sin Grid Search ni filtraciones.
    # No participa en la elección entre los dos algoritmos exigidos.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_val = scores(y_val, dummy.predict_proba(X_val)[:, 1], 0.5)
    with mlflow.start_run(run_name="DummyClassifier_baseline") as baseline_run:
        baseline_run_id = baseline_run.info.run_id
        mlflow.set_tags({"actividad": "5", "algoritmo": "DummyClassifier",
                         "fase": "baseline", "version_dataset": "A5-v1"})
        mlflow.log_params({"modelo": "DummyClassifier", "strategy": "prior",
                           "umbral": 0.5, "semilla": SEED})
        mlflow.log_metrics({"validation_" + k: float(v)
                            for k, v in dummy_val.items() if k != "threshold"})

    for name in ("LogisticRegression", "RandomForest"):
        model, grid_params = pipeline_and_grid(name)
        grid = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            scoring="average_precision",
            cv=cv,
            n_jobs=2,
            refit=True,
            return_train_score=True,
            error_score="raise",
        )
        print(f"\nGrid Search {name}: iniciando 5-fold estratificado...")
        with mlflow.start_run(run_name=f"{name}_best") as outer:
            run_ids[name] = outer.info.run_id
            mlflow.set_tags({
                "actividad": "5", "algoritmo": name,
                "dataset": "AI4I2020", "fase": "entrenamiento_gridsearch",
                "version_dataset": "A5-v1"
            })
            mlflow.log_params({
                "modelo": name, "semilla": SEED, "cv_folds": 5,
                "criterio_cv": "average_precision",
                "datos_entrenamiento": len(X_train),
                "datos_validacion": len(X_val),
                "datos_prueba_reservados": len(X_test),
                "min_precision_validation": THRESHOLD_PRECISION_MIN,
                "dataset_sha256": quality["sha256_origen"],
                "splits_sha256": quality["sha256_splits"],
            })
            tic = perf_counter()
            grid.fit(X_train, y_train)
            fit_seconds = perf_counter() - tic
            mlflow.log_metric("fit_seconds", fit_seconds)
            mlflow.log_metric("cv_best_average_precision", float(grid.best_score_))
            for k, v in grid.best_params_.items():
                mlflow.log_param("best_" + k.replace("est__", ""), str(v))

            all_cv = pd.DataFrame(grid.cv_results_)
            all_cv.insert(0, "Modelo", name)
            grids.append(all_cv.copy())
            path = OUT / f"gridsearch_{name}.csv"
            all_cv.to_csv(path, index=False)
            mlflow.log_artifact(str(path), artifact_path="cv")

            # MLflow: jerarquía padre / hijo, una ejecución por combinación.
            for ix, row in all_cv.iterrows():
                with mlflow.start_run(run_name=f"{name}_candidate_{ix:02d}", nested=True):
                    mlflow.set_tags({
                        "tipo": "combinacion_CV",
                        "algoritmo": name,
                        "is_best": str(ix == grid.best_index_),
                    })
                    for key, value in row["params"].items():
                        mlflow.log_param(key.replace("est__", ""), str(value))
                    mlflow.log_metrics({
                        "cv_mean_average_precision": float(row["mean_test_score"]),
                        "cv_std_average_precision": float(row["std_test_score"]),
                        "cv_mean_train_AP": float(row["mean_train_score"]),
                        "cv_mean_fit_time_s": float(row["mean_fit_time"]),
                        "cv_rank": float(row["rank_test_score"]),
                    })

            fitted[name] = grid.best_estimator_
            pval = fitted[name].predict_proba(X_val)[:, 1]
            threshold, grid_table, feasible = select_threshold(
                y_val, pval, min_precision=THRESHOLD_PRECISION_MIN
            )
            thresholds.extend({"Modelo": name, **s} for s in grid_table)
            mlflow.log_metrics({
                "validation_" + k: float(v)
                for k, v in threshold.items()
                if k != "threshold"
            })
            mlflow.log_param("umbral_validacion", threshold["threshold"])
            mlflow.set_tag("cumple_min_precision_validacion", str(feasible))
            # Serializa el pipeline completo (imputación + transformación + modelo).
            mlflow.sklearn.log_model(
                sk_model=fitted[name], name=f"{name}_pipeline",
                input_example=X_train.head(2),
                registered_model_name=f"PM_Actividad5_{name}",
            )
            mlflow.log_artifact(str(OUT / "calidad_datos.json"), artifact_path="data")
            mlflow.log_artifact(str(OUT / "datos_manifest.json"), artifact_path="data")
            mlflow.log_artifacts(str(FIG), artifact_path="figuras_EDA")

            print(f"{name}: CV AP={grid.best_score_:.4f}, "
                  f"validacion F2={threshold['f2']:.4f}, "
                  f"umbral={threshold['threshold']}, "
                  f"run={outer.info.run_id}")
            results.append({
                "Modelo": name,
                "cv_average_precision": float(grid.best_score_),
                "cv_fit_seconds": fit_seconds,
                "validation_f2": threshold["f2"],
                "validation_recall": threshold["recall"],
                "validation_precision": threshold["precision"],
                "validation_average_precision": threshold["average_precision"],
                "validation_threshold": threshold["threshold"],
                "cumple_precision_minima": feasible,
                "mlflow_run_id": outer.info.run_id,
                "best_params_json": json.dumps(grid.best_params_, sort_keys=True),
            })

    # SELECCIÓN CONGELADA ANTES DE CUALQUIER CONSULTA AL TEST.
    ordering = sorted(
        results,
        key=lambda d: (d["cumple_precision_minima"], d["validation_f2"],
                       d["validation_recall"], d["validation_average_precision"]),
        reverse=True,
    )
    champion = ordering[0]["Modelo"]
    champion_threshold = ordering[0]["validation_threshold"]
    print("\nMODELO ELEGIDO SOLO CON VALIDACION:", champion, champion_threshold)
    decisions = {
        "modelo_seleccionado": champion,
        "umbral_congelado": champion_threshold,
        "criterio": "F2 en validación con precision>=0.50 si es factible",
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "runs_mlflow": {**run_ids, "DummyClassifier": baseline_run_id},
    }
    (OUT / "seleccion_congelada.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # PRUEBA FINAL: ningún GridSearchCV ni selección utiliza test.
    baseline_test = scores(y_test, dummy.predict_proba(X_test)[:, 1], 0.5)
    with mlflow.start_run(run_id=baseline_run_id):
        mlflow.log_metrics({"test_" + k: float(v)
                            for k, v in baseline_test.items() if k != "threshold"})
    (OUT / "baseline_dummy.json").write_text(
        json.dumps({"run_id": baseline_run_id,
                    "validation": dummy_val, "test": baseline_test},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    pred_df = pd.DataFrame({
        "indice_original": ids["test"], "valor_real": y_test.to_numpy()
    })
    output = {}
    for result in results:
        name = result["Modelo"]
        tic = perf_counter()
        probabilities = fitted[name].predict_proba(X_test)[:, 1]
        infer_seconds = perf_counter() - tic
        threshold = float(result["validation_threshold"])
        metric = scores(y_test, probabilities, threshold)
        result.update({"test_" + k: v for k, v in metric.items()})
        result["test_inference_seconds"] = infer_seconds
        result["test_ms_per_row"] = 1000 * infer_seconds / len(X_test)
        pred_df[name + "_probabilidad"] = probabilities
        pred_df[name + "_prediccion"] = (probabilities >= threshold).astype(int)
        output[name] = {
            "probabilities": probabilities,
            "labels": pred_df[name + "_prediccion"],
            "threshold": threshold,
        }
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_metrics({
                "test_" + k: float(v) for k, v in metric.items()
                if k != "threshold"
            })
            mlflow.log_metrics({
                "test_inference_seconds": infer_seconds,
                "test_ms_per_row": result["test_ms_per_row"],
            })
            mlflow.set_tag("seleccionado_en_validacion", str(name == champion))

    # Bootstrap pareado: las dos predicciones se remuestrean sobre los mismos
    # registros del test; es un análisis descriptivo de incertidumbre.
    rng = np.random.default_rng(SEED)
    a, b = output["LogisticRegression"], output["RandomForest"]
    y_array = y_test.to_numpy(dtype=int)
    from sklearn.metrics import f1_score
    delta_obs = f1_score(y_array, b["labels"], zero_division=0) - (
        f1_score(y_array, a["labels"], zero_division=0)
    )
    delta_boot = []
    for _ in range(2000):
        ids_b = rng.integers(0, len(y_array), size=len(y_array))
        delta_boot.append(
            f1_score(y_array[ids_b], np.asarray(b["labels"])[ids_b],
                     zero_division=0)
            - f1_score(y_array[ids_b], np.asarray(a["labels"])[ids_b],
                       zero_division=0)
        )
    ci_low, ci_high = np.quantile(delta_boot, [0.025, 0.975])
    boot = pd.DataFrame([{
        "Comparacion": "RandomForest - LogisticRegression",
        "metrica": "Diferencia F1 (test)",
        "delta_observada": float(delta_obs),
        "IC95_inferior": float(ci_low),
        "IC95_superior": float(ci_high),
        "repeticiones": 2000,
        "semilla": SEED,
        "metodo": "Bootstrap pareado percentil",
        "contiene_cero": bool(ci_low <= 0 <= ci_high),
    }])
    boot.to_csv(OUT / "bootstrap_pareado.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(delta_boot, bins=35)
    ax.axvline(0, linestyle="--", linewidth=1.5, label="Diferencia cero")
    ax.axvline(delta_obs, linestyle="-", linewidth=1.5,
               label=f"Observada = {delta_obs:.3f}")
    ax.set_title("Figura 10. Bootstrap pareado de diferencia F1 (2,000)")
    ax.set_xlabel("F1(Random Forest) - F1(Reg. Logistica)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "10_bootstrap_f1.png", dpi=170)
    plt.close(fig)

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUT / "comparacion_modelos.csv", index=False)
    pred_df.to_csv(OUT / "predicciones_test.csv", index=False)
    pd.concat(grids, ignore_index=True).to_csv(OUT / "gridsearch_cv.csv", index=False)
    pd.DataFrame(thresholds).to_csv(OUT / "umbrales_validacion.csv", index=False)
    save_visualizations(comparison, output, y_test)
    for name, run_id in run_ids.items():
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifacts(str(FIG), artifact_path="figuras_finales")
            for file in ("bootstrap_pareado.csv", "comparacion_modelos.csv", "predicciones_test.csv",
                         "umbrales_validacion.csv", "seleccion_congelada.json"):
                mlflow.log_artifact(str(OUT / file), artifact_path="resultados")
    (OUT / "resumen_experimento.json").write_text(
        json.dumps({
            "campeon": champion, "umbral": champion_threshold, "metadata": metadata,
            "resultados": results, "bootstrap": boot.to_dict(orient="records")[0], "baseline": baseline_test, "experiment_name": EXPERIMENT_NAME,
            "tracking_uri": tracking_uri
        }, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("\n=== RESULTADOS (reales, generados en esta ejecución) ===")
    print(comparison[[
        "Modelo", "cv_average_precision", "validation_f2", "test_f1",
        "test_recall", "test_precision", "test_average_precision",
        "test_ms_per_row"
    ]].round(4).to_string(index=False))
    print("Panel MLflow:", tracking_uri)
    print("Artefactos en:", OUT)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        help="Por defecto, MLflow local de Windows. Para Colab usar sqlite:////content/mlflow.db",
    )
    args = parser.parse_args()
    run(args.tracking_uri)


if __name__ == "__main__":
    main()
