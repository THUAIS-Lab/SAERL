"""Fit a difficulty regressor and produce the shared curriculum input table.

Input is a tabular Parquet with one row per sample, explicit f_* feature
columns, lengths, and a nullable difficulty label. Neither clustering nor
prediction uses target labels as features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LENGTH_COLUMNS = ("prompt_len_tok", "sol_len_tok", "full_len_tok")


def fit_prediction_table(
    frame: pd.DataFrame,
    *,
    backend: str,
    feature_prefix: str = "f_",
    label_column: str = "difficulty_true",
    n_clusters: int = 10,
    seed: int = 42,
    labeled_samples: int = 3000,
    shrinkage: float = 20.0,
) -> tuple[pd.DataFrame, object, dict[str, object]]:
    """Train on labeled rows; predict and cluster every row using features only."""
    feature_columns = sorted(c for c in frame if c.startswith(feature_prefix))
    raw = frame[feature_columns].to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="median")
    features = imputer.fit_transform(raw)
    labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    candidates = np.flatnonzero(np.isfinite(labels))
    labeled = np.zeros(len(labels), dtype=bool)
    chosen = np.random.default_rng(seed).choice(candidates, size=min(labeled_samples, len(candidates)), replace=False)
    labeled[chosen] = True
    labels[~labeled] = np.nan

    if backend == "elasticnet":
        regressor = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=20_000, random_state=seed),
        )
    elif backend == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install lightgbm to use --backend lightgbm") from exc
        regressor = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, verbosity=-1, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
    regressor.fit(features[labeled], labels[labeled])
    prediction = np.asarray(regressor.predict(features), dtype=np.float64)

    cluster_scaler = StandardScaler()
    scaled = cluster_scaler.fit_transform(features)
    clusterer = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=10, random_state=seed)
    cluster_ids = clusterer.fit_predict(scaled)
    calibration = IsotonicRegression(out_of_bounds="clip")
    calibration.fit(prediction[labeled], labels[labeled])
    calibrated = calibration.predict(prediction)
    residuals = pd.DataFrame({"cluster_id": cluster_ids[labeled], "residual": labels[labeled] - calibrated[labeled]})
    cluster_residual = residuals.groupby("cluster_id").residual.agg(["mean", "count"])
    correction = cluster_residual["mean"] * cluster_residual["count"] / (cluster_residual["count"] + shrinkage)
    score = calibrated + pd.Series(cluster_ids).map(correction).fillna(0.0).to_numpy()
    if min(scaled.shape) >= 2:
        embedding = PCA(n_components=2, random_state=seed).fit_transform(scaled)
    else:
        embedding = np.column_stack((scaled[:, 0], np.zeros(len(scaled))))

    result = frame[["sample_id", *LENGTH_COLUMNS]].copy()
    result["cluster_id"] = cluster_ids.astype(np.int32)
    result["has_difficulty_label"] = labeled
    result["difficulty_true"] = labels
    result["difficulty_pred"] = prediction
    result["difficulty_hat"] = score
    result["ranking_score"] = score
    result["pred_percentile_global"] = pd.Series(score).rank(pct=True).to_numpy()
    result["pred_percentile_in_cluster"] = (
        result.groupby("cluster_id")["ranking_score"].rank(pct=True).to_numpy()
    )
    result["emb_0"] = embedding[:, 0]
    result["emb_1"] = embedding[:, 1]
    result = result.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    metadata: dict[str, object] = {
        "backend": backend, "feature_columns": feature_columns,
        "label_column": label_column, "n_rows": len(frame),
        "n_labeled": int(labeled.sum()), "n_clusters": n_clusters, "seed": seed,
        "labeled_samples": labeled_samples,
        "shrinkage": shrinkage,
        "ranking_score": "isotonic-calibrated prediction plus shrunken cluster residual",
        "cluster_policy": "MiniBatchKMeans on standardized features without labels",
    }
    return result, {"imputer": imputer, "regressor": regressor,
                    "cluster_scaler": cluster_scaler,
                    "clusterer": clusterer, "calibration": calibration,
                    "cluster_correction": correction}, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-parquet", type=Path, required=True)
    parser.add_argument("--backend", choices=("elasticnet", "lightgbm"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-prefix", default="f_")
    parser.add_argument("--label-column", default="difficulty_true")
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labeled-samples", type=int, default=3000)
    parser.add_argument("--shrinkage", type=float, default=20.0)
    args = parser.parse_args()
    frame = pd.read_parquet(args.features_parquet)
    table, fitted, metadata = fit_prediction_table(
        frame, backend=args.backend, feature_prefix=args.feature_prefix,
        label_column=args.label_column, n_clusters=args.clusters,
        seed=args.seed, labeled_samples=args.labeled_samples, shrinkage=args.shrinkage,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output_dir / "predictions_clustered.parquet", index=False)
    joblib.dump(fitted, args.output_dir / "difficulty_model.joblib")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
