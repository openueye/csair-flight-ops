"""Pre-flight delay-risk model and at-gate comparison.

Run after 01_prepare.py:
    python scripts/03_model.py

The main model predicts whether a completed flight will arrive 15+ minutes
late using only information available before departure:
  * scheduled calendar / time features
  * carrier, origin, destination, route
  * historical delay-rate encodings and traffic volumes computed on the
    training period only (out-of-fold for training rows)

A second "at-gate" variant adds DEP_DELAY, which is known at pushback, to
show the value of departure-delay information.  The test period is never
used for fitting or threshold selection.

Outputs
-------
outputs/tables/model_metrics.csv
outputs/tables/threshold_metrics.csv
outputs/tables/feature_importance.csv
outputs/tables/test_predictions_sample.csv
outputs/tables/calibration_bins.csv
outputs/figures/roc_pr_curves.png
outputs/figures/calibration.png
outputs/figures/feature_importance.png
outputs/metrics_model.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "flights.parquet"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)

TRAIN_END = pd.Timestamp("2018-02-10")
VALID_END = pd.Timestamp("2018-02-18")
SMOOTHING = 20.0
N_FOLDS = 5
RANDOM_STATE = 42

NUMERIC = [
    "month",
    "day",
    "day_of_week",
    "is_weekend",
    "dep_hour",
    "dep_minute",
    "sched_dep_min",
    "distance",
    "crs_elapsed_time",
]
KEYS = [
    "carrier",
    "origin",
    "dest",
    "route",
    "origin_hour",
    "dow_hour",
    "carrier_origin",
    "carrier_route",
]


def load() -> pd.DataFrame:
    cols = [
        "flight_date",
        "carrier",
        "origin",
        "dest",
        "route",
        "month",
        "day",
        "day_of_week",
        "is_weekend",
        "dep_hour",
        "dep_minute",
        "sched_dep_min",
        "distance",
        "crs_elapsed_time",
        "dep_delay",
        "arr_delay",
        "completed",
    ]
    df = pd.read_parquet(DATA, columns=cols)
    df = df[df["completed"].eq(1) & df["arr_delay"].notna()].copy()
    df["is_delayed"] = (df["arr_delay"] > 15).astype("int8")
    df["carrier"] = df["carrier"].astype(str)
    df["origin"] = df["origin"].astype(str)
    df["dest"] = df["dest"].astype(str)
    df["route"] = df["route"].astype(str)
    df["origin_hour"] = df["origin"] + "_" + df["dep_hour"].astype(str)
    df["dow_hour"] = df["day_of_week"].astype(str) + "_" + df["dep_hour"].astype(str)
    df["carrier_origin"] = df["carrier"] + "_" + df["origin"]
    df["carrier_route"] = df["carrier"] + "_" + df["route"]
    return df


def fit_encoding(
    frame: pd.DataFrame, key: str, target: str, smoothing: float = SMOOTHING
) -> tuple[pd.Series, float]:
    stats = frame.groupby(key, observed=True)[target].agg(["mean", "count"])
    global_mean = float(frame[target].mean())
    encoded = (stats["mean"] * stats["count"] + global_mean * smoothing) / (
        stats["count"] + smoothing
    )
    return encoded, global_mean


def add_target_encodings(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    test = test.reset_index(drop=True)
    te_cols: list[str] = []

    for key in KEYS:
        col = f"{key}_te"
        te_cols.append(col)
        train[col] = np.nan
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fit_idx, enc_idx in kf.split(train):
            enc, global_mean = fit_encoding(train.iloc[fit_idx], key, "is_delayed")
            train.loc[enc_idx, col] = (
                train.iloc[enc_idx][key].map(enc).astype(float).fillna(global_mean)
            )
        enc, global_mean = fit_encoding(train, key, "is_delayed")
        valid[col] = valid[key].map(enc).astype(float).fillna(global_mean)
        test[col] = test[key].map(enc).astype(float).fillna(global_mean)

        vol_col = f"{key}_volume"
        te_cols.append(vol_col)
        counts = train.groupby(key, observed=True).size()
        train[vol_col] = np.log1p(train[key].map(counts).fillna(0).astype(float))
        valid[vol_col] = np.log1p(valid[key].map(counts).fillna(0).astype(float))
        test[vol_col] = np.log1p(test[key].map(counts).fillna(0).astype(float))

    # Distance / elapsed time are complete in this dataset, but fill defensively.
    for frame in (train, valid, test):
        frame["distance"] = frame["distance"].fillna(train["distance"].median())
        frame["crs_elapsed_time"] = frame["crs_elapsed_time"].fillna(
            train["crs_elapsed_time"].median()
        )
    return train, valid, test, te_cols


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
        "log_loss": round(float(log_loss(y_true, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best_idx])


def top_decile_lift(y_true: np.ndarray, y_prob: np.ndarray, frac: float = 0.10) -> dict:
    n = max(1, int(len(y_prob) * frac))
    order = np.argsort(-y_prob)[:n]
    base = float(np.mean(y_true))
    top = float(np.mean(y_true[order]))
    return {
        "fraction": frac,
        "n": n,
        "base_delay_rate": round(base, 4),
        "top_delay_rate": round(top, 4),
        "lift": round(top / base, 3),
    }


def plot_curves(y_valid, p_valid, y_test, p_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for label, y, p, color in [
        ("Validation", y_valid, p_valid, "#3b6ea5"),
        ("Test", y_test, p_test, "#b5653a"),
    ]:
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y, p):.3f})", color=color)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{label} (AP={average_precision_score(y, p):.3f})", color=color)
    axes[0].plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="Chance")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC curve")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_pr_curves.png")
    plt.close(fig)


def plot_calibration(y_test, p_test) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=10, strategy="quantile")
    bins = pd.DataFrame({"mean_predicted": mean_pred, "fraction_positive": frac_pos})
    bins.to_csv(TABLES / "calibration_bins.csv", index=False)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#999", linewidth=1, label="Perfect")
    ax.plot(mean_pred, frac_pos, "o-", color="#3b6ea5", label="Model")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed delay rate")
    ax.set_title("Calibration (test, quantile bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png")
    plt.close(fig)
    return bins


def plot_importance(importance: pd.DataFrame) -> None:
    top = importance.sort_values("importance_mean", ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#5c7f67")
    ax.set_xlabel("Permutation importance (ROC-AUC drop)")
    ax.set_title("Pre-flight model feature importance (test sample)")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png")
    plt.close(fig)


def main() -> None:
    df = load()
    train = df[df["flight_date"] < TRAIN_END].copy()
    valid = df[(df["flight_date"] >= TRAIN_END) & (df["flight_date"] < VALID_END)].copy()
    test = df[df["flight_date"] >= VALID_END].copy()
    print(f"train={len(train):,} valid={len(valid):,} test={len(test):,}")

    train, valid, test, te_cols = add_target_encodings(train, valid, test)
    features = NUMERIC + te_cols
    print("features:", features)

    y_train = train["is_delayed"].to_numpy(dtype=int)
    y_valid = valid["is_delayed"].to_numpy(dtype=int)
    y_test = test["is_delayed"].to_numpy(dtype=int)

    results: list[dict] = []
    thresholds: list[dict] = []

    # 1. Prior baseline
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(train[features], y_train)
    p_valid = dummy.predict_proba(valid[features])[:, 1]
    p_test = dummy.predict_proba(test[features])[:, 1]
    results.append(evaluate("Dummy prior", y_test, p_test, 0.5))
    thresholds.append({"model": "Dummy prior", "threshold": 0.5, "note": "constant prior"})

    # 2. Logistic regression baseline
    logit = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    logit.fit(train[features], y_train)
    p_valid = logit.predict_proba(valid[features])[:, 1]
    p_test = logit.predict_proba(test[features])[:, 1]
    thr = best_f1_threshold(y_valid, p_valid)
    results.append(evaluate("Logistic regression", y_test, p_test, 0.5))
    results.append(evaluate("Logistic regression", y_test, p_test, thr))
    thresholds.append({"model": "Logistic regression", "threshold": round(thr, 4), "note": "best validation F1"})

    # 3. Gradient boosting (main pre-flight model)
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    hgb.fit(train[features], y_train)
    p_valid = hgb.predict_proba(valid[features])[:, 1]
    p_test = hgb.predict_proba(test[features])[:, 1]
    thr = best_f1_threshold(y_valid, p_valid)
    results.append(evaluate("Gradient boosting (pre-flight)", y_test, p_test, 0.5))
    results.append(evaluate("Gradient boosting (pre-flight)", y_test, p_test, thr))
    thresholds.append({"model": "Gradient boosting (pre-flight)", "threshold": round(thr, 4), "note": "best validation F1"})

    plot_curves(y_valid, p_valid, y_test, p_test)
    plot_calibration(y_test, p_test)

    # Feature importance on a fixed test sample to keep runtime bounded.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(test), size=min(25000, len(test)), replace=False)
    X_sample = test.iloc[sample_idx][features]
    y_sample = y_test[sample_idx]
    perm = permutation_importance(
        hgb,
        X_sample,
        y_sample,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": features,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .round(6)
    )
    importance.to_csv(TABLES / "feature_importance.csv", index=False)
    plot_importance(importance)

    # 4. At-gate comparison: DEP_DELAY is known at pushback.
    gate_mask_train = train["dep_delay"].notna()
    gate_mask_valid = valid["dep_delay"].notna()
    gate_mask_test = test["dep_delay"].notna()
    gate_features = features + ["dep_delay"]
    for frame in (train, valid, test):
        frame["dep_delay"] = frame["dep_delay"].clip(-60, 360)
    hgb_gate = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )
    hgb_gate.fit(train.loc[gate_mask_train, gate_features], y_train[gate_mask_train])
    p_valid_gate = hgb_gate.predict_proba(valid.loc[gate_mask_valid, gate_features])[:, 1]
    p_test_gate = hgb_gate.predict_proba(test.loc[gate_mask_test, gate_features])[:, 1]
    thr_gate = best_f1_threshold(y_valid[gate_mask_valid], p_valid_gate)
    results.append(evaluate("Gradient boosting (at-gate)", y_test[gate_mask_test], p_test_gate, 0.5))
    results.append(evaluate("Gradient boosting (at-gate)", y_test[gate_mask_test], p_test_gate, thr_gate))
    thresholds.append({"model": "Gradient boosting (at-gate)", "threshold": round(thr_gate, 4), "note": "best validation F1"})

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(TABLES / "model_metrics.csv", index=False)
    pd.DataFrame(thresholds).to_csv(TABLES / "threshold_metrics.csv", index=False)

    lift = top_decile_lift(y_test, p_test)
    lift_gate = top_decile_lift(y_test[gate_mask_test], p_test_gate)

    # Save a small, anonymised-ish sample of test predictions for inspection.
    pred = test.loc[:, ["flight_date", "carrier", "origin", "dest", "dep_hour", "arr_delay", "is_delayed"]].copy()
    pred["risk_preflight"] = p_test
    pred["risk_at_gate"] = np.nan
    pred.loc[gate_mask_test, "risk_at_gate"] = p_test_gate
    pred.sample(n=min(20000, len(pred)), random_state=RANDOM_STATE).to_csv(
        TABLES / "test_predictions_sample.csv", index=False
    )

    best_row = metrics_df[
        (metrics_df["model"] == "Gradient boosting (pre-flight)")
        & (metrics_df["threshold"] != 0.5)
    ].iloc[0]
    metrics = {
        "split": {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "test_rows": int(len(test)),
            "train_end": str(TRAIN_END.date()),
            "valid_end": str(VALID_END.date()),
        },
        "features": features,
        "preflight_test": best_row.to_dict(),
        "at_gate_test": metrics_df[
            (metrics_df["model"] == "Gradient boosting (at-gate)")
            & (metrics_df["threshold"] != 0.5)
        ].iloc[0].to_dict(),
        "top_decile_lift_preflight": lift,
        "top_decile_lift_at_gate": lift_gate,
        "validation_f1_threshold_preflight": round(float(thr), 4),
        "validation_f1_threshold_at_gate": round(float(thr_gate), 4),
    }
    with open(ROOT / "outputs" / "metrics_model.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)

    print(metrics_df.to_string(index=False))
    print("\nTop-decile lift (pre-flight):", lift)
    print("Top-decile lift (at-gate):", lift_gate)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
