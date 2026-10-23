#!/usr/bin/env python3
"""Reproducible experiment for knowledge-enhanced opportunity prediction."""
from __future__ import annotations
import io
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
OUT = Path("outputs")
TARGET = "Revenue"

def load_data() -> pd.DataFrame:
    """Download the official UCI archive and return its CSV dataset."""
    print("Downloading UCI Online Shoppers Purchasing Intention dataset...")
    req = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0 research-reproducibility"})
    with urlopen(req, timeout=45) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError("No CSV file found in the UCI archive.")
        with archive.open(sorted(csv_names, key=len)[0]) as handle:
            df = pd.read_csv(handle)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET!r}; got {list(df.columns)}")
    return df

def add_knowledge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent features motivated by online customer-journey knowledge."""
    x = df.copy()
    required = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration", "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay", "VisitorType", "Weekend", "Month"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    x["KE_EngagementPages"] = x["Administrative"] + x["Informational"] + x["ProductRelated"]
    x["KE_EngagementDuration"] = x["Administrative_Duration"].clip(lower=0) + x["Informational_Duration"].clip(lower=0) + x["ProductRelated_Duration"].clip(lower=0)
    x["KE_ProductFocus"] = x["ProductRelated"] / (1.0 + x["Administrative"] + x["Informational"] + x["ProductRelated"])
    x["KE_Friction"] = (x["BounceRates"].clip(lower=0) + x["ExitRates"].clip(lower=0)) / 2
    x["KE_ValueSignal"] = np.log1p(x["PageValues"].clip(lower=0))
    x["KE_SpecialDayProximity"] = x["SpecialDay"].clip(lower=0)
    x["KE_OpportunityRuleScore"] = (0.30 * np.log1p(x["ProductRelated"].clip(lower=0)) + 0.25 * np.log1p(x["ProductRelated_Duration"].clip(lower=0)) + 0.35 * np.log1p(x["PageValues"].clip(lower=0)) - 0.10 * x["BounceRates"].clip(lower=0) - 0.10 * x["ExitRates"].clip(lower=0))
    return x

def make_pipeline(model, numeric_cols, categorical_cols, scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    prep = ColumnTransformer([("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)])
    return Pipeline([("prep", prep), ("model", model)])

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    y = df[TARGET].astype(bool).astype(int)
    raw_x = df.drop(columns=[TARGET])
    knowledge_x = add_knowledge_features(raw_x)
    categorical = raw_x.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric = [c for c in raw_x.columns if c not in categorical]
    knowledge_numeric = [c for c in knowledge_x.columns if c not in categorical]
    Xr_train, Xr_test, y_train, y_test = train_test_split(raw_x, y, test_size=0.20, random_state=SEED, stratify=y)
    Xk_train, Xk_test, _, _ = train_test_split(knowledge_x, y, test_size=0.20, random_state=SEED, stratify=y)
    models = {
        "Logistic regression (raw)": (make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced"), numeric, categorical, scale=True), Xr_train, Xr_test),
        "Random forest (raw)": (make_pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), numeric, categorical), Xr_train, Xr_test),
        "Random forest (knowledge-enhanced)": (make_pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), knowledge_numeric, categorical), Xk_train, Xk_test),
    }
    metrics, fitted = [], {}
    plt.figure(figsize=(7.2, 5.5))
    for name, (pipe, x_train, x_test) in models.items():
        print(f"Training: {name}")
        pipe.fit(x_train, y_train)
        probs = pipe.predict_proba(x_test)[:, 1]
        pred = (probs >= 0.5).astype(int)
        row = {"model": name, "roc_auc": roc_auc_score(y_test, probs), "average_precision": average_precision_score(y_test, probs), "f1_at_0.5": f1_score(y_test, pred, zero_division=0), "balanced_accuracy_at_0.5": balanced_accuracy_score(y_test, pred), "precision_at_0.5": precision_score(y_test, pred, zero_division=0), "recall_at_0.5": recall_score(y_test, pred, zero_division=0)}
        metrics.append(row)
        fitted[name] = (pipe, x_test)
        RocCurveDisplay.from_predictions(y_test, probs, name=f"{name} (AUC={row['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")
    plt.title("Figure 1. Test-set ROC curves")
    plt.tight_layout()
    plt.savefig(OUT / "figure1_model_comparison.png", dpi=220)
    plt.close()
    metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(OUT / "table1_model_metrics.csv", index=False)
    raw_rf = metrics_df.loc[metrics_df["model"] == "Random forest (raw)"].iloc[0]
    enhanced_rf = metrics_df.loc[metrics_df["model"] == "Random forest (knowledge-enhanced)"].iloc[0]
    ablation = pd.DataFrame([
        {"configuration": "Random forest, raw predictors", "roc_auc": raw_rf.roc_auc, "average_precision": raw_rf.average_precision, "delta_roc_auc_vs_raw_rf": 0.0},
        {"configuration": "Random forest + knowledge features", "roc_auc": enhanced_rf.roc_auc, "average_precision": enhanced_rf.average_precision, "delta_roc_auc_vs_raw_rf": enhanced_rf.roc_auc - raw_rf.roc_auc},
    ])
    ablation.to_csv(OUT / "table2_ablation.csv", index=False)
    enhanced_pipe, X_test_enhanced = fitted["Random forest (knowledge-enhanced)"]
    print("Calculating held-out permutation importance...")
    perm = permutation_importance(enhanced_pipe, X_test_enhanced, y_test, scoring="roc_auc", n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = pd.DataFrame({"feature": X_test_enhanced.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False).head(12)
    importance.to_csv(OUT / "feature_importance.csv", index=False)
    plot_data = importance.sort_values("importance_mean")
    plt.figure(figsize=(8.2, 6.0))
    plt.barh(plot_data["feature"], plot_data["importance_mean"], xerr=plot_data["importance_std"], capsize=2)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("Decrease in ROC-AUC after permutation")
    plt.title("Figure 2. Held-out permutation importance")
    plt.tight_layout()
    plt.savefig(OUT / "figure2_feature_importance.png", dpi=220)
    plt.close()
    n, positives = len(df), int(y.sum())
    with open(OUT / "results.md", "w", encoding="utf-8") as f:
        f.write("# Automatically generated experiment results\n\n")
        f.write(f"- Dataset rows: {n}\n- Positive sessions: {positives} ({positives/n:.1%})\n- Random seed: {SEED}\n\n")
        f.write("## Table 1. Test-set performance\n\n")
        f.write(metrics_df.round(4).to_markdown(index=False))
        f.write("\n\n## Table 2. Ablation analysis\n\n")
        f.write(ablation.round(4).to_markdown(index=False))
        f.write("\n\n## Interpretation guardrails\n\n")
        f.write("These are predictive associations on one held-out split, not causal effects. The dataset measures online shopping sessions, not firm-level digital transformation or innovation outcomes. Report the limitation explicitly.\n")
        f.write("\n## Figure files\n\n- figure1_model_comparison.png\n- figure2_feature_importance.png\n")
    print(f"\nFinished. Tables, figures, and results are in: {OUT.resolve()}")
    print(metrics_df.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
