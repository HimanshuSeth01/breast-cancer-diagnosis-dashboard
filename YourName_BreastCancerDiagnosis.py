"""
Breast Cancer Diagnosis Decision Dashboard
==========================================
AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares

Single-file project (data + ML back-end + Streamlit front-end).

Business-Intelligence flow followed:
    Data -> KPIs -> Trends -> Drivers -> Risk / Opportunity -> Action

Run the dashboard :  streamlit run YourName_BreastCancerDiagnosis.py
Run in terminal   :  python YourName_BreastCancerDiagnosis.py --cli

Dataset: Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 numeric features.
  Kaggle : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
  UCI    : https://archive.ics.uci.edu/dataset/17
The identical dataset ships inside scikit-learn (sklearn.datasets.load_breast_cancer),
so the project runs offline with no manual download.
"""

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
POSITIVE_LABEL = "Malignant"   # the class the hospital cannot afford to miss


# ----------------------------------------------------------------------------
# 1. DATA LAYER
# ----------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """Load dataset and give the target a business-readable label."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # sklearn encodes 0 = malignant, 1 = benign -> convert to readable text
    df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})
    df["is_malignant"] = (df["target"] == 0).astype(int)
    return df.drop(columns=["target"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic quality checks: duplicates and missing values."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.fillna(df.median(numeric_only=True))
    return df


def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in ("diagnosis", "is_malignant")]


# ----------------------------------------------------------------------------
# 2. MODEL LAYER
# ----------------------------------------------------------------------------
def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        "Support Vector Machine": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
    }


def train_and_evaluate(df: pd.DataFrame):
    """Train all models, compare them, return everything the dashboard needs."""
    feats = feature_columns(df)
    X, y = df[feats], df["is_malignant"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, model in build_models().items():
        cv_recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall").mean()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "CV Recall": cv_recall,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba),
        })
        fitted[name] = model

    results = pd.DataFrame(rows).sort_values(["Recall", "ROC-AUC"], ascending=False)
    results = results.reset_index(drop=True)
    best_name = results.loc[0, "Model"]
    best = fitted[best_name]

    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    imp = permutation_importance(best, X_test, y_test, n_repeats=10,
                                 random_state=RANDOM_STATE, scoring="roc_auc")
    importance = (pd.DataFrame({"Feature": feats, "Importance": imp.importances_mean})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))

    return {
        "feats": feats, "results": results, "best_name": best_name, "best": best,
        "cm": confusion_matrix(y_test, pred), "fpr": fpr, "tpr": tpr,
        "importance": importance, "X_test": X_test, "y_test": y_test,
        "proba": proba, "pred": pred,
    }


# ----------------------------------------------------------------------------
# 3. INSIGHT LAYER  (KPIs -> Risk -> Opportunity -> Action)
# ----------------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame, art: dict) -> dict:
    best_row = art["results"].iloc[0]
    tn, fp, fn, tp = art["cm"].ravel()
    return {
        "Total Cases": len(df),
        "Malignancy Rate": df["is_malignant"].mean() * 100,
        "Model Recall (cancer caught)": best_row["Recall"] * 100,
        "Model Precision": best_row["Precision"] * 100,
        "Model ROC-AUC": best_row["ROC-AUC"],
        "Missed Cancers (test set)": int(fn),
        "False Alarms (test set)": int(fp),
    }


def build_insights(df: pd.DataFrame, art: dict) -> dict:
    top = art["importance"].head(3)["Feature"].tolist()
    grp = df.groupby("diagnosis")[["mean radius", "mean concavity", "mean area"]].mean()
    area_ratio = grp.loc["Malignant", "mean area"] / grp.loc["Benign", "mean area"]
    tn, fp, fn, tp = art["cm"].ravel()
    recall = tp / (tp + fn)
    return {
        "fact": (f"{df['is_malignant'].mean()*100:.1f}% of {len(df)} cases are malignant, and malignant "
                 f"tumours have a mean area {area_ratio:.1f}x larger than benign ones."),
        "drivers": top,
        "risk": (f"{fn} malignant case(s) missed and {fp} false alarm(s) on the held-out test set. "
                 f"A missed cancer is the costliest error, so recall ({recall*100:.1f}%) is the "
                 f"decision metric."),
        "opportunity": (f"Cell-nucleus measurements ({', '.join(top)}) separate the classes strongly, "
                        f"so a quick, low-cost screening aid can prioritise high-risk patients."),
        "action": ("Use the model as a second-opinion triage tool: send every patient with predicted "
                   "malignancy probability above the threshold for urgent specialist review, and "
                   "audit all missed cases monthly."),
    }


def predict_patient(model, feats, values: dict, threshold: float):
    row = pd.DataFrame([values])[feats]
    p = float(model.predict_proba(row)[0, 1])
    return p, ("Malignant" if p >= threshold else "Benign")


# ----------------------------------------------------------------------------
# 4. CHART HELPERS
# ----------------------------------------------------------------------------
PALETTE = {"Malignant": "#C0392B", "Benign": "#2E86C1"}


def fig_class_balance(df):
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    counts = df["diagnosis"].value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    ax.set_title("Cases by Diagnosis"); ax.set_ylabel("Patients")
    sns.despine(); fig.tight_layout(); return fig


def fig_distribution(df, col):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    sns.histplot(data=df, x=col, hue="diagnosis", palette=PALETTE, kde=True,
                 element="step", ax=ax)
    ax.set_title(f"{col.title()} by Diagnosis"); sns.despine(); fig.tight_layout(); return fig


def fig_trend_by_size(df):
    """'Trend' view: malignancy rate rises as tumour radius grows."""
    d = df.copy()
    d["radius band"] = pd.qcut(d["mean radius"], 8, duplicates="drop")
    rate = d.groupby("radius band", observed=True)["is_malignant"].mean() * 100
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(range(len(rate)), rate.values, marker="o", color="#C0392B", lw=2.5)
    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([f"{i.mid:.1f}" for i in rate.index], rotation=0)
    ax.set_xlabel("Mean radius (band midpoint)"); ax.set_ylabel("Malignancy rate (%)")
    ax.set_title("Trend: Malignancy Rate Rises With Tumour Size")
    sns.despine(); fig.tight_layout(); return fig


def fig_importance(imp, n=10):
    top = imp.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top["Feature"], top["Importance"], color="#1F618D")
    ax.set_title(f"Top {n} Drivers (permutation importance)")
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    sns.despine(); fig.tight_layout(); return fig


def fig_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    fig.tight_layout(); return fig


def fig_roc(art):
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    auc = art["results"].iloc[0]["ROC-AUC"]
    ax.plot(art["fpr"], art["tpr"], color="#C0392B", lw=2.5, label=f"{art['best_name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right"); sns.despine(); fig.tight_layout(); return fig


def fig_model_comparison(results):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    m = results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]]
    m.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
    ax.set_ylim(0.85, 1.01); ax.set_ylabel("Score"); ax.set_title("Model Comparison")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(loc="lower right", ncol=2, fontsize=8); sns.despine(); fig.tight_layout(); return fig


def fig_correlation(df):
    cols = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
            "mean compactness", "mean concavity", "mean concave points", "is_malignant"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation of Key Features With Malignancy"); fig.tight_layout(); return fig


# ----------------------------------------------------------------------------
# 5. COMMAND-LINE MODE
# ----------------------------------------------------------------------------
def run_cli():
    df = clean_data(load_data())
    art = train_and_evaluate(df)
    print("\n=== KPIs ===")
    for k, v in compute_kpis(df, art).items():
        print(f"{k:32s}: {v:.2f}" if isinstance(v, float) else f"{k:32s}: {v}")
    print("\n=== Model comparison ===")
    print(art["results"].round(4).to_string(index=False))
    print(f"\nSelected model: {art['best_name']}")
    print("\n=== Top 5 drivers ===")
    print(art["importance"].head(5).round(4).to_string(index=False))
    print("\n=== Insights ===")
    for k, v in build_insights(df, art).items():
        print(f"{k.upper():12s}: {v}")


# ----------------------------------------------------------------------------
# 6. STREAMLIT FRONT-END
# ----------------------------------------------------------------------------
def run_app():
    import streamlit as st

    st.set_page_config(page_title="Breast Cancer Decision Dashboard", page_icon="🩺", layout="wide")

    @st.cache_data
    def get_data():
        return clean_data(load_data())

    @st.cache_resource
    def get_artifacts():
        return train_and_evaluate(get_data())

    df, art = get_data(), get_artifacts()
    kpis, insights = compute_kpis(df, art), build_insights(df, art)

    st.sidebar.title("🩺 Diagnosis Decision Dashboard")
    page = st.sidebar.radio("Go to", [
        "1. Executive Overview", "2. Data & Drivers", "3. Model Performance",
        "4. Risk • Opportunity • Action", "5. Patient Risk Predictor"])
    st.sidebar.caption("Dataset: Breast Cancer Wisconsin (Diagnostic)\n\nFor decision support only - "
                       "not a substitute for clinical judgement.")

    if page.startswith("1"):
        st.title("Executive Overview")
        st.caption("Fact → Insight → Action: what the data says about diagnosis outcomes.")
        c = st.columns(4)
        c[0].metric("Total Cases", kpis["Total Cases"])
        c[1].metric("Malignancy Rate", f"{kpis['Malignancy Rate']:.1f}%")
        c[2].metric("Cancers Caught (Recall)", f"{kpis['Model Recall (cancer caught)']:.1f}%")
        c[3].metric("ROC-AUC", f"{kpis['Model ROC-AUC']:.3f}")
        c = st.columns(3)
        c[0].metric("Model Precision", f"{kpis['Model Precision']:.1f}%")
        c[1].metric("Missed Cancers (test)", kpis["Missed Cancers (test set)"])
        c[2].metric("False Alarms (test)", kpis["False Alarms (test set)"])
        st.success(f"**Key fact:** {insights['fact']}")
        l, r = st.columns(2)
        l.pyplot(fig_class_balance(df)); r.pyplot(fig_trend_by_size(df))

    elif page.startswith("2"):
        st.title("Data & Drivers")
        l, r = st.columns(2)
        col = l.selectbox("Explore a feature", art["feats"], index=art["feats"].index("mean radius"))
        l.pyplot(fig_distribution(df, col))
        r.pyplot(fig_importance(art["importance"]))
        st.pyplot(fig_correlation(df))
        with st.expander("Preview dataset"):
            st.dataframe(df.head(50))
            st.dataframe(df.describe().T)

    elif page.startswith("3"):
        st.title("Model Performance")
        st.write(f"Best model by recall then ROC-AUC: **{art['best_name']}**")
        st.dataframe(art["results"].style.format({c: "{:.3f}" for c in art["results"].columns[1:]}))
        l, r, r2 = st.columns(3)
        l.pyplot(fig_confusion(art["cm"])); r.pyplot(fig_roc(art)); r2.pyplot(fig_model_comparison(art["results"]))

    elif page.startswith("4"):
        st.title("Risk • Opportunity • Action")
        st.error(f"**RISK** - {insights['risk']}")
        st.info(f"**OPPORTUNITY** - {insights['opportunity']}")
        st.success(f"**ACTION** - {insights['action']}")
        st.subheader("Top drivers behind the insight")
        st.table(art["importance"].head(5).round(4))

    else:
        st.title("Patient Risk Predictor")
        st.caption("Adjust the main measurements; remaining features default to the dataset average.")
        feats = art["feats"]
        defaults = df[feats].mean().to_dict()
        thr = st.slider("Alert threshold (lower = catch more cancers, more false alarms)", 0.05, 0.95, 0.40, 0.05)
        vals = dict(defaults)
        cols = st.columns(3)
        for i, f in enumerate(art["importance"].head(6)["Feature"]):
            lo, hi = float(df[f].min()), float(df[f].max())
            vals[f] = cols[i % 3].slider(f, lo, hi, float(defaults[f]))
        p, label = predict_patient(art["best"], feats, vals, thr)
        st.metric("Predicted malignancy probability", f"{p*100:.1f}%")
        (st.error if label == "Malignant" else st.success)(f"Triage result: **{label}**")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_app()
