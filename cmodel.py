import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ================================================================
# CONFIGURATION
# ================================================================
# Path to the CSV produced by csiz.py
DATA_FILE = "raw_csi_gesture_data_YYYYMMDD_HHMMSS.csv"

# Classes to train on.
# csiz.py records three classes: "still", "walk", "wave"
# Change this list to train on all three or any subset of two.
# Examples:
#   Two-class  → ["still", "walk"]
#   Three-class → ["still", "walk", "wave"]
TARGET_CLASSES = ["still", "walk"]
# ================================================================

TARGET_COLUMN = "label"
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_SPLITS = 5
SELECTED_FEATURE_COUNT = 12

print("=" * 58)
n_cls = len(TARGET_CLASSES)
print(f"  WiFi Gesture Classifier — {n_cls}-Class CSI Trainer")
print(f"  Classes: {TARGET_CLASSES}")
print("=" * 58)

# ── 1. LOAD & FILTER DATA ────────────────────────────────────
print("\n[1/7] Loading and filtering data...")

if not Path(DATA_FILE).exists():
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. Check the file name and folder."
    )

df = pd.read_csv(DATA_FILE)
print(f"  Raw Rows: {df.shape[0]} | Columns: {df.shape[1]}")

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET_COLUMN]).dropna(axis=0)

df = df[df[TARGET_COLUMN].isin(TARGET_CLASSES)].copy()
print(f"  Filtered Rows ({n_cls} classes): {len(df)}")

print("\n  Samples per class:")
counts = df[TARGET_COLUMN].value_counts().sort_index()
for label, count in counts.items():
    bar = "█" * max(1, count // 3)
    print(f"    {str(label):<12} {count:>5}  {bar}")

if len(counts) < 2:
    raise ValueError(
        f"Need at least 2 classes, but only found: {list(counts.keys())}"
    )
if set(counts.keys()) != set(TARGET_CLASSES):
    missing = set(TARGET_CLASSES) - set(counts.keys())
    raise ValueError(
        f"These classes were not found in the CSV: {missing}. "
        "Record them with csiz.py first."
    )

# ── 2. SELECT RAW CSI FEATURES ───────────────────────────────
print("\n[2/7] Preparing features...")

metadata_columns = {TARGET_COLUMN, "timestamp", "frame_number", "session_id"}
feature_columns = [col for col in df.columns if col not in metadata_columns]

X_raw = df[feature_columns].apply(pd.to_numeric, errors="coerce")
y = df[TARGET_COLUMN].astype(str)

valid_rows = X_raw.notna().all(axis=1)
X_raw = X_raw.loc[valid_rows].reset_index(drop=True)
y = y.loc[valid_rows].reset_index(drop=True)

# ── 3. FEATURE ENGINEERING ───────────────────────────────────
print("\n[3/7] Engineering features...")

def engineer_features(data):
    features = data.copy()
    eps = 1e-8

    if {"std_signal", "mean_signal"}.issubset(features.columns):
        features["signal_cv"] = features["std_signal"] / (features["mean_signal"].abs() + eps)

    if {"range_signal", "mean_signal"}.issubset(features.columns):
        features["signal_norm_range"] = features["range_signal"] / (features["mean_signal"].abs() + eps)

    if {"temporal_change", "std_signal"}.issubset(features.columns):
        features["temporal_to_std_ratio"] = features["temporal_change"] / (features["std_signal"].abs() + eps)

    if {"temporal_change_max", "temporal_change"}.issubset(features.columns):
        features["temporal_peak_ratio"] = features["temporal_change_max"] / (features["temporal_change"].abs() + eps)

    if {"range_subcarrier_max", "range_subcarrier_mean"}.issubset(features.columns):
        features["subcarrier_range_ratio"] = features["range_subcarrier_max"] / (features["range_subcarrier_mean"].abs() + eps)

    if {"std_subcarrier_max", "std_subcarrier_mean"}.issubset(features.columns):
        features["subcarrier_std_ratio"] = features["std_subcarrier_max"] / (features["std_subcarrier_mean"].abs() + eps)

    if {"max_change_max", "max_change_mean"}.issubset(features.columns):
        features["change_peak_ratio"] = features["max_change_max"] / (features["max_change_mean"].abs() + eps)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features

X_engineered = engineer_features(X_raw)

# ── 4. TRAIN/TEST SPLIT & SELECTION ──────────────────────────
print("\n[4/7] Splitting and selecting features...")

X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

k = min(SELECTED_FEATURE_COUNT, X_train.shape[1])
selector = SelectKBest(score_func=f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()].tolist()

X_train_df = pd.DataFrame(X_train_selected, columns=selected_features)
X_test_df = pd.DataFrame(X_test_selected, columns=selected_features)

# ── 5. TRAIN MODELS ──────────────────────────────────────────
print("\n[5/7] Training models...")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=15, min_samples_leaf=2,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
)
gb = GradientBoostingClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    random_state=RANDOM_STATE,
)
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="rbf", C=10, gamma="scale",
        class_weight="balanced", probability=True,
        random_state=RANDOM_STATE,
    )),
])
ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb), ("svm", svm)],
    voting="soft",
)

models = {
    "Random Forest": rf,
    "Gradient Boosting": gb,
    "SVM": svm,
    "Ensemble": ensemble,
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}

for name, model in models.items():
    scores = cross_val_score(
        model, X_train_df, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    cv_scores[name] = scores
    print(f"    {name:<20} {scores.mean() * 100:.1f}% ± {scores.std() * 100:.1f}%")

best_name = max(cv_scores, key=lambda n: cv_scores[n].mean())
best_model = models[best_name]

print(f"\n  Best model: {best_name}")
best_model.fit(X_train_df, y_train)

y_pred = best_model.predict(X_test_df)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print(f"  Test Accuracy: {accuracy * 100:.1f}% | Macro F1: {macro_f1:.3f}")
print()
print(classification_report(y_test, y_pred))

# ── 6. VISUALIZATIONS ────────────────────────────────────────
print("\n[6/7] Generating performance report...")

class_labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=class_labels)

plt.style.use("seaborn-v0_8-white")
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(
    2, 2,
    width_ratios=[0.82, 1.18],
    height_ratios=[1, 1.15],
    hspace=0.45,
    wspace=0.35,
)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# Confusion Matrix
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
    xticklabels=class_labels, yticklabels=class_labels,
    linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.7},
)
ax1.set_title(f"Confusion Matrix ({best_name})", fontsize=11, pad=10)
ax1.set_xlabel("Predicted", fontsize=9)
ax1.set_ylabel("Actual", fontsize=9)

# Per-Class Accuracy
per_class_acc = (np.diag(cm) / cm.sum(axis=1)) * 100
colors = ["#22c55e" if acc >= 85 else "#3b82f6" for acc in per_class_acc]
bars = ax2.barh(class_labels, per_class_acc, color=colors, edgecolor="white", height=0.5)
ax2.set_xlim(0, 115)
ax2.set_title("Per-Class Accuracy (%)", fontsize=11, pad=10)
ax2.set_xlabel("Accuracy (%)", fontsize=9)
for bar, val in zip(bars, per_class_acc):
    ax2.text(val + 2.0, bar.get_y() + bar.get_height() / 2,
             f"{val:.0f}%", va="center", fontsize=9)

# CV Comparison
model_names = list(models.keys())
means = [cv_scores[n].mean() * 100 for n in model_names]
stds = [cv_scores[n].std() * 100 for n in model_names]
bar_colors = ["#22c55e" if n == best_name else "#3b82f6" for n in model_names]
ax3.bar(model_names, means, yerr=stds, color=bar_colors,
        capsize=4, edgecolor="white", error_kw={"linewidth": 1.0})
ax3.set_ylim(0, 115)
ax3.set_title("Cross-Validation Accuracy", fontsize=11, pad=10)
ax3.set_ylabel("CV Accuracy (%)", fontsize=9)
ax3.tick_params(axis="x", rotation=20, labelsize=8.5)
for i, (m, s) in enumerate(zip(means, stds)):
    ax3.text(i, m + s + 2.0, f"{m:.0f}%", ha="center", fontsize=8.5)

# Feature Importance
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "named_estimators_"):
    importances = best_model.named_estimators_["rf"].feature_importances_
else:
    importances = np.zeros(len(selected_features))

fi_series = pd.Series(importances, index=selected_features).sort_values()
ax4.barh(fi_series.index, fi_series.values, color="#8b5cf6", edgecolor="white", height=0.6)
ax4.set_title("Feature Importance", fontsize=11, pad=10)
ax4.set_xlabel("Importance", fontsize=9)

title_str = f"WiFi Gesture Recognition ({n_cls}-Class: {' vs '.join(TARGET_CLASSES)})"
fig.suptitle(title_str, fontsize=14, fontweight="bold", y=0.98)

plt.savefig("raw_csi_model_report.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print("  Saved: raw_csi_model_report.png")

# ── 7. SHAP EXPLAINABILITY ───────────────────────────────────
print("\n[7/7] Computing SHAP values...")

try:
    if hasattr(best_model, "feature_importances_"):
        shap_model = best_model
    elif hasattr(best_model, "named_estimators_"):
        shap_model = best_model.named_estimators_["rf"]
    else:
        shap_model = rf.fit(X_train_df, y_train)

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_test_df)

    # Select SHAP values for the last class (positive class in binary,
    # or the final class in multi-class)
    if isinstance(shap_values, list):
        target_shap_values = shap_values[-1]
    elif len(np.array(shap_values).shape) == 3:
        target_shap_values = shap_values[:, :, -1]
    else:
        target_shap_values = shap_values

    shap_class_label = TARGET_CLASSES[-1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(target_shap_values, X_test_df, show=False)
    plt.title(f"SHAP Feature Contribution (Class: '{shap_class_label}')", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("csi_shap_summary.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.summary_plot(target_shap_values, X_test_df, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig("csi_shap_bar.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print("  SHAP analysis complete.")
    print("  Saved: csi_shap_summary.png & csi_shap_bar.png")

except Exception as e:
    print(f"  Warning: SHAP analysis failed ({e})")

# ── SAVE MODEL PACKAGE ───────────────────────────────────────
save_package = {
    "model": best_model,
    "selector": selector,
    "selected_features": selected_features,
    "all_engineered_features": X_engineered.columns.tolist(),
    "classes": list(best_model.classes_),
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "model_name": best_name,
    "data_file": DATA_FILE,
    "target_classes": TARGET_CLASSES,
}

joblib.dump(save_package, "raw_csi_gesture_model.pkl")
print("\nTraining complete. Model saved to: raw_csi_gesture_model.pkl")
