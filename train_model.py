"""
Train and evaluate the Network Intrusion Detection model.

Dataset: NSL-KDD (Canadian Institute for Cybersecurity, UNB)
https://www.unb.ca/cic/datasets/nsl.html
125,973 training connections / 22,544 test connections.
Each record: 41 network connection features + label.

Methodology (deliberately honest):
- Model selection uses a stratified validation split taken from KDDTrain+
  ONLY. The official KDDTest+ file is never used to pick or tune a model.
- Preprocessors (one-hot encoder, scaler) are fit on the training portion
  only, inside a scikit-learn Pipeline, so no information leaks from
  validation or test data into training.
- KDDTest+ is touched exactly once: the final evaluation of the single
  chosen model.
- NSL-KDD is class-imbalanced (R2L and U2R are rare). We report per-class
  precision/recall/F1 and the confusion matrix, not just accuracy.

Run:  python train_model.py
Output: model/ids_model.joblib, model/metrics.json, printed report.
"""

import json
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TRAIN_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_PATH = os.path.join(DATA_DIR, "KDDTest+.txt")

# The 41 NSL-KDD features, then the label, then a difficulty score we drop.
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL + ["label", "difficulty"]]
FEATURES = CATEGORICAL + NUMERIC

# Every NSL-KDD attack name mapped to its category (standard mapping).
ATTACK_CATEGORIES = {
    "normal": "normal",
    # Denial of Service
    **{a: "DoS" for a in ["back", "land", "neptune", "pod", "smurf",
                           "teardrop", "apache2", "udpstorm", "processtable",
                           "worm", "mailbomb"]},
    # Scanning / probing
    **{a: "Probe" for a in ["satan", "ipsweep", "nmap", "portsweep",
                             "mscan", "saint"]},
    # Remote-to-local: attacker tries to gain a local account
    **{a: "R2L" for a in ["guess_passwd", "ftp_write", "imap", "phf",
                           "multihop", "warezmaster", "warezclient", "spy",
                           "xlock", "xsnoop", "snmpguess", "snmpgetattack",
                           "httptunnel", "sendmail", "named"]},
    # User-to-root: local user tries to become root
    **{a: "U2R" for a in ["buffer_overflow", "loadmodule", "rootkit",
                           "perl", "sqlattack", "xterm", "ps"]},
}

CATEGORY_INFO = {
    "normal": "Ordinary, legitimate network traffic.",
    "DoS": "Denial of Service - floods a machine so real users cannot use it (e.g. SYN flood).",
    "Probe": "Scanning - the attacker maps hosts/ports/services before attacking (e.g. port scan).",
    "R2L": "Remote to Local - an outside attacker tries to get a user account (e.g. password guessing).",
    "U2R": "User to Root - a local user tries to gain administrator/root control (e.g. buffer overflow).",
}


def load_nsukdd(path):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    unknown = sorted(set(df["label"]) - set(ATTACK_CATEGORIES))
    if unknown:
        print(f"WARNING: unmapped labels {unknown} -> grouped as 'other_attack'")
    df["category"] = df["label"].map(ATTACK_CATEGORIES).fillna("other_attack")
    return df


def make_pipeline(classifier):
    preprocess = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("numeric", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()

    print("Loading KDDTrain+ ...")
    train_full = load_nsukdd(TRAIN_PATH)
    X_full = train_full[FEATURES]
    y_full = train_full["category"]

    # Stratified split: model selection happens on this validation set only.
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    print(f"train: {len(X_train)}  validation: {len(X_val)}")

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }

    validation_results = []
    fitted = {}
    for name, clf in candidates.items():
        print(f"Training {name} ...")
        pipe = make_pipeline(clf)
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        acc = accuracy_score(y_val, val_pred)
        report = classification_report(y_val, val_pred, output_dict=True, zero_division=0)
        validation_results.append({
            "model": name,
            "validation_accuracy": round(acc, 4),
            "validation_macro_f1": round(report["macro avg"]["f1-score"], 4),
        })
        fitted[name] = pipe
        print(f"  validation accuracy={acc:.4f}  macro F1={report['macro avg']['f1-score']:.4f}")

    # Select by validation macro F1 (robust under class imbalance), not test data.
    best_name = max(validation_results, key=lambda r: r["validation_macro_f1"])["model"]
    print(f"\nSelected model: {best_name} (by validation macro F1)")

    # Refit the chosen configuration on ALL of KDDTrain+ (train + validation),
    # then touch the official test set exactly once.
    best_pipe = make_pipeline(candidates[best_name])
    best_pipe.fit(X_full, y_full)

    print("Final evaluation on untouched KDDTest+ ...")
    test_df = load_nsukdd(TEST_PATH)
    X_test, y_test = test_df[FEATURES], test_df["category"]
    test_pred = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    labels_order = sorted(set(y_test) | set(test_pred))
    cm = confusion_matrix(y_test, test_pred, labels=labels_order)

    print(f"\n=== FINAL TEST RESULTS ({best_name}) ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, test_pred, zero_division=0))

    metrics = {
        "selected_model": best_name,
        "validation_results": validation_results,
        "selection_rule": "Highest macro F1 on a stratified validation split of KDDTrain+. KDDTest+ was not used for selection.",
        "test_accuracy": round(float(test_acc), 4),
        "test_macro_f1": round(float(test_report["macro avg"]["f1-score"]), 4),
        "test_per_class": {
            k: {m: round(v, 4) for m, v in test_report[k].items() if m != "support"} | {"support": int(test_report[k]["support"])}
            for k in test_report if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "confusion_matrix": {"labels": labels_order, "matrix": cm.tolist()},
        "train_class_distribution": {k: int(v) for k, v in y_full.value_counts().items()},
        "test_class_distribution": {k: int(v) for k, v in y_test.value_counts().items()},
        "category_info": CATEGORY_INFO,
        "dataset": {
            "name": "NSL-KDD",
            "source": "Canadian Institute for Cybersecurity, University of New Brunswick - https://www.unb.ca/cic/datasets/nsl.html",
            "train_records": int(len(X_full)),
            "test_records": int(len(X_test)),
            "features": 41,
        },
        "training_time_seconds": round(time.time() - t0, 1),
    }

    joblib.dump(best_pipe, os.path.join(MODEL_DIR, "ids_model.joblib"), compress=3)
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    size_mb = os.path.getsize(os.path.join(MODEL_DIR, "ids_model.joblib")) / 1e6
    print(f"Saved model ({size_mb:.1f} MB) and metrics to model/")


if __name__ == "__main__":
    main()
