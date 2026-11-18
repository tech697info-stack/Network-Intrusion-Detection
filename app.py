"""
Network Intrusion Detection - demo web app.

Loads the model trained by train_model.py and lets you:
  1. Overview      - what the project does and how the model was built.
  2. Model results - the measured validation/test metrics.
  3. Try a prediction - classify a real connection from the test set,
     or upload your own CSV of connection records.

Run:  streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ids_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
TEST_PATH = os.path.join(BASE_DIR, "data", "KDDTest+.txt")

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
FEATURES = COLUMNS[:-2]

SHOW_FIELDS = ["duration", "protocol_type", "service", "flag", "src_bytes",
               "dst_bytes", "count", "srv_count", "serror_rate",
               "same_srv_rate", "dst_host_count", "dst_host_srv_count"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_test_sample():
    df = pd.read_csv(TEST_PATH, header=None, names=COLUMNS)
    df = df.drop(columns=["difficulty"])
    return df


st.set_page_config(page_title="Network Intrusion Detection (ML)", layout="wide")
page = st.sidebar.radio(
    "Page", ["Overview", "Model results", "Try a prediction"])

model = load_model()
metrics = load_metrics()

if page == "Overview":
    st.title("Network Intrusion Detection using Machine Learning")
    st.write(
        "This project trains a machine-learning model to classify a network "
        "connection record as **normal** traffic or one of four attack "
        "categories: **DoS**, **Probe**, **R2L** and **U2R**. It is an "
        "educational demonstration of ML applied to network security data - "
        "it classifies recorded connection data, it does not monitor or "
        "protect a live network.")

    st.subheader("Dataset")
    ds = metrics["dataset"]
    st.write(
        f"**{ds['name']}** - {ds['source']}. "
        f"{ds['train_records']:,} training connections and "
        f"{ds['test_records']:,} test connections, each described by "
        f"{ds['features']} network connection features (protocol, service, "
        f"bytes transferred, error rates, login behavior, and more).")

    st.subheader("Attack categories")
    for name, desc in metrics["category_info"].items():
        st.write(f"- **{name}**: {desc}")

    st.subheader("How the model was built")
    st.write(metrics["selection_rule"])
    st.write(
        f"The selected model is **{metrics['selected_model']}**. "
        "Preprocessing (encoding and scaling) was fit on the training "
        "portion only, and the official test set was used once, for the "
        "final evaluation shown on the Model results page.")

    st.subheader("Honest limitations")
    st.write(
        "- NSL-KDD is based on 1999-era network traffic. It is the standard "
        "teaching dataset for this task, but real modern traffic looks "
        "different.\n"
        "- The data is imbalanced: R2L and U2R attacks are rare, and the "
        "model detects them poorly. The per-class numbers on the next page "
        "show this openly.\n"
        "- This is an offline classifier for recorded data, not a real-time "
        "protection system.")

elif page == "Model results":
    st.title("Measured results")
    st.subheader("Model comparison (validation split of the training data)")
    st.dataframe(pd.DataFrame(metrics["validation_results"]),
                 width="stretch")
    st.caption(metrics["selection_rule"])

    st.subheader("Final evaluation on the untouched official test set")
    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
    c2.metric("Test macro F1", f"{metrics['test_macro_f1']:.2f}")
    st.write(
        "The test set is harder than the training data: it contains attack "
        "types the model never saw in training. The gap between validation "
        "and test scores is expected on NSL-KDD and is honestly reported "
        "rather than hidden.")

    st.subheader("Per-class results on the test set")
    pc = pd.DataFrame(metrics["test_per_class"]).T
    pc = pc[["precision", "recall", "f1-score", "support"]]
    st.dataframe(pc, width="stretch")
    st.caption(
        "Recall for the rare classes (R2L, U2R) is low: the model misses "
        "most of those attacks. That is a known, documented limitation of "
        "models trained on this imbalanced dataset.")

    st.subheader("Confusion matrix (test set)")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])
    st.dataframe(cm_df, width="stretch")

    st.subheader("Class distribution (training data)")
    dist = pd.Series(metrics["train_class_distribution"], name="records")
    st.bar_chart(dist)

else:
    st.title("Try a prediction")
    mode = st.radio("Input source",
                    ["Pick a random real connection from the test set",
                     "Upload a CSV of connection records"])

    if mode.startswith("Pick"):
        if st.button("Pick a random connection"):
            st.session_state["row"] = load_test_sample().sample(
                1, random_state=None).iloc[0]
        row = st.session_state.get("row")
        if row is None:
            st.info("Press the button to draw a connection record.")
        else:
            st.write("Connection record (key fields):")
            st.dataframe(row[SHOW_FIELDS].to_frame("value").T,
                         width="stretch")
            pred = model.predict(row[FEATURES].to_frame().T)[0]
            proba = (model.predict_proba(row[FEATURES].to_frame().T)[0]
                     if hasattr(model, "predict_proba") else None)
            c1, c2 = st.columns(2)
            c1.metric("Model prediction", pred)
            c2.metric("True label (dataset)", row["label"])
            if pred == "normal" and row["label"] == "normal":
                st.success("Correct: normal traffic recognized.")
            elif pred == row["label"]:
                st.success("Correct: attack category recognized.")
            else:
                st.warning("The model's answer differs from the true label - "
                           "see the limitations on the results page.")
            if proba is not None:
                st.write("Prediction confidence by category:")
                st.bar_chart(pd.Series(proba, index=model.classes_))
    else:
        uploaded = st.file_uploader(
            "CSV with the 41 NSL-KDD feature columns, no header "
            "(an optional 42nd label column is allowed)", type="csv")
        if uploaded is not None:
            df = pd.read_csv(uploaded, header=None)
            if df.shape[1] not in (41, 42, 43):
                st.error(f"Expected 41-43 columns, got {df.shape[1]}.")
            else:
                df = df.iloc[:, :41]
                df.columns = COLUMNS[:-2]
                preds = model.predict(df)
                st.subheader("Summary")
                st.bar_chart(pd.Series(preds).value_counts())
                st.subheader("First 100 rows")
                out = df[SHOW_FIELDS].copy()
                out["prediction"] = preds
                st.dataframe(out.head(100), width="stretch")
