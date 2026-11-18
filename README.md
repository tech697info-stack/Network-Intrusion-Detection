# Network Intrusion Detection using Machine Learning

A machine-learning project that classifies a network connection record as
**normal** traffic or one of four attack categories (**DoS**, **Probe**,
**R2L**, **U2R**). It comes with a small web app for demonstrating the
trained model and its measured results.

This is an educational, offline classifier for recorded connection data.
It does not monitor or protect a live network.

## What is inside

| File | Purpose |
| --- | --- |
| `train_model.py` | Downloads nothing - trains on the bundled dataset, compares 3 models honestly, saves the winner and all metrics |
| `app.py` | Streamlit web app: overview, measured results, live predictions |
| `data/KDDTrain+.txt` | Official NSL-KDD training data (125,973 records) |
| `data/KDDTest+.txt` | Official NSL-KDD test data (22,544 records) |
| `model/ids_model.joblib` | The trained model (already trained - no need to retrain) |
| `model/metrics.json` | All measured results the app displays |
| `sample_connections.csv` | 200 real records taken from the official test file, labels removed - for demonstrating the CSV-upload feature only; NOT part of any training or evaluation |
| `requirements.txt` | Exact Python package versions this was tested with |
| `VIVA_NOTES.md` | Explanation and likely viva questions with honest answers |

## How to run (Windows)

You need Python 3.10 or newer (built and tested with Python 3.13 in mind).

Open a terminal in this folder, then:

**Command Prompt (CMD):**
```
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

**PowerShell:**
```
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell refuses to run the activation script, run this once in the
same window and repeat the activation line:
```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

Your browser opens at http://localhost:8501 with three pages:
**Overview**, **Model results**, **Try a prediction**.

Optional - retrain the model yourself (about a minute):

```
python train_model.py
```

## Dataset

NSL-KDD, the standard teaching dataset for network intrusion detection,
from the Canadian Institute for Cybersecurity, University of New Brunswick:
https://www.unb.ca/cic/datasets/nsl.html

Each record is one network connection described by 41 features (protocol,
service, bytes sent/received, error rates, login behavior, traffic counts),
plus a label: normal or a specific attack name. Attack names group into
DoS, Probe, R2L and U2R.

## Methodology (short version)

- Three models were trained (Logistic Regression, Random Forest,
  Histogram Gradient Boosting) and compared on a **validation split of the
  training data only**.
- Preprocessing (one-hot encoding, scaling) was fit on training data only,
  inside a scikit-learn Pipeline, so no test information leaks in.
- The winner (Random Forest, chosen by validation macro F1) was evaluated
  **once** on the official untouched test set.
- Final measured test results: **73.8% accuracy, 0.48 macro F1**.
  Per-class details are in the app and in `VIVA_NOTES.md`.

## Honest limitations

- NSL-KDD is based on 1999-era traffic; it is the standard teaching
  dataset, not a picture of the modern internet.
- The test set contains attack types never seen in training, so test
  scores are much lower than validation scores. That gap is expected and
  is reported openly.
- R2L and U2R attacks are extremely rare in the data (995 and 52 training
  records), so the model misses most of them. See per-class metrics.

## Dataset license and citation

The ISCX NSL-KDD dataset license permits redistribution, republication and
mirroring in any form, provided any use or redistribution includes a
citation (https://www.unb.ca/cic/datasets/nsl.html):

[1] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, "A Detailed
Analysis of the KDD CUP 99 Data Set," Second IEEE Symposium on
Computational Intelligence for Security and Defense Applications (CISDA),
2009.
