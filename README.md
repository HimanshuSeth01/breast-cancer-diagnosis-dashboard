# Breast Cancer Diagnosis Decision Dashboard

**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 | BharatCares**

A single-file Python project that turns diagnostic measurements into a business-intelligence
dashboard and a machine-learning triage aid. It follows the flow
**Data -> KPIs -> Trends -> Drivers -> Risk / Opportunity -> Action**.

## Problem Statement
Hospitals need to prioritise patients for urgent specialist review. Missing a malignant case is far
costlier than a false alarm. This project analyses tumour cell-nucleus measurements, identifies what
drives malignancy, and predicts the risk of malignancy so decision makers can act early.

## Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic) - 569 samples, 30 numeric features, no missing values
- **Kaggle:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **UCI:** https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- The same data is bundled in scikit-learn (`sklearn.datasets.load_breast_cancer`), so the project
  runs offline with no manual download.

## Technologies Used
| Layer | Tools |
|---|---|
| Back-end / data | Python, pandas, NumPy |
| Machine learning | scikit-learn (Logistic Regression, SVM, Random Forest, Gradient Boosting) |
| Visualisation | Matplotlib, Seaborn |
| Front-end | Streamlit (5-page dashboard) |

## Project Structure
```
YourName_BreastCancerDiagnosis.py   # complete code: data, model, insights, dashboard
requirements.txt                    # dependencies
YourName_ProjectReport.docx         # full project documentation
README.md                           # this file
```

## Setup and Run
```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3a. launch the dashboard (opens at http://localhost:8501)
streamlit run YourName_BreastCancerDiagnosis.py

# 3b. or print KPIs, model comparison and insights in the terminal
python YourName_BreastCancerDiagnosis.py --cli
```

## Dashboard Pages
1. **Executive Overview** - KPI cards (cases, malignancy rate, recall, precision, ROC-AUC, missed cancers) and the size-vs-malignancy trend.
2. **Data & Drivers** - feature distributions, correlation heat map, top drivers by permutation importance.
3. **Model Performance** - comparison of four models, confusion matrix, ROC curve.
4. **Risk - Opportunity - Action** - business recommendations generated from the results.
5. **Patient Risk Predictor** - adjustable alert threshold and key measurements give a malignancy probability.

## Key Results (20% stratified hold-out test set, 114 patients)
| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (selected) | 96.5% | 97.5% | 92.9% | 0.996 |
| Support Vector Machine | 97.4% | 100% | 92.9% | 0.995 |
| Random Forest | 97.4% | 100% | 92.9% | 0.994 |
| Gradient Boosting | 96.5% | 100% | 90.5% | 0.995 |

- Malignancy rate in the data: **37.3%**; malignant tumours have a mean area about **2.1x** larger than benign.
- Selected model: 39 of 42 malignant cases caught, 1 false alarm. The model is chosen on recall, then ROC-AUC.
- Top drivers: worst symmetry, worst texture, worst concavity.
- **Recommended action:** use the model as a second-opinion triage tool, escalate patients above the alert threshold, and audit missed cases monthly.

## Disclaimer
Educational decision-support project. It is not a medical device and must not replace clinical judgement.
