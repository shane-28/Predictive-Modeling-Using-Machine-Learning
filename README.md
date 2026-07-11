# 🤖 Predictive Modeling Using Machine Learning

A supervised learning project that predicts **customer churn** (Yes/No) from account and usage data, comparing three classification algorithms and evaluating them with confusion matrices, ROC curves, and standard metrics.

---

## 📁 Project Structure

```
predictive-modeling-ml/
├── data/
│   ├── generate_data.py           # Creates the synthetic churn dataset
│   └── customer_churn.csv         # The dataset used for training/testing
├── visuals/
│   ├── 01_target_distribution.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_confusion_matrix_logistic_regression.png
│   ├── 03_confusion_matrix_decision_tree.png
│   ├── 03_confusion_matrix_random_forest.png
│   ├── 04_roc_curves_comparison.png
│   ├── 05_model_comparison.png
│   ├── 06_feature_importance_rf.png
│   └── model_comparison_metrics.csv
├── model_training.py              # Main pipeline: preprocess, train, evaluate
├── requirements.txt
└── README.md
```

---

## 🎯 Objective

Learn and demonstrate:
- Supervised learning for a binary classification problem
- Training and comparing multiple algorithms (Logistic Regression, Decision Tree, Random Forest)
- Model evaluation using accuracy, precision, recall, F1, confusion matrices, and ROC/AUC
- Interpreting feature importance to explain *why* a model predicts what it predicts

> **Note on algorithms:** The brief mentions Linear Regression, Decision Trees, or Random Forest. Since the target here (`Churn`) is categorical, plain Linear Regression doesn't apply directly (it can't produce a confusion matrix or ROC curve). **Logistic Regression** is used instead as the linear baseline — it's the natural classification counterpart to Linear Regression and plays the same role architecturally.

---

## 🗃️ Dataset

A synthetic telecom customer dataset (`data/customer_churn.csv`, 2,000 rows) with realistic, non-trivial signal:

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `TenureMonths` | Months as a customer |
| `ContractType` | Month-to-Month / One Year / Two Year |
| `InternetService` | DSL / Fiber Optic / No |
| `PaymentMethod` | Electronic Check / Mailed Check / Bank Transfer / Credit Card |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount billed to date |
| `SupportCallsLastYear` | Number of support calls |
| `Churn` | **Target** — Yes / No |

Regenerate anytime with:
```bash
python data/generate_data.py
```

---

## 🧪 Modeling Pipeline

1. **Preprocessing**
   - Impute missing `TotalCharges` with the median
   - One-hot encode categorical features (`ContractType`, `InternetService`, `PaymentMethod`)
   - Label-encode the target (`Yes` → 1, `No` → 0)
   - Standardize features for Logistic Regression
2. **Train/Test Split** — 75% train / 25% test, stratified on the target
3. **Train 3 models:**
   - Logistic Regression (linear baseline)
   - Decision Tree (`max_depth=6`)
   - Random Forest (`n_estimators=200`, `max_depth=8`)
4. **Evaluate** each model on the held-out test set

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.662 | 0.620 | 0.600 | 0.610 | 0.728 |
| Decision Tree | 0.630 | 0.576 | 0.605 | 0.590 | 0.684 |
| Random Forest | 0.644 | 0.603 | 0.559 | 0.580 | 0.707 |

*(Exact numbers will vary slightly if you regenerate the dataset with a different random seed.)*

**Logistic Regression edges out the tree-based models here** — a reminder that more complex models don't automatically win, especially when the underlying relationship between features and target is fairly linear (as constructed in this synthetic data).

### Visual Evaluation
- **Confusion matrices** for each model show the split of true/false positives and negatives
- **ROC curve comparison** shows each model's true-positive vs. false-positive trade-off across thresholds
- **Feature importance** (Random Forest) shows `ContractType`, `TenureMonths`, and `SupportCallsLastYear` as the strongest churn drivers

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/predictive-modeling-ml.git
cd predictive-modeling-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) regenerate the dataset
python data/generate_data.py

# 4. Train and evaluate all models
python model_training.py
```

All metrics print to the console, and every chart is saved to `visuals/`.

---

## 🧠 Key Learnings

- A stratified train/test split keeps class balance consistent when evaluating a classifier.
- Accuracy alone can be misleading on imbalanced-ish data — precision, recall, F1, and ROC AUC give a fuller picture.
- Tree-based ensembles (Random Forest) aren't guaranteed to beat simpler linear models — it depends on the true structure of the relationship in the data.
- Feature importance from a Random Forest is a quick, useful way to explain model behavior to non-technical stakeholders.

---

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn

---

## 📌 Possible Extensions

- Add hyperparameter tuning (`GridSearchCV` / `RandomizedSearchCV`)
- Try gradient boosting models (XGBoost, LightGBM)
- Add cross-validation instead of a single train/test split
- Swap in a real-world dataset (e.g., the classic Telco Customer Churn dataset on Kaggle) using the same pipeline
- Deploy the trained model behind a simple Flask/FastAPI endpoint or a Streamlit demo app

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
