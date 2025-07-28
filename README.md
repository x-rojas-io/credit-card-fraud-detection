# 🛡️ Credit Card Fraud Detection — ML Case Study

This is a practical Data Science/ML project that investigates the detection of fraudulent credit card transactions using a real-world dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The goal is not just to classify transactions, but to prioritize high-risk cases effectively under real-world constraints, where fraud is extremely rare and false positives are costly.

⸻
## 📦 Dataset

- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Size: 284,807 transactions, 492 frauds (~0.17%)
- Features: PCA-transformed numerical variables (`V1` to `V28`), `Time`, `Amount`, `Class` (target: 0 = legit, 1 = fraud)



## 📊 Problem Statement

### 🧠 Rethinking Fraud Detection: It’s Not About Accuracy It’s About Prioritization Under Uncertainty

Most fraud detection systems fail not because they lack data or computing power, but because they optimize for the wrong thing. In high volume environments like credit card networks, the real challenge isn’t prediction, it’s prioritization.

You’re not trying to classify every transaction perfectly. You’re trying to elevate the few critical cases where intervention actually matters despite limited signal, overwhelming noise, and asymmetric cost.

⸻

### 🎯 The Real Objective: Risk-Sensitive Triage, Not Binary Classification

A fraud detection system isn’t a yes/no engine. It’s a risk triage system—ranking transactions by their likelihood of being fraudulent and enabling actionable thresholds based on operational trade-offs.

In this context:
* A “missed fraud” can mean financial loss, reputational damage, or regulatory penalties.
* A “false alarm” is just an inconvenient flag—tolerable up to a point.

So the goal shifts from maximizing accuracy to optimizing intervention impact under strict false/positive constraints.

⸻

### 📚 Theory-Aligned Modeling Approach

This requires models and evaluation strategies designed for:
* Extreme class asymmetry
* Decision prioritization
* Cost-aware intervention

Rather than:

`“How well can we classify fraud?”`

Ask:

`“How effectively can we surface the top 0.1% riskiest cases with acceptable noise?”`

⸻

#### Key Principles:
1. Rank-based Evaluation
    * Use precision-recall curves to evaluate performance across decision thresholds.
	* Prioritize PR/AUC, not ROC-AUC, for model selection.
2.	Threshold-Aware Metrics
	* Pick a working threshold that aligns with operational tolerance for false positives.
	* Measure F1, recall, and precision at that point.
3.	Probability Calibration
	* Don’t rely on raw scores—calibrate your outputs (e.g., Platt scaling, isotonic regression) so they reflect true intervention likelihood.
4.	Risk Framing over Classification
	* Use models that output well ranked risk scores, not just class predictions.
	* Gradient boosting and anomaly detection models often excel here.

⸻

### 📌 Bottom Line

You don’t need a perfect classifier you need a reliable, rank-aware signal that enables smart intervention. That’s where true impact lies in fraud detection.

---

## 🧠 Theoretical Framework: Hybrid Fraud Detection

### 🎯 Problem Definition

* We have a highly imbalanced binary classification problem: Legitimate ≫ Fraudulent (≈0.17%).  it is required to:

	1.	Detect rare frauds with minimal false positives.

	2.	Work under uncertainty (labels may lag, new fraud patterns emerge).

	3.	Provide a prioritization signal — not just a binary classifier.

### 🧭 Solution Strategy: Hybrid Anomaly Detection

Combine:

* Unsupervised anomaly detection (Isolation Forest) detects unknown fraud patterns, no label needed.

* Supervised learning learns from known frauds and real transaction history.

* Threshold tuning + PR-AUC/F1 metrics ensures real-world deployment value.

           Raw Data (creditcard.csv)
                    |
           ┌────────▼────────┐
           │  Preprocessing  │  (Standardize, clean)
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  IsolationForest│  (unsupervised)
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  Add Anomaly    │
           │     Scores      │
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │ Supervised Model│
           └────────┬────────┘
                    │
           ┌────────▼────────┐
           │  Threshold Tune │ (PR/F1/ConfMatrix)
           └────────┬────────┘
                    ▼
              Deployment / Reporting

## 🔍 Preprocessing

**Input Dataset:**
- Features: `Time`, `Amount`, `V1–V28`, `Class`
- Fraud Rate: ~0.17%

**Steps:**
- Scaled `Time` and `Amount` into `normTime`, `normAmount`
- Retained anonymized PCA components
- Verified class imbalance and explored feature distributions

**Output:**
- `processed dataset`

---

## 📊 Exploratory Data Analysis (EDA)

EDA focused on understanding the distribution of transaction features and the nature of fraud cases.
[Exploratory Data Analysis](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/notebooks/01_exploration.ipynb)
**Highlights:**
- Confirmed severe class imbalance
- Explored `Amount`, `Time`, and PCA components
- Compared feature distributions for fraud vs. legit cases

**Visuals:**
1. Class Distribution
![Class imbalance bar plot](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/class_distribution.png)
---
Less than 0.2% of all transactions are fraudulent a textbook example of class imbalance. This log-scaled chart visualizes the massive skew in class frequencies, with each bar labeled by percentage. Such imbalance challenges traditional classifiers, which tend to favor the majority class and ignore rare fraud cases.
----
2. Amount Distribution by Class
![Log-scaled histogram of transaction amounts by class](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/amount_distribution_by_class.png)
----
This plot shows how transaction amounts are distributed across classes. Fraudulent transactions tend to cluster around lower values, while legitimate purchases span a wider range. The log-scale x-axis helps surface these subtle fraud patterns that would otherwise be hidden.
----
3. Time Distribution by Class
![Histogram of transaction times by class](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/hourly_transaction_density.png)
----
This density plot reveals that while legitimate transactions peak during typical business hours, fraudulent activity appears more uniformly distributed possibly taking place during off-peak hours to avoid detection.
----
4. Correlation Heatmap
![Heatmap of correlations between all numeric features](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/correlation_matrix.png)
----
5. Boxplots of PCA features
- V1
![Boxplots comparing PCA features V1–V5](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/boxplot_V1_by_class.png)
- V2
![Boxplots comparing PCA features V1–V5](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/boxplot_V2_by_class.png)
- V3
![Boxplots comparing PCA features V1–V5](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/boxplot_V3_by_class.png)
- V4
![Boxplots comparing PCA features V1–V5](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/boxplot_V4_by_class.png)
- V5
![Boxplots comparing PCA features V1–V5](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/boxplot_V5_by_class.png)
----
6. Violin Plot V14
![Distribution of V14 shaped by class](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/violinplot_V14.png)
----
7. KDE Plot V14
![KDE density overlay comparing fraud vs legit for V14](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/kde_V14_fraud_vs_legit.png)

---
## 🧪 Unsupervised Modeling — Isolation Forest

**What it is:**
Isolation Forest isolates points by randomly selecting features and splitting values. Anomalies are isolated quicker due to their uniqueness.

**Pros:**
- Label-free detection
- Fast and scalable

**Cons:**
- Sensitive to sparse regions
- Doesn't use true fraud labels

**Setup:**
- Input: Preprocessed dataset
- Model: `IsolationForest(n_estimators=100, contamination='auto')`
- Outputs:
  - `anomaly_score` (continuous)
  - `predicted` (binary flag)

**Artifacts:**
- `isoforest_scored.csv`

**Visuals:**

1. Distribution of anomaly scores
![isoforest_score_hist.png](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/isoforest_score_hist.png)
----
2. Precision-recall of anomaly predictions
![Precision-recall of anomaly predictions](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/isoforest_pr_curve.png)
----
3. Isolation Forest fraud catch performance
![Isolation Forest fraud catch performance](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/isoforest_confusion_matrix.png)

---
## ➕ Augmenting with Anomaly Scores

**Purpose:**
Inject `anomaly_score` as an additional feature in supervised learning to enrich feature space.

**Input:** `processed_creditcard.csv + anomaly_score`
**Output:** Augmented dataset with `anomaly_score` as a new column


## 🌲 Supervised Modeling — Random Forest

**Model:** `RandomForestClassifier(n_estimators=100, class_weight='balanced')`

**Pros:**
- Robust against overfitting
- Handles feature interactions well

**Cons:**
- Less interpretable
- Slower in large-scale deployment

**Input:**
- All features + `anomaly_score`

**Output:**
- Trained model: `rf_model.joblib`
- Evaluation data: `X_test.csv`, `y_test.csv`, `y_probs.npy`

**Visuals:**
1. Confusion matrix at threshold=0.5
![Confusion matrix at threshold=0.5](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/rf_confusion_matrix.png)
----
2. PR curve for classifier probabilities
![PR curve for classifier probabilities](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/rf_pr_curve.png)
----
3. Feature ranking by Gini importance
![Feature ranking by Gini importance](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/rf_feature_importance.png)
----
4. Raw metrics export
![ Metrics](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/keyresult1.jpeg)

---
## 🎯 Threshold Tuning

**What it is:**
Rather than predicting fraud if probability > 0.5, we evaluate different thresholds to optimize recall or F1 score.

**Input:** `y_probs`, `y_test`

**Processing:**
- Precision–Recall analysis
- Threshold vs F1 plotting
- Confusion matrix generation

**Output:**
- Tuned threshold
- Final metrics at best threshold

**Visuals:**
1. Precision/Recall/F1 vs threshold
![Precision/Recall/F1 vs threshold](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/threshold_vs_metrics.png)
----
2. Confusion matrix @ tuned threshold
![Confusion matrix @ tuned threshold](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/final_confusion_matrix_thresh.png)
----
3. Report with new metrics
<pre>
<code>

              precision    recall  f1-score   support

           0     0.9997    0.9999    0.9998     56864
           1     0.9318    0.8367    0.8817        98

    accuracy                         0.9996     56962
   macro avg     0.9658    0.9183    0.9408     56962
weighted avg     0.9996    0.9996    0.9996     56962
</code>
</pre>


---

## 📤 Deployment and Reporting

**Artifacts saved:**
- Models in `/models`
- Data splits and scores in `/data`
- Visualizations and logs in `/assets`

**Use cases:**
- Batch scoring pipeline
- Streamlit dashboard
- Reporting via automated summaries or plots

---

## 📈 Key Results

![ Metrics](https://raw.githubusercontent.com/x-rojas-io/credit-card-fraud-detection/main/assets/keyresult.jpeg)

> Tuning improved recall and F1 score while trading off a minor precision loss suitable when minimizing false negatives is the priority.

----
