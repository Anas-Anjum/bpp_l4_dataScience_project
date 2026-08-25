
# Executive Summary
As a Data Scientist working at one of the largest banks in the UK, the effects of fraud on the institution and on customers are all too apparent. Whilst there are many rules-based fraud detection systems utilised by the bank, the high false positive rates and inability to dynamically adapt to trends can lead to friction with genuine everyday customers. Using a dataset of anonymised credit card transactions, this project addresses the challenge of creating a model which can detect and predict fraud in an extremely imbalanced dataset where fraud only accounts for 0.172% of total volume (492/284,807). The main goal is to maximise fraud capture (Recall) whilst strictly limiting false positives to preserve systems integrity, operational efficiency and customer trust.
This project implements an end-to-end data science pipeline built to demonstrate production-grade data engineering, machine learning, and interactive visualization practices:
•	Data Infrastructure & Engineering: Designed an automated ETL1 pipeline using Python (Pandas2, NumPy3, SQLAlchemy4) that ingests raw transaction feeds, applies robust scaling4 to temporal and monetary features, and handles severe class imbalance using SMOTE5 (Synthetic Minority Over-sampling Technique) combined with Tomek Links6.
•	Analytical Techniques & Modelling: Trained and evaluated multiple anomaly detection and supervised learning models (Logistic Regression, Random Forest, and XGBoost)7. Optimized thresholds using Precision-Recall (PR-AUC)8 curves rather than standard ROC-AUC9 to account for skewness.
•	Visualization & Reporting: Developed an interactive dashboard (Streamlit / Power BI) displaying key operational KPIs, real-time transaction scoring simulations, and SHAP (SHapley Additive exPlanations) visual insights for model explainability.

# ETL Pipline:
1.	Robust Scaling (Amount & Time)
Financial transaction amounts are heavily skewed (most purchases are small, but a few are thousands of dollars). Standard scaling methods fail here because extreme outliers pull the mean and variance.
•	Method used: RobustScaler from scikit-learn.
•	How it works: Instead of using the mean and standard deviation, it subtracts the median and divides by the Interquartile Range (IQR) (the spread between the 25th and 75th percentiles).
•	New columns created: scaled_amount and scaled_time.
2. Feature Engineering: Cyclic Time Transformation
The raw dataset stores time as continuous seconds elapsed since the first transaction over a 48-hour period. Machine learning models cannot easily recognize daily recurring fraud patterns from raw seconds.
Calculation: (Time // 3600) % 24
How it works: Converts raw seconds into an integer representing the hour of the day (0 through 23).
New column created: hour_of_day.
Why it matters: Fraud rates spike at specific times of day (e.g., late-night transactions when cardholders are asleep). This feature explicitly gives models time-of-day context.
3. Redundancy Cleanup & Column Dropping
To avoid feature redundancy and keep the dataset lean:
Action: Drops the original unscaled Amount and Time columns.
Result: Replaces raw values with scaled_amount, scaled_time, and hour_of_day, while preserving the anonymized PCA features (V1 through V28) and target variable (Class).





## Before vs. After Transformation

Feature	Raw Data (creditcard.csv)	Transformed Data (creditcard_clean.parquet)
Transaction Amount	Amount (€0.00 to €25,691.16)	scaled_amount (Outlier-resistant zero-centered range)
Transaction Time	Time (0 to 172,792 seconds)	scaled_time (Scaled) + hour_of_day (0–23 hour integers)
Anonymized V-Features	V1 to V28	V1 to V28 (Preserved intact)
Fraud Label	Class (0 = Legitimate, 1 = Fraud)	Class (Preserved intact)

# 📊 Exploratory Data Analytics & Domain Insights > 🔗 **Deep Dive:** View the complete, fully documented [Exploratory Data Analysis Notebook](notebooks/01_eda_and_analytics.ipynb) containing step-by-step statistical calculations and visualisations.

All exploratory data analysis was conducted on the transformed data extracted directly from the SQLite Data Warehouse (`data/processed/fraud_warehouse.db`). Transaction amounts are originally denominated in **Euros (€)**.

---

## 1. Extreme Class Imbalance Analysis
![Class Imbalance Distribution](docs/class_imbalance.png)

* **Key Finding:** The dataset exhibits extreme class imbalance: **99.83% Legitimate** (€284,315 transactions) vs. **0.17% Fraudulent** (€492 transactions).
* **Analytical Impact:** Standard evaluation metrics like accuracy are misleading (a dummy model predicting 'Legitimate' achieves 99.83% accuracy). Downstream models must be evaluated using **Precision-Recall AUC (PR-AUC)** and **F1-Score**.

---

## 2. Temporal Behavior & Hourly Spikes
![Hourly Fraud Pattern](docs/hourly_fraud_pattern.png)

* **Key Finding:** While overall legitimate transaction volume drops significantly during off-peak hours (2:00 AM – 5:00 AM), the **relative fraud rate (%) spikes drastically**.
* **Business Takeaway:** Fraudulent actors exploit off-peak hours when victims are asleep and unlikely to notice real-time push notifications or block compromised cards.

---

## 3. Key Predictive Feature Drivers
![Feature Correlations](docs/feature_correlations.png)

* **Key Finding:** Anonymized PCA features **`V17`**, **`V14`**, **`V12`**, and **`V10`** show strong negative correlations with fraud, while **`V4`** and **`V11`** show strong positive correlations.
* **Engineering Impact:** These features provide clear signal-to-noise ratios and serve as top candidate features for classifier training.

## ⚙️ Detailed Explanation of Model Types & Implementation Differences

### 1. How Model Types Work

* **Random Forest (`RandomForestClassifier`)**: An **ensemble bagging (bootstrap aggregating) method**. It constructs multiple decision trees in parallel on bootstrapped subsets of the training data and averages their predictions (or takes a majority vote) to reduce variance. It is naturally resistant to overfitting and performs exceptionally well out-of-the-box on structured tabular data.
  * **Reference:** Read the official [`RandomForestClassifier` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
* **XGBoost (`xgb.XGBClassifier`)**: An **ensemble gradient boosting method**. It constructs decision trees sequentially, where each new tree is built to minimize residual errors made by previous trees using gradient descent on a specified loss function. It typically offers higher predictive power on complex tabular datasets but requires careful hyperparameter tuning.
  * **Reference:** Read the official [XGBoost Python API Reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier).

---

### 2. How to Code Imbalance Handling: Random Forest vs. XGBoost

Because fraudulent transactions account for only **~0.17%** of instances, standard objective functions default to predicting the majority class. Both libraries handle this using specialized cost-sensitive weighting parameters:

| Parameter / Technique | Random Forest Implementation | XGBoost Implementation |
| :--- | :--- | :--- |
| **Imbalance Parameter** | `class_weight='balanced'` | `scale_pos_weight = negative_count / positive_count` |
| **How it Works** | Automatically adjusts class weights inversely proportional to class frequencies during tree node splits: $w_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_j}$. | Scales the gradient calculations for positive instances to heavily penalize missing fraud cases during sequential boosting iterations. |
| **Execution Syntax** | `RandomForestClassifier(class_weight='balanced')` | `xgb.XGBClassifier(scale_pos_weight=num_neg/num_pos)` |

* **Documentation Links:**
  * Learn how [`class_weight='balanced'` calculates penalties](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) in Scikit-Learn.
  * Read the guide on [tuning `scale_pos_weight` in XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).

---

### 3. Why We Avoid Simple Accuracy & Use Imbalanced Metrics

#### The Accuracy Trap
In a dataset where **99.83%** of transactions are legitimate and **0.17%** are fraudulent:

$$\text{Dummy Model Accuracy} = \frac{284,315 \text{ (Legitimate)}}{284,807 \text{ (Total)}} = 99.83\%$$

A naive classifier that blindly predicts *every* transaction as "Legitimate" achieves **99.83% accuracy** while failing to detect a single instance of fraud.

#### Selected Evaluation Metrics

* **Precision ($\text{Positive Predictive Value}$)**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  * **Meaning:** Out of all transactions flagged as fraud by the model, how many were actual fraud? High precision minimizes false alarms, avoiding unnecessary account blocks and customer friction.
* **Recall ($\text{Sensitivity / True Positive Rate}$)**:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  * **Meaning:** Out of all actual fraudulent transactions that occurred, how many did the model successfully catch? High recall minimizes uncaptured fraud losses and direct financial risk.
* **F1-Score**:
  $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  * **Meaning:** The harmonic mean balancing Precision and Recall into a single scalar metric.
* **PR-AUC (Precision-Recall Area Under Curve)**:
  * **Meaning:** Measures the trade-off between Precision and Recall across all decision probability thresholds ($0.0$ to $1.0$).
  * **Why it matters:** Unlike ROC-AUC (which yields inflated performance scores on imbalanced data due to large True Negative counts), PR-AUC focuses exclusively on minority class detection performance.
  * **Documentation Link:** Read the [`scikit-learn` Precision-Recall Guide](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).
