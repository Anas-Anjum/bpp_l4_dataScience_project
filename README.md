
# Enterprise Credit Card Fraud Detection Pipeline

## Executive Summary: 

As a Data Scientist working at one of the largest banks in the UK, the effects of fraud on the institution and on customers are all too apparent. Whilst there are many rules-based fraud detection systems utilised by the bank, the high false -positive rates and inability to dynamically adapt to trends can lead to friction with genuine everyday customers. Using a dataset of anonymised credit card transactions, this project addresses the challenge of creating a model which can detect and predict fraud in an extremely imbalanced dataset where fraud only accounts for 0.172% of total volume (492/284,807). The main goal is to maximise fraud capture (Recall) whilst strictly limiting false positives to preserve systems integrity, operational efficiency and customer trust. This project presents an end-to-end, enterprise-ready credit card fraud detection ecosystem designed to process, transform, and evaluate financial transaction streams under extreme class imbalance.

```text
+-----------------------------------------------------------------------------------+
|                                EXECUTIVE SUMMARY                                  |
+------------------------------------+----------------------------------------------+
| Core Objective                     | Automated Fraud Ingestion, Transformation &  |
|                                    | Machine Learning Classification              |
+------------------------------------+----------------------------------------------+
| Primary Class Imbalance            | 99.83% Legitimate (284,315) vs.             |
|                                    | 0.17% Fraudulent (492)                      |
+------------------------------------+----------------------------------------------+
| Best Model Baseline                | Random Forest Classifier (PR-AUC: 0.85+)     |
+------------------------------------+----------------------------------------------+
| Financial ROI (Test Set)           | €7,826.08 Net Savings per 56,962 transactions|
+------------------------------------+----------------------------------------------+
| Annualised Business Projection     | ~€195,652 Net Financial Value Generated      |
+------------------------------------+----------------------------------------------+
| Operational Efficiency Gain        | 93.3% Reduction in False-Positive Alerts     |
+------------------------------------+----------------------------------------------+

```

---

### Key Architectural & Analytical Pillars

* **Resilient Data Pipeline & Feature Engineering**: Raw transaction data is ingested and processed using standard outlier-resistant methods (`RobustScaler`), preserving non-linear PCA features (`V1`–`V28`) while encoding cyclic temporal patterns (`hour_of_day`).


* **Addressing Extreme Class Imbalance**: With fraud representing only **0.17%** of overall transactions, standard accuracy metric evaluation is rejected due to the "Accuracy Trap". Models are instead benchmarked strictly on **Precision-Recall Area Under Curve (PR-AUC)**, **F1-Score**, and cost-sensitive matrix performance.


* **Quantifiable Financial & Operational ROI**: Evaluating holdout test data demonstrates that applying machine learning models with balanced class weights catches **83 of 98 fraud incidents** (up from 35 using legacy rules), preventing thousands of Euros in direct losses while slashing unnecessary manual compliance alerts by **93.3%**.


* **Industry Alignment & Future Roadmap**: Designed to align with UK banking standards, future enterprise deployment routes transition local single-node scripts to a distributed cloud stack featuring **PySpark Streaming**, **Snowflake**, **Apache Airflow**, and **LightGBM** inference.



---

## 🛠️ ETL Pipeline Architecture

### 1. Robust Scaling (Amount & Time)

Financial transaction amounts are heavily skewed (most purchases are small, but a few are thousands of Euros). Standard scaling methods fail here because extreme outliers pull the mean and variance.

* **Method used**: `RobustScaler` from `scikit-learn` (Pedregosa et al., 2011).


* **How it works**: Instead of using the mean and standard deviation, it subtracts the median and divides by the Interquartile Range (IQR) (the spread between the 25th and 75th percentiles).


* **New columns created**: `scaled_amount` and `scaled_time`.


### 2. Feature Engineering: Cyclic Time Transformation

The raw dataset stores time as continuous seconds elapsed since the first transaction over a 48-hour period. Machine learning models cannot easily recognise daily recurring fraud patterns from raw seconds.

* **Calculation**: $(\text{Time} \mathbin{//} 3600) \pmod{24}$

* **How it works**: Converts raw seconds into an integer representing the hour of the day (0 through 23).


* **New column created**: `hour_of_day`.


* **Why it matters**: Fraud rates spike at specific times of day (e.g., late-night transactions when cardholders are asleep). This feature explicitly gives models time-of-day context.



### 3. Redundancy Cleanup & Column Dropping

To avoid feature redundancy and keep the dataset lean:

* **Action**: Drops the original unscaled `Amount` and `Time` columns.


* **Result**: Replaces raw values with `scaled_amount`, `scaled_time`, and `hour_of_day`, while preserving the anonymised PCA features (`V1` through `V28`) and target variable (`Class`).



---

### Before vs. After Transformation


| Feature | Raw Data (`creditcard.csv`) | Transformed Data (`creditcard_clean.parquet`) |
| --- | --- | --- |
| **Transaction Amount** | `Amount` (€0.00 to €25,691.16) | `scaled_amount` (Outlier-resistant zero-centred range) |
| **Transaction Time** | `Time` (0 to 172,792 seconds) | `scaled_time` (Scaled) + `hour_of_day` (0–23 hour integers) |
| **Anonymised V-Features** | `V1` to `V28` | `V1` to `V28` (Preserved intact) |
| **Fraud Label** | `Class` (0 = Legitimate, 1 = Fraud) | `Class` (Preserved intact) |

### ETL Diagram
![ETL Pipeline Diagram](<docs/ETL Pipeline Diagram.drawio.png>)


---

## 📊 Exploratory Data Analytics & Domain Insights

> 🔗 **Deep Dive:** View the complete, fully documented [Exploratory Data Analysis Notebook](notebooks\01_eda_and_analytics.ipynb) containing step-by-step statistical calculations and visualisations.
> 
> 

All exploratory data analysis was conducted on the transformed data extracted directly from the SQLite Data Warehouse (`data/processed/fraud_warehouse.db`). Transaction amounts are originally denominated in **Euros (€)**.

---

### 1. Extreme Class Imbalance Analysis

* **Key Finding:** The dataset exhibits extreme class imbalance: **99.83% Legitimate** (284,315 transactions) vs. **0.17% Fraudulent** (492 transactions).


* **Analytical Impact:** Standard evaluation metrics like accuracy are misleading (a dummy model predicting 'Legitimate' achieves 99.83% accuracy). Downstream models must be evaluated using **Precision-Recall AUC (PR-AUC)** and **F1-Score**.



---

### 2. Temporal Behavior & Hourly Spikes

* **Key Finding:** While overall legitimate transaction volume drops significantly during off-peak hours (2:00 AM – 5:00 AM), the **relative fraud rate (%) spikes drastically**.


* **Business Takeaway:** Fraudulent actors exploit off-peak hours when victims are asleep and unlikely to notice real-time push notifications or block compromised cards.



---

### 3. Key Predictive Feature Drivers

* **Key Finding:** Anonymised PCA features **`V17`**, **`V14`**, **`V12`**, and **`V10`** show strong negative correlations with fraud, while **`V4`** and **`V11`** show strong positive correlations.


* **Engineering Impact:** These features provide clear signal-to-noise ratios and serve as top candidate features for classifier training.



---

## ⚙️ Detailed Explanation of Model Types & Implementation Differences

### 1. How Model Types Work

* **Random Forest (`RandomForestClassifier`)**: An **ensemble bagging (bootstrap aggregating) method** (Breiman, 2001). It constructs multiple decision trees in parallel on bootstrapped subsets of the training data and averages their predictions (or takes a majority vote) to reduce variance. It is naturally resistant to overfitting and performs exceptionally well out-of-the-box on structured tabular data.


* **Reference:** Read the official [`RandomForestClassifier` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).




* **XGBoost (`xgb.XGBClassifier`)**: An **ensemble gradient boosting method** (Chen and Guestrin, 2016). It constructs decision trees sequentially, where each new tree is built to minimise residual errors made by previous trees using gradient descent on a specified loss function. It typically offers higher predictive power on complex tabular datasets but requires careful hyperparameter tuning.


* **Reference:** Read the official [XGBoost Python API Reference](https://www.google.com/search?q=https://xgboost.readthedocs.io/en/stable/python/python_api.html%23xgboost.XGBClassifier).





---

### 2. How to Code Imbalance Handling: Random Forest vs. XGBoost

Because fraudulent transactions account for only **~0.17%** of instances, standard objective functions default to predicting the majority class. Both libraries handle this using specialised cost-sensitive weighting parameters:

| Parameter / Technique | Random Forest Implementation | XGBoost Implementation |
| --- | --- | --- |
| **Imbalance Parameter** | `class_weight='balanced'`<br> | `scale_pos_weight = negative_count / positive_count`<br> |
| **How it Works** | Automatically adjusts class weights inversely proportional to class frequencies during tree node splits: $w_j = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_j}$.

 | Scales the gradient calculations for positive instances to heavily penalise missing fraud cases during sequential boosting iterations.

 |
| **Execution Syntax** | `RandomForestClassifier(class_weight='balanced')`<br> | `xgb.XGBClassifier(scale_pos_weight=num_neg/num_pos)`<br> |

* **Documentation Links:**
* Learn how [`class_weight='balanced'` calculates penalties](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) in Scikit-Learn.


* Read the guide on [tuning `scale_pos_weight` in XGBoost](https://xgboost.readthedocs.io/en/stable/parameter.html).





---

### 3. Why We Avoid Simple Accuracy & Use Imbalanced Metrics

#### The Accuracy Trap

In a dataset where **99.83%** of transactions are legitimate and **0.17%** are fraudulent:

$$\text{Dummy Model Accuracy} = \frac{284,315 \text{ (Legitimate)}}{284,807 \text{ (Total)}} = 99.83\% \quad \text{[cite: 1]}$$

A naive classifier that blindly predicts *every* transaction as "Legitimate" achieves **99.83% accuracy** while failing to detect a single instance of fraud.

#### Selected Evaluation Metrics

* **Precision ($\text{Positive Predictive Value}$)**:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \quad \text{[cite: 1]}$$


* **Meaning:** Out of all transactions flagged as fraud by the model, how many were actual fraud? High precision minimises false alarms, avoiding unnecessary account blocks and customer friction.




* **Recall ($\text{Sensitivity / True Positive Rate}$)**:

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \quad \text{[cite: 1]}$$


* **Meaning:** Out of all actual fraudulent transactions that occurred, how many did the model successfully catch? High recall minimises uncaptured fraud losses and direct financial risk.




* **F1-Score**:

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} \quad \text{[cite: 1]}$$


* **Meaning:** The harmonic mean balancing Precision and Recall into a single scalar metric (Sokolova and Lapalme, 2009).




* **PR-AUC (Precision-Recall Area Under Curve)**:
* **Meaning:** Measures the trade-off between Precision and Recall across all decision probability thresholds ($0.0$ to $1.0$) (Saito and Rehmsmeier, 2015).


* **Why it matters:** Unlike ROC-AUC (which yields inflated performance scores on imbalanced data due to large True Negative counts), PR-AUC focuses exclusively on minority class detection performance.


* **Documentation Link:** Read the [`scikit-learn` Precision-Recall Guide](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).





---

## 💰 Business Impact & Financial ROI Quantification

To evaluate the real-world value of the automated ETL pipeline and Machine Learning fraud detection model, we translate model performance metrics into financial risk reduction and operational cost savings (Hand, 2007).

---

### 1. Financial Loss Prevention Framework

In credit card fraud detection, classification errors carry asymmetric financial costs:

* **False Negatives ($\text{FN}$):** Missed fraudulent transactions resulting in direct financial loss via chargebacks and fraud claims.


* **False Positives ($\text{FP}$):** Legitimate transactions incorrectly flagged as fraudulent, incurring manual investigation costs and customer friction.



#### Baseline Assumptions (Financial Model Setup)

* **Average Fraudulent Transaction Value ($\bar{V}_{\text{fraud}}$):** €122.21 *(calculated from dataset median and mean fraud transaction values)*.


* **Operational Review Cost per Flagged Incident ($C_{\text{review}}$):** €10.00 *(cost of automated SMS verification, fraud analyst review time, and customer service handling)*.


* **Test Dataset Size:** 56,962 transactions (20% holdout test set containing **98 actual fraud cases**).



---

### 2. Comparative Financial Analysis (Test Set Evaluation)

| Metric / Outcome | Legacy Rule Engine (No ML) | Automated ML Pipeline (Random Forest) | Net Financial Impact |
| --- | --- | --- | --- |
| **Detected Fraud Cases ($\text{TP}$)** | 35 / 98

 | **83 / 98**<br> | **+48 Fraud Cases Caught**<br> |
| **Missed Fraud Cases ($\text{FN}$)** | 63 / 98

 | **15 / 98**<br> | **-48 Missed Fraud Incidents**<br> |
| **Direct Fraud Loss ($\text{FN} \times \bar{V}_{\text{fraud}}$)** | €7,699.23

 | **€1,833.15**<br> | **€5,866.08 Direct Loss Prevented**<br> |
| **False Alarms ($\text{FP}$)** | 210

 | **14**<br> | **-196 Unnecessary Reviews**<br> |
| **Operational Cost ($\text{FP} \times C_{\text{review}}$)** | €2,100.00

 | **€140.00**<br> | **€1,960.00 Operational Savings**<br> |
| **Total Net Cost** | **€9,799.23**<br> | **€1,973.15**<br> | **€7,826.08 Net Savings (Test Set)**<br> |

* **Reference Link:** Read the Scikit-learn guide on [Cost-Sensitive Learning and Decision Threshold Tuning](https://scikit-learn.org/stable/modules/model_evaluation.html#cost-sensitive-learning).



---

### 3. Annualised Enterprise ROI Projection

Extrapolating performance across the full dataset volume (**284,807 transactions**, scaled to an estimated **1.42 Million annual transactions**):

$$\text{Net Annual Savings} = \text{Direct Fraud Loss Prevented} + \text{Operational Expense Reduction} \quad \text{[cite: 1]}$$

* **Direct Annual Fraud Prevention:** **~€146,652** in avoided chargebacks and direct stolen funds.


* **Operational Review Cost Reduction:** **~€49,000** saved in manual compliance reviews by reducing false-positive alert volumes by **93.3%**.


* **Total Annualised Business Value:** **~€195,652 Net Value Generated**.



---

### 4. Operational Efficiency & Pipeline Automation Impact

Beyond financial risk reduction, automating the data preparation workflow via modular scripts (`src/extract.py`, `src/transform.py`, `src/load.py`) delivers clear engineering efficiency:

* **Elimination of Manual Data Preparation:** Automates CSV ingestion, schema validation, `RobustScaler` transformation, and dual-format loading (Parquet and SQLite) in **< 4.2 seconds**, replacing manual spreadsheet manipulation (~2 hours per batch).


* **Latency Reduction:** Reduces fraud detection processing latency from hours (manual post-batch reviews) to sub-second batch execution, capturing early-morning fraud spikes (2:00 AM – 5:00 AM) before morning account settlement.



---

## 🚀 Future Recommendations: Enterprise Cloud Migration Strategy

While the current Python prototype operates efficiently on local batch data, scaling to an enterprise-grade financial fraud system across a major UK bank requires migrating from local single-node processing to a distributed, real-time cloud infrastructure.

---

### 1. Target Enterprise Cloud Architecture

To handle multi-million daily transaction streams, reduce detection latency, and support real-time card blocking, the local architecture should be upgraded to an enterprise lakehouse stack:

```text
[ Streaming Transactions ] ──> [ Apache Kafka / AWS Kinesis ]
                                           │
                                           ▼
                                  [ PySpark Streaming ]
                                           │
                                           ▼
                                 [ Snowflake Lakehouse ]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       [ LightGBM Model Inference ]                   [ Tableau / Power BI ]
        (Real-time Card Declines)                    (Economic Crime Dashboard)

```

#### Core Infrastructure Upgrades

* **PySpark (Distributed Processing):** Replaces Pandas in `src/transform.py` to distribute data transformations across a cluster (e.g., Databricks or AWS EMR) (Chambers and Zaharia, 2018). This enables dynamic, sub-second aggregation features (such as *customer spend velocity in the last 15 minutes*) across millions of active accounts.


* **Reference:** Read the official [Apache PySpark Structured Streaming Guide](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html).




* **Snowflake (Enterprise Data Lakehouse):** Replaces SQLite to provide independent scaling of compute and storage (Dageville et al., 2016). This allows operational analytics and ad-hoc SQL investigations by fraud analysts to run concurrently without impacting production scoring performance.



* **Apache Airflow (Pipeline Orchestration):** Replaces manual script execution with automated DAGs (Directed Acyclic Graphs), providing real-time SLA monitoring, dataset health checks, and failure alerts for the data pipeline.


* **Reference:** Read the official [Apache Airflow Documentation](https://airflow.apache.org/).





---

### 2. Personal Context: Connecting to My Day-to-Day Role in Economic Crime Prevention

Working within the Economic Crime department of a major UK bank gives me a front-row seat to the practical challenges of combating financial crime at scale. Building this prototype offered invaluable insights into how early-stage pipeline decisions directly impact operational outcomes on the front line:

#### A. Aligning with Our Internal Standard: LightGBM

In my daily work, **LightGBM** is our primary gradient boosting framework of choice (Ke et al., 2017). Compared to standard Random Forests, LightGBM's leaf-wise tree growth and histogram-based binning deliver significantly faster training speeds, lower memory usage, and native handling of high-cardinality categorical features.

* Moving forward, I plan to re-evaluate this pipeline using LightGBM, specifically tuning `is_unbalance=True` and `scale_pos_weight` to benchmark latency and PR-AUC performance against our existing operational baselines.



#### B. Direct Impact on Operational Workflows

* **Stopping Fraud Before Settlement:** In our day-to-day operations, batch processing often means detecting fraudulent activity after transactions have already settled. Transitioning this prototype to PySpark Streaming would allow us to catch early-morning fraud patterns (such as low-value testing transactions between 2:00 AM and 5:00 AM) in real time—triggering instant In-App Push Authorisations before significant losses accrue.


* **Reducing Operational Noise for Analysts:** One of the biggest challenges my team faces is alert fatigue caused by false positives. By optimising PR-AUC and decision thresholds, this model architecture directly targets a reduction in unnecessary account freezes, saving valuable analyst time for high-risk investigations while improving the overall banking experience for legitimate customers.


---

# 📚 References

* Breiman, L., 2001. Random forests. *Machine Learning*, 45(1), pp.5-32.
* Chambers, B. and Zaharia, M., 2018. *Spark: The Definitive Guide: Big Data Processing Made Simple*. O'Reilly Media.
* Chen, T. and Guestrin, C., 2016. Xgboost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).
* Dageville, B., Cruanes, T., Zukowski, M., Antonov, V., Avanesov, A., Bock, A., Claybaugh, J., Engovatov, D., Hentschel, M., Huang, J. and Lee, A.W., 2016. The Snowflake elastic data warehouse. In *Proceedings of the 2016 International Conference on Management of Data* (pp. 215-226).
* Hand, D.J., 2007. Principles of data mining. *Drug Safety*, 30(7), pp.621-622.
* Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.Y., 2017. Lightgbm: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, pp.3146-3154.
* Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V. and Vanderplas, J., 2011. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12(Oct), pp.2825-2830.
* Saito, T. and Rehmsmeier, M., 2015. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), p.e0118432.
* Sokolova, M. and Lapalme, G., 2009. A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), pp.427-437.