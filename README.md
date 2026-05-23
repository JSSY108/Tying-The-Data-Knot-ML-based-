# Tying the (Data) Knot: Love, Life & Likes 💘

An interactive Streamlit web application serving a pre-trained **LightGBM Classifier** to simulate and evaluate dating app match outcomes based on user behavioral telemetry. 

This repository was created as part of the group assignment for the **WIA1006/WID3006 Machine Learning** course, Faculty of Computer Science & Information Technology (FCSIT), **University of Malaya** (Semester 2, Session 2025/2026).

---

## 🚀 Live Application & Deployment

The application is configured to run out-of-the-box on local development environments and is pre-packaged with all dependencies required for hosting on the **Streamlit Community Cloud**.

### Installation & Local Run
To run the dashboard locally, make sure you have Python (version 3.9 to 3.12 recommended) installed, and follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JSSY108/Tying-The-Data-Knot-ML-based-.git
   cd Tying-The-Data-Knot-ML-based-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---

## 🔮 Core Dashboard Views

The interface has been split into three distinct views accessible via the sidebar navigation:

### 1. Scenario & Match Simulator
Allows evaluators to input a simulated user profile dividing background demographics (gender, orientation, zodiac, income, body type, interests) from active app telemetry (swipe right ratio, app usage time, messages sent, and emoji usage rate).
*   **The Preprocessing Pipeline:** Raw inputs undergo manual feature engineering (cyclic sin/cos hour conversion, high-cardinality multi-label interest binarization), percentile clipping, ordinal/nominal encoding, and scaling (`RobustScaler`) inside a pipeline fitted on the training split (`X_train`) to guarantee mathematical parity with training.
*   **Inference:** The processed 1-row input is subset to the 30 features selected by Mutual Information, scoring the predicted match state (e.g. *Mutual Match, Date Happened, Ghosted, Blocked, Catfished*) and displaying all category probabilities.

### 2. Behavioral Contrast Engine
Allows researchers to slice demographics to identify behavioral clusters:
*   Includes a **Mandatory Outcome Comparison** filter to isolate specific groups (such as comparing *Mutual Match* vs. *Ghosted*) to avoid cluttered charts.
*   Plots Plotly scatter charts of interaction habits (swipe ratio vs app usage time) and communication habits (message volume vs emoji rate) with marker opacities set to `0.40` to visualize dense overlaps.
*   **Interactive Guide:** Features an integrated accordion panel prompting users to explore the **"Power Swiper" Paradox** (does indiscriminate swiping actually secure matches, or does it lead directly to being ghosted?).

### 3. Model Benchmarks & Critique
Displays the final tuning leaderboard evaluating the LightGBM classifier against other models (Logistic Regression, Random Forest, Extra Trees, XGBoost, and **auto-sklearn**).

*   **Best Evaluation Method:** Since this is a 10-class classification task with perfectly balanced classes (~5,000 instances each, representing exactly 10% of the dataset), **Classification Accuracy** and **Macro-averaged F1-Score** are the best evaluation metrics. While accuracy gives an overall performance summary, the **Macro F1-Score** is the gold standard here because it calculates the unweighted mean of F1-scores across all 10 target outcomes. This ensures that any performance drops in minor or highly noisy classes are not masked.
*   **Model Comparison Framework:** To compare our models, we used a Stratified Train-Test split (70% train, 30% test) to preserve outcome proportions across splits. We evaluated each framework on:
    1.  **Macro F1-score & Accuracy:** Evaluating classification capability against the random baseline.
    2.  **Confusion Matrix Analysis:** Plotting error distributions to check if the models isolated any real target signals.
    3.  **Tuning Efficiency:** Comparing search runtime duration (seconds).
*   **Comparison to auto-sklearn:**
    *   **Custom LightGBM Performance:** Our tuned LightGBM classifier achieved a Macro F1 of **0.0997** (9.97%) and test accuracy of **0.0998** (9.98%), taking 172.93 seconds to execute randomized search cross-validation.
    *   **auto-sklearn Baseline:** The automated machine learning framework (`auto-sklearn`) was configured with a 1-hour time budget (3600 seconds) and achieved a validation Macro F1 of **0.1000** and test accuracy of **0.1012**.
    *   **Academic Verdict:** Our custom tuned LightGBM performs identically to the auto-sklearn automated baseline. Both converge precisely at the **10.0% random baseline**, proving that neither automated search nor high-capacity ensembling can extract predictive signal from a synthetic target vector that consists of uniform random noise.

---

## ⚙️ Mathematical Preprocessing Alignment

To prevent feature distribution drift and ensure zero prediction skew, the interactive simulator's input pipeline is mathematically aligned with the training configurations:

1. **Outlier Capping:** App telemetry inputs (`app_usage_time_min`, `message_sent_count`, `likes_received`, and `mutual_matches`) are clipped using the exact 1st and 99th percentile bounds calculated from the training split.
2. **Cyclic Time Conversions:** The user's active hour input ($H \in [0, 23]$) is mapped to circular dimensions using $\sin(2\pi H / 24)$ and $\cos(2\pi H / 24)$ trigonometric transforms.
3. **Ordinal Mapping:** Education levels and income brackets are encoded using a pre-fit `OrdinalEncoder` using the exact hierarchical categories defined during training.
4. **One-Hot Nominal Encoding:** Context variables (gender, orientation, zodiac, location, body type, intent) are expanded into dummy indicator variables via `OneHotEncoder`.
5. **Robust Scaling:** Numeric metrics are scaled via `RobustScaler` using the medians and Interquartile Ranges (IQR) of the training dataset split, rather than standard normalizing.
6. **Mutual Information Subset:** Out of the resulting 114 preprocessed features, the app extracts the exact 30 features selected by Mutual Information to match the LightGBM input vector.

---

## 📂 Project Structure

*   `app.py` — The core executable Streamlit application.
*   `requirements.txt` — Python libraries needed for Cloud/local deployment.
*   `final_model.pkl` — The pre-trained LightGBM classification pipeline.
*   `dating_app_behavior_dataset_extended.csv` — The dating app behaviors dataset (50,000 records).
*   `WIA1006 Machine Learning Group Assignment.ipynb` — Jupyter Notebook documenting EDA, feature selection, and model training.
*   `.gitignore` — Prevents committing Python and notebook caches (`__pycache__`, etc.).
