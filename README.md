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
*   **The Academic Critique:** Discusses why all frameworks converge at **~10% classification accuracy** (equal to random guessing on a 10-class uniform split). It critiques the synthetic dataset, illustrating that when targets contain uniform noise, high-capacity boosting models simply overfit local noise patterns instead of learning generalizable relationships.

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
