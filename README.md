# 🎓 AI-Powered Student Placement Prediction & Career Guidance System

An end-to-end machine learning system that predicts a student's placement
probability, diagnoses skill gaps, and generates personalized AI career
guidance — wrapped in an interactive Streamlit dashboard.

---

## ✨ Features

- **Synthetic placement dataset** — 1200+ realistic student records across 12 features
- **Full preprocessing pipeline** — missing value imputation, IQR outlier capping, feature scaling
- **Exploratory Data Analysis** — correlation heatmap, distributions, pair plots, feature importance
- **5 ML models compared** — Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM
- **Hyperparameter tuning** via `GridSearchCV` with before/after comparison
- **Placement probability prediction** with confidence levels
- **Rule-based Skill Gap Analysis Engine** — flags weak areas with targeted recommendations
- **AI Career Guidance** powered by the **Gemini API** — strengths/weaknesses analysis, tech
  stack suggestions, project ideas, certifications, interview strategy, career path ranking
- **Interactive Streamlit dashboard** — student form, probability meter, skill-gap cards,
  radar chart, benchmark comparisons, downloadable report

---

## 🗂️ Project Structure

```
placement_prediction_system/
├── data/
│   ├── generate_dataset.py      # synthetic dataset generator
│   └── placement_dataset.csv    # generated dataset (1200 records)
├── notebooks/
│   └── Model_Training.ipynb     # full Colab notebook: EDA → training → tuning → Gemini
├── models/
│   └── placement_model.pkl      # trained model + scaler (pickled)
├── app/
│   └── streamlit_app.py         # interactive dashboard
├── utils/
│   ├── prediction.py            # preprocessing, training, evaluation, inference
│   └── recommendations.py       # rule-based skill-gap engine
├── requirements.txt
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & install
```bash
git clone https://github.com/kamatiramya256-ui/PlacementPredictionSystem.git
cd PlacementPredictionSystem
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python data/generate_dataset.py
```
This writes `placement_dataset.csv` to the current directory — move it into `data/` if needed:
```bash
# Windows (PowerShell)
Move-Item -Path placement_dataset.csv -Destination data\ -Force
# macOS/Linux
mv placement_dataset.csv data/
```

### 3. Train the model
Either run `notebooks/Model_Training.ipynb` end-to-end (recommended — includes all EDA charts
and the Gemini career-guidance demo), or train quickly from the command line:

```bash
python -c "import sys; sys.path.insert(0,'utils'); import pandas as pd; from prediction import preprocess, split_and_scale, train_all_models, save_artifacts; df = pd.read_csv('data/placement_dataset.csv'); df = preprocess(df); X_train, X_test, y_train, y_test, scaler = split_and_scale(df); fitted, results = train_all_models(X_train, X_test, y_train, y_test); best = max(results, key=lambda n: results[n]['F1 Score']); save_artifacts(fitted[best], scaler, 'models/placement_model.pkl'); print('Saved:', best)"
```

### 4. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```
Open the printed local URL (usually `http://localhost:8501`).

### 5. (Optional) Enable AI Career Guidance
Check **"Generate AI career guidance"** in the sidebar form and paste a free Gemini API key
from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

---

## ☁️ Deploy to Streamlit Cloud

1. Push this repo to GitHub (see [Git setup](#-git-setup) below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account.
3. Select this repository and set the main file path to `app/streamlit_app.py`.
4. (Optional) Add `GEMINI_API_KEY` under **App Settings → Secrets** if you want a default key
   baked in — the app also accepts a key typed in by users at runtime.
5. Click **Deploy**.

---

## 🔧 Git Setup

```bash
git init
git add .
git commit -m "Initial commit: AI placement prediction system"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

---

## 🧠 Model Performance (sample run)

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.82 | 0.88 | 0.87 |
| Random Forest | 0.81 | 0.88 | 0.84 |
| SVM | 0.81 | 0.87 | 0.87 |
| Decision Tree | 0.74 | 0.82 | 0.69 |

*Results vary slightly by random seed and after hyperparameter tuning.*

---

## 🛠️ Tech Stack

Python · scikit-learn · XGBoost · pandas · NumPy · Matplotlib · Seaborn · Streamlit ·
Google Gemini API

---

## 📌 Future Enhancements

- Replace synthetic dataset with real institutional placement data
- Add SHAP-based explainability for individual predictions
- Multi-language support for career guidance
- User authentication and prediction history tracking
- A/B test alternative skill-gap rule thresholds against real outcomes

---

## 📄 License

This project is open for educational and personal use.
