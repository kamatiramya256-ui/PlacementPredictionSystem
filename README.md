# AI-Powered Student Placement Prediction and Career Guidance System

## Run locally

```bash
pip install -r requirements.txt

# 1. Generate the dataset (if not already present)
python data/generate_dataset.py
mv placement_dataset.csv data/   # if it landed in cwd

# 2. Train the model (or just run the notebook in Colab)
python -c "
import sys; sys.path.insert(0,'utils')
import pandas as pd
from prediction import preprocess, split_and_scale, train_all_models, save_artifacts
df = pd.read_csv('data/placement_dataset.csv')
df = preprocess(df)
X_train, X_test, y_train, y_test, scaler = split_and_scale(df)
fitted, results = train_all_models(X_train, X_test, y_train, y_test)
best = max(results, key=lambda n: results[n]['F1 Score'])
save_artifacts(fitted[best], scaler, 'models/placement_model.pkl')
print('Saved:', best)
"

# 3. Launch the dashboard
streamlit run app/streamlit_app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

For the AI Career Guidance feature, enable the checkbox in the sidebar form and paste a
Gemini API key (free at https://aistudio.google.com/app/apikey).

## Streamlit Cloud deployment
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, connect your GitHub, select the repo.
3. Set the main file path to `app/streamlit_app.py`.
4. Deploy. Add your `GEMINI_API_KEY` under app Secrets if you want a default key
   (the app also accepts a key typed in by the user at runtime).
