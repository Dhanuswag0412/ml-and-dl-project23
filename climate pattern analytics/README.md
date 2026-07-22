# 🌍 An Intelligent Environmental Data Analytics System for Climate Pattern Analysis Using Machine Learning

A production-style Streamlit dashboard that classifies climate risk (Normal, Heatwave,
Drought, Flood, Extreme Weather) from environmental sensor readings using a
**Random Forest Classifier**, with **K-Means** clustering for unsupervised pattern
discovery and a full interactive analytics dashboard.

## 📁 Project Structure
```
climate_ml_project/
├── app.py                     # Streamlit app (UI + inference)
├── train_model.py             # Generates data + trains + saves model.pkl
├── requirements.txt           # Python dependencies
├── .streamlit/config.toml     # Dashboard theme
├── model/
│   └── climate_model.pkl      # Trained model bundle
└── data/
    └── environmental_data.csv # Dataset (synthetic, 10 yrs x 6 regions)
```

## ▶️ Run locally
```bash
pip install -r requirements.txt
python train_model.py        # (already generated — only needed to retrain)
streamlit run app.py
```

---

## 🚀 Deploy to Streamlit Community Cloud

### 1. Push this project to GitHub
```bash
cd climate_ml_project
git init
git add .
git commit -m "Initial commit: Climate Pattern Analytics ML app"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```
> Create the empty repo first at https://github.com/new (don't initialize it with a README).

### 2. Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. Click **"Create app" → "From existing repo"**.
3. Select your repository, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and launch the app.
5. Your live URL will look like: `https://<your-app-name>.streamlit.app`

### 3. Updating the app later
Any `git push` to `main` automatically redeploys the app on Streamlit Cloud.

---

## 🧠 Model Details
| Item | Value |
|---|---|
| Algorithm | RandomForestClassifier (scikit-learn) |
| Unsupervised model | KMeans (4 clusters) |
| Features | Temperature, Humidity, Rainfall, CO₂, Wind Speed, Pressure, Region |
| Target | Climate Risk Category (5 classes) |
| Test Accuracy | ~74% |
| Weighted F1 | ~75% |

Retrain anytime with:
```bash
python train_model.py
```
This regenerates `data/environmental_data.csv` and `model/climate_model.pkl`.

## 📊 App Pages
- **Overview Dashboard** — KPIs, temperature trends, risk distribution, rainfall histogram, correlation heatmap
- **Data Explorer** — filter by region/year/risk category, download filtered CSV
- **Predict Climate Risk** — live inference form with probability breakdown & recommendations
- **Model Insights** — feature importance, PCA cluster visualization, performance metrics
- **About** — methodology & tech stack

## 🔄 Using your own real data
Replace `data/environmental_data.csv` with your own file using the same columns
(`Region, Year, Month, Temperature_C, Humidity_pct, Rainfall_mm, CO2_ppm,
Wind_Speed_kmh, Pressure_hPa[, Climate_Risk_Category]`), then run
`python train_model.py` to retrain the model on real observations. You can also
just upload a CSV directly in the app sidebar for exploration/dashboard views
without retraining.
