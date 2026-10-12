# Predicting When a Person's Digital Activity Pattern Becomes Unusually Distracted

Mini project · **AD23532 Principles of Data Science** · B.Tech Artificial Intelligence and Data Science · Rajalakshmi Engineering College · Semester V (2026-27 ODD)

Total screen time alone does not tell the whole story. This project classifies each day of a person's phone/computer use as **Normal**, **Increased Distraction** or **Unusually Distracted** by comparing it with **that person's own previous 14 days** (a personal baseline), not with a population average.

> Disclaimer: the project identifies usage-pattern changes only. It does not diagnose addiction or any medical or psychological condition.

## Live app
* **Web app:** https://jeshwanth-tkd.github.io/Distraction-Pattern-Detector/ (GitHub Pages, served from `docs/index.html`)
* The trained model runs entirely in the browser: pick an example person and day, or enter your own typical day and today.

## Results (test month: 1–29 April, 1,160 user-days)

| Model | Accuracy | Macro-F1 | F1 (Unusual) | ROC-AUC |
|---|---|---|---|---|
| Z-score statistical baseline (1σ/2σ) | 0.716 | 0.540 | 0.554 | 0.853 |
| Isolation Forest | 0.588 | 0.518 | 0.510 | 0.705 |
| K-Means (k=3) | 0.718 | 0.674 | 0.676 | 0.847 |
| Decision Tree (tuned) | 0.761 | 0.703 | 0.712 | 0.858 |
| Random Forest (tuned) | 0.793 | 0.750 | 0.762 | 0.893 |
| **Logistic Regression (tuned) — final** | **0.812** | **0.763** | **0.786** | **0.898** |

The tuned Logistic Regression is selected: best on 5 of 7 metrics, statistically tied with the Random Forest (McNemar p = 0.063, bootstrap 95% CI of the F1 gap −0.015 to +0.042), and simpler, more stable across unseen people, calibrated and interpretable. Personal-baseline features add +0.069 macro-F1 to it (ablation).

## Repository layout
```
data/
  raw_digital_activity.csv      raw data with injected quality issues (4,885 rows × 14 cols)
  clean_digital_activity.csv    cleaned data (4,800 rows: 40 users × 120 days)
notebooks/
  Distraction_Detection.ipynb   executed notebook: Review 1 → Review 2 → Review 3
  distraction_project.py        same notebook as a plain Python script (percent format)
src/
  generate_dataset.py           builds the raw dataset (seed 42)
  run_notebook.py               executes the script and writes the .ipynb with outputs
  build_app.py                  bundles the model + results into the web app
models/
  final_model.pkl               tuned Logistic Regression pipeline (scikit-learn)
  parity_check.json             Python predictions used to verify the browser model
figures/                        all 18 charts from the notebook
app/                            web app source (template, exported model, dashboard data)
docs/index.html                 deployed web app (GitHub Pages)
reports/Technical_Report.md     technical report
reports/Learning_Guide.pdf      beginner guide: rebuild the project from zero
```

## Reproduce
```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
python src/generate_dataset.py                       # 1. raw data
python src/run_notebook.py notebooks/distraction_project.py Distraction_Detection.ipynb   # 2. full analysis (~5 min)
python src/build_app.py                              # 3. web app -> app/index.html, docs/index.html
```
Or open `notebooks/Distraction_Detection.ipynb` in Jupyter / VS Code / Google Colab and run all cells.

## Method in brief
1. **Data quality audit:** missing values, 85 duplicate syncs, mixed date formats, inconsistent spellings, negative durations, >24 h totals, a 9999 sensor glitch.
2. **Cleaning:** standardise text, unify dates, de-duplicate on user + date, invalid → missing → person's own median, recompute totals; validation checks all pass.
3. **Features:** charter ratios (distraction, productive, switch rate, night usage) and **personal z-scores** from the previous 14 days only (no look-ahead leakage); mutual-information feature selection.
4. **EDA:** univariate, bivariate (Kruskal-Wallis effect sizes), multivariate (Spearman heatmap, PCA), trends by day of week.
5. **Models:** Z-score rule, Isolation Forest, K-Means, Logistic Regression, Decision Tree, Random Forest; chronological train/test split; person-grouped 5-fold CV; Grid Search tuning; McNemar + bootstrap significance tests.

## Data note
Per-app usage logs with day-level labels are private, so the dataset is **simulated** (`src/generate_dataset.py`) to follow the structure in the project charter and patterns reported in screen-time studies: personal baselines, persistent distraction spells, weekend effects and 7% label noise. Results should be validated on real, consented logs.
