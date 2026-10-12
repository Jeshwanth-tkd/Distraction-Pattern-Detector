"""Builds the self-contained web app: app/template.html + app/model.json + app/dashboard_data.json -> app/index.html (and docs/index.html for GitHub Pages)."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
model = json.loads((ROOT / "app/model.json").read_text())
dash = json.loads((ROOT / "app/dashboard_data.json").read_text())

# Typical day-to-day spread (coefficient of variation) for the ratio features, used by "Enter your own day"
df = pd.read_csv(ROOT / "data/clean_digital_activity.csv")
tot = df["screen_time_hrs"] * 60
df["distraction_ratio"] = (df["social_media_min"] + df["gaming_min"]) / tot
df["productive_ratio"] = df["productive_min"] / tot
for c in ["distraction_ratio", "productive_ratio"]:
    g = df.groupby("user_id")[c]
    dash["user_cv"][c] = round(float((g.std() / g.mean()).median()), 3)

html = (ROOT / "app/template.html").read_text()
html = html.replace("__MODEL__", json.dumps(model, separators=(",", ":"))).replace("__DASH__", json.dumps(dash, separators=(",", ":")))
(ROOT / "app/index.html").write_text(html)
(ROOT / "docs").mkdir(exist_ok=True)
full = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
        '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style></head><body>' + html + "</body></html>")
(ROOT / "docs/index.html").write_text(full)
print(f"app/index.html {len(html)/1024:.0f} kB, docs/index.html written")
