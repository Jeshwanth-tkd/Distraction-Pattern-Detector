"""
generate_dataset.py
-------------------
Builds the raw dataset used in the mini-project:
"Predicting When a Person's Digital Activity Pattern Becomes Unusually Distracted".

Why simulated?  Real per-app screen-time logs (Android Digital Wellbeing /
iOS Screen Time) are private and cannot be shared publicly. This script
simulates 40 users x 120 days of daily logs, following the structure shown
in the Review-1 charter (screen time, social, gaming, productive time,
app switches, night usage) and the behaviour patterns reported in
published screen-time studies (evening/weekend peaks, higher switching on
distracted days, personal baselines that differ a lot between people).

The generator deliberately injects the kinds of data-quality problems found
in real exported logs (missing values, duplicate syncs, mixed date formats,
inconsistent category spellings, impossible values and sensor glitches),
so that Review-2 has genuine cleaning work to do.

Run:  python src/generate_dataset.py      -> data/raw_digital_activity.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 40
N_DAYS = 120
START = "2024-01-01"
STATES = ["Normal", "Increased Distraction", "Unusually Distracted"]

rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "data" / "raw_digital_activity.csv"


def make_users():
    users = []
    for i in range(N_USERS):
        utype = "Student" if i < 22 else "Professional"
        base_screen = float(np.clip(rng.normal(5.0 if utype == "Student" else 4.6, 1.1), 2.5, 9.0))  # hours
        shares = rng.dirichlet([3.0, 1.6, 4.0, 3.0])  # social, gaming, productive, other
        if utype == "Professional":
            shares = shares * np.array([0.9, 0.6, 1.35, 1.0])
            shares = shares / shares.sum()
        users.append(
            dict(
                user_id=f"U{i + 1:03d}",
                user_type=utype,
                base_screen=base_screen,
                shares=shares,
                base_switch=float(np.clip(rng.normal(55, 14), 25, 100)),
                base_night=float(np.clip(rng.normal(25 if utype == "Student" else 15, 10), 3, 60)),
                base_unlock=float(np.clip(rng.normal(70, 18), 30, 130)),
                base_notif=float(np.clip(rng.normal(110, 35), 30, 220)),
                volatility=float(rng.uniform(0.15, 0.30)),
            )
        )
    return users


# Transition matrix of the latent "distraction state" (Markov chain).
# Distraction tends to persist for a few days, and weekends push it up.
T_WEEKDAY = np.array([[0.82, 0.14, 0.04], [0.45, 0.40, 0.15], [0.30, 0.35, 0.35]])
T_WEEKEND = np.array([[0.70, 0.22, 0.08], [0.35, 0.43, 0.22], [0.25, 0.35, 0.40]])

# Multipliers applied to a user's personal baseline for each state
MULT = {
    "social":     [1.00, 1.55, 2.50],
    "gaming":     [1.00, 1.45, 2.30],
    "productive": [1.00, 0.80, 0.52],
    "other":      [1.00, 1.05, 1.10],
    "switch":     [1.00, 1.35, 2.00],
    "night":      [1.00, 1.50, 2.40],
    "unlock":     [1.00, 1.25, 1.70],
    "notif":      [1.00, 1.10, 1.25],
}


def simulate():
    users = make_users()
    dates = pd.date_range(START, periods=N_DAYS, freq="D")
    rows = []
    for u in users:
        state = 0
        for d in dates:
            weekend = d.dayofweek >= 5
            T = T_WEEKEND if weekend else T_WEEKDAY
            state = rng.choice(3, p=T[state])
            vol = u["volatility"]

            def ln(x):  # multiplicative day-to-day noise
                return x * rng.lognormal(0, vol)

            base_min = u["base_screen"] * 60
            s, g, p, o = u["shares"] * base_min
            # weekend effect (a confounder: more leisure, but not "distracted")
            wk_social = 1.20 if weekend else 1.0
            wk_prod = (0.55 if u["user_type"] == "Professional" else 0.80) if weekend else 1.0

            social = ln(s * MULT["social"][state] * wk_social)
            gaming = ln(g * MULT["gaming"][state] * (1.15 if weekend else 1.0))
            productive = ln(p * MULT["productive"][state] * wk_prod)
            other = ln(o * MULT["other"][state])
            screen_hrs = (social + gaming + productive + other) / 60

            rows.append(
                dict(
                    user_id=u["user_id"],
                    user_type=u["user_type"],
                    date=d,
                    screen_time_hrs=round(screen_hrs, 2),
                    social_media_min=round(social, 1),
                    gaming_min=round(gaming, 1),
                    productive_min=round(productive, 1),
                    other_min=round(other, 1),
                    app_switches=int(round(ln(u["base_switch"] * MULT["switch"][state] * (1.1 if weekend else 1.0)))),
                    night_usage_min=round(ln(u["base_night"] * MULT["night"][state] * (1.25 if weekend else 1.0)), 1),
                    phone_unlocks=int(round(ln(u["base_unlock"] * MULT["unlock"][state]))),
                    notifications=int(round(ln(u["base_notif"] * MULT["notif"][state]))),
                    label=STATES[state],
                )
            )
    df = pd.DataFrame(rows)

    # ~7% label noise: people's self-reports are not perfect (adjacent class only)
    noise_idx = rng.choice(len(df), size=int(0.07 * len(df)), replace=False)
    for i in noise_idx:
        k = STATES.index(df.at[i, "label"])
        k = k + rng.choice([-1, 1]) if k == 1 else (1 if k == 0 else 1)
        df.at[i, "label"] = STATES[k]
    return df


def inject_quality_issues(df):
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d").astype(object)
    n = len(df)

    def pick(frac_or_n):
        k = frac_or_n if isinstance(frac_or_n, int) else int(frac_or_n * n)
        return rng.choice(n, size=k, replace=False)

    # 1. Missing values (sync failures / permission not granted)
    for col, frac in [("social_media_min", 0.025), ("gaming_min", 0.02), ("night_usage_min", 0.03),
                      ("notifications", 0.04), ("phone_unlocks", 0.02), ("user_type", 0.01)]:
        df.loc[pick(frac), col] = np.nan

    # 2. Mixed date formats (exports from different phone locales)
    idx = pick(0.05)
    df.loc[idx, "date"] = pd.to_datetime(df.loc[idx, "date"]).dt.strftime("%d/%m/%Y").values

    # 3. Inconsistent category spellings
    label_variants = {
        "Normal": ["normal", "NORMAL ", " Normal"],
        "Increased Distraction": ["Increased", "increased distraction", "INCREASED DISTRACTION"],
        "Unusually Distracted": ["Unusual", "unusually distracted", "Unusually  Distracted"],
    }
    for i in pick(0.06):
        df.at[i, "label"] = rng.choice(label_variants[df.at[i, "label"]])
    for i in pick(0.04):
        if isinstance(df.at[i, "user_type"], str):
            df.at[i, "user_type"] = rng.choice([df.at[i, "user_type"].lower(), df.at[i, "user_type"].upper(),
                                                df.at[i, "user_type"] + " "])

    # 4. Invalid / impossible values
    df.loc[pick(12), "gaming_min"] = -rng.integers(5, 40, 12)          # negative durations
    df.loc[pick(8), "screen_time_hrs"] = rng.uniform(24.5, 30, 8).round(2)  # > 24 h in a day
    df.loc[pick(14), "app_switches"] = 9999                               # sensor overflow glitch
    df.loc[pick(10), "night_usage_min"] = rng.uniform(900, 1500, 10).round(1)  # > 24 h night usage

    # 5. Duplicate records (the same day synced twice)
    dups = df.loc[pick(85)]
    df = pd.concat([df, dups], ignore_index=True)

    # shuffle so duplicates are not adjacent, then add a record id
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "record_id", np.arange(1, len(df) + 1))
    return df


if __name__ == "__main__":
    clean = simulate()
    raw = inject_quality_issues(clean)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT, index=False)
    print(f"Saved {raw.shape[0]} rows x {raw.shape[1]} cols -> {OUT}")
    print(clean["label"].value_counts(normalize=True).round(3))
