# streamlit_app.py
import os
import json
import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Evidence Graph – Weight-Loss Coaching Insights",
                   page_icon="🏋️", layout="wide")

# -----------------------
# Helpers
# -----------------------
def compute_bmi(weight_kg, height_cm):
    h_m = height_cm / 100.0
    return weight_kg / (h_m * h_m)

def weekly_deficit_kcal(row):
    """
    Very simple proxy:
    If dataset has 'calorie_intake' and 'activity_level' we assume:
    - Baseline TDEE rough rule-of-thumb via Mifflin-like proxy:
      base = (10*weight + 6.25*height - 5*age + s), s=+5 (M) / -161 (F)
    - activity_level categorical: sedentary(1.2), light(1.375), moderate(1.55), active(1.725)
    Weekly deficit = 7*(TDEE - intake). If negative => surplus
    """
    sex = row.get("sex", "F")
    s_bias = 5 if str(sex).strip().lower().startswith("m") else -161
    activity_map = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
    mult = activity_map.get(str(row.get("activity_level","moderate")).lower(), 1.55)

    tdee = (10*row["weight"] + 6.25*row["height"] - 5*row["age"] + s_bias) * mult
    intake = row.get("calorie_intake", np.nan)
    if pd.isna(intake):
        return np.nan
    return 7 * (tdee - intake)

def detect_plateau(ser_wt):
    """
    Plateau: 3+ consecutive weeks with |Δweight| < 0.2 kg
    """
    deltas = ser_wt.diff().fillna(0).abs()
    # count consecutive < 0.2 from the end
    cnt, best = 0, 0
    for d in deltas[::-1]:
        if d < 0.2:
            cnt += 1
            best = max(best, cnt)
        else:
            cnt = 0
    return best >= 3

def summarize_coaching(df_user):
    """
    Human-readable suggestion per user based on last 3 weeks patterns.
    """
    if df_user.empty:
        return "No data."
    df_user = df_user.sort_values("week_no")
    last = df_user.iloc[-1]
    start_w = df_user.iloc[0]["weight"]
    end_w = last["weight"]
    loss = start_w - end_w

    plateau_flag = detect_plateau(df_user["weight"])
    mean_def = df_user["weekly_deficit_kcal"].tail(3).mean()

    if plateau_flag:
        return "Plateau detected (≥3 weeks). Try progressive overload, vary intensity, or reassess calorie target."
    if mean_def is not None and mean_def < 3500:  # <500/day approx
        return "Deficit likely insufficient. Consider a modest calorie reduction and increase NEAT/steps."
    if loss >= 5:
        return "Great progress! Maintain consistency; consider deload weeks to sustain adherence."
    return "On track. Maintain current plan and monitor sleep, steps, and protein intake."

def daily_deficit_from_weekly(kcal):
    # simple conversion so UI can show /day
    return kcal / 7.0

# -----------------------
# Sidebar: data upload / sample
# -----------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload patients.csv", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.sidebar.info("No file uploaded. Using bundled / example dataset if present.")
    try:
        df = pd.read_csv("patients.csv")
    except Exception:
        st.sidebar.warning("No patients.csv found. Please upload one.")
        df = pd.DataFrame(columns=[
            "user_id","age","sex","height","weight","target_weight",
            "activity_level","calorie_intake","workout_mins","week_no","weight_change"
        ])

# Basic schema cleaning
expected_cols = ["user_id","age","sex","height","weight","target_weight",
                 "activity_level","calorie_intake","workout_mins","week_no","weight_change"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}. Please supply required schema.")
    st.stop()

# Convert types where sensible
num_cols = ["age","height","weight","target_weight","calorie_intake","workout_mins","week_no","weight_change"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["sex"] = df["sex"].astype(str)
df["activity_level"] = df["activity_level"].astype(str)

# Compute BMI and weekly deficit
df["bmi"] = compute_bmi(df["weight"], df["height"])
df["weekly_deficit_kcal"] = df.apply(weekly_deficit_kcal, axis=1)
df["daily_deficit_kcal"]  = df["weekly_deficit_kcal"].apply(lambda x: np.nan if pd.isna(x) else daily_deficit_from_weekly(x))

# Identify at-risk users
user_flags = []
for uid, g in df.groupby("user_id"):
    g = g.sort_values("week_no")
    plateau = detect_plateau(g["weight"])
    low_def = (g["daily_deficit_kcal"].mean() if not g["daily_deficit_kcal"].isna().all() else np.nan)
    low_def_flag = (not pd.isna(low_def)) and (low_def < 500)
    user_flags.append({"user_id": uid, "plateau_3w": plateau, "low_deficit": low_def_flag})
flags = pd.DataFrame(user_flags)
df = df.merge(flags, on="user_id", how="left")
df["at_risk"] = df["plateau_3w"] | df["low_deficit"]

# KPIs
n_users = df["user_id"].nunique()
loss_8w = (
    df.sort_values(["user_id","week_no"]).groupby("user_id")
      .agg(start_w=("weight", "first"), end_w=("weight","last"))
)
loss_8w["loss_kg"] = loss_8w["start_w"] - loss_8w["end_w"]
achieved_5kg = (loss_8w["loss_kg"] >= 5).sum()

avg_daily_def = df.groupby("user_id")["daily_deficit_kcal"].mean().mean()
plateau_users = flags["plateau_3w"].sum()

# Header
st.title("🏋️ AI Evidence Graph – Weight-Loss Coaching Insights")
st.caption("Prototype dashboard for evidence-based health coaching (test assignment)")

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", int(n_users))
c2.metric("≥5 kg loss (8w)", int(achieved_5kg))
c3.metric("Users plateaued ≥3w", int(plateau_users))
if not pd.isna(avg_daily_def):
    c4.metric("Mean deficit (/day)", f"{avg_daily_def:,.0f} kcal")

st.divider()

# -----------------------
# Visuals
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 Correlation", "🧠 Coach Suggestions", "🔮 Forecast (Bonus)"])

with tab1:
    st.subheader("Weight trends & BMI over 8 weeks")
    sel_user = st.selectbox("Select user", sorted(df["user_id"].unique()))
    g = df[df["user_id"] == sel_user].sort_values("week_no")
    lcol, rcol = st.columns(2)
    with lcol:
        st.line_chart(g.set_index("week_no")["weight"], height=300)
        st.caption("Weight trend by week")
    with rcol:
        st.line_chart(g.set_index("week_no")["bmi"], height=300)
        st.caption("BMI trend by week")

with tab2:
    st.subheader("Calorie deficit vs. weight delta correlation")
    # weight delta per week already provided; use daily_deficit average or weekly_deficit for same week
    gg = df.copy()
    gg = gg.dropna(subset=["weekly_deficit_kcal","weight_change"])
    st.scatter_chart(gg[["weekly_deficit_kcal","weight_change"]].rename(
        columns={"weekly_deficit_kcal":"Weekly Deficit (kcal)", "weight_change":"Weekly Weight Δ (kg)"}), height=380)
    st.caption("Expect negative correlation: higher deficit → more negative (larger) weekly weight loss.")

with tab3:
    st.subheader("Coach suggestion cards")
    st.write("Auto-generated per user based on plateau/deficit logic.")
    for uid, g in df.groupby("user_id"):
        with st.container(border=True):
            colA, colB, colC, colD = st.columns([2,2,2,4])
            last = g.sort_values("week_no").iloc[-1]
            start = g.sort_values("week_no").iloc[0]
            loss = start["weight"] - last["weight"]
            colA.metric("User", uid)
            colB.metric("Total Loss (kg)", f"{loss:.1f}")
            colC.metric("Mean Deficit (/day)", f"{g['daily_deficit_kcal'].mean():.0f} kcal" if not g['daily_deficit_kcal'].isna().all() else "—")
            colD.write(summarize_coaching(g))

with tab4:
    st.subheader("One-step ahead forecast (Linear Regression; bonus)")
    sel_user2 = st.selectbox("Pick user for forecast", sorted(df["user_id"].unique()), key="fuser")
    g2 = df[df["user_id"] == sel_user2].sort_values("week_no")
    X = g2[["week_no"]].values
    y = g2["weight"].values
    if len(g2) >= 3:
        model = LinearRegression().fit(X, y)
        next_w = int(g2["week_no"].max() + 1)
        pred = float(model.predict(np.array([[next_w]]))[0])
        st.write(f"Predicted weight for week {next_w}: **{pred:.1f} kg**")
    else:
        st.info("Need at least 3 points for a reasonable linear forecast.")

st.divider()

# -----------------------
# Export JSON / CSV summary for AVA
# -----------------------
st.subheader("📤 Export insights (JSON / CSV)")

summary_payload = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "summary": f"Out of {int(n_users)} users, {int(achieved_5kg)} achieved ≥5 kg loss in 8 weeks.",
    "insights": [
        f"Mean daily deficit ≈ {0 if pd.isna(avg_daily_def) else int(round(avg_daily_def))} kcal",
        f"{int(plateau_users)} users plateaued ≥3 weeks",
    ],
}

# Optional: simple AI summary (only if OPENAI_API_KEY present)
ai_summary = None
if os.getenv("OPENAI_API_KEY"):
    try:
        import openai  # for older environments
        from openai import OpenAI
        client = OpenAI()
        prompt = (f"Create a concise coaching summary for a weight-loss program using this JSON: "
                  f"{json.dumps(summary_payload)}. Tone: supportive, professional. 2–3 sentences.")
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        ai_summary = resp.output_text.strip()
    except Exception:
        ai_summary = None

if ai_summary:
    summary_payload["ai_summary"] = ai_summary

st.json(summary_payload)

csv_buf = io.StringIO()
loss_8w.reset_index()[["user_id","start_w","end_w","loss_kg"]].to_csv(csv_buf, index=False)
st.download_button("Download summary CSV", data=csv_buf.getvalue(),
                   file_name="summary_loss_8w.csv", mime="text/csv")

st.download_button("Download insights JSON", data=json.dumps(summary_payload, indent=2),
                   file_name="insights.json", mime="application/json")

st.caption("JSON/CSV intended for ingestion by AVA.")

# Footer
with st.expander("Assumptions & Notes"):
    st.markdown("""
- **At-risk definition:** plateau (≥3 consecutive weeks with <0.2kg change) **or** mean daily deficit <500 kcal.
- **Deficit calc:** rough TDEE proxy with activity multipliers; intended for **relative** trends, not clinical prescription.
- **Forecast:** simple linear regression to satisfy bonus; production should consider seasonality and behaviour.
""")
    
    