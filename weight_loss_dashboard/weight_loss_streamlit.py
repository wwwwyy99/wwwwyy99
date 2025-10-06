import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

st.set_page_config(page_title="AI Evidence Graph – Weight-Loss Coaching Insights", layout="wide")

# --- Load & prepare data ---
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "patients.csv")

    if not os.path.exists(csv_path):
        st.error("❌ patients.csv not found. Please ensure the file is in the same folder as this app.")
        st.stop()

    df = pd.read_csv(csv_path)

    if "height" not in df.columns or "weight" not in df.columns:
        st.error("❌ Dataset missing required columns: 'height' or 'weight'.")
        st.stop()

    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["daily_deficit_kcal"] = (2400 - df["calorie_intake"]) + (df["workout_mins"] * 5)
    return df

df = load_data()

# --- Build user risk flags safely ---
flags = []

if "user_id" not in df.columns:
    st.error("❌ Dataset error: 'user_id' column missing from patients.csv. Please check the uploaded data.")
    st.stop()

for uid, g in df.groupby("user_id"):
    g = g.sort_values("week_no")

    # Detect plateau ≥3 weeks
    streak = 0
    plateau_3w = False
    for i in range(1, len(g)):
        if abs(g["weight"].iloc[i] - g["weight"].iloc[i - 1]) < 0.1:
            streak += 1
            if streak >= 3:
                plateau_3w = True
        else:
            streak = 0

    avg_deficit = g["daily_deficit_kcal"].mean() if "daily_deficit_kcal" in g.columns else np.nan
    total_loss = g["weight"].iloc[0] - g["weight"].iloc[-1]
    achieved_5kg = total_loss >= 5

    flags.append({
        "user_id": uid,
        "plateau_3w": plateau_3w,
        "total_loss": total_loss,
        "avg_deficit": avg_deficit,
        "achieved_5kg": achieved_5kg
    })

if len(flags) == 0:
    st.warning("⚠️ No user data detected — the dataset might be empty or missing key fields.")
    flags = pd.DataFrame(columns=["user_id", "plateau_3w", "total_loss", "avg_deficit", "achieved_5kg"])
else:
    flags = pd.DataFrame(flags)

# --- Safe merge back into df ---
if "user_id" not in flags.columns:
    st.error("⚠️ Internal error: Missing 'user_id' column in flags DataFrame. Please check preprocessing logic.")
    st.stop()
else:
    df = df.merge(flags[["user_id", "plateau_3w"]], on="user_id", how="left")

st.success("✅ Data loaded and merged successfully!")

# --- KPIs ---
n_users = df["user_id"].nunique()
achieved_5kg = flags["achieved_5kg"].sum()
plateau_users = flags["plateau_3w"].sum()
avg_daily_def = flags["avg_deficit"].mean()

st.title("🏋️‍♀️ AI Evidence Graph – Weight-Loss Coaching Insights")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", int(n_users))
c2.metric("≥5 kg loss (8w)", int(achieved_5kg))
c3.metric("Users plateaued ≥3w", plateau_users)
if not pd.isna(avg_daily_def):
    c4.metric("Mean deficit (/day)", f"{avg_daily_def:,.0f} kcal")

st.divider()

# --- Plateaued User Summary ---
st.subheader("🚨 Plateaued User Summary")
user_summary = (
    df.sort_values("week_no")
      .groupby("user_id")
      .agg(
          age=("age","first"),
          sex=("sex","first"),
          activity_level=("activity_level","first"),
          current_weight=("weight","last"),
          target_weight=("target_weight","first"),
          avg_deficit_per_day=("daily_deficit_kcal","mean")
      )
      .reset_index()
)
summary_with_flags = user_summary.merge(flags[["user_id","plateau_3w"]], on="user_id", how="left")
plateau_df = summary_with_flags[summary_with_flags["plateau_3w"]].copy()

if plateau_df.empty:
    st.info("🎉 Great news! No users currently plateaued.")
else:
    st.dataframe(
        plateau_df.rename(columns={
            "user_id":"User",
            "current_weight":"Current Weight (kg)",
            "target_weight":"Target Weight (kg)",
            "avg_deficit_per_day":"Avg Deficit (/day, kcal)",
            "plateau_3w":"Plateau ≥3w"
        }).style.format({
            "Current Weight (kg)":"{:.1f}",
            "Target Weight (kg)":"{:.1f}",
            "Avg Deficit (/day, kcal)":"{:.0f}"
        }),
        use_container_width=True
    )

st.divider()

# --- User selection ---
user_ids = sorted(df["user_id"].unique())
selected_user = st.selectbox("Select User to Explore", user_ids)

user_data = df[df["user_id"] == selected_user].sort_values("week_no")
user_flag = flags[flags["user_id"] == selected_user].iloc[0]

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📉 BMI & Weight Trend", "💡 Forecast", "🧠 Coach Suggestion"])

with tab1:
    st.subheader(f"User {selected_user}: BMI & Weight Trend")
    st.line_chart(user_data[["week_no","BMI"]].set_index("week_no"), height=350)
    st.line_chart(user_data[["week_no","weight"]].set_index("week_no"), height=350)
    st.caption(f"Total Loss: {user_flag['total_loss']:.1f} kg | Avg Deficit: {user_flag['avg_deficit']:.0f} kcal/day")

with tab2:
    st.subheader(f"Forecast: Week 8 Weight (User {selected_user})")
    sample = user_data[user_data["week_no"] <= 7].copy()
    X = sample[["week_no"]]
    y = sample["weight"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(pd.DataFrame({"week_no":[8]}))
    st.metric("Predicted Week 8 Weight", f"{pred.mean():.1f} kg")

with tab3:
    st.subheader(f"Coach Suggestion for User {selected_user}")
    msg = []
    if user_flag["plateau_3w"]:
        msg.append("Plateau ≥3 weeks – vary workout intensity or adjust calorie plan.")
    else:
        msg.append("On track – continue current plan and monitor progress weekly.")
    st.success(" ".join(msg))

st.divider()

# --- Overall Correlation ---
st.subheader("📊 Overall: Calorie Deficit vs Weight Change Correlation")
df["weekly_deficit_kcal"] = df["daily_deficit_kcal"] * 7
corr_df = df.dropna(subset=["weekly_deficit_kcal","weight_change"])
st.scatter_chart(
    corr_df.rename(columns={
        "weekly_deficit_kcal":"Weekly Deficit (kcal)",
        "weight_change":"Weekly Weight Δ (kg)"
    })[["Weekly Deficit (kcal)","Weekly Weight Δ (kg)"]],
    height=400
)
corr = corr_df["weekly_deficit_kcal"].corr(corr_df["weight_change"])
st.caption(f"Correlation = {corr:.2f} (expect negative: higher deficit → greater weight loss)")

# --- Weekly Calorie Deficit vs Target Loss ---
st.divider()
st.subheader("🔥 Weekly Calorie Deficit vs Target Loss")

weekly_summary = (
    df.groupby("week_no")
      .agg(
          avg_deficit=("daily_deficit_kcal", "mean"),
          avg_weight_change=("weight_change", "mean")
      )
      .reset_index()
)
weekly_summary["avg_weight_change"] = weekly_summary["avg_weight_change"].abs()
weekly_summary["target_loss_per_week"] = 10 / 8

st.line_chart(
    weekly_summary.set_index("week_no")[["avg_deficit", "avg_weight_change"]],
    height=350
)

st.caption("Average weekly deficit vs achieved loss. Target = 1.25 kg/week.")

# --- Export summary ---
summary_json = {
    "summary": f"Out of {n_users} users, {achieved_5kg} achieved ≥5 kg loss in 8 weeks.",
    "insights": [
        f"Avg daily deficit = {avg_daily_def:.0f} kcal",
        f"{plateau_users} plateaued ≥3 weeks"
    ]
}
st.download_button("📥 Download Insights JSON", data=str(summary_json), file_name="summary.json")
st.download_button("📊 Download CSV Summary", data=df.to_csv(index=False), file_name="patients_summary.csv")
