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
    csv_path = os.path.join(base_dir, "patients_2.csv")

    if not os.path.exists(csv_path):
        st.error("❌ patients_2.csv not found. Please ensure the file is in the same folder as this app.")
        st.stop()

    df = pd.read_csv(csv_path)

    if "height" not in df.columns or "weight" not in df.columns:
        st.error("❌ Dataset missing required columns: 'height' or 'weight'.")
        st.stop()

    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["daily_deficit_kcal"] = (2400 - df["calorie_intake"]) + (df["workout_mins"] * 5)
    return df

df = load_data()

# --- Build user flags ---
flags = []

if "user_id" not in df.columns:
    st.error("❌ Dataset error: 'user_id' column missing from patients_2.csv.")
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
    achieved_2kg = total_loss >= 2

    flags.append({
        "user_id": uid,
        "plateau_3w": plateau_3w,
        "total_loss": total_loss,
        "avg_deficit": avg_deficit,
        "achieved_2kg": achieved_2kg
    })

flags = pd.DataFrame(flags)
df = df.merge(flags[["user_id", "plateau_3w"]], on="user_id", how="left")

st.title("🏋️‍♀️ AI Evidence Graph – Weight-Loss Coaching Insights")

# --- KPIs ---
n_users = df["user_id"].nunique()
achieved_2kg = flags["achieved_2kg"].sum()
plateau_users = flags["plateau_3w"].sum()
avg_daily_def = flags["avg_deficit"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Users", int(n_users))
c2.metric("≥2 kg loss (8w)", int(achieved_2kg))
c3.metric("Users plateaued ≥3w", plateau_users)
if not pd.isna(avg_daily_def):
    c4.metric("Mean deficit (/day)", f"{avg_daily_def:,.0f} kcal")

st.divider()

# --- User selection ---
user_ids = sorted(df["user_id"].unique())
selected_user = st.selectbox("Select User to Explore", user_ids)

user_data = df[df["user_id"] == selected_user].sort_values("week_no")
user_flag = flags[flags["user_id"] == selected_user].iloc[0]

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 BMI & Weight Trend",
    "🔥 Weekly Calorie Deficit & Target Loss",
    "💡 Forecast",
    "🧠 Coach Suggestion"
])

# --- TAB 1: BMI & Weight Trend ---
with tab1:
    st.subheader(f"User {selected_user}: BMI & Weight Trend")
    st.line_chart(user_data[["week_no", "BMI"]].set_index("week_no"), height=350)
    st.line_chart(user_data[["week_no", "weight"]].set_index("week_no"), height=350)
    st.caption(f"Total Loss: {user_flag['total_loss']:.1f} kg | Avg Deficit: {user_flag['avg_deficit']:.0f} kcal/day")

# --- TAB 2: Weekly Calorie Deficit & Target Loss ---
with tab2:
    st.subheader(f"User {selected_user}: Weekly Calorie Deficit vs Target Loss")

    weekly_summary = (
        user_data.groupby("week_no")
        .agg(
            avg_deficit=("daily_deficit_kcal", "mean"),
            avg_weight_change=("weight_change", "mean")
        )
        .reset_index()
    )

    weekly_summary["avg_weight_change"] = weekly_summary["avg_weight_change"].abs()
    weekly_summary["target_loss_per_week"] = 10 / 8

    st.dataframe(
        weekly_summary.rename(columns={
            "week_no": "Week",
            "avg_deficit": "Avg Calorie Deficit (kcal/day)",
            "avg_weight_change": "Avg Weight Loss (kg)",
            "target_loss_per_week": "Target Loss (kg)"
        }).style.format({
            "Avg Calorie Deficit (kcal/day)": "{:.0f}",
            "Avg Weight Loss (kg)": "{:.2f}",
            "Target Loss (kg)": "{:.2f}"
        }),
        use_container_width=True
    )

    avg_deficit_all = weekly_summary["avg_deficit"].mean()
    avg_loss_all = weekly_summary["avg_weight_change"].mean()
    st.markdown(
        f"**📈 Overall averages:** Daily Deficit ≈ {avg_deficit_all:,.0f} kcal/day | "
        f"Weekly Loss ≈ {avg_loss_all:.2f} kg | Target = 1.25 kg/week"
    )

# --- TAB 3: Forecast ---
with tab3:
    st.subheader(f"Forecast: Week 8 Weight (User {selected_user})")
    sample = user_data[user_data["week_no"] <= 7].copy()
    X = sample[["week_no"]]
    y = sample["weight"]
    model = LinearRegression().fit(X, y)
    pred = model.predict(pd.DataFrame({"week_no": [8]}))
    st.metric("Predicted Week 8 Weight", f"{pred.mean():.1f} kg")

# --- TAB 4: Coach Suggestion ---
with tab4:
    st.subheader(f"Coach Suggestion for User {selected_user}")
    msg = []
    if user_flag["plateau_3w"]:
        msg.append("⚠️ Plateau ≥3 weeks – vary workout intensity or adjust calorie plan.")
    else:
        msg.append("✅ On track – continue current plan and monitor progress weekly.")
    st.success(" ".join(msg))

st.divider()

# --- Overall Comparison: Selected User vs All Users ---
st.subheader("📊 Overall: Weight Loss Progress vs Average Trend")

import altair as alt

# Compute group average weight per week
group_avg = (
    df.groupby("week_no", as_index=False)["weight"]
    .mean()
    .rename(columns={"weight": "avg_weight"})
)

# Merge group average with user data
user_trend = df[df["user_id"] == selected_user][["week_no", "weight"]]

# Create Altair line for overall average
avg_line = (
    alt.Chart(group_avg)
    .mark_line(color="#888", strokeDash=[5, 3], size=2)
    .encode(
        x=alt.X("week_no:O", title="Week Number"),
        y=alt.Y("avg_weight:Q", title="Weight (kg)", scale=alt.Scale(zero=False)),
        tooltip=["week_no", "avg_weight"]
    )
)

# Create line for selected user
user_line = (
    alt.Chart(user_trend)
    .mark_line(point=True, color="#E45756", size=3)
    .encode(
        x=alt.X("week_no:O"),
        y=alt.Y("weight:Q"),
        tooltip=["week_no", "weight"]
    )
)

# Combine both charts
combined = (avg_line + user_line).properties(height=400)

st.altair_chart(combined, use_container_width=True)

# Optional regression trendline for user
from sklearn.linear_model import LinearRegression

X_user = user_trend[["week_no"]]
y_user = user_trend["weight"]
model = LinearRegression().fit(X_user, y_user)
trend_user = model.coef_[0]

st.caption(f"User {selected_user} trend ≈ {trend_user:.2f} kg/week (negative = weight loss).")

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

