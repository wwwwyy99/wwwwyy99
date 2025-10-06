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
    low_deficit = avg_deficit < 500 if not np.isnan(avg_deficit) else False
    total_loss = g["weight"].iloc[0] - g["weight"].iloc[-1]
    achieved_5kg = total_loss >= 5

    flags.append({
        "user_id": uid,
        "plateau_3w": plateau_3w,
        "low_deficit": low_deficit,
        "at_risk": plateau_3w or low_deficit,
        "total_loss": total_loss,
        "avg_deficit": avg_deficit,
        "achieved_5kg": achieved_5kg
    })

# --- Convert to DataFrame safely ---
if len(flags) == 0:
    st.warning("⚠️ No user data detected — the dataset might be empty or missing key fields.")
    flags = pd.DataFrame(columns=[
        "user_id", "plateau_3w", "low_deficit",
        "at_risk", "total_loss", "avg_deficit", "achieved_5kg"
    ])
else:
    flags = pd.DataFrame(flags)

# --- Safe merge back into df ---
if "user_id" not in flags.columns:
    st.error("⚠️ Internal error: Missing 'user_id' column in flags DataFrame. Please check preprocessing logic.")
    st.stop()
else:
    df = df.merge(flags[["user_id", "plateau_3w", "low_deficit", "at_risk"]],
                  on="user_id", how="left")
