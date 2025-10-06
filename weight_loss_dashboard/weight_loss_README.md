# 🧠 AI Evidence Graph – Weight-Loss Coaching Insights

### 🎯 Goal
An interactive prototype dashboard that visualises **evidence-based coaching data** for users targeting a **10 kg reduction over 8 weeks**.  
The dashboard identifies behavioural patterns, calorie trends, and progress plateaus to support data-driven health coaching decisions.

### 🛠️ Tech Stack
**Python · Pandas · Streamlit · scikit-learn (Linear Regression)**  
Deployed on **Streamlit Community Cloud**.

---

## 📊 Dataset Overview
**File:** `patients.csv`  
Each record represents a weekly data point for one of 50 users over 8 weeks.

| Column | Description |
|:--------|:-------------|
| `user_id` | Unique participant ID |
| `age`, `sex`, `height` | Demographic / biometric data |
| `weight`, `target_weight` | Actual vs goal weight (kg) |
| `activity_level` | Sedentary / Light / Moderate / Active |
| `calorie_intake` | Average daily energy intake (kcal) |
| `workout_mins` | Weekly exercise duration (minutes) |
| `week_no` | Week index (1–8) |
| `weight_change` | Weight delta vs previous week (kg) |

**Derived fields**
- **BMI** = `weight / (height_m²)`  
- **Daily deficit (kcal)** ≈ `(2400 − calorie_intake) + (workout_mins × 5 / 7)`  
- **Plateau flag** = ≥ 3 consecutive weeks with `|Δweight| < 0.1 kg`  

---

## 🎯 KPIs Displayed
| KPI | Definition |
|:----|:------------|
| **Users** | Total participants in dataset |
| **≥ 5 kg loss (8 w)** | Users whose final–initial weight ≥ 5 kg |
| **Users plateaued ≥ 3 w** | Users showing no progress for ≥ 3 weeks |
| **Mean Deficit (/day)** | Average daily calorie deficit across users |

> *For this sample dataset, no user achieved ≥ 5 kg loss or plateaued ≥ 3 weeks — consistent with the realistic synthetic data distribution.*

---

## 📈 Visuals & Interaction

The dashboard provides four tabs to toggle between user-level insights:

1. **📉 BMI & Weight Trend**  
   - Displays individual trajectories across 8 weeks  
   - Highlights weekly weight change alongside BMI evolution  

2. **🔥 Weekly Calorie Deficit & Target Loss**  
   - Computes weekly mean calorie deficit and weight loss per week  
   - Compares to a fixed target of 1.25 kg/week (10 kg ÷ 8)  
   - Displays as a clean numeric summary table and overall averages  

3. **📈 Forecast**  
   - Uses **Linear Regression** to predict next-week weight based on previous 7 weeks  
   - Provides one-step-ahead forecast for trend continuation  

4. **💬 Coach Suggestion**  
   - Rule-based recommendations depending on plateau detection or consistent progress  
   - Example: *“On track – maintain current plan” or “Possible plateau – review diet/exercise balance.”*

---

## 🤖 Model & Analytical Logic
- **Aggregation:** Weekly averages by `user_id` and `week_no`  
- **Trend Analysis:** Weight, BMI, and deficit evolution  
- **Correlation Check:** Scatter of `daily_deficit_kcal` vs `weight_change` to confirm expected negative relationship  
- **Forecast Model:** `LinearRegression()` fitted to predict Week 8 from Weeks 1–7 per user  
- **Rule Engine:** Plateau & progress rules generate text recommendations  

---

## 📦 Export Summary
When exported as JSON or CSV, the dashboard produces a concise report for integration with AVA:

```json
{
  "summary": "Out of 50 users, 0 achieved ≥5 kg loss in 8 weeks.",
  "insights": [
    "Average deficit ≈ 620 kcal/day",
    "0 users plateaued ≥3 weeks"
  ],
  "ai_summary": "Most users show gradual, consistent progress with minor weekly fluctuations."
}
