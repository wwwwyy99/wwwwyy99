# 🧠 AI Evidence Graph – Weight-Loss Coaching Insights

## Goal  
Prototype dashboard to visualise evidence-based coaching data for users targeting **–10 kg** over 8 weeks.

## Tech  
**Python · Pandas · Streamlit · scikit-learn (Linear Regression)**

---

## 📊 Dataset
**File:** `patients.csv`  
**Columns:**  
`user_id, age, sex, height, weight, target_weight, activity_level, calorie_intake, workout_mins, week_no, weight_change`

Each user contributes 8 weekly records.  

**Derived metrics:**  
- **BMI:** `weight_kg / (height_m^2)`  
- **Weekly calorie deficit:** `7 × (TDEE – calorie_intake)`  
- **TDEE (Total Daily Energy Expenditure):** estimated via activity multiplier  
  *(sedentary=1.2, light=1.375, moderate=1.55, active=1.725)*  
- **At-risk users:** plateau **≥3 weeks** (|Δweight| < 0.2 kg)  
  or mean daily deficit < **500 kcal**

---

## 🎯 KPIs  
- **Total users**  
- **Users with ≥ 5 kg loss** in 8 weeks  
- **Users plateaued ≥ 3 weeks**  
- **Mean daily deficit** across all users  

---

## 📈 Visuals & Tabs  
1. **BMI & Weight Trend:** per-user line chart showing progress vs target.  
2. **Weekly Calorie Deficit & Target Loss:** displays weekly calorie deficit and expected loss values directly from dataset.  
3. **Forecast:** one-step-ahead prediction using a simple **Linear Regression** model.  
4. **Coach Suggestion:** rule-based text guidance (e.g., “Increase activity” or “Good progress!”).  

Users can toggle between these tabs and select individual users from a sidebar filter.

---

## 🤖 Model Logic  
- **Linear Regression** trained on `week_no` vs `weight` for each user.  
- Produces a **forecasted next-week weight** to visualise potential outcomes.  
- Model re-fits dynamically when a different user is selected.

---

## 📦 Export Summary  
A JSON-style summary (for optional API/AI integration) includes:  
```json
{
  "summary": "Out of <N> users, <K> achieved ≥5 kg loss in 8 weeks.",
  "insights": [
    "Mean deficit ≈ <X> kcal/day",
    "<P> users plateaued ≥3 weeks"
  ],
  "ai_summary": "(optional, if OPENAI_API_KEY set)"
}
