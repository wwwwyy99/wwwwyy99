# AI Evidence Graph – Weight-Loss Coaching Insights

**Goal**  
Prototype dashboard to visualise evidence-based coaching data for users targeting **–10 kg**.

**Tech**  
Python · Pandas · Streamlit · scikit-learn (linear regression for bonus)

## Dataset
`patients.csv` columns:
`user_id, age, sex, height, weight, target_weight, activity_level, calorie_intake, workout_mins, week_no, weight_change`

- **BMI:** `weight_kg / (height_m^2)`
- **Weekly deficit:** 7 × (TDEE – calorie_intake), TDEE via simple proxy with activity multiplier.
- **At-risk users:** plateau **≥3 weeks** (|Δweight| < 0.2kg) **or** mean daily deficit < **500 kcal**.
- **Trends:** 8 weeks per user.

## KPIs
- Total users; users with **≥5 kg** loss in 8 weeks
- Users plateaued ≥3 weeks
- Mean daily deficit across users

## Visuals
1. **Weight trend** (per user, line) and **BMI trend**  
2. **Correlation:** weekly deficit vs weekly weight delta (scatter)  
3. **Coach suggestion cards:** rule-based text per user  
4. **Bonus:** linear regression forecast (one-step ahead)

## Exports
- **JSON** summary for AVA:
```json
{
  "summary": "Out of <N> users, <K> achieved ≥5 kg loss in 8 weeks.",
  "insights": ["Mean deficit ≈ <X> kcal/day", "<P> users plateaued ≥3 weeks"],
  "ai_summary": "(optional, if OPENAI_API_KEY set) ..."
}