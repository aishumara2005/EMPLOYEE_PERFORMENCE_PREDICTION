import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# 1. Load Dataset (CRITICAL FIX)
# =========================
df = pd.read_excel("JUSTIN_TITTO.xlsx")

# Force DATE
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

# 🔴 ABSOLUTE FIX: force OVERALL numeric
df['OVERALL'] = pd.to_numeric(df['OVERALL'], errors='coerce')

# Remove broken rows BEFORE aggregation
df = df.dropna(subset=['DATE', 'OVERALL']).reset_index(drop=True)

# =========================
# 2. Monthly Aggregation
# =========================
monthly = (
    df.resample('ME', on='DATE')
      .mean(numeric_only=True)
      .reset_index()
)

# =========================
# 3. FINAL SAFETY CLEAN
# =========================
monthly['OVERALL'] = pd.to_numeric(monthly['OVERALL'], errors='coerce')
monthly = monthly.dropna(subset=['OVERALL']).reset_index(drop=True)

# Time index AFTER cleaning
monthly['t'] = np.arange(len(monthly))

# 🚨 HARD STOP IF STILL NaN (DEBUG GUARANTEE)
if monthly['OVERALL'].isna().any():
    raise ValueError("OVERALL still has NaN after cleaning")

# =========================
# 4. XGBoost Model
# =========================
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =========================
# 5. 6-Month Backtesting
# =========================
if len(monthly) > 6:
    train = monthly.iloc[:-6]
    test = monthly.iloc[-6:]

    model.fit(train[['t']], train['OVERALL'])
    test_pred = model.predict(test[['t']])

    print("\n📐 6-Month Model Accuracy")
    print("R²   :", round(r2_score(test['OVERALL'], test_pred), 3))
    print("MAE  :", round(mean_absolute_error(test['OVERALL'], test_pred), 3))
    print("RMSE :", round(mean_squared_error(test['OVERALL'], test_pred, squared=False), 3))
else:
    print("⚠ Not enough data for backtesting")

# =========================
# 6. Train on FULL Data
# =========================
model.fit(monthly[['t']], monthly['OVERALL'])

# =========================
# 7. Predict Next 6 Months
# =========================
future_t = pd.DataFrame({
    't': np.arange(len(monthly), len(monthly) + 6)
})

future_dates = pd.date_range(
    start=monthly['DATE'].iloc[-1] + pd.offsets.MonthEnd(1),
    periods=6,
    freq='ME'
)

future_predictions = model.predict(future_t)

# HR-safe range
future_predictions = np.clip(future_predictions, 0, 100)

# =========================
# 8. Visualization
# =========================
plt.figure(figsize=(10,5))
plt.plot(monthly['DATE'], monthly['OVERALL'], marker='o', label='Historical')
plt.plot(future_dates, future_predictions, marker='o', linestyle='--', label='Forecast')
plt.title("6-Month Performance Prediction (XGBoost)")
plt.xlabel("Month")
plt.ylabel("OVERALL Score")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 9. Prediction Table
# =========================
prediction_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted_OVERALL": future_predictions
})

print("\n📅 Next 6 Months Prediction")
print(prediction_df)

# =========================
# 10. Performance Drop Analysis
# =========================
numeric_cols = df.select_dtypes(include=np.number)
features = numeric_cols.drop(columns=['OVERALL'], errors='ignore')

corr = features.corrwith(df['OVERALL']).dropna().sort_values()

# =========================
# 11. Correlation Plot
# =========================
plt.figure(figsize=(10,5))
corr.plot(kind='barh')
plt.axvline(0, color='black')
plt.title("Parameters Affecting Performance")
plt.xlabel("Correlation with OVERALL")
plt.grid(True)
plt.show()

# =========================
# 12. Negative Factors
# =========================
low_perf = corr[corr < -0.2]

print("\n⚠ Performance Downfall Reasons")
print(low_perf)

# =========================
# 13. Best & Worst Months
# =========================
best = monthly.nlargest(3, 'OVERALL')
worst = monthly.nsmallest(3, 'OVERALL')

plt.figure(figsize=(10,4))
plt.bar(best['DATE'].dt.strftime('%b %Y'), best['OVERALL'], color='green', label='Best')
plt.bar(worst['DATE'].dt.strftime('%b %Y'), worst['OVERALL'], color='red', label='Worst')
plt.title("Best vs Worst Performance Months")
plt.ylabel("OVERALL Score")
plt.legend()
plt.grid(True)
plt.show()
