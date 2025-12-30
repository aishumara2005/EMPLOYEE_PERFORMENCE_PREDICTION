import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =========================
# 1. Load Dataset
# =========================
df = pd.read_excel("ADHARSH.xlsx")
df['DATE'] = pd.to_datetime(df['DATE'])

# =========================
# 2. Monthly Aggregation
# =========================
monthly = (
    df.resample('ME', on='DATE')
      .mean(numeric_only=True)
      .reset_index()
)

monthly['t'] = np.arange(len(monthly))

# =========================
# 3. Initialize Model
# =========================
model = LinearRegression()

# =========================
# 4. 6-Month Accuracy (ONLY if possible)
# =========================
if len(monthly) > 6:
    train = monthly.iloc[:-6]
    test = monthly.iloc[-6:]

    X_train = train[['t']]
    y_train = train['OVERALL']
    X_test = test[['t']]
    y_test = test['OVERALL']

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    print("6-Month Accuracy (Back-Tested)")
    print("R² Score :", round(r2_score(y_test, y_test_pred), 3))
    print("MAE      :", round(mean_absolute_error(y_test, y_test_pred), 3))
    print("RMSE     :", round(mean_squared_error(y_test, y_test_pred, squared=False), 3))



else:
    print("⚠ Not enough historical months to compute 6-month accuracy.")
    print("Training model using all available data.")

    model.fit(monthly[['t']], monthly['OVERALL'])

# =========================
# 5. Retrain on Full Data
# =========================
model.fit(monthly[['t']], monthly['OVERALL'])

# =========================
# 6. Predict Next 6 Months
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

# =========================
# 7. Visualization
# =========================
plt.figure()
plt.plot(monthly['DATE'], monthly['OVERALL'],
         marker='o', label='Historical Performance')

plt.plot(future_dates, future_predictions,
         marker='o', linestyle='--',
         label='Predicted Next 6 Months')

plt.xlabel("Date")
plt.ylabel("Overall Performance Score")
plt.title("6-Month Performance Prediction Using Linear Regression")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# 8. Prediction Table
# =========================
prediction_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted_OVERALL": future_predictions
})

print("\nNext 6 Months Performance Prediction")
print(prediction_df)
# =========================
# 9. PERFORMANCE DROP ANALYSIS
# =========================

# Select numeric columns only
numeric_cols = df.select_dtypes(include=np.number)

# Remove OVERALL from features
features = numeric_cols.drop(columns=['OVERALL'], errors='ignore')

# Calculate correlation with OVERALL
corr = features.corrwith(df['OVERALL']).sort_values()

# =========================
# 10. Correlation Visualization
# =========================
plt.figure(figsize=(10, 5))
corr.plot(kind='barh')
plt.axvline(0)
plt.title("Parameters Affecting Employee Performance")
plt.xlabel("Correlation with OVERALL Performance")
plt.ylabel("Performance Factors")
plt.grid(True)
plt.show()

# =========================
# 11. LOW PERFORMANCE PARAMETERS (Negative Impact)
# =========================
low_perf = corr[corr < 0]

print("\n⚠ Performance Down Reasons (Negative Correlation):")
print(low_perf)
best = monthly.nlargest(3, 'OVERALL')
worst = monthly.nsmallest(3, 'OVERALL')

plt.figure(figsize=(10,4))
plt.bar(best['DATE'].dt.strftime('%b %Y'), best['OVERALL'], label='Best Months')
plt.bar(worst['DATE'].dt.strftime('%b %Y'), worst['OVERALL'], label='Worst Months')
plt.title("Best & Worst Performance Months")
plt.xlabel("Month")
plt.ylabel("OVERALL Score")
plt.legend()
plt.grid(True)
plt.show()
