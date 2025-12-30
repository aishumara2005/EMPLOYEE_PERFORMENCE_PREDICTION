import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# 1. Load Dataset
# =========================
df = pd.read_excel("PAVITHRA.xlsx")

# Convert DATE column
df['DATE'] = pd.to_datetime(df['DATE'])

# =========================
# 2. Monthly Aggregation
# =========================
# Take monthly average performance
monthly = df.resample('M', on='DATE').mean(numeric_only=True).reset_index()

# Create time index for regression
monthly['t'] = np.arange(len(monthly))

# =========================
# 3. Train Regression Model
# =========================
X = monthly[['t']]
y = monthly['OVERALL']

model = LinearRegression()
model.fit(X, y)

# =========================
# 4. Predict Next 6 Months
# =========================
future_t = np.arange(len(monthly), len(monthly) + 6).reshape(-1, 1)

future_dates = pd.date_range(
    start=monthly['DATE'].iloc[-1] + pd.offsets.MonthEnd(1),
    periods=6,
    freq='M'
)

future_predictions = model.predict(future_t)

# =========================
# 5. Visualization
# =========================
plt.figure()
plt.plot(monthly['DATE'], y, marker='o', label='Historical Performance')
plt.plot(future_dates, future_predictions, marker='o',
         linestyle='--', label='Predicted (Next 6 Months)')

plt.xlabel("Date")
plt.ylabel("Overall Performance Score")
plt.title("Overall Performance Prediction Using Regression")
plt.legend()
plt.show()

# =========================
# 6. Print Predictions
# =========================
prediction_df = pd.DataFrame({
    "Month": future_dates,
    "Predicted_OVERALL": future_predictions
})

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

# =========================
# 12. Scatter Plots for Top 3 Low Parameters
# =========================
for col in low_perf.index[:3]:
    plt.figure()
    plt.scatter(df[col], df['OVERALL'])
    plt.xlabel(col)
    plt.ylabel("OVERALL Performance")
    plt.title(f"{col} vs Overall Performance")
    plt.grid(True)
    plt.show()
