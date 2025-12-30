import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Senchola Performance Portal", layout="wide")
os.makedirs("employees_data", exist_ok=True)

# =========================
# Logo
# =========================
logo_path = "senchola-logo.jpeg"
if os.path.exists(logo_path):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(logo_path, width=700)

# =========================
# HR Credentials
# =========================
HR_USERS = {"HR001": "senchola@hr"}

# =========================
# Employee Master Data
# =========================
EMPLOYEES = {
    "EMP1001": {"name": "ADHARSH", "dob": "1998-08-20", "file": "ADHARSH.xlsx"},
    "EMP1002": {"name": "BALASUBRAMANIAN", "dob": "1999-01-15", "file": "BALASUBRAMANIAN.xlsx"},
    "EMP1003": {"name": "GUNASEELAN", "dob": "1997-06-10", "file": "GUNASEELAN.xlsx"},
    "EMP1004": {"name": "NAGARAJAN", "dob": "1997-06-10", "file": "NAGARAJAN.xlsx"},
    "EMP1005": {"name": "PAVITHRA", "dob": "1997-06-10", "file": "PAVITHRA.xlsx"},
    "EMP1006": {"name": "SANA", "dob": "1997-06-10", "file": "SANA.xlsx"},
    "EMP1007": {"name": "JUSTIN TITTO", "dob": "1997-06-10", "file": "JUSTIN_TITTO.xlsx"},
    "EMP1008": {"name": "SUSINTHRAN", "dob": "1997-06-10", "file": "SUSINTHRAN.xlsx"},
}

# =========================
# Session State
# =========================
if "role" not in st.session_state:
    st.session_state.role = None
    st.session_state.user = None

# =========================
# Authentication
# =========================
def authenticate(role, user, pwd):
    if role == "HR":
        return HR_USERS.get(user) == pwd
    if role == "Employee":
        return user in EMPLOYEES and EMPLOYEES[user]["dob"] == pwd
    return False

# =========================
# Login Screen
# =========================
if st.session_state.role is None:
    st.title("🔐 Senchola Login")

    role = st.radio("Login As", ["HR", "Employee"])
    uid = st.text_input("User ID")
    pwd = st.text_input("Password / DOB", type="password")

    if st.button("Login"):
        if authenticate(role, uid, pwd):
            st.session_state.role = role
            st.session_state.user = uid
            st.rerun()
        else:
            st.error("Invalid credentials")

# =========================
# Dashboard
# =========================
else:
    st.sidebar.success(f"{st.session_state.role} : {st.session_state.user}")

    # =========================
    # Employee Selection
    # =========================
    if st.session_state.role == "HR":
        selected_emp = st.sidebar.selectbox(
            "Select Employee",
            list(EMPLOYEES.keys()),
            format_func=lambda x: EMPLOYEES[x]["name"]
        )
    else:
        selected_emp = st.session_state.user

    emp = EMPLOYEES[selected_emp]
    st.title(f"📊 Performance Dashboard – {emp['name']}")

    # =========================
    # Load & CLEAN Data
    # =========================
    if not os.path.exists(emp["file"]):
        st.error("Performance file not found")
        st.stop()

    df = pd.read_excel(emp["file"], engine="openpyxl")

    # 🔥 CRITICAL CLEANING
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["OVERALL"] = pd.to_numeric(df["OVERALL"], errors="coerce")

    df = df.dropna(subset=["DATE", "OVERALL"]).reset_index(drop=True)

    if len(df) < 5:
        st.error("Not enough valid data after cleaning")
        st.stop()

    # =========================
    # Monthly Aggregation
    # =========================
    monthly = (
        df.resample("M", on="DATE")
        .mean(numeric_only=True)
        .reset_index()
    )

    monthly = monthly.dropna(subset=["OVERALL"]).reset_index(drop=True)
    monthly["t"] = np.arange(len(monthly))

    if len(monthly) < 3:
        st.error("Not enough monthly data")
        st.stop()

    model = LinearRegression()

    # =========================
    # 6-Month Backtest
    # =========================
    st.subheader("📐 Model Accuracy")

    if len(monthly) > 6:
        train = monthly.iloc[:-6]
        test = monthly.iloc[-6:]

        model.fit(train[["t"]], train["OVERALL"])
        pred = model.predict(test[["t"]])

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", round(r2_score(test["OVERALL"], pred), 3))
        c2.metric("MAE", round(mean_absolute_error(test["OVERALL"], pred), 3))
        c3.metric("RMSE", round(mean_squared_error(test["OVERALL"], pred, squared=False), 3))
    else:
        st.info("Not enough data for 6-month backtest")

    # =========================
    # Train Full Data & Forecast
    # =========================
    model.fit(monthly[["t"]], monthly["OVERALL"])

    future_t = pd.DataFrame({
        "t": np.arange(len(monthly), len(monthly) + 6)
    })

    future_dates = pd.date_range(
        start=monthly["DATE"].iloc[-1] + pd.offsets.MonthEnd(1),
        periods=6,
        freq="M"
    )

    future_pred = model.predict(future_t)

    pred_df = pd.DataFrame({
        "Month": future_dates,
        "Predicted OVERALL": future_pred.round(2)
    })

    st.subheader("📅 Next 6 Months Prediction")
    st.dataframe(pred_df, use_container_width=True)

    # =========================
    # Trend Plot
    # =========================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly["DATE"], monthly["OVERALL"], marker="o", label="Actual")
    ax.plot(future_dates, future_pred, linestyle="--", marker="o", label="Forecast")
    ax.set_title("Performance Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("OVERALL")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # =========================
    # PERFORMANCE DROP ANALYSIS
    # =========================
    numeric_cols = df.select_dtypes(include=np.number)
    features = numeric_cols.drop(columns=["OVERALL"], errors="ignore")

    corr = features.corrwith(df["OVERALL"]).dropna().sort_values()

    st.subheader("⚠ Performance Impact Analysis")

    if not corr.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        corr.plot(kind="barh", ax=ax2)
        ax2.axvline(0, color="black")
        ax2.set_title("Correlation with OVERALL")
        ax2.grid(True)
        st.pyplot(fig2)

        negative = corr[corr < 0]
        if not negative.empty:
            st.error("Negative Impact Parameters")
            st.dataframe(negative)
        else:
            st.success("No strong negative indicators found")
    else:
        st.info("No numeric performance factors found")

    # =========================
    # Logout
    # =========================
    if st.sidebar.button("Logout"):
        st.session_state.role = None
        st.session_state.user = None
        st.rerun()
