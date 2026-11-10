
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page configuration
st.set_page_config(
    page_title="Weather Prediction",
    page_icon="🌦️",
    layout="wide"
)

# Main heading
st.title("🌦️ Weather Data Analysis and Prediction")

st.write(
    "This project analyzes historical weather data "
    "and predicts future temperature trends using Linear Regression."
)

st.divider()

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("data/weather.csv")
    data["date"] = pd.to_datetime(data["date"])
    data["temperature_c"] = pd.to_numeric(
        data["temperature_c"], errors="coerce"
    )
    data = data.dropna(subset=["date", "temperature_c"])
    data = data.sort_values("date").drop_duplicates("date")
    return data

try:
    data = load_data()
except Exception as e:
    st.error(f"Unable to load weather.csv: {e}")
    st.stop()

if len(data) < 10:
    st.error("Please provide at least 10 valid weather records.")
    st.stop()

# Dataset
st.header("1. Historical Weather Dataset")
st.dataframe(data, use_container_width=True)

# Statistical analysis
st.header("2. Weather Data Analysis")

average = data["temperature_c"].mean()
maximum = data["temperature_c"].max()
minimum = data["temperature_c"].min()

col1, col2, col3 = st.columns(3)

col1.metric("Average Temperature", f"{average:.2f} °C")
col2.metric("Maximum Temperature", f"{maximum:.2f} °C")
col3.metric("Minimum Temperature", f"{minimum:.2f} °C")

# Historical graph
st.header("3. Historical Temperature Trend")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    data["date"],
    data["temperature_c"],
    marker="o",
    linewidth=2
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Historical Temperature Analysis")
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Machine learning
st.header("4. Machine Learning - Linear Regression")

temperatures = data["temperature_c"].values

X = np.arange(len(temperatures)).reshape(-1, 1)
y = temperatures

# Chronological train-test split
split = int(len(data) * 0.8)

if split < 2 or len(data) - split < 2:
    st.error("Not enough records for training and testing.")
    st.stop()

model = LinearRegression()

model.fit(X[:split], y[:split])

test_predictions = model.predict(X[split:])

mae = mean_absolute_error(y[split:], test_predictions)
rmse = np.sqrt(mean_squared_error(y[split:], test_predictions))
r2 = r2_score(y[split:], test_predictions)

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:.2f} °C")
c2.metric("RMSE", f"{rmse:.2f} °C")
c3.metric("R² Score", f"{r2:.3f}")

st.caption("Evaluation uses the final 20% of records as test data.")

# Actual vs predicted
st.subheader("Actual vs Predicted Temperature")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(data["date"].iloc[split:], y[split:], label="Actual")
ax.plot(data["date"].iloc[split:], test_predictions, label="Predicted")

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.legend()
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Future prediction
st.header("5. Future Temperature Prediction")

days = st.slider(
    "Select number of future days",
    min_value=1,
    max_value=30,
    value=7
)

# Retrain on all available historical data
final_model = LinearRegression()
final_model.fit(X, y)

future_X = np.arange(
    len(data), len(data) + days
).reshape(-1, 1)

predictions = final_model.predict(future_X)

future_dates = pd.date_range(
    start=data["date"].max() + pd.Timedelta(days=1),
    periods=days
)

result = pd.DataFrame({
    "Date": future_dates,
    "Predicted Temperature (°C)": predictions.round(2)
})

st.subheader("Prediction Results")
st.dataframe(result, use_container_width=True)

# Future prediction graph
fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    data["date"].tail(30),
    data["temperature_c"].tail(30),
    label="Historical Temperature"
)

ax.plot(
    future_dates,
    predictions,
    marker="o",
    color="red",
    label="Predicted Temperature"
)

ax.set_xlabel("Date")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Future Temperature Prediction")
ax.legend()
ax.grid(alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Download predictions
st.download_button(
    label="Download Prediction CSV",
    data=result.to_csv(index=False),
    file_name="weather_prediction.csv",
    mime="text/csv"
)

# Conclusion
st.divider()

st.header("6. Conclusion")

st.success(
    "Weather Data Analysis and Prediction completed successfully!"
)

st.write(
    "This project demonstrates historical weather analysis, "
    "data visualization, Linear Regression and future "
    "temperature trend prediction."
)

st.warning(
    "This is an educational trend-prediction model, "
    "not a professional weather forecasting system."
)
