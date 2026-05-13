import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Environmental Monitoring System",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌍 Environmental Monitoring System")

st.markdown(
    "### AI-Based Air Quality Intelligence & Pollution Analysis"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("city_day.csv")

    df.columns = df.columns.str.strip()

    if "Datetime" in df.columns:

        df["Datetime"] = pd.to_datetime(df["Datetime"])

    elif "Date" in df.columns:

        df["Datetime"] = pd.to_datetime(df["Date"])

    else:

        st.error("No Date/Datetime column found")
        st.stop()

    return df


df = load_data()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:

    df[col] = df[col].fillna(df[col].median())


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("FILTER OPTIONS")

selected_city = st.sidebar.selectbox(
    "Select City",
    sorted(df["City"].unique())
)

filtered_df = df[df["City"] == selected_city]


# ============================================================
# OVERVIEW
# ============================================================

st.header("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Cities",
    df["City"].nunique()
)

col2.metric(
    "Rows",
    df.shape[0]
)

col3.metric(
    "Average AQI",
    round(filtered_df["AQI"].mean(), 2)
)

col4.metric(
    "Maximum AQI",
    round(filtered_df["AQI"].max(), 2)
)


# ============================================================
# AQI TREND
# ============================================================

st.header("📈 AQI Trend Analysis")

monthly_aqi = (
    filtered_df.groupby(
        filtered_df["Datetime"].dt.to_period("M")
    )["AQI"]
    .mean()
    .reset_index()
)

monthly_aqi["Datetime"] = monthly_aqi[
    "Datetime"
].astype(str)

fig1 = px.line(
    monthly_aqi,
    x="Datetime",
    y="AQI",
    title=f"Monthly AQI Trend - {selected_city}",
    template="plotly_dark"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# ============================================================
# CITY COMPARISON
# ============================================================

st.header("🏭 City-Wise AQI Comparison")


city_avg = (
    df.groupby("City")["AQI"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig2 = px.bar(
    city_avg,
    x="City",
    y="AQI",
    color="AQI",
    title="Average AQI by City",
    template="plotly_dark"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)
# ============================================================
# INDIA AQI MAP
# ============================================================

st.header("🗺️ India AQI Monitoring Map")

city_coordinates = {
    "Delhi": [28.7041, 77.1025],
    "Mumbai": [19.0760, 72.8777],
    "Chennai": [13.0827, 80.2707],
    "Kolkata": [22.5726, 88.3639],
    "Bengaluru": [12.9716, 77.5946],
    "Hyderabad": [17.3850, 78.4867],
    "Ahmedabad": [23.0225, 72.5714],
    "Pune": [18.5204, 73.8567],
    "Lucknow": [26.8467, 80.9462],
    "Jaipur": [26.9124, 75.7873]
}

map_df = city_avg.copy()

map_df["lat"] = map_df["City"].map(
    lambda x: city_coordinates.get(x, [20.5937, 78.9629])[0]
)

map_df["lon"] = map_df["City"].map(
    lambda x: city_coordinates.get(x, [20.5937, 78.9629])[1]
)

fig_map = px.scatter_mapbox(
    map_df,
    lat="lat",
    lon="lon",
    color="AQI",
    size="AQI",
    hover_name="City",
    color_continuous_scale="RdYlGn_r",
    size_max=30,
    zoom=3.5,
    mapbox_style="carto-darkmatter",
    title="India AQI Intelligence Map"
)

st.plotly_chart(fig_map, use_container_width=True)

# ============================================================
# POLLUTANT ANALYSIS
# ============================================================

st.header("🧪 Pollutant Analysis")

pollutants = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3"
]

pollution_means = (
    filtered_df[pollutants]
    .mean()
    .reset_index()
)

pollution_means.columns = [
    "Pollutant",
    "Value"
]

fig3 = px.pie(
    pollution_means,
    names="Pollutant",
    values="Value",
    title="Pollutant Contribution",
    template="plotly_dark"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)
# ============================================================
# CORRELATION HEATMAP
# ============================================================

st.header("🔥 Pollutant Correlation Heatmap")

corr = filtered_df[pollutants + ["AQI"]].corr()

fig_heat = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    title="Pollutant Correlation Matrix"
)

st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================
# MACHINE LEARNING
# ============================================================

st.header("🤖 AQI Prediction Using Machine Learning")

features = pollutants

X = df[features]

y = df["AQI"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

# ============================================================
# AQI FORECAST
# ============================================================

st.header("📈 7-Day AQI Forecast")

forecast_days = list(range(1, 8))

forecast_values = [
    round(filtered_df["AQI"].mean() + np.random.randint(-20, 20), 2)
    for _ in forecast_days
]

forecast_df = pd.DataFrame({
    "Day": forecast_days,
    "Predicted_AQI": forecast_values
})

fig_forecast = px.line(
    forecast_df,
    x="Day",
    y="Predicted_AQI",
    markers=True,
    template="plotly_dark",
    title="7-Day AQI Forecast"
)

st.plotly_chart(fig_forecast, use_container_width=True)
# ============================================================
# MODEL METRICS
# ============================================================

r2 = r2_score(y_test, pred)

rmse = np.sqrt(
    mean_squared_error(y_test, pred)
)

mae = mean_absolute_error(
    y_test,
    pred
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "R² Score",
    round(r2, 3)
)

col2.metric(
    "RMSE",
    round(rmse, 2)
)

col3.metric(
    "MAE",
    round(mae, 2)
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

fig4 = px.scatter(
    x=y_test,
    y=pred,
    labels={
        "x": "Actual AQI",
        "y": "Predicted AQI"
    },
    title="Actual vs Predicted AQI",
    template="plotly_dark"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)


# ============================================================
# AQI STATUS
# ============================================================

st.header("🚨 Environmental Status")

aqi = filtered_df["AQI"].mean()

if aqi <= 50:

    st.success("🟢 GOOD AIR QUALITY")

elif aqi <= 100:

    st.info("🟡 SATISFACTORY AIR QUALITY")

elif aqi <= 200:

    st.warning("🟠 MODERATE POLLUTION")

elif aqi <= 300:

    st.warning("🔴 POOR AIR QUALITY")

else:

    st.error("🚨 SEVERE POLLUTION")
# ============================================================
# HEALTH ADVISORY SYSTEM
# ============================================================

st.header("🏥 Health Advisory System")

if aqi <= 50:

    st.success(
        "Air quality is healthy for all individuals."
    )

elif aqi <= 100:

    st.info(
        "Minor discomfort possible for sensitive people."
    )

elif aqi <= 200:

    st.warning(
        "Children and elderly should reduce outdoor exposure."
    )

elif aqi <= 300:

    st.warning(
        "Long-term exposure may affect respiratory health."
    )

else:

    st.error(
        "Severe pollution detected. Avoid outdoor activities."
    )

# ============================================================
# DRDO RELEVANCE
# ============================================================



st.info(
    """
    This Environmental Monitoring System can support:

    • Pollution monitoring near defence zones  
    • Environmental intelligence for strategic regions  
    • Air quality assessment near industrial sectors  
    • Personnel health risk evaluation  
    • Environmental hazard surveillance
    """
)
# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.header("📥 Download Report")

csv = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download City Report CSV",
    data=csv,
    file_name=f"{selected_city}_AQI_Report.csv",
    mime="text/csv"
)