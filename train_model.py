import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("global_air_quality_data_10000.csv")

# =========================
# AQI FUNCTIONS
# =========================
def compute_sub_aqi(C, breakpoints):
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= C <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (C - C_low) + I_low
    return np.nan

pm25_bp = [
    (0, 12, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200)
]

pm10_bp = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200)
]

no2_bp = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150)
]

# =========================
# CREATE TARGET
# =========================
df["AQI_PM25"] = df["PM2.5"].apply(lambda x: compute_sub_aqi(x, pm25_bp))
df["AQI_PM10"] = df["PM10"].apply(lambda x: compute_sub_aqi(x, pm10_bp))
df["AQI_NO2"]  = df["NO2"].apply(lambda x: compute_sub_aqi(x, no2_bp))

df["AQI_Global"] = df[["AQI_PM25", "AQI_PM10", "AQI_NO2"]].max(axis=1)

def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy"
    else:
        return "Very Unhealthy"

df["Air_Quality"] = df["AQI_Global"].apply(aqi_category)

# =========================
# FEATURES / TARGET
# =========================
X = df.drop(
    ["Air_Quality", "AQI_Global", "AQI_PM25", "AQI_PM10", "AQI_NO2"],
    axis=1
)
y = df["Air_Quality"]

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# COLUMNS
# =========================
num_cols = [
    'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3',
    'Temperature', 'Humidity', 'Wind Speed'
]

cat_cols = ['City', 'Country', 'Date']

# =========================
# PIPELINES
# =========================
num_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

# =========================
# TRAIN
# =========================
final_pipeline.fit(X_train, y_train)

# =========================
# SAVE
# =========================
joblib.dump(final_pipeline, "final_model.pkl")

categories = {
    "City": sorted(df["City"].dropna().unique().tolist()),
    "Country": sorted(df["Country"].dropna().unique().tolist()),
    "Date": sorted(df["Date"].astype(str).dropna().unique().tolist())
}

joblib.dump(categories, "categories.pkl")

print("✅ Model trained and saved successfully")
