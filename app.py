import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

# =========================
# GLOBAL DARK CSS (NO WHITE ANYWHERE)
# =========================
st.markdown("""
<style>

/* Remove Streamlit UI */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* App background */
.stApp {
    background: radial-gradient(1200px 600px at 20% 0%, rgba(59,130,246,.15), transparent 60%),
                radial-gradient(900px 500px at 90% 10%, rgba(16,185,129,.12), transparent 55%),
                #0b1220;
    color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220, #0e1627);
    border-right: 1px solid rgba(255,255,255,.08);
}
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Inputs (select, text, date) */
.stSelectbox > div > div,
.stTextInput > div > div,
.stDateInput > div > div {
    background: #1f2937 !important;
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 10px;
}

/* Dropdown list */
ul[role="listbox"] {
    background: #020617 !important;
    border: 1px solid rgba(255,255,255,.12);
}
ul[role="listbox"] li {
    color: #e5e7eb !important;
}
ul[role="listbox"] li:hover {
    background: #1f2937 !important;
}

/* Buttons */
.stButton > button {
    background: #1f2937;
    color: #e5e7eb;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.15);
    font-weight: 700;
}
.stButton > button:hover {
    background: #374151;
}

/* Tabs colors */
button[data-baseweb="tab"] {
    color: rgba(229,231,235,.70) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #e5e7eb !important;
    border-bottom: 2px solid rgba(59,130,246,.9) !important;
}

/* Cards */
.card {
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,.35);
}

/* Result */
.result {
    border-radius: 18px;
    padding: 16px;
    color: white;
    box-shadow: 0 14px 36px rgba(0,0,0,.35);
}

/* Custom dark table */
.table-card{
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px;
  overflow: hidden;
}
.table-title{
  padding: 14px 16px;
  border-bottom: 1px solid rgba(255,255,255,.08);
  font-weight: 800;
}
.dark-table{
  width: 100%;
  border-collapse: collapse;
}
.dark-table th{
  background: #020617;
  padding: 12px 14px;
  text-align: left;
}
.dark-table td{
  padding: 12px 14px;
  border-bottom: 1px solid rgba(255,255,255,.06);
}
.dark-table tr:nth-child(even){
  background: rgba(255,255,255,.03);
}
.dark-table tr:hover{
  background: rgba(59,130,246,.12);
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
pipeline = joblib.load("final_model.pkl")
categories = joblib.load("categories.pkl")

# =========================
# AQI STYLE
# =========================
def aqi_style(aqi):
    styles = {
        "Good": ("🟢", "#16a34a", "Air quality is good."),
        "Moderate": ("🟡", "#ca8a04", "Sensitive people should be cautious."),
        "Unhealthy": ("🟠", "#ea580c", "Limit outdoor activity."),
        "Very Unhealthy": ("🔴", "#dc2626", "Avoid outdoor exposure.")
    }
    return styles.get(aqi, ("⚪", "#64748b", "No description available."))

# =========================
# HEADER (CENTERED)
# =========================
st.markdown("""
<div style="text-align:center;margin-bottom:20px;">
    <h1>🌍 Air Quality Prediction</h1>
    <p style="opacity:.7;">Smart environmental monitoring </p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.markdown("## ⚙️ Inputs")

with st.sidebar.expander("🌫️ Pollution", True):
    pm25 = st.slider("PM2.5", 0, 500, 35)
    pm10 = st.slider("PM10", 0, 500, 30)
    no2  = st.slider("NO2", 0, 300, 40)
    so2  = st.slider("SO2", 0, 300, 20)
    co   = st.slider("CO", 0, 50, 2)
    o3   = st.slider("O3", 0, 300, 60)

with st.sidebar.expander("🌡️ Weather", True):
    temperature = st.slider("Temperature", -10, 60, 25)
    humidity = st.slider("Humidity", 0, 100, 45)
    wind_speed = st.slider("Wind Speed", 0, 150, 15)

with st.sidebar.expander("📍 Location", True):
    city_choice = st.selectbox("City", categories["City"] + ["Other"])
    city = st.text_input("Enter city") if city_choice == "Other" else city_choice

    country_choice = st.selectbox("Country", categories["Country"] + ["Other"])
    country = st.text_input("Enter country") if country_choice == "Other" else country_choice

    date = st.selectbox("Date", categories["Date"])

# =========================
# INPUT DATAFRAME
# =========================
input_df = pd.DataFrame({
    "PM2.5":[pm25],
    "PM10":[pm10],
    "NO2":[no2],
    "SO2":[so2],
    "CO":[co],
    "O3":[o3],
    "Temperature":[temperature],
    "Humidity":[humidity],
    "Wind Speed":[wind_speed],
    "City":[city],
    "Country":[country],
    "Date":[date]
})

# =========================
# LAYOUT
# =========================
left, right = st.columns([2, 1])

# =========================
# LEFT – INPUT SUMMARY (HTML DARK TABLE)
# =========================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧾 Input Summary")

    rows = "".join([
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in input_df.iloc[0].items()
    ])

    st.markdown(f"""
    <div class="table-card">
      <table class="dark-table">
        <thead><tr><th>Feature</th><th>Value</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# RIGHT – PREDICTION + TABS
# =========================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Prediction")

    if st.button("🚀 Run Prediction", use_container_width=True):
        pred = pipeline.predict(input_df)[0]
        icon, color, msg = aqi_style(pred)

        # RESULT BOX
        st.markdown(f"""
        <div class="result" style="background:linear-gradient(135deg,{color},#020617)">
            <h3>{icon} {pred}</h3>
            <p>{msg}</p>
        </div>
        """, unsafe_allow_html=True)

        # POLLUTANTS DICT
        pollutants = {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3
        }

        # TABS
        tab1, tab2, tab3 = st.tabs(["📈 Charts", "📝 Insights", "📄 PDF"])

        # TAB 1 - CHARTS
        with tab1:
            fig, ax = plt.subplots()
            bars = ax.bar(pollutants.keys(), pollutants.values())
            ax.set_ylabel("Concentration")
            ax.set_title("Pollution Indicators")
            ax.grid(axis="y", alpha=0.3)

            for b in bars:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height(),
                    f"{int(b.get_height())}",
                    ha="center",
                    va="bottom"
                )

            st.pyplot(fig, use_container_width=True)

        # TAB 2 - INSIGHTS
        with tab2:
            dominant_pollutant = max(pollutants, key=pollutants.get)

            st.markdown("#### 🔍 Key Insights")
            st.write(f"- **Air Quality Level:** `{pred}`")
            st.write(f"- **Dominant Pollutant:** `{dominant_pollutant}` ({pollutants[dominant_pollutant]})")
            st.write(f"- **Location:** {city}, {country}")
            st.write(f"- **Weather:** {temperature}°C, {humidity}% humidity, wind {wind_speed} km/h")

            if pred in ["Unhealthy", "Very Unhealthy"]:
                st.warning("⚠️ Sensitive groups should reduce outdoor activities.")
            elif pred == "Moderate":
                st.info("ℹ️ Air quality is acceptable, but caution is advised for sensitive people.")
            else:
                st.success("✅ Air quality is good. Enjoy outdoor activities.")

        # TAB 3 - PDF
        with tab3:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            pdf.cell(0, 10, "Air Quality Prediction Report", ln=True)
            pdf.ln(4)
            pdf.cell(0, 10, f"Prediction: {pred}", ln=True)
            pdf.cell(0, 10, f"City: {city}", ln=True)
            pdf.cell(0, 10, f"Country: {country}", ln=True)
            pdf.cell(0, 10, f"Date: {date}", ln=True)
            pdf.ln(4)
            pdf.cell(0, 10, "Input Parameters:", ln=True)

            for k, v in input_df.iloc[0].items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.output(tmp.name)

            with open(tmp.name, "rb") as f:
                st.download_button(
                    "📄 Download PDF Report",
                    f,
                    file_name="air_quality_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    else:
        st.caption("Click **Run Prediction** to show charts, insights and export PDF.")

    st.markdown("</div>", unsafe_allow_html=True)

st.caption("© Air Quality Dashboard")





