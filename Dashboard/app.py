import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from streamlit_autorefresh import st_autorefresh
from gtts import gTTS
import base64
import io

# ================= CONFIG =================
st.set_page_config(page_title="AI Disease Voice Intelligence", layout="wide")
st_autorefresh(interval=15000, key="refresh")

# ================= PREMIUM UI =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#000000);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}
.card {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
    padding: 20px;
    border-radius: 18px;
    color:white;
    box-shadow: 0 8px 32px rgba(0,255,255,0.2);
}
h1,h2,h3 {color:#00ffe0;}
.wave-container {
    display:flex;
    justify-content:center;
    align-items:center;
    height:80px;
}
.bar {
    width:6px;
    height:30px;
    margin:0 3px;
    background:#00ffe0;
    animation:wave 1s infinite ease-in-out;
}
.bar:nth-child(2) { animation-delay:0.1s;}
.bar:nth-child(3) { animation-delay:0.2s;}
.bar:nth-child(4) { animation-delay:0.3s;}
.bar:nth-child(5) { animation-delay:0.4s;}
@keyframes wave {
    0%,100% { height:20px; }
    50% { height:60px; }
}
.speaking-text {
    text-align:center;
    color:#00ffe0;
    font-size:18px;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Disease Early Warning Voice Intelligence System")

# ================= LOAD DATA =================
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
if uploaded_file is None:
    st.warning("Upload dataset to continue.")
    st.stop()

df = pd.read_csv(uploaded_file)

# ================= SIDEBAR OPTIONS =================
disease_list = df["Disease"].unique()
selected_disease = st.sidebar.selectbox("Select Disease", disease_list)

state_list = df["state_ut"].unique()
selected_state = st.sidebar.selectbox("Select State", state_list)

city_list = df[df["state_ut"] == selected_state]["district"].unique()
selected_city = st.sidebar.selectbox("Select City", city_list)

model_name = st.sidebar.selectbox(
    "Select ML Model",
    ["XGBoost", "Random Forest", "Gradient Boosting", "Extra Trees", "SVM", "Logistic Regression"]
)

# ================= PREPARE DATA =================
df_disease = df[df["Disease"] == selected_disease].copy()
median_cases = df_disease["Cases"].median()
df_disease["Target"] = (df_disease["Cases"] > median_cases).astype(int)

features = ['Deaths','preci','LAI','Temp']
X = df_disease[features]
y = df_disease["Target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# ================= MODEL SELECTION =================
if model_name == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
elif model_name == "Random Forest":
    model = RandomForestClassifier()
elif model_name == "Gradient Boosting":
    model = GradientBoostingClassifier()
elif model_name == "Extra Trees":
    model = ExtraTreesClassifier()
elif model_name == "SVM":
    model = SVC(probability=True)
else:
    model = LogisticRegression(max_iter=1000)

model.fit(X_train,y_train)
accuracy = accuracy_score(y_test,model.predict(X_test))

# ================= MODEL INFO =================
col1,col2 = st.columns(2)
col1.markdown(f'<div class="card"><h3>Selected Model</h3>{model_name}</div>',unsafe_allow_html=True)
col2.markdown(f'<div class="card"><h3>Accuracy</h3>{accuracy:.2%}</div>',unsafe_allow_html=True)

# ================= LIVE RISK + AUTO VOICE =================
st.header("🌍 Live Risk Assessment")

if st.button("Analyze Risk"):

    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={selected_city}"
    geo_data = requests.get(geo_url).json()

    if "results" in geo_data:

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_data = requests.get(weather_url).json()

        current_temp = weather_data["current_weather"]["temperature"]
        lai = df_disease["LAI"].mean()
        avg_prec = df_disease["preci"].mean()

        input_data = np.array([[0,avg_prec,lai,current_temp]])
        probability = model.predict_proba(input_data)[0][1]

        risk_label = "HIGH RISK" if probability > 0.5 else "LOW RISK"

        # Risk Gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability*100,
            title={'text':f"{selected_state} Risk (%)"},
            gauge={'axis':{'range':[0,100]}}
        ))
        gauge.update_layout(template="plotly_dark")
        st.plotly_chart(gauge,use_container_width=True)

        # Voice Message
        message = f"{selected_state}, {selected_city} has {risk_label} for {selected_disease}"
        st.success(message)

        # Generate audio
        tts = gTTS(text=message, lang='en')
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Convert to base64
        audio_base64 = base64.b64encode(audio_buffer.read()).decode()

        # Animated Waveform
        st.markdown("""
        <div class="wave-container">
          <div class="bar"></div>
          <div class="bar"></div>
          <div class="bar"></div>
          <div class="bar"></div>
          <div class="bar"></div>
        </div>
        <div class="speaking-text">🎙 AI Speaking...</div>
        """, unsafe_allow_html=True)

        # Auto-play audio
        st.markdown(f"""
        <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """, unsafe_allow_html=True)

    else:
        st.error("City not found.")

st.markdown("### 🚀 AI Early Warning Voice Intelligence Platform (Fully Automatic)")