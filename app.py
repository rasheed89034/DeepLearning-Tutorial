import streamlit as st
import requests
import time

# Page config
st.set_page_config(page_title="AI Activity Tracker", page_icon="🏃‍♂️", layout="centered")

st.title("🏃‍♂️ Real-Time Human Activity Recognition")
st.markdown("Live predictions powered by **LSTM & FastAPI**")
st.divider()

# UI Container
status_placeholder = st.empty()

while True:
    try:
        # FastAPI se current status mangwana
        # Agar FastAPI alag machine par hai toh 127.0.0.1 ki jagah uska IP likhein
        response = requests.get("http://127.0.0.1:8000/status").json()
        activity = response['activity']
        buffer_len = response['buffer_size']
        
        with status_placeholder.container():
            # Khubsurat metric display
            st.metric(label="Current Detected Activity", value=activity)
            
            # Progress bar for data buffer
            progress = buffer_len / 128.0
            st.progress(progress, text=f"Data Buffer Status: {buffer_len}/128 readings")
            
            if activity == "WALKING":
                st.success("User is in motion! 🚶‍♂️")
            elif activity == "SITTING":
                st.info("User is resting. 🪑")
                
    except requests.exceptions.ConnectionError:
        status_placeholder.error("Waiting for FastAPI server to start...")
        
    time.sleep(1) # Har 1 second baad dashboard update hoga