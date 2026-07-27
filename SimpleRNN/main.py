import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_imdb_word_index():
    word_index = imdb.get_word_index()
    return word_index

import os
@st.cache_resource
def load_keras_model():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'simple_rnn_imdb.h5')
    
    return load_model(model_path)


word_index = load_imdb_word_index()
model = load_keras_model()


def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [word_index.get(word, 2) + 3 for word in words]

    encoded_review = [i if i < 10000 else 2 for i in encoded_review] 
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


with st.sidebar:

    st.title("👨‍💻 About the Developer")
    st.write("""
    **Name:** Rasheed Ahmad
    **Role:** Deep Learning / AI Enthusiast
    """)
    st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/rasheed-ahmad-ml-engineer-b56037311/) | [GitHub](https://github.com/rasheed89034)")
    
    st.divider() 
    
    st.title("ℹ️ About the Project")
    st.info("""
    **End-to-End Deep Learning Project**.
    
    * **Model:** Simple RNN (Recurrent Neural Network)
    * **Dataset:** 50,000 IMDB Reviews
    * **Objective:** Binary Text Classification (Positive / Negative)
    * **Max Vocabulary:** 10,000 words
    * **Sequence Length:** 500 words
    """)


st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("Provide an English movie review below, and our AI model will analyze whether the sentiment is Positive or Negative.")

# User Input Text Area
user_review = st.text_area(
    "✍️ Write moive review:", 
    height=150, 
    placeholder="Example: The movie was...."
)

# Predict Button
if st.button("Predict Sentiment 🚀", use_container_width=True):
    if user_review.strip() == "":
        st.warning("⚠️ Prediction ke liye pehle koi review enter karein!")
    else:
        with st.spinner("Analyzing Sentiment... 🧠"):
            # Preprocess and Predict
            processed_input = preprocess_text(user_review)
            prediction = model.predict(processed_input)
            score = prediction[0][0]
            
        
            st.divider()
            st.subheader("📊 Prediction Result")
            
            if score >= 0.5:
                st.success("🌟 **Positive Review!**")
                st.metric(label="Confidence Score", value=f"{score:.2%}")
                st.progress(float(score))
            else:
                st.error("**Negative Review!**")
                negative_confidence = 1 - score
                st.metric(label="Confidence Score", value=f"{negative_confidence:.2%}")
                st.progress(float(negative_confidence))
                
            with st.expander("Show Processed Tensor (Debugging)"):
                st.write(processed_input)
