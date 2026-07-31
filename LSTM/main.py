import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = FastAPI(title="Next Word Predictor API")

# CORS setup so the HTML frontend can talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model and Tokenizer globally so they only load once
print("Loading model and tokenizer...")
try:
    model = tf.keras.models.load_model("next_word_lstm.h5")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # Calculate max sequence length based on the model's input shape
    # model.input_shape is usually (None, max_sequence_len - 1)
    max_sequence_len = model.input_shape[1] + 1 
    print(f"Model loaded successfully. Max sequence length: {max_sequence_len}")
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    model = None
    tokenizer = None

# Input Data Schema
class TextRequest(BaseModel):
    text: str
    num_words: int = 1  # How many future words to predict

def predict_next_words(text, num_words=1):
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded properly.")
    
    predicted_text = text
    for _ in range(num_words):
        # Tokenize the input text
        token_list = tokenizer.texts_to_sequences([predicted_text])[0]
        # Pad the sequence to match the input shape of the model
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        
        # Predict the next word probabilities
        predicted_probs = model.predict(token_list, verbose=0)
        # Get the index of the highest probability
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]
        
        # Convert index back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        predicted_text += " " + output_word
    
    return predicted_text

@app.get("/")
def read_root():
    return {"message": "Next Word Prediction API is running. Send POST request to /predict."}

@app.post("/predict")
def predict_word(request: TextRequest):
    try:
        input_text = request.text.strip()
        if not input_text:
             raise HTTPException(status_code=400, detail="Input text cannot be empty.")
            
        result = predict_next_words(input_text, request.num_words)
        
        # Extract just the newly predicted words
        new_words = result[len(input_text):].strip()
        
        return {
            "input": input_text,
            "full_prediction": result,
            "predicted_words": new_words
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)