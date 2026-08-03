from fastapi import FastAPI, Request
import tensorflow as tf
import numpy as np
from collections import deque

app = FastAPI()

model = tf.keras.models.load_model("lstm_har_model.keras")

# 2. 128 Readings ka Buffer banayein
buffer_x = deque(maxlen=128)
buffer_y = deque(maxlen=128)
buffer_z = deque(maxlen=128)

# UCI HAR Labels
activities = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
current_state = {"activity": "Waiting for data...", "buffer_size": 0}

@app.post("/webhook")
async def receive_sensor_data(request: Request):
    data = await request.json()
    
    try:
        # Sensor Logger app ka JSON parse karna
        payload = data.get('payload', [])
        for item in payload:
            if item.get('name') == 'accelerometer':
                vals = item.get('values')
                buffer_x.append(vals['x'])
                buffer_y.append(vals['y'])
                buffer_z.append(vals['z'])
        
        current_state["buffer_size"] = len(buffer_x)

        # 3. Jab Buffer mein 128 readings jama ho jayein toh Predict karein
        if len(buffer_x) == 128:
            sequence = np.column_stack((buffer_x, buffer_y, buffer_z))
            sequence = np.expand_dims(sequence, axis=0) # Shape: (1, 128, 3)
            
            prediction = model.predict(sequence, verbose=0)
            class_index = np.argmax(prediction[0])
            current_state["activity"] = activities[class_index]
            
    except Exception as e:
        print("Data parsing error:", e)
        
    return {"status": "success"}

@app.get("/status")
def get_status():
    return current_state