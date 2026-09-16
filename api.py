from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from pydantic import BaseModel
import scipy.ndimage as ndimage

# 1. Initialize the FastAPI servere
app = FastAPI()

# 2. Set up CORS so your React frontend is allowed to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 3. Load your trained neural network into memory
print("Loading the brain...")
model = tf.keras.models.load_model('handwritten_model.keras')
print("Brain loaded successfully!")

# 4. Create the rulebook for incoming data
class ImageData(BaseModel):
    #we expect React to send  signal list containing 784 pixel values
    pixels: list[float]

# 5. Create the upgraded prediction endpoint
@app.post("/predict")
async def predict_digit(data: ImageData):
    # Step A: Convert to a 28x28 grid immediately
    pixel_array = np.array(data.pixels)
    image = pixel_array.reshape(28, 28)
    
    # Step B: Calculate the "Center of Mass" of the ink
    cy, cx = ndimage.center_of_mass(image)
    
    # Step C: If the canvas isn't totally blank, calculate how far off-center it is
    if not np.isnan(cy) and not np.isnan(cx):
        # The true center of a 28x28 grid is (14, 14)
        shift_y = 14.0 - cy
        shift_x = 14.0 - cx
        
        # Slide the pixels mathematically to perfectly center the drawing
        image = ndimage.shift(image, [shift_y, shift_x])
    
    # Step D: Package it back up for the neural network
    image_ready = image.reshape(1, 28, 28)
    
    # Step E: Ask the brain to make a prediction
    prediction_probabilities = model.predict(image_ready)
    predicted_number = np.argmax(prediction_probabilities[0])
    
    return {"prediction": int(predicted_number)}