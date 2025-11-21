from fastapi import FastAPI,UploadFile,File
import numpy as np
import cv2 
import tensorflow as tf
import os

# model = tf.keras.models.load_model('mask_detector.keras')
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mask_detector.h5")
model = tf.keras.models.load_model(MODEL_PATH)


class_names = ['no_mask', 'with mask']

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Predict whether has mask or not '}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    '''
    Predicts whether someone has a mask or not for a given presented image
    '''
    # Read uploaded file as bytes
    contents = await file.read()

    # Convert bytes → numpy array → OpenCV image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #preprocess
    
    input_image_resized = cv2.resize(img, (128,128))
    input_image_scaled = input_image_resized/255
    input_image_reshaped = np.reshape(input_image_scaled, [1,128,128,3])
    input_prediction = model.predict(input_image_reshaped)
    input_pred_label = np.argmax(input_prediction)
    
    # Map index → label
    label = class_names[input_pred_label]     # "no_mask" or "mask"

    # Log for terminal
    print(f'Prediction: {input_pred_label} (label: {label})')
    results = 'The person is wearing a mask' if input_pred_label ==1 else 'The person is not wearing a mask'
    print(results)

    return {
        'prediction':int(input_pred_label),
        'label':label,
        'description': '1 = mask, 0 = no_mask'
    }