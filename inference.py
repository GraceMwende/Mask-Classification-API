import numpy as np
import cv2 
import tensorflow as tf
import os
import sys

# 0 = no_mask
# 1 = with mask
class_names = ['no_mask', 'with mask']

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading model from: {model_path}")
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load the image. Check the file type and path.")

    img_resized = cv2.resize(img, (128, 128))
    img_scaled = img_resized / 255.0
    img_final = np.reshape(img_scaled, (1, 128, 128, 3))
    return img_final

def predict_mask(model_path, image_path):
    # Load model
    model = load_model(model_path)

    # Preprocess image
    image_processed = preprocess_image(image_path)

    # Predict
    prediction_vector = model.predict(image_processed)[0]  # shape (2,)
    predicted_index = int(np.argmax(prediction_vector))
    predicted_label = class_names[predicted_index]

    # Message
    message = "The person is wearing a mask" if predicted_index == 1 else "The person is not wearing a mask"

    # Print results
    print("\n=== Prediction Result ===")
    print(f"Image: {image_path}")
    print(f"Prediction vector: {prediction_vector}")
    print(f"Predicted index: {predicted_index}")
    print(f"Predicted label: {predicted_label}")
    print(f"Message: {message}")

    return {
        "prediction": predicted_index,
        "label": predicted_label,
        "message": message
    }

# --------------------
# For command-line use
# --------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\nUsage:")
        print("  python inference.py mask_detector.h5 image.jpg\n")
        sys.exit(1)

    model_file = sys.argv[1]
    image_file = sys.argv[2]

    predict_mask(model_file, image_file)