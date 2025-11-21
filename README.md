
# Face Mask Detection (CNN + FastAPI + Docker)

This project detects whether a person is **wearing a face mask or not** using a Convolutional Neural Network (CNN) built with **TensorFlow / Keras**, served via a **FastAPI** backend, and containerized with **Docker**.

## 📁 Project Structure

```
Face Mask Detection/
├─ facemaskclassification.ipynb
├─ server.py
├─ inference.py
├─ mask_detector.h5
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

##  Problem & Approach

- **Goal:** Detect if a person is wearing a mask.
- **Labels:**
  - `0` → without mask
  - `1` → with mask

##  Data & Preprocessing

1. Load image
2. Resize → (128,128)
3. Normalize `/255`
4. Expand dims → `(1,128,128,3)`

##  Model Architecture

- Conv2D (32) → MaxPool
- Conv2D (64) → MaxPool
- Flatten
- Dense(128) + Dropout
- Dense(64) + Dropout
- Dense(2) output

##  Local Inference

```
python inference.py mask_detector.h5 path/image.jpg
```

##  FastAPI Endpoints

- `POST /predict` → Upload an image

##  Docker

Build Docker image:
```
docker build -t face-mask-api .
```

Run Container:
```
docker run --name face-mask-api-container -p 8000:8000 face-mask-api
```

Query model Via the web interface(chrome):
```
http://127.0.0.1:8000/docs -> test model
```
### How To run the inference script
- `Locally` - python inference.py mask_detector.h5 images/imageswithout.jpg

- `API` -Start Docker container with your FastAPI app

        Open http://localhost:8000/docs

        Use /predict, upload the same image

        Compare the output with what inference.py printed
        
##  Common Issues

- CUDA warnings → safe to ignore
- Missing `python-multipart` → install it
- OpenCV DLL error → ensure correct environment
