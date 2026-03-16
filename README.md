# PureCheck — Food Adulteration Detection System

A full-stack web application for detecting food adulteration using a
**Hybrid ResNet + 3D CNN + PCA** deep learning model trained on hyperspectral imaging data.

---

## Project Structure

```
food_adulteration/
├── app.py                          # Flask backend + model inference
├── requirements.txt
├── hsi_ultimate_model.keras        # ← Copy your model here
├── hsi_ultimate_model_pca_models.pkl  # ← Copy PCA models here
├── templates/
│   ├── index.html                  # Cover / landing page
│   └── test.html                   # Sample analysis page
├── uploads/                        # Temp upload directory (auto-created)
└── README.md
```

---

## Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place model files

Copy your model files into the project root:

```bash
cp /path/to/hsi_ultimate_model.keras ./
cp /path/to/hsi_ultimate_model_pca_models.pkl ./
```

### 3. Run the server

```bash
python app.py
```

Visit: **http://localhost:5000**

---

## Features

| Feature | Description |
|---|---|
| Cover Page | Animated landing page explaining food adulteration |
| Image Upload | Drag-and-drop or click to upload (JPG, PNG, TIFF, NPY, HDR, BMP) |
| AI Prediction | Real model inference via Hybrid ResNet + 3D CNN + PCA |
| Confidence Score | Visual confidence bar + probability breakdown |
| Adulterant Detection | Lists detected substances when adulteration found |
| PDF Report | Downloadable professional report with full analysis details |
| Simulation Mode | Runs with heuristic analysis if TensorFlow not installed |

---

## API Endpoint

### `POST /api/predict`

Upload a food sample image and get an adulteration prediction.

**Request:** `multipart/form-data` with `file` field

**Response:**
```json
{
  "prediction": "Adulterated",
  "is_adulterated": true,
  "confidence": 87.3,
  "raw_scores": {
    "pure": 12.7,
    "adulterated": 87.3
  },
  "detected_adulterants": ["Sudan dye contamination", "Artificial colorant traces"],
  "severity": "High",
  "recommendation": "Do not consume. Report to food safety authorities.",
  "inference_time_ms": 143.2,
  "model_info": {
    "architecture": "Hybrid ResNet + 3D CNN + PCA",
    "spectral_bands": 30,
    "pca_components": "Variable per region"
  },
  "timestamp": "2024-01-15T10:30:45.123456",
  "filename": "sample.jpg",
  "mode": "model"
}
```

---

## Model Architecture

```
Input (HSI Image)
    ↓
PCA Dimensionality Reduction (per-region, saved in .pkl)
    ↓
ResNet Branch (spatial features)
    ↓
3D CNN Branch (spectral-spatial features)
    ↓
Feature Fusion
    ↓
Dense Classifier → Pure / Adulterated
```

---

## Tech Stack

- **Backend:** Python, Flask
- **ML:** TensorFlow/Keras, scikit-learn (PCA), NumPy
- **Frontend:** Vanilla HTML/CSS/JS (no framework dependency)
- **PDF:** jsPDF (client-side generation)
- **Fonts:** Playfair Display + DM Sans (Google Fonts)

---

## Notes

- If TensorFlow is not installed, the app runs in **simulation mode** using image
  color heuristics. Install TensorFlow for full model inference.
- The PCA `.pkl` file uses a custom serialization format — if unpickling fails,
  the model runs without PCA preprocessing.
- For hyperspectral `.npy` files, the array should be in `(H, W, Bands)` format.
