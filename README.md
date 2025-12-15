## Klasifikasi Gender pada Cosplayer Berdasarkan Citra Wajah Menggunakan Ekstraksi Fitur Multi-scale Block LBP dan Histogram of Oriented Gradients dengan Metode k-Nearest Neighbor

UAS Project for machine learning class using wikimedia cosplay database 

### Prerequisites
- Python 3.8+
- Virtualenv to create an isolated environment

### Install
- **Create & activate venv (Windows PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate
```

- **Install required packages:**
```bash
pip install -r requirements.txt
```

#### Project layout
- `detector/face_detection_yunet_2023mar.onnx`: required ONNX face detector model.
- `features/`: precomputed features, `gender_features.npz`, are stored.
- `main.py`: launches GUI demo.
- `inference.py`: example inference script used by the app.

#### Run the demo GUI
- Start the GUI (launches the gradio demo defined in `gui`):
```powershell
py main.py
```

- Open the website on localhost
```powershell
http://localhost:7860/
```

#### Training
- Load your own dataset into `dataset/` folder and use `train.ipynb` or your own script to extract features and train the model.