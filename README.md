# WiFi CSI Gesture Recognition

Passive, device-free gesture detection using **Channel State Information (CSI)** from an ESP32-S3. No camera, no wearable — just ambient WiFi signals passing through the air around you.

The system captures raw CSI amplitudes, extracts statistical and temporal features from sliding windows, trains a machine-learning ensemble, and runs live classification through a browser dashboard.

---

## How it works

```
ESP32-S3 (csi_recorder_v4.ino)
   │  promiscuous-mode CSI over USB serial
   ▼
csiz.py  ←── records labelled windows → CSV
   │
   ▼
cmodel.py ←── trains RF / GB / SVM ensemble → .pkl
   │
   ▼
dash.py  ←── live inference + breathing monitor → browser
```

---

## Hardware required

| Item | Notes |
|------|-------|
| **ESP32-S3** development board | Any variant with PSRAM works |
| USB cable | Data-capable, not charge-only |
| 2.4 GHz WiFi router or AP | Must be nearby; ambient traffic drives CSI |
| PC running Windows / macOS / Linux | For serial capture and Python scripts |

> The ESP32-S3 connects to WiFi and enables **promiscuous mode** — it passively sniffs all nearby packets to trigger CSI callbacks. No dedicated transmitter is needed.

---

## Repository structure

```
├── csi_recorder_v4.ino   # Arduino sketch — flash to the ESP32-S3
├── csiz.py               # Data recorder — captures labelled CSI windows
├── cmodel.py             # Model trainer — outputs raw_csi_gesture_model.pkl
├── dash.py               # Live dashboard — browser UI with real-time inference
├── flowchart.html
├── requirements.txt      # Python dependencies
├── wifi_credentials.h.example
├── LICENSE
├── .gitignore
└── README.md
```
You can look in flowchart.html for a visual overview
---

## Quick start

### 1 — Flash the ESP32

1. Open `csi_recorder_v4.ino` in **Arduino IDE 2** (or PlatformIO).
2. Install the **Arduino ESP32 core** (≥ 3.x) via Boards Manager.
3. Copy `wifi_credentials.h.example` → `wifi_credentials.h` and fill in your WiFi SSID and password.  
   (`wifi_credentials.h` is git-ignored so your credentials stay private.)
4. Select your board (`ESP32S3 Dev Module`) and port, then **Upload**.
5. Open the Serial Monitor at **115200 baud** — you should see lines like:
   ```
   1023,0,64,12.4,11.8,13.1,...
   ```

### 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Python 3.9 or newer is recommended.

### 3 — Record training data

```bash
python csiz.py
```

- Edit `SERIAL_PORT` at the top of `csiz.py` to match your port (e.g. `COM3`, `/dev/ttyUSB0`).
- The recorder supports **three classes**: `still`, `walk`, `wave`.
- Press `1` → **still**, `2` → **walk**, `3` → **wave**, `q` → quit.
- Each session records for `RECORD_SECONDS` (default 300 s) and saves a timestamped CSV.

> **About the three-class vs two-class split:** `csiz.py` records all three gestures, but `cmodel.py` trains on whichever subset you choose. By default `TARGET_CLASSES = ["still", "walk"]`. To train on all three, change it to `["still", "walk", "wave"]` and collect `wave` data first.

### 4 — Train the model

```bash
python cmodel.py
```

- Update `DATA_FILE` at the top of `cmodel.py` with your CSV filename.
- Update `TARGET_CLASSES` to match the gestures you recorded.
- Outputs: `raw_csi_gesture_model.pkl`, `raw_csi_model_report.png`, `csi_shap_summary.png`, `csi_shap_bar.png`.

### 5 — Run the live dashboard

```bash
python dash.py
```

- Edit `PORT` and `MODEL_PKG` at the top of `dash.py` if needed.
- Open **http://127.0.0.1:5000** in your browser.
- The dashboard shows real-time gesture prediction, confidence, breathing rate estimate, subcarrier spectrum, and a scrolling spectrogram.

---

## Configuration reference

### `csiz.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `SERIAL_PORT` | `"COM3"` | Serial port of the ESP32 |
| `BAUD_RATE` | `115200` | Must match the Arduino sketch |
| `WINDOW_SIZE` | `5` | Frames per ML sample |
| `RECORD_SECONDS` | `300` | Duration of each labelled session |
| `NON_OVERLAPPING_WINDOWS` | `True` | No overlap between samples |
| `USE_MEDIAN_FILTER` | `False` | Optional temporal smoothing |

### `cmodel.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_FILE` | `"raw_csi_gesture_data_file.csv"` | CSV from csiz.py |
| `TARGET_CLASSES` | `["still", "walk"]` | Gesture subset to train on |
| `SELECTED_FEATURE_COUNT` | `12` | Features kept after SelectKBest |
| `TEST_SIZE` | `0.25` | Fraction held out for evaluation |

### `dash.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `"COM3"` | Serial port |
| `BAUD` | `115200` | Baud rate |
| `MODEL_PKG` | `"raw_csi_gesture_model.pkl"` | Trained model file |
| `WINDOW_SIZE` | `5` | Must match training value |

---

## Features extracted per window

Each window of `WINDOW_SIZE` frames produces 26 raw features (mean, std, range, energy, temporal change, per-subcarrier statistics, frame correlation, motion signal) plus 7 engineered ratio features. `SelectKBest` then keeps the top 12 for training.

---

## Tips for better accuracy

- Keep the ESP32 and router in **fixed positions** during recording and inference.
- Record at least **150–200 samples per class** for reliable generalisation.
- Perform gestures at a **consistent distance** (1–3 m from the ESP32 works well).
- Avoid moving other objects in the room while recording a class.
- If `WALK` and `STILL` are hard to separate, increase `WINDOW_SIZE` to 7 or 10.

---
## Notes

- serial ports default to COM3 (Windows) and must be changed for Linux/macOS.
- csiz.py / dash.py both hard-code COM3
- DATA_FILE = "raw_csi_gesture_data_YYYYMMDD_HHMMSS.csv" in cmodel.py is hard-coded, rename/update it after recording. 

---
## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
