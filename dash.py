import serial
import numpy as np
from flask import Flask, render_template_string, jsonify
import threading
from collections import deque, Counter
import time
import joblib
import os
from scipy.signal import butter, filtfilt
import pandas as pd

# ================================================================
# SETTINGS — edit these before running
# ================================================================
PORT        = "COM3"              # Windows: "COM3"  Linux/Mac: "/dev/ttyUSB0"
BAUD        = 115200              # Must match csi_recorder_v4.ino
MODEL_PKG   = "raw_csi_gesture_model.pkl"   # output of cmodel.py
WINDOW_SIZE = 5                   # Must match WINDOW_SIZE used during recording
# ================================================================

app = Flask(__name__)

# ── Load model once at startup ────────────────────
print("Loading model...")
if not os.path.exists(MODEL_PKG):
    raise FileNotFoundError(
        f"Model file '{MODEL_PKG}' not found. "
        "Make sure it is in the same folder as this script."
    )

model_pkg        = joblib.load(MODEL_PKG)
clf              = model_pkg["model"]
selected_features = model_pkg["selected_features"]
classes          = model_pkg["classes"]
model_name       = model_pkg.get("model_name", "RandomForest")
model_accuracy   = model_pkg.get("accuracy", 0.0)

print(f"  Model      : {model_name}")
print(f"  Classes    : {classes}")
print(f"  Features   : {selected_features}")
print(f"  Accuracy   : {model_accuracy*100:.1f}%")
print("Model loaded OK.")

# ── Buffers ───────────────────────────────────────
MAX_HISTORY   = 512
BREATH_WINDOW = 300

raw_history        = deque(maxlen=MAX_HISTORY)
frame_window       = deque(maxlen=WINDOW_SIZE)   # holds raw amplitude arrays
subcarrier_history = deque(maxlen=128)

shared = {
    "status":          "Connecting...",
    "fps":             0.0,
    "motion":          "WAITING",
    "motion_conf":     0.0,
    "motion_probs":    {c: 0.0 for c in classes},
    "breath_rate":     0.0,
    "breath_wave":     [0.0] * 150,
    "breath_quality":  "WEAK",
    "spectrogram":     [[0.0]*64 for _ in range(80)],
    "spectrum_now":    [0.0] * 64,
    "subcarrier_cnt":  0,
    "model_name":      model_name,
    "model_accuracy":  round(model_accuracy * 100, 1),
}
lock = threading.Lock()

# ── Signal processing ─────────────────────────────
def bandpass(data, lo, hi, fs, order=3):
    nyq  = fs / 2.0
    lo_n = max(0.001, min(lo / nyq, 0.999))
    hi_n = max(0.001, min(hi / nyq, 0.999))
    if lo_n >= hi_n:
        return data
    try:
        b, a = butter(order, [lo_n, hi_n], btype='band')
        if len(data) > max(len(a), len(b)) * 3:
            return filtfilt(b, a, data)
    except Exception:
        pass
    return data

def dominant_freq(signal, fs):
    if len(signal) < 32:
        return 0.0
    w     = np.hanning(len(signal))
    fft   = np.abs(np.fft.rfft(signal * w))
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    mask  = (freqs >= 0.1) & (freqs <= 0.6)
    if not np.any(mask):
        return 0.0
    return float(freqs[mask][np.argmax(fft[mask])])

def safe_zscore(values):
    std = np.std(values)
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - np.mean(values)) / std

# ── Feature extraction — matches training exactly ─
def extract_features(window):
    """
    Extract the same 26 features used during training,
    then select only the ones the model needs.
    window: list of numpy arrays (amplitude vectors)
    """
    if len(window) < WINDOW_SIZE:
        return None

    counts = [len(f) for f in window]
    if len(set(counts)) != 1:
        return None

    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2 or not np.all(np.isfinite(arr)):
        return None

    mean_sub  = np.mean(arr, axis=0)
    std_sub   = np.std(arr,  axis=0)
    max_sub   = np.max(arr,  axis=0)
    min_sub   = np.min(arr,  axis=0)
    range_sub = max_sub - min_sub
    diffs     = np.diff(arr, axis=0)

    if diffs.shape[0] > 0:
        abs_diff = np.abs(diffs)
        mc_sub   = np.mean(abs_diff, axis=0)
        maxc_sub = np.max(abs_diff,  axis=0)
    else:
        mc_sub   = np.zeros(arr.shape[1])
        maxc_sub = np.zeros(arr.shape[1])

    norm = np.asarray([safe_zscore(f) for f in arr], dtype=np.float32)
    nd   = np.diff(norm, axis=0)

    activity = np.ptp(arr, axis=0)
    act_thr  = np.mean(activity)

    corrs = []
    for i in range(len(arr) - 1):
        if np.std(arr[i]) > 1e-8 and np.std(arr[i+1]) > 1e-8:
            c = np.corrcoef(arr[i], arr[i+1])[0, 1]
            if np.isfinite(c):
                corrs.append(c)

    # Motion signal
    motion_sig = np.mean(np.abs(diffs), axis=1) if diffs.shape[0] > 0 else np.zeros(1)

    std_sig = float(np.std(arr))
    tc      = float(np.mean(np.abs(diffs)))

    all_features = {
        "mean_signal":             float(np.mean(arr)),
        "std_signal":              std_sig,
        "range_signal":            float(np.ptp(arr)),
        "energy_signal":           float(np.mean(arr**2)),
        "temporal_change":         tc,
        "temporal_change_max":     float(np.max(np.abs(diffs))),
        "mean_subcarrier_mean":    float(np.mean(mean_sub)),
        "mean_subcarrier_std":     float(np.std(mean_sub)),
        "std_subcarrier_mean":     float(np.mean(std_sub)),
        "std_subcarrier_max":      float(np.max(std_sub)),
        "std_subcarrier_std":      float(np.std(std_sub)),
        "range_subcarrier_mean":   float(np.mean(range_sub)),
        "range_subcarrier_max":    float(np.max(range_sub)),
        "mean_change_mean":        float(np.mean(mc_sub)),
        "mean_change_max":         float(np.max(mc_sub)),
        "max_change_mean":         float(np.mean(maxc_sub)),
        "max_change_max":          float(np.max(maxc_sub)),
        "normalized_change_mean":  float(np.mean(np.abs(nd))),
        "normalized_change_max":   float(np.max(np.abs(nd))),
        "active_ratio":            float(np.max(activity) / (np.mean(activity) + 1e-6)),
        "fraction_active":         float(np.sum(activity > act_thr) / len(activity)),
        "mean_frame_corr":         float(np.mean(corrs)) if corrs else 1.0,
        "motion_mean":             float(np.mean(motion_sig)),
        "motion_std":              float(np.std(motion_sig)),
        "motion_range":            float(np.ptp(motion_sig)),
        "motion_peak":             float(np.max(motion_sig)),
        # engineered features the trainer added
        "temporal_to_std_ratio":   tc / (std_sig + 1e-6),
        "cv":                      std_sig / (float(np.mean(arr)) + 1e-6),
        "norm_range":              float(np.ptp(arr)) / (abs(float(np.mean(arr))) + 1e-6),
        "energy_ratio":            float(np.mean(arr**2)) / (float(np.mean(arr))**2 + 1e-6),
        "skew_proxy":              float(np.mean(arr)) - float(np.median(arr)),
        "upper_dev":               float(np.max(arr)) - float(np.mean(arr)),
        "lower_dev":               float(np.mean(arr)) - float(np.min(arr)),
        "dev_asymmetry":           (float(np.max(arr)) - float(np.mean(arr))) -
                                   (float(np.mean(arr)) - float(np.min(arr))),
    }

    # Select only what the model needs — in the right order
    try:
        row = [all_features[f] for f in selected_features]
        return pd.DataFrame([row], columns=selected_features)
    except KeyError as e:
        print(f"Missing feature: {e}")
        return None

# ── Serial reader thread ──────────────────────────
def parse_line(line):
    try:
        parts = line.strip().split(",")
        if len(parts) < 4:
            return None, None
        ts   = int(parts[0])
        num  = int(parts[2])
        amps = np.array([float(x) for x in parts[3:3+num]], dtype=np.float32)
        if len(amps) == 0 or not np.all(np.isfinite(amps)):
            return None, None
        return ts, amps
    except Exception:
        return None, None

def read_serial():
    frame_times = deque(maxlen=30)
    prediction_buffer = deque(maxlen=5)  # ← add this

    while True:
        try:
            ser = serial.Serial(PORT, BAUD, timeout=1)
            time.sleep(2)
            ser.reset_input_buffer()
            with lock:
                shared["status"] = "Connected — waiting for data"

            while True:
                if not ser.in_waiting:
                    time.sleep(0.005)
                    continue

                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue

                ts, amps = parse_line(line)
                if amps is None:
                    continue

                now = time.time()
                frame_times.append(now)
                fps_est = (
                    (len(frame_times) - 1) /
                    (frame_times[-1] - frame_times[0])
                    if len(frame_times) >= 2 and
                    (frame_times[-1] - frame_times[0]) > 0
                    else 0
                )

                mean_amp = float(np.mean(amps))

                with lock:
                    raw_history.append(mean_amp)
                    frame_window.append(amps)

                    # Subcarrier history for spectrogram
                    sc = np.zeros(64, dtype=np.float32)
                    n  = min(len(amps), 64)
                    sc[:n] = amps[:n]
                    subcarrier_history.append(sc.tolist())

                    arr = np.array(raw_history)
                    fs  = fps_est if fps_est > 0.5 else 1.0

                    # ── ML prediction ─────────────────────
                    motion      = "WAITING"
                    motion_conf = 0.0
                    motion_probs = {c: 0.0 for c in classes}

                    if len(frame_window) >= WINDOW_SIZE:
                        feat = extract_features(list(frame_window))
                        if feat is not None:
                            pred        = clf.predict(feat)[0]
                            probs       = clf.predict_proba(feat)[0]
                            motion_conf = float(np.max(probs))
                    
                            # Only add to buffer if confidence is high enough
                            if motion_conf > 0.70:
                                prediction_buffer.append(pred)
                    
                            # Majority vote from buffer
                            if prediction_buffer:
                                counts = Counter(prediction_buffer)
                                motion = counts.most_common(1)[0][0].upper()
                    
                            motion_probs = {
                                c: round(float(p), 2)
                                for c, p in zip(classes, probs)
                            }


                    # ── Breathing ─────────────────────────
                    breath_wave    = [0.0] * 150
                    breath_rate    = 0.0
                    breath_quality = "WEAK"

                    if len(arr) >= 60:
                        seg      = arr[-min(len(arr), BREATH_WINDOW):]
                        filtered = bandpass(seg, 0.1, 0.5, fs)
                        rng      = np.max(filtered) - np.min(filtered)
                        if rng > 1e-6:
                            norm_b = (filtered - np.min(filtered)) / rng * 2 - 1
                        else:
                            norm_b = filtered * 0
                        idx        = np.linspace(0, len(norm_b)-1, 150).astype(int)
                        breath_wave = norm_b[idx].tolist()
                        df         = dominant_freq(filtered, fs)
                        breath_rate = df * 60.0
                        amp_std     = float(np.std(filtered))
                        breath_quality = (
                            "STRONG"   if amp_std > 0.3 else
                            "MODERATE" if amp_std > 0.1 else
                            "WEAK"
                        )

                    # ── Spectrogram ───────────────────────
                    sg = list(subcarrier_history)[-80:]
                    while len(sg) < 80:
                        sg.insert(0, [0.0]*64)

                    shared.update({
                        "status":         "Live",
                        "fps":            round(fps_est, 1),
                        "motion":         motion,
                        "motion_conf":    round(motion_conf * 100, 1),
                        "motion_probs":   motion_probs,
                        "breath_rate":    round(breath_rate, 1),
                        "breath_wave":    breath_wave,
                        "breath_quality": breath_quality,
                        "spectrogram":    sg,
                        "spectrum_now":   np.log1p(sc).tolist(),
                        "subcarrier_cnt": int(n),
                    })

        except serial.SerialException as e:
            with lock:
                shared["status"] = f"Serial error: {e}"
            time.sleep(3)
        except Exception as e:
            with lock:
                shared["status"] = f"Error: {e}"
            time.sleep(1)

# ── Flask routes ──────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/data")
def data():
    with lock:
        return jsonify(dict(shared))

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>WiFi CSI Monitor</title>
<style>
:root {
  --bg:      #080e1a;
  --surface: #0f1d30;
  --card:    #111e33;
  --border:  #1a3050;
  --accent:  #38bdf8;
  --green:   #34d399;
  --yellow:  #fbbf24;
  --red:     #f87171;
  --purple:  #a78bfa;
  --muted:   #3d5878;
  --text:    #e2eaf4;
  --sub:     #6a8faf;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;padding:18px;min-height:100vh}

/* header */
header{display:flex;align-items:center;gap:12px;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--border)}
.logo{width:36px;height:36px;background:linear-gradient(135deg,#0ea5e9,#7c3aed);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0}
header h1{font-size:18px;font-weight:700}
header .sub{font-size:12px;color:var(--sub)}
.badges{margin-left:auto;display:flex;gap:8px;flex-wrap:wrap}
.badge{padding:3px 10px;border-radius:99px;font-size:11px;font-weight:600;border:1px solid var(--border);color:var(--sub);background:var(--surface)}
.badge.live{border-color:var(--green);color:var(--green)}
.badge.err{border-color:var(--red);color:var(--red)}

/* stat row */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.stat{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 16px;position:relative;overflow:hidden}
.stat::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent);opacity:.5}
.stat.g::after{background:var(--green)}
.stat.y::after{background:var(--yellow)}
.stat.p::after{background:var(--purple)}
.stat-label{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--sub);margin-bottom:6px}
.stat-val{font-size:28px;font-weight:800;line-height:1;color:var(--accent)}
.stat-val.g{color:var(--green)}
.stat-val.y{color:var(--yellow)}
.stat-val.r{color:var(--red)}
.stat-val.p{color:var(--purple)}
.stat-sub{font-size:11px;color:var(--sub);margin-top:4px}

/* confidence bar */
.conf-bar-wrap{margin-top:8px;height:5px;border-radius:3px;background:var(--border);overflow:hidden}
.conf-bar{height:100%;border-radius:3px;background:var(--green);transition:width .3s}

/* prob pills */
.probs{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.prob-pill{font-size:10px;padding:2px 8px;border-radius:99px;background:var(--surface);border:1px solid var(--border);color:var(--sub)}
.prob-pill.top{border-color:var(--green);color:var(--green)}

/* charts */
.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.chart-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px}
.chart-card.full{grid-column:1/-1}
.chart-title{font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:var(--sub);margin-bottom:10px;display:flex;align-items:center;gap:7px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--accent);animation:pulse 2s infinite}
.dot.g{background:var(--green)}
.dot.y{background:var(--yellow)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.2}}
canvas{display:block;width:100%;border-radius:7px}

footer{margin-top:12px;font-size:11px;color:var(--muted);text-align:center}
</style>
</head>
<body>

<header>
  <div class="logo">📡</div>
  <div>
    <h1>WiFi CSI Monitor</h1>
    <div class="sub" id="modelBadge">Loading model...</div>
  </div>
  <div class="badges">
    <div class="badge" id="statusBadge">Connecting</div>
    <div class="badge" id="fpsBadge">0 fps</div>
  </div>
</header>

<div class="stats">
  <!-- Motion -->
  <div class="stat g">
    <div class="stat-label">Motion (ML)</div>
    <div class="stat-val g" id="motion">—</div>
    <div class="stat-sub">Confidence: <span id="conf">0</span>%</div>
    <div class="conf-bar-wrap"><div class="conf-bar" id="confBar" style="width:0%"></div></div>
    <div class="probs" id="probs"></div>
  </div>

  <!-- Breathing rate -->
  <div class="stat y">
    <div class="stat-label">Breathing Rate</div>
    <div class="stat-val y" id="brate">—</div>
    <div class="stat-sub">breaths / minute</div>
  </div>

  <!-- Breath quality -->
  <div class="stat">
    <div class="stat-label">Signal Quality</div>
    <div class="stat-val" id="bquality">—</div>
    <div class="stat-sub">breathing clarity</div>
  </div>

  <!-- Subcarriers -->
  <div class="stat p">
    <div class="stat-label">Subcarriers</div>
    <div class="stat-val p" id="subcnt">—</div>
    <div class="stat-sub">per CSI frame</div>
  </div>
</div>

<div class="charts">
  <div class="chart-card">
    <div class="chart-title"><span class="dot"></span>Breathing Waveform (0.1–0.5 Hz)</div>
    <canvas id="breathCanvas" height="110"></canvas>
  </div>

  <div class="chart-card">
    <div class="chart-title"><span class="dot y"></span>Subcarrier Amplitudes</div>
    <canvas id="specCanvas" height="110"></canvas>
  </div>

  <div class="chart-card full">
    <div class="chart-title"><span class="dot g"></span>Scrolling Spectrogram</div>
    <canvas id="sgCanvas" height="140"></canvas>
  </div>
</div>

<footer id="footer">Waiting for ESP32...</footer>

<script>
function resize(c){ c.width = (c.clientWidth||400)*window.devicePixelRatio; }

function drawBreath(canvas, wave){
  resize(canvas);
  const W=canvas.width, H=canvas.height, ctx=canvas.getContext('2d');
  ctx.fillStyle='#080e1a'; ctx.fillRect(0,0,W,H);
  // zero line
  ctx.strokeStyle='#1a3050'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();
  if(!wave||wave.length<2) return;
  const grad=ctx.createLinearGradient(0,0,W,0);
  grad.addColorStop(0,'rgba(56,189,248,.2)');
  grad.addColorStop(.5,'#38bdf8');
  grad.addColorStop(1,'#818cf8');
  ctx.beginPath(); ctx.strokeStyle=grad; ctx.lineWidth=2.5; ctx.lineJoin='round';
  wave.forEach((v,i)=>{
    const x=(i/(wave.length-1))*W, y=H/2-v*(H*.42);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  });
  ctx.stroke();
  // fill
  ctx.lineTo(W,H/2); ctx.lineTo(0,H/2); ctx.closePath();
  const fill=ctx.createLinearGradient(0,0,0,H);
  fill.addColorStop(0,'rgba(56,189,248,.18)'); fill.addColorStop(1,'rgba(56,189,248,0)');
  ctx.fillStyle=fill; ctx.fill();
}

function drawSpectrum(canvas, spec){
  resize(canvas);
  const W=canvas.width, H=canvas.height, ctx=canvas.getContext('2d');
  ctx.fillStyle='#080e1a'; ctx.fillRect(0,0,W,H);
  if(!spec||!spec.length) return;
  const maxV=Math.max(...spec,.01), barW=W/spec.length;
  spec.forEach((v,i)=>{
    const bH=(v/maxV)*H*.9, hue=190+(i/spec.length)*60;
    ctx.fillStyle=`hsl(${hue},80%,55%)`;
    ctx.fillRect(i*barW, H-bH, barW-1, bH);
  });
}

function drawSpectrogram(canvas, sg){
  resize(canvas);
  const W=canvas.width, H=canvas.height, ctx=canvas.getContext('2d');
  ctx.fillStyle='#080e1a'; ctx.fillRect(0,0,W,H);
  if(!sg||!sg.length||!sg[0].length) return;

  const timeSteps = sg.length;
  const numFreq   = sg[0].length;
  const cW = W / timeSteps;
  const cH = H / numFreq;

  // Find global max with percentile clipping so bright outliers
  // don't wash out the rest of the heatmap
  const flat = sg.flatMap(r => r).filter(v => v > 0);
  flat.sort((a,b) => a-b);
  const maxV = flat.length
    ? flat[Math.floor(flat.length * 0.95)]   // 95th percentile cap
    : 0.01;

  // Custom 5-stop colormap: black → deep blue → cyan → green → yellow → white-hot
  function heatColor(norm){
    const n = Math.min(Math.max(norm, 0), 1);
    let r, g, b;
    if(n < 0.2){
      const t=n/0.2; r=0; g=0; b=Math.floor(t*200);
    } else if(n < 0.4){
      const t=(n-0.2)/0.2; r=0; g=Math.floor(t*220); b=Math.floor(200+t*55);
    } else if(n < 0.6){
      const t=(n-0.4)/0.2; r=0; g=220; b=Math.floor(255-t*255);
    } else if(n < 0.8){
      const t=(n-0.6)/0.2; r=Math.floor(t*255); g=220; b=0;
    } else {
      const t=(n-0.8)/0.2; r=255; g=Math.floor(220-t*220); b=0;
    }
    return `rgb(${r},${g},${b})`;
  }

  // Draw cells
  sg.forEach((row, t) => {
    row.forEach((v, f) => {
      ctx.fillStyle = heatColor(v / maxV);
      ctx.fillRect(
        Math.floor(t * cW),
        Math.floor(H - (f + 1) * cH),
        Math.ceil(cW) + 1,
        Math.ceil(cH) + 1
      );
    });
  });

  // Overlay: subtle vertical grid every 10 time steps
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth   = 1;
  for(let t = 0; t < timeSteps; t += 10){
    ctx.beginPath();
    ctx.moveTo(t * cW, 0);
    ctx.lineTo(t * cW, H);
    ctx.stroke();
  }

  // Colorbar on the right edge
  const barW = Math.max(6, Math.floor(W * 0.012));
  for(let py = 0; py < H; py++){
    ctx.fillStyle = heatColor(1 - py / H);
    ctx.fillRect(W - barW, py, barW, 1);
  }
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = `${Math.max(9, Math.floor(H*0.1))}px monospace`;
  ctx.fillText('hi', W - barW - 18, 10);
  ctx.fillText('lo', W - barW - 14, H - 4);
}

function motionColor(m){
  if(m==='STILL')   return 'var(--green)';
  if(m==='WALK')    return 'var(--red)';
  if(m==='WALKING') return 'var(--red)';
  return 'var(--sub)';
}

const breathCanvas=document.getElementById('breathCanvas');
const specCanvas=document.getElementById('specCanvas');
const sgCanvas=document.getElementById('sgCanvas');

async function update(){
  try{
    const d=await fetch('/data').then(r=>r.json());

    // badges
    const sb=document.getElementById('statusBadge');
    sb.innerText=d.status; sb.className='badge'+(d.status==='Live'?' live':' err');
    document.getElementById('fpsBadge').innerText=d.fps+' fps';
    document.getElementById('modelBadge').innerText=
      `${d.model_name} · ${d.model_accuracy}% accuracy`;

    // motion
    const mel=document.getElementById('motion');
    mel.innerText=d.motion; mel.style.color=motionColor(d.motion);
    document.getElementById('conf').innerText=d.motion_conf;
    document.getElementById('confBar').style.width=d.motion_conf+'%';

    // prob pills
    const pEl=document.getElementById('probs');
    pEl.innerHTML='';
    const probs=d.motion_probs||{};
    const topClass=Object.entries(probs).sort((a,b)=>b[1]-a[1])[0]?.[0];
    Object.entries(probs).forEach(([cls,p])=>{
      const pill=document.createElement('span');
      pill.className='prob-pill'+(cls===topClass?' top':'');
      pill.innerText=`${cls}: ${(p*100).toFixed(0)}%`;
      pEl.appendChild(pill);
    });

    // breathing
    const br=document.getElementById('brate');
    br.innerText=d.breath_rate>0?d.breath_rate:'—';
    const bq=document.getElementById('bquality');
    bq.innerText=d.breath_quality;
    bq.style.color=
      d.breath_quality==='STRONG'  ?'var(--green)':
      d.breath_quality==='MODERATE'?'var(--yellow)':'var(--red)';

    document.getElementById('subcnt').innerText=d.subcarrier_cnt;

    // charts
    drawBreath(breathCanvas, d.breath_wave);
    drawSpectrum(specCanvas, d.spectrum_now);
    drawSpectrogram(sgCanvas, d.spectrogram);

    document.getElementById('footer').innerText=
      `Status: ${d.status}  ·  ${d.subcarrier_cnt} subcarriers  ·  ${d.fps} fps`;

  } catch(e){
    document.getElementById('statusBadge').innerText='Disconnected';
    document.getElementById('statusBadge').className='badge err';
  }
}
setInterval(update, 250);
update();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=read_serial, daemon=True)
    t.start()
    print("=" * 50)
    print("  WiFi CSI Dashboard v2 — ML Edition")
    print(f"  Model : {MODEL_PKG}")
    print(f"  Port  : {PORT}")
    print("  Open  → http://127.0.0.1:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)