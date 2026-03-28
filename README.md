# SIATA Temperature Anomaly Detection

Deep learning project for detecting temperature anomalies in meteorological stations from the SIATA network (Medellín, Colombia).

---

## Problem Description

The SIATA (Sistema de Alerta Temprana de Medellín y el Valle de Aburrá) network monitors 4 urban meteorological stations at 1-minute resolution. Sensor failures and transmission errors produce temperature anomalies. The `temperatura_dudosa` quality flag provides ground-truth labels (~3.6% anomaly rate across ~1.66M records in 2025).

**ML task:** Binary classification of 30-minute sliding windows as normal or anomalous.

**Stations:**

| Code | Name | Anomaly rate |
|------|------|-------------|
| 68 | Jardin Botanico | ~1.8% |
| 201 | Torre SIATA | ~2.3% |
| 203 | UNAN | ~1.5% |
| 478 | Fiscalia General | 0% |

---

## Dataset

Source: `data/temperatura_estaciones_2025.csv` — 1.66M minutely records from 4 stations, year 2025.

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `codigo` | int | — | Station identifier (68, 201, 203, 478) |
| `estacion_nombre` | str | — | Station name |
| `fecha_hora` | datetime | YYYY-MM-DD HH:MM:SS | Timestamp, 1-minute frequency |
| `h` | float | % | Relative humidity. Valid range: [0, 100] |
| `t` | float | °C | Ambient temperature. Expected range for Medellín (~1500 m.a.s.l.): [5, 45], typical: [15, 30] |
| `pr` | float | hPa | Atmospheric pressure. Expected range: [800, 900] |
| `vv` | float | m/s | Average wind speed. Valid range: [0, 50] |
| `vv_max` | float | m/s | Maximum wind gust. Valid range: [0, 80], must be ≥ `vv` |
| `dv` | float | ° | Average wind direction (0–360, clockwise from North) |
| `dv_max` | float | ° | Wind direction at maximum gust |
| `p` | float | mm | Cumulative precipitation during the minute. Valid: ≥ 0 |
| `calidad` | int | — | SIATA quality index. `1`/`2` = reliable; `153` = doubtful temperature; `154` = doubtful humidity; `155` = doubtful pressure |
| `calidad_dudosa` | bool | — | `True` if any variable has doubtful quality |
| `temperatura_dudosa` | bool | — | `True` if temperature quality is doubtful — used as the **anomaly label** (~3.6% positive rate) |

> Missing values in the original SIATA data were encoded as `-999` and replaced with `NaN` during cleaning.

---

## Experimental Design

### Stages

1. **EDA** — visualize temperature time series per station, anomaly distribution, feature correlations.
2. **Preprocessing** — normalize features with `StandardScaler`, build 30-minute sliding windows (stride=5), stratified train/val/test split (70/15/15).
3. **Training** — each model is trained with `EarlyStopping(patience=5)` on the training set, monitored on validation loss.
4. **Threshold calibration** — the decision threshold is swept from 0.05 to 0.95 on the validation set; the value that maximizes F1 is selected (not fixed at 0.5).
5. **Evaluation** — final metrics are computed on the held-out test set.

### Experiments

| Exp | Model | Training strategy |
|-----|-------|------------------|
| E1 | MLP Baseline | Supervised, all 4 stations |
| E2 | 1D-CNN + Residual Block | Supervised, all 4 stations |
| E3a | CNN — frozen backbone | Pre-train on stations 68+201, freeze backbone, train head on 20% of station 203 |
| E3b | CNN — full fine-tune | Same backbone, unfreeze all layers, fine-tune on 20% of station 203 |
| E3c | CNN — scratch | Same architecture trained from random weights on 20% of station 203 |

### Evaluation Metrics

**Accuracy is not used** — with 96.4% normal records, a model that always predicts "normal" achieves 96.4% accuracy but detects zero anomalies.

| Metric | Description |
|--------|-------------|
| **Precision** | Of all predicted anomalies, how many are real anomalies |
| **Recall** | Of all real anomalies, how many were detected |
| **F1** | Harmonic mean of precision and recall — main ranking metric |
| **AUC-PR** | Area under the precision-recall curve — threshold-independent, preferred for imbalanced datasets |

**Class imbalance** is handled with weighted binary crossentropy: the anomaly class receives a loss multiplier of `n_negative / n_positive ≈ 27×`.

---

## Architecture

### E1 — MLP Baseline
```
Input [30×4] → Flatten → Dense(128) + BatchNorm + Dropout(0.3) → Dense(64) → sigmoid
```

### E2 — 1D-CNN with Residual Block
```
Input [30, 4]
  → Conv1D(64, k=3) → BatchNorm
  → [Conv1D(128, k=3) → BatchNorm] + skip(Conv1D(128, k=1))   ← residual block
  → GlobalAveragePooling1D
  → Dense(64) + Dropout(0.3) → sigmoid
```

### E3 — Transfer Learning
```
Phase 1 (source: stations 68+201):
  Train full CNN backbone + head

Phase 2a (target: station 203, 20% data, backbone frozen):
  backbone.trainable = False → only head trains

Phase 2b (target: station 203, 20% data, full fine-tune):
  backbone.trainable = True  → lower lr (1e-4)
```

---

## Library Structure

```
siata_anomaly/
├── __init__.py       exports all public functions
├── data.py           load_csv, preprocess, make_windows, split_data, compute_class_weight
├── models.py         build_mlp, build_cnn_backbone, attach_head, weighted_binary_crossentropy
├── detector.py       AnomalyDetector (fit_threshold, predict, evaluate)
└── metrics.py        precision_recall_f1, plot_confusion_matrix, plot_training_history, summary_table
```

**SOLID principles:**
- **S** — each file has one responsibility (data, models, detection, evaluation)
- **O** — `attach_head(backbone, trainable=...)` extends behavior without modifying the backbone
- **I** — small, focused function signatures with no unnecessary dependencies
- **D** — `AnomalyDetector` receives a trained model; it does not build or own one

---

## How to Run in Google Colab

### Step 1 — Upload data to Drive
Upload `temperatura_estaciones_2025.csv` to your Google Drive at:
```
MyDrive/data/temperatura_estaciones_2025.csv
```

### Step 2 — Upload the project files
Upload the repo root (or clone it) to your Drive at:
```
MyDrive/TrabajoFinal_DeepL/
```

### Step 3 — Open the notebook
Open `anomaly_detection.ipynb` in Colab.

### Step 4 — Configure paths (first code cell)
Verify these two variables match your Drive layout:
```python
REPO_PATH = '/content/drive/MyDrive/TrabajoFinal_DeepL'
DATA_PATH = '/content/drive/MyDrive/TrabajoFinal_DeepL/data/temperatura_estaciones_2025.csv'
```

### Step 5 — Run all cells
`Runtime → Run all`

Expected total runtime: ~15–25 minutes on Colab GPU.

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone <repo-url>
cd TrabajoFinal_DeepL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook anomaly_detection.ipynb
```

Set `IN_COLAB = False` at the top of the second cell if not in Colab (it is auto-detected).

---

## Results & Conclusions

Results are generated at runtime in Section 6 of the notebook. The comparison table covers Precision, Recall, F1, and AUC-PR for all 5 variants.

Key conclusions documented in the notebook:
- **Weighted loss is essential** — without it, models collapse to always predicting "normal"
- **Threshold calibration** significantly improves recall without sacrificing too much precision
- **Residual block** improves over MLP by capturing local temporal patterns (sudden spikes)
- **Transfer learning** (E3b fine-tuned) is expected to outperform training from scratch (E3c) when only 20% of target station data is available, validating cross-station feature reuse
- **Station 478 (Fiscalia General)** has zero labeled anomalies in 2025 — a candidate for future unsupervised anomaly detection


---

## Dependencies

```
tensorflow>=2.12
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
numpy>=1.23
```
