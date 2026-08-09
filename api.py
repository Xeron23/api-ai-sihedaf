from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from scipy.signal import find_peaks
import torch
import torch.nn as nn
from typing import List

app = FastAPI(title="Sihedaf PPG to HRV Prediction API (PyTorch)")

data = joblib.load("final.joblib")
config = data['config']
mu = data['mu']
sigma = data['sigma']
state_dict = data['state_dict']

LABEL_MAP = {
    0: "Normal (N)",
    1: "Atrial Fibrillation (AFIB)",
    2: "Atrial Flutter (AFL)"
}

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(1, max_len, d_model)
        self.register_buffer('pe', pe) # Will be loaded from state_dict

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class MicroTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=32, nhead=2, num_layers=1, dim_ff=64, num_classes=3):
        super(MicroTransformer, self).__init__()
        # Based on state_dict keys:
        # embedding.weight/bias
        self.embedding = nn.Linear(input_dim, d_model)
        
        # pos_enc.pe
        self.pos_enc = PositionalEncoding(d_model, max_len=50)
        
        # encoder.layers.0...
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_ff,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # fc.weight/bias
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src shape: [batch, seq, 2]
        x = self.embedding(src)
        x = self.pos_enc(x)
        x = self.encoder(x)
        # Average pooling based on sequence length as typical
        x = x.mean(dim=1)
        out = self.fc(x)
        return out

model_dl = MicroTransformer(
    input_dim=config.get('input_dim', 2),
    d_model=config.get('d_model', 32),
    nhead=config.get('nhead', 2),
    num_layers=1, # Since only encoder.layers.0 is in state_dict
    dim_ff=config.get('dim_ff', 64),
    num_classes=config.get('num_classes', 3)
)
model_dl.load_state_dict(state_dict)
model_dl.eval()

class RawPPGPayload(BaseModel):
    raw_ppg: List[float]
    sampling_rate: int = 50

@app.post("/predict")
def predict_raw_ppg(payload: RawPPGPayload):
    ppg_signal = np.array(payload.raw_ppg)
    
    if len(ppg_signal) < payload.sampling_rate * 3:
        raise HTTPException(status_code=400, detail="Data terlalu pendek (Butuh setidaknya 3 detik data)")

    min_distance = int(0.33 * payload.sampling_rate)
    peaks, _ = find_peaks(ppg_signal, distance=min_distance)
    
    if len(peaks) < 3:
        raise HTTPException(status_code=400, detail="Tidak cukup detak jantung (peaks) yang ditemukan untuk dianalisa.")

    jarak_sample = np.diff(peaks)
    ibi_ms = (jarak_sample / payload.sampling_rate) * 1000.0
    
    diff_ibi = np.diff(ibi_ms)
    # Pad diff_ibi to match ibi_ms length
    diff_ibi_padded = np.insert(diff_ibi, 0, 0)
    
    # Sequence of [IBI, Delta]
    seq_data = np.stack([ibi_ms, diff_ibi_padded], axis=1)
    
    # Truncate or pad to max_len = 50 (based on pos_enc.pe torch.Size([1, 50, 32]))
    max_len = 50
    if len(seq_data) > max_len:
        seq_data = seq_data[:max_len]
    elif len(seq_data) < max_len:
        pad_size = max_len - len(seq_data)
        seq_data = np.pad(seq_data, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
    
    # Standardization (Z-score normalization)
    seq_data = (seq_data - mu) / (sigma + 1e-8)
    
    tensor_input = torch.tensor(seq_data, dtype=torch.float32).unsqueeze(0) # [1, 50, 2]
    
    with torch.no_grad():
        logits = model_dl(tensor_input)
        probs = torch.softmax(logits, dim=1).squeeze().numpy()
        pred_class = int(torch.argmax(logits, dim=1).item())
        
    features_dict = {
        "mean_ibi": round(float(np.mean(ibi_ms)), 4),
        "mean_delta": round(float(np.mean(diff_ibi)), 4) if len(diff_ibi) > 0 else 0,
        "beat_count": len(peaks)
    }
        
    return {
        "status": "success",
        "signal_stats": {
            "total_samples": len(ppg_signal),
            "detected_peaks": len(peaks)
        },
        "extracted_features": features_dict,
        "prediction_class": pred_class,
        "prediction_label": LABEL_MAP.get(pred_class, "Unknown"),
        "confidence": {
            "Normal": round(float(probs[0]), 4),
            "AFIB": round(float(probs[1]), 4),
            "AFL": round(float(probs[2]), 4)
        }
    }
