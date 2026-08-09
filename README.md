# API AI SiHEDAF

Microservice berbasis FastAPI dan PyTorch (Micro-Transformer) untuk memprediksi Atrial Fibrillation (AFIB) & Atrial Flutter (AFL) berdasarkan rentang waktu detak jantung (IBI) dari sinyal PPG.

## Cara Menjalankan

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
2. Jalankan server:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```
