
# Deteksi Penyakit Daun Tomat (Streamlit + TensorFlow)

Disesuaikan dengan pendekatan Anda: model `.h5` di root, scaling 1/255, tab Upload & Kamera,
label kelas: Late_blight, Leaf_Mold, Septoria_leaf_spot, healthy.

## Cara Pakai (Lokal)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Letakkan file **model_daun_tomat_mobilenetv2.h5** di **root** repo (sejajar `app.py`).

## Deploy ke Streamlit Community Cloud
1. Push ke GitHub.
2. Saat **Deploy**, buka **Advanced settings** → pilih **Python 3.11**.
3. Pastikan `requirements.txt` tetap seperti repo ini.

## Catatan
- Gunakan `from tensorflow import keras` / `tf.keras` (bukan `import keras` terpisah).
- Jika Anda ingin memindahkan model ke folder `models/`, ubah `MODEL_PATH` di `app.py`.

