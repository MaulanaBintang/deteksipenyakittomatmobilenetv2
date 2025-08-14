
import streamlit as st
import numpy as np
from PIL import Image
import os, json, io, pathlib, contextlib

st.set_page_config(page_title="MobileNetV2 Tomato Leaf Classifier", page_icon="🍅", layout="wide")

st.title("🍅 MobileNetV2 Tomato Leaf Classifier")
st.caption("App Streamlit siap jalan di Streamlit Community Cloud. Upload gambar daun tomat untuk diprediksi.")

# ---------- Utils ----------
@st.cache_resource(show_spinner=False)
def _import_backends():
    """Try tf.keras first, then keras 3.x fallback."""
    backends = {}
    err = {}
    with contextlib.suppress(Exception):
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras as tfk
        backends["tfk"] = tfk
    if "tfk" not in backends:
        with contextlib.suppress(Exception):
            import keras as k3  # Keras 3 standalone
            backends["k3"] = k3
    return backends

BACKENDS = _import_backends()
if not BACKENDS:
    st.error("Tidak menemukan backend Keras/TensorFlow di environment. Tambahkan `tensorflow` atau `keras` di requirements.txt.")
    st.stop()

def _list_candidate_models():
    names = [
        "mymodel.keras",
        "model_daun_tomat_mobilenetv2.h5",
        "model.h5",
        "model.keras",
    ]
    places = [pathlib.Path("."), pathlib.Path("./mobilenetv2")]
    found = []
    for p in places:
        for n in names:
            f = p / n
            if f.exists():
                found.append(str(f))
    # Also scan for any .h5/.keras in those places
    for p in places:
        for ext in ("*.h5", "*.keras"):
            for f in p.glob(ext):
                s = str(f)
                if s not in found:
                    found.append(s)
    return found

@st.cache_resource(show_spinner=True)
def load_model(model_path: str):
    last_err = None
    # Try tf.keras first
    if "tfk" in BACKENDS:
        try:
            tfk = BACKENDS["tfk"]
            model = tfk.models.load_model(model_path, compile=False)
            preprocess = tfk.applications.mobilenet_v2.preprocess_input
            target_size = (224, 224)
            return {"model": model, "preprocess": preprocess, "target_size": target_size, "backend": "tf.keras"}
        except Exception as e:
            last_err = e
    # Fallback to keras 3
    if "k3" in BACKENDS:
        try:
            k3 = BACKENDS["k3"]
            model = k3.saving.load_model(model_path)
            # Try to get MobileNetV2 preprocess if present
            try:
                preprocess = k3.applications.mobilenet_v2.preprocess_input
            except Exception:
                # Generic: scale to [-1,1] like MobileNetV2
                def preprocess(x):
                    return (x.astype("float32") / 127.5) - 1.0
            target_size = (224, 224)
            return {"model": model, "preprocess": preprocess, "target_size": target_size, "backend": "keras3"}
        except Exception as e2:
            last_err = e2
    raise RuntimeError(f"Gagal memuat model: {last_err}")

def load_labels(default_n: int | None = None):
    # Order of preference: class_names.json, labels.txt (comma/line separated), manual input, generic
    candidates = ["class_names.json", "labels.txt", "mobilenetv2/class_names.json", "mobilenetv2/labels.txt"]
    labels = None
    for c in candidates:
        p = pathlib.Path(c)
        if p.exists():
            try:
                if p.suffix == ".json" :
                    labels = json.loads(p.read_text(encoding="utf-8"))
                else:
                    txt = p.read_text(encoding="utf-8")
                    if "," in txt:
                        labels = [s.strip() for s in txt.split(",") if s.strip()]
                    else:
                        labels = [s.strip() for s in txt.splitlines() if s.strip()]
                break
            except Exception:
                pass
    manual = st.sidebar.text_area("Label (opsional, pisahkan dengan koma)", value=",")  # allows user override
    if manual and manual.strip(", \n\t "):
        labels = [s.strip() for s in manual.split(",") if s.strip()]
    if labels is None and default_n:
        labels = [f"Class {i}" for i in range(default_n)]
    return labels

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Pengaturan")
candidates = _list_candidate_models()
if not candidates:
    st.sidebar.warning("Tidak menemukan file model (.h5 / .keras). Taruh file di root repo atau folder `mobilenetv2/`.")
model_path = st.sidebar.selectbox("Pilih file model", candidates, index=0 if candidates else None, placeholder="Pilih model...")

# ---------- Main UI ----------
st.subheader("1) Upload gambar daun tomat (JPG/PNG)")
uploaded = st.file_uploader("Pilih gambar", type=["jpg","jpeg","png"], accept_multiple_files=False)
colp = st.empty()

if model_path:
    try:
        bundle = load_model(model_path)
        model = bundle["model"]
        preprocess = bundle["preprocess"]
        target_size = bundle["target_size"]
        backend = bundle["backend"]
        st.success(f"Model dimuat ✅ ({backend}): {model_path}")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()
else:
    st.info("Pilih model terlebih dahulu di sidebar.")
    st.stop()

# Determine output classes
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
try:
    out = model.predict(dummy, verbose=0)
    n_classes = int(out.shape[-1]) if out.ndim > 1 else 1
except Exception:
    n_classes = None
labels = load_labels(default_n=n_classes)

st.subheader("2) Pratinjau & Prediksi")
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Gambar diupload", use_column_width=True)

    # Preprocess
    img_resized = image.resize(target_size)
    arr = np.array(img_resized).astype("float32")
    arr = preprocess(arr)
    arr = np.expand_dims(arr, axis=0)

    with st.spinner("Menghitung prediksi..."):
        preds = model.predict(arr, verbose=0)
        if preds.ndim == 1:
            probs = preds
        else:
            probs = preds[0]

    # Normalize if necessary
    probs = np.array(probs).astype("float32")
    if probs.min() < 0 or probs.max() > 1.0:
        # attempt softmax
        e = np.exp(probs - np.max(probs))
        probs = e / (e.sum() + 1e-9)

    # Build labels
    if labels is None:
        labels = [f"Class {i}" for i in range(len(probs))]

    # Show top results
    import pandas as pd
    df = pd.DataFrame({"label": labels, "prob": probs})
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    st.write("### Hasil Prediksi (Top 5)")
    st.dataframe(df.head(5), use_container_width=True)

    try:
        import altair as alt
        chart = alt.Chart(df).mark_bar().encode(x=alt.X("label:N", sort="-y"), y="prob:Q", tooltip=["label","prob"]).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        pass

    st.success(f"Prediksi utama: **{df.iloc[0]['label']}** (prob ~ {df.iloc[0]['prob']:.2f})")
else:
    st.info("Silakan upload gambar untuk memulai prediksi.")

st.divider()
with st.expander("📄 Petunjuk Deploy (Ringkas)", expanded=False):
    st.markdown(
        """
        1. Upload file **app.py** dan file model (`.h5` / `.keras`) ke repo GitHub (root atau folder `mobilenetv2/`).  
        2. Tambahkan `requirements.txt` minimal berisi `streamlit` dan salah satu: `tensorflow` **atau** `keras`.  
        3. Di Streamlit Cloud: **New app** → pilih repo & branch → set **Main file** ke `app.py` → **Deploy**.
        4. Jika error backend, coba gunakan hanya salah satu: *TensorFlow 2.x* (untuk `.h5`) atau *Keras 3* (untuk `.keras`).
        """
    )
