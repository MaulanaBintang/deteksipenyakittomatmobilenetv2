
import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # ensure Keras 3 uses TF backend if available

import streamlit as st
import numpy as np
from PIL import Image
import pathlib, contextlib, io, sys, traceback

st.set_page_config(page_title="🍅 Tomato Leaf Classifier (MobileNetV2)", page_icon="🍅", layout="wide")
st.title("🍅 Tomato Leaf Classifier")
st.caption("Mendukung model .h5 (tf.keras) dan .keras (Keras 3). App siap untuk Streamlit Community Cloud.")

# ---------------- Utils ----------------
@st.cache_resource(show_spinner=False)
def _import_backends():
    backends = {}
    with contextlib.suppress(Exception):
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras as tfk
        backends["tfk"] = tfk
    with contextlib.suppress(Exception):
        import keras as k3  # Keras 3
        backends["k3"] = k3
    return backends

BACKENDS = _import_backends()
if not BACKENDS:
    st.error("Backend Keras/TensorFlow tidak ditemukan. Tambahkan `tensorflow` atau `keras` di requirements.txt.")
    st.stop()

def _candidate_models():
    places = [pathlib.Path("."), pathlib.Path("./mobilenetv2")]
    exts = ("*.h5", "*.keras")
    found = []
    for p in places:
        for ext in exts:
            for f in p.glob(ext):
                found.append(str(f))
    return sorted(found)

def _extension(path: str) -> str:
    return pathlib.Path(path).suffix.lower()

@st.cache_resource(show_spinner=True)
def load_any_model(path: str):
    last_err = None
    ext = _extension(path)

    # Heuristic: prefer loader by extension first
    order = []
    if ext == ".keras":
        order = ["k3", "tfk"]
    elif ext == ".h5":
        order = ["tfk", "k3"]
    else:
        order = ["tfk", "k3"]

    for key in order:
        if key not in BACKENDS:
            continue
        try:
            if key == "tfk":
                tfk = BACKENDS["tfk"]
                model = tfk.models.load_model(path, compile=False)
                backend = "tf.keras"
                preprocess = getattr(tfk.applications.mobilenet_v2, "preprocess_input", None)
            else:
                k3 = BACKENDS["k3"]
                model = k3.saving.load_model(path)
                backend = "keras3"
                preprocess = None
                with contextlib.suppress(Exception):
                    preprocess = k3.applications.mobilenet_v2.preprocess_input
            # Target size: infer from model input if possible
            try:
                shape = model.inputs[0].shape
                h = int(shape[1]) if shape[1] and shape[1] != -1 else 224
                w = int(shape[2]) if shape[2] and shape[2] != -1 else 224
                target_size = (w, h)
            except Exception:
                target_size = (224, 224)
            return {"model": model, "backend": backend, "preprocess": preprocess, "target_size": target_size}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Gagal memuat model dengan kedua backend. Penyebab terakhir: {repr(last_err)}")

def _safe_preprocess(arr: np.ndarray, preprocess_fn):
    # If model already has a Rescaling layer, both will still work; this is a common MobileNetV2 scheme.
    if preprocess_fn is not None:
        try:
            return preprocess_fn(arr.astype("float32"))
        except Exception:
            pass
    # Fallback to MobileNetV2 scale [-1, 1]
    return (arr.astype("float32") / 127.5) - 1.0

def _format_exception(e: BaseException) -> str:
    tb = traceback.format_exc()
    return f"{e.__class__.__name__}: {e}\n\nTraceback:\n{tb}"

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Pengaturan")
models = _candidate_models()
if not models:
    st.sidebar.warning("Tidak menemukan file model (.h5 / .keras). Taruh di root repo atau folder `mobilenetv2/`.")
model_path = st.sidebar.selectbox("Pilih file model", models, index=0 if models else None, placeholder="Pilih model...")
show_debug = st.sidebar.toggle("Tampilkan debug detail", False)

# ---------------- Main ----------------
st.subheader("1) Upload gambar (JPG/PNG)")
uploaded = st.file_uploader("Pilih gambar daun tomat", type=["jpg","jpeg","png"])

if not model_path:
    st.info("Pilih model terlebih dahulu di sidebar.")
    st.stop()

try:
    bundle = load_any_model(model_path)
    model = bundle["model"]
    backend = bundle["backend"]
    preprocess = bundle["preprocess"]
    target_size = bundle["target_size"]
    st.success(f"Model dimuat ✅ ({backend}) · Input size: {target_size[1]}×{target_size[0]}")
except Exception as e:
    if show_debug:
        st.exception(e)
    else:
        st.error(f"Gagal memuat model: {e}")
    st.stop()

# Model summary (optional)
with st.expander("🧠 Model summary", expanded=False):
    try:
        sio = io.StringIO()
        model.summary(print_fn=lambda x: sio.write(x + "\n"))
        st.code(sio.getvalue(), language="text")
    except Exception as e:
        st.caption(f"(Tidak bisa menampilkan summary: {e})")

st.subheader("2) Prediksi")
if uploaded is None:
    st.info("Silakan upload gambar untuk memulai prediksi.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Gambar diupload", use_column_width=True)

# Resize sesuai target size yang diinferensikan
img_resized = image.resize(target_size)
arr = np.array(img_resized)
arr = _safe_preprocess(arr, preprocess)
arr = np.expand_dims(arr, axis=0)

# Prediksi
try:
    with st.spinner("Menghitung prediksi..."):
        preds = model.predict(arr, verbose=0)
except Exception as e:
    if show_debug:
        st.exception(e)
    else:
        st.error(f"Gagal menjalankan prediksi: {e}")
    st.stop()

# Probabilities
probs = preds[0] if getattr(preds, "ndim", 1) > 1 else preds
probs = np.array(probs).astype("float32")
# Normalize if not in [0,1]
if probs.min() < 0 or probs.max() > 1.0:
    e = np.exp(probs - np.max(probs))
    probs = e / (e.sum() + 1e-9)

# Labels
def load_labels():
    for p in ["class_names.json", "labels.txt", "mobilenetv2/class_names.json", "mobilenetv2/labels.txt"]:
        path = pathlib.Path(p)
        if path.exists():
            try:
                if path.suffix == ".json":
                    import json
                    return json.loads(path.read_text(encoding="utf-8"))
                else:
                    txt = path.read_text(encoding="utf-8")
                    if "," in txt:
                        return [s.strip() for s in txt.split(",") if s.strip()]
                    return [s.strip() for s in txt.splitlines() if s.strip()]
            except Exception:
                pass
    return None

labels = load_labels() or [f"Class {i}" for i in range(len(probs))]

import pandas as pd
df = pd.DataFrame({"label": labels, "prob": probs})
df = df.sort_values("prob", ascending=False).reset_index(drop=True)

st.write("### Hasil Prediksi (Top 5)")
st.dataframe(df.head(5), use_container_width=True)

try:
    import altair as alt
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("label:N", sort="-y"),
        y="prob:Q",
        tooltip=["label","prob"]
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)
except Exception:
    pass

st.success(f"Prediksi utama: **{df.iloc[0]['label']}** (≈ {df.iloc[0]['prob']:.2f})")

st.divider()
with st.expander("🔧 Troubleshooting umum", expanded=False):
    st.markdown(
        """
        **Error `unsupported operand type(s) for %: 'tuple' and 'int'`** biasanya terjadi karena:
        - Bentuk input model tidak sesuai dengan ukuran gambar yang diberikan, **atau**
        - Loader yang dipakai tidak cocok dengan tipe file model (mis. file `.keras` dibuka dengan `tf.keras`).

        Yang sudah diatasi pada app ini:
        - Ukuran input otomatis diinferensikan dari `model.inputs[0].shape` (default 224×224 bila tidak diketahui).
        - Pemilihan loader otomatis: `.h5` → **tf.keras** lebih dulu, `.keras` → **Keras 3** lebih dulu.
        - Opsi **Tampilkan debug detail** di sidebar untuk melihat traceback lengkap.
        """
    )
