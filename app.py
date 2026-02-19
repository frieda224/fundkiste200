
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from datetime import date

# ----------------------------
# DATEINAMEN (alles im ROOT)
# ----------------------------
MODEL_FILE = "keras_modell.h5"
LABEL_FILE = "labels.txt"
DATA_FILE = "data.json"

# ----------------------------
# MODELL & LABELS LADEN
# ----------------------------
if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Modell-Datei fehlt: {MODEL_FILE}")
    st.stop()

if not os.path.exists(LABEL_FILE):
    st.error(f"❌ labels.txt fehlt")
    st.stop()

model = tf.keras.models.load_model(MODEL_FILE)

with open(LABEL_FILE, "r") as f:
    labels = [line.strip().lower() for line in f.readlines()]

st.write("📦 Modell geladen:", MODEL_FILE)
st.write("🏷️ Labels:", labels)

# ----------------------------
# HILFSFUNKTIONEN
# ----------------------------
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = int(np.argmax(prediction))
    confidence = float(prediction[0][index])

    return labels[index], confidence

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="📦 Digitales Fundbüro", page_icon="📦")
st.title("📦 Digitales Fundbüro (Schule)")

tab1, tab2 = st.tabs(["📸 Fund erfassen", "🔍 Fund suchen"])

# ============================
# TAB 1 – FUND ERFASSEN
# ============================
with tab1:
    st.header("Gefundenen Gegenstand erfassen")

    quelle = st.radio(
        "Bildquelle auswählen:",
        ["📷 Kamera verwenden", "📁 Bild hochladen"]
    )

    image = None

    if quelle == "📷 Kamera verwenden":
        cam = st.camera_input("Foto aufnehmen")
        if cam:
            image = Image.open(cam)

    if quelle == "📁 Bild hochladen":
        file = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])
        if file:
            image = Image.open(file)

    if image:
        st.image(image, use_column_width=True)

        beschreibung = st.text_input("Kurze Beschreibung")
        fundort = st.text_input("Fundort")
        funddatum = st.date_input("Funddatum", value=date.today())

        label, confidence = predict_image(image)

        st.info(f"🤖 KI erkennt: **{label}** ({confidence:.1%})")

        if st.button("Fund speichern"):
            data = load_data()

            img_name = f"fund_{len(data)}.jpg"
            image.save(img_name)

            data.append({
                "label": label.strip().lower(),
                "confidence": confidence,
                "beschreibung": beschreibung,
                "fundort": fundort,
                "funddatum": str(funddatum),
                "image": img_name
            })

            save_data(data)
            st.success("✅ Fund erfolgreich gespeichert!")

# ============================
# TAB 2 – FUND SUCHEN
# ============================
with tab2:
    st.header("Verlorenen Gegenstand suchen")

    suchwort = st.selectbox(
        "Was suchst du?",
        labels
    )

    data = load_data()

    treffer = [
        d for d in data
        if d["label"].strip().lower() == suchwort.strip().lower()
    ]

    if treffer:
        for d in treffer:
            st.image(d["image"], width=250)
            st.write(f"**Beschreibung:** {d['beschreibung']}")
            st.write(f"**Fundort:** {d['fundort']}")
            st.write(f"**Datum:** {d['funddatum']}")
            st.write(f"**KI-Sicherheit:** {d['confidence']:.1%}")
            st.markdown("---")
    else:
        st.info("❌ Kein passender Fund gefunden.")
