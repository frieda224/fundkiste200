import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from datetime import date

# ----------------------------
# Dateinamen (alles im ROOT!)
# ----------------------------
MODEL_FILE = "dein_modell.h5"   # ⬅ GENAU so muss deine .h5 heißen
LABEL_FILE = "labels.txt"
DATA_FILE = "data.json"

# ----------------------------
# Modell & Labels laden
# ----------------------------
if not os.path.exists(MODEL_FILE):
    st.error(f"❌ Modell-Datei nicht gefunden: {MODEL_FILE}")
    st.stop()

if not os.path.exists(LABEL_FILE):
    st.error(f"❌ Label-Datei nicht gefunden: {LABEL_FILE}")
    st.stop()

model = tf.keras.models.load_model(MODEL_FILE)

with open(LABEL_FILE, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Debug (sehr wichtig bei Streamlit Cloud)
st.write("📦 Modell geladen:", MODEL_FILE)
st.write("🏷️ Labels:", labels)

# ----------------------------
# Hilfsfunktionen
# ----------------------------
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
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
# TAB 1: Fund erfassen
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
        st.info(f"🤖 KI-Erkennung: **{label}** ({confidence:.2%})")

        if st.button("Fund speichern"):
            data = load_data()

            img_name = f"fund_{len(data)}.jpg"
            image.save(img_name)

            data.append({
                "label": label,
                "confidence": confidence,
                "beschreibung": beschreibung,
                "fundort": fundort,
                "funddatum": str(funddatum),
                "image": img_name
            })

            save_data(data)
            st.success("✅ Fund gespeichert!")

# ============================
# TAB 2: Fund suchen
# ============================
with tab2:
    st.header("Verlorenen Gegenstand suchen")

    suchwort = st.selectbox(
        "Was suchst du?",
        ["Flasche", "Stift", "Brotdose"]
    )

    data = load_data()
    treffer = [d for d in data if d["label"] == suchwort]

    if treffer:
        for d in treffer:
            st.image(d["image"], width=250)
            st.write(f"**Beschreibung:** {d['beschreibung']}")
            st.write(f"**Fundort:** {d['fundort']}")
            st.write(f"**Datum:** {d['funddatum']}")
            st.write(f"**KI-Sicherheit:** {d['confidence']:.2%}")
            st.markdown("---")
    else:
        st.info("❌ Keine passenden Funde gefunden.")
