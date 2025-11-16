import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64


# =========================================================
# LOAD MODEL + TOKENIZER + LABEL ENCODER
# =========================================================
@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model("icd11_classifier_model.keras")

    with open("icd11_tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)

    with open("icd11_label_encoder.pickle", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_model_and_tools()
vocab_size = len(tokenizer.word_index) + 1


# =========================================================
# ICD-11 CHAPTERS (COLORS + CODES)
# =========================================================
ICD_CHAPTERS = {
    'Certain infectious or parasitic diseases': {'code': '01', 'color': '#fee2e2', 'description': 'Diseases caused by infectious agents or parasites'},
    'Neoplasms': {'code': '02', 'color': '#f3e8ff', 'description': 'Cancers, tumors, and abnormal tissue growths'},
    'Diseases of the blood or blood-forming organs': {'code': '03', 'color': '#fce7f3', 'description': 'Blood disorders'},
    'Diseases of the immune system': {'code': '04', 'color': '#e0e7ff', 'description': 'Immune disorders'},
    'Endocrine, nutritional or metabolic diseases': {'code': '05', 'color': '#fef3c7', 'description': 'Hormone, nutritional, and metabolic disorders'},
    'Mental, behavioural or neurodevelopmental disorders': {'code': '06', 'color': '#dbeafe', 'description': 'Mental health conditions'},
    'Sleep-wake disorders': {'code': '07', 'color': '#f1f5f9', 'description': 'Sleep disorders'},
    'Diseases of the nervous system': {'code': '08', 'color': '#cffafe', 'description': 'Brain and nerve disorders'},
    'Diseases of the visual system': {'code': '09', 'color': '#ccfbf1', 'description': 'Eye diseases'},
    'Diseases of the ear or mastoid process': {'code': '10', 'color': '#d1fae5', 'description': 'Ear disorders'},
    'Diseases of the circulatory system': {'code': '11', 'color': '#ffe4e6', 'description': 'Heart & vessel diseases'},
    'Diseases of the respiratory system': {'code': '12', 'color': '#e0f2fe', 'description': 'Lung diseases'},
    'Diseases of the digestive system': {'code': '13', 'color': '#ffedd5', 'description': 'Gastrointestinal diseases'},
    'Diseases of the skin': {'code': '14', 'color': '#fef08a', 'description': 'Skin disorders'},
    'Diseases of the musculoskeletal system or connective tissue': {'code': '15', 'color': '#d9f99d', 'description': 'Bone, joint & muscle diseases'},
    'Diseases of the genitourinary system': {'code': '16', 'color': '#ddd6fe', 'description': 'Urinary & reproductive system disorders'},
    'Conditions related to sexual health': {'code': '17', 'color': '#f5d0fe', 'description': 'Sexual health conditions'},
    'Pregnancy, childbirth or the puerperium': {'code': '18', 'color': '#fbcfe8', 'description': 'Pregnancy conditions'},
    'Certain conditions originating in the perinatal period': {'code': '19', 'color': '#bfdbfe', 'description': 'Newborn conditions'},
    'Developmental anomalies': {'code': '20', 'color': '#a7f3d0', 'description': 'Congenital defects'},
    'Symptoms, signs or clinical findings, not elsewhere classified': {'code': '21', 'color': '#e5e7eb', 'description': 'General symptoms'},
    'Injury, poisoning or certain other consequences of external causes': {'code': '22', 'color': '#fecaca', 'description': 'Injuries & poisoning'}
}


# =========================================================
# TEXT PREPROCESSING
# =========================================================
def preprocess_text(text: str):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\-/]", " ", text)
    text = " ".join(text.split())
    return text


# =========================================================
# PREDICTION FUNCTION
# =========================================================
def predict_category(text, top_n=3):
    processed = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=600, padding="post")

    prediction = model.predict(padded)[0]

    top_idx = np.argsort(prediction)[-top_n:][::-1]

    top_preds = [
        {
            "category": label_encoder.inverse_transform([idx])[0],
            "confidence": float(prediction[idx])
        }
        for idx in top_idx
    ]

    main = top_preds[0]
    return main, top_preds


# =========================================================
# STREAMLIT PAGE UI
# =========================================================
st.set_page_config(
    page_title="ICD-11 Classifier",
    page_icon="🏥",
    layout="wide",
)

# Custom CSS
st.markdown(
    """
    <style>
    .result-card {
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏥 Medical Notes ICD-11 Classifier")
st.write("AI-powered classification of clinical notes into ICD-11 chapters.")


# =========================================================
# INPUT AREA
# =========================================================
st.subheader("📄 Enter or Upload Medical Notes")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
else:
    text = st.text_area(
        "Paste medical notes here:",
        height=220,
        placeholder="E.g., Patient presents with chest pain radiating to the left arm..."
    )

# Example buttons
col1, col2, col3 = st.columns(3)

if col1.button("🫁 Respiratory Example"):
    text = """Patient presents with shortness of breath and productive cough..."""
if col2.button("💓 Circulatory Example"):
    text = """Patient presents with chest pain radiating to the left arm..."""
if col3.button("🍽️ Digestive Example"):
    text = """Patient presents with abdominal pain, nausea, vomiting..."""

if text:
    st.text_area("Medical Notes (Editable)", text, height=220, key="editable_text")


# =========================================================
# CLASSIFY BUTTON
# =========================================================
if st.button("🔍 Classify Notes", type="primary"):
    with st.spinner("Analyzing medical notes..."):
        main, alternatives = predict_category(st.session_state.editable_text)

    category = main["category"]
    confidence = main["confidence"]
    info = ICD_CHAPTERS.get(category, {"color": "#e5e7eb", "code": "??", "description": "Unknown"})

    # MAIN RESULT
    st.markdown(
        f"""
        <div class="result-card" style="background-color:{info['color']};">
            <h2>Primary Classification</h2>
            <h3><strong>{category}</strong> (ICD-11 Chapter {info['code']})</h3>
            <p>{info['description']}</p>
            <h2>Confidence: {confidence*100:.1f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ALTERNATIVES
    st.subheader("🔄 Alternative Classifications")
    for alt in alternatives[1:]:
        alt_info = ICD_CHAPTERS.get(alt["category"], {"code": "??", "description": "Unknown"})

        st.info(
            f"""
            **{alt['category']}**  
            ICD-11 Chapter **{alt_info['code']}**  
            Confidence: **{alt['confidence']*100:.1f}%**
            """
        )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "⚠️ *This tool is for educational and research purposes only. Not for clinical diagnosis.*"
)
