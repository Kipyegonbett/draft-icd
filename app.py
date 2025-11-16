import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# Page config
st.set_page_config(
    page_title="ICD-11 Medical Notes Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ICD-11 CHAPTER INFORMATION
# ============================================================================
ICD_CHAPTERS = {
    'Certain infectious or parasitic diseases': {
        'code': '01',
        'description': 'Diseases caused by infectious agents or parasites',
        'color': '#fee2e2'
    },
    'Neoplasms': {
        'code': '02',
        'description': 'Cancers, tumors, and abnormal tissue growths',
        'color': '#f3e8ff'
    },
    'Diseases of the blood or blood-forming organs': {
        'code': '03',
        'description': 'Disorders affecting blood and blood-forming tissues',
        'color': '#fce7f3'
    },
    'Diseases of the immune system': {
        'code': '04',
        'description': 'Immune system disorders and deficiencies',
        'color': '#e0e7ff'
    },
    'Endocrine, nutritional or metabolic diseases': {
        'code': '05',
        'description': 'Hormonal, nutritional, and metabolic disorders',
        'color': '#fef3c7'
    },
    'Mental, behavioural or neurodevelopmental disorders': {
        'code': '06',
        'description': 'Mental health and behavioral conditions',
        'color': '#dbeafe'
    },
    'Sleep-wake disorders': {
        'code': '07',
        'description': 'Sleep disturbances and disorders',
        'color': '#f1f5f9'
    },
    'Diseases of the nervous system': {
        'code': '08',
        'description': 'Neurological conditions affecting the brain and nerves',
        'color': '#cffafe'
    },
    'Diseases of the visual system': {
        'code': '09',
        'description': 'Eye and vision-related disorders',
        'color': '#ccfbf1'
    },
    'Diseases of the ear or mastoid process': {
        'code': '10',
        'description': 'Ear and hearing-related conditions',
        'color': '#d1fae5'
    },
    'Diseases of the circulatory system': {
        'code': '11',
        'description': 'Heart and blood vessel diseases',
        'color': '#ffe4e6'
    },
    'Diseases of the respiratory system': {
        'code': '12',
        'description': 'Lung and breathing-related conditions',
        'color': '#e0f2fe'
    },
    'Diseases of the digestive system': {
        'code': '13',
        'description': 'Gastrointestinal and digestive disorders',
        'color': '#ffedd5'
    },
    'Diseases of the skin': {
        'code': '14',
        'description': 'Skin and subcutaneous tissue conditions',
        'color': '#fef08a'
    },
    'Diseases of the musculoskeletal system or connective tissue': {
        'code': '15',
        'description': 'Bone, joint, and muscle disorders',
        'color': '#d9f99d'
    },
    'Diseases of the genitourinary system': {
        'code': '16',
        'description': 'Urinary and reproductive system disorders',
        'color': '#ddd6fe'
    },
    'Conditions related to sexual health': {
        'code': '17',
        'description': 'Sexual and reproductive health conditions',
        'color': '#f5d0fe'
    },
    'Pregnancy, childbirth or the puerperium': {
        'code': '18',
        'description': 'Conditions related to pregnancy and childbirth',
        'color': '#fbcfe8'
    },
    'Certain conditions originating in the perinatal period': {
        'code': '19',
        'description': 'Newborn and early infant conditions',
        'color': '#bfdbfe'
    },
    'Developmental anomalies': {
        'code': '20',
        'description': 'Congenital malformations and birth defects',
        'color': '#a7f3d0'
    },
    'Symptoms, signs or clinical findings, not elsewhere classified': {
        'code': '21',
        'description': 'General symptoms and clinical findings',
        'color': '#e5e7eb'
    },
    'Injury, poisoning or certain other consequences of external causes': {
        'code': '22',
        'description': 'Injuries, poisoning, and external causes',
        'color': '#fecaca'
    }
}

# ============================================================================
# LOAD MODEL (with caching)
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained ICD-11 model and preprocessors"""
    try:
        # Load model
        model = tf.keras.models.load_model('icd11_classifier_model.keras')
        
        # Load tokenizer
        with open('icd11_tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
            
        # Load label encoder
        with open('icd11_label_encoder.pickle', 'rb') as f:
            label_encoder = pickle.load(f)
            
        st.success("✅ Model loaded successfully!")
        return model, tokenizer, label_encoder
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please make sure these files are in the same directory:")
        st.code("""
        - icd11_classifier_model.keras
        - icd11_tokenizer.pickle  
        - icd11_label_encoder.pickle
        """)
        return None, None, None

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================
def preprocess_text(text):
    """Enhanced preprocessing that preserves medical terms"""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\-/]', ' ', text)
    text = ' '.join(text.split())
    return text

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_category(text, model, tokenizer, label_encoder, show_top_n=3):
    """Predict ICD-11 category from medical notes"""
    if not text.strip():
        return None
    
    # Preprocess
    processed = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=600, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    
    category = label_encoder.inverse_transform([predicted_class])[0]
    
    # Get top N predictions
    top_indices = np.argsort(prediction[0])[-show_top_n:][::-1]
    top_predictions = []
    for idx in top_indices:
        top_predictions.append({
            'category': label_encoder.inverse_transform([idx])[0],
            'confidence': float(prediction[0][idx])
        })
    
    return {
        'category': category,
        'confidence': float(confidence),
        'top_predictions': top_predictions
    }

# ============================================================================
# EXAMPLE NOTES
# ============================================================================
EXAMPLE_NOTES = {
    "Respiratory": "Patient presents with shortness of breath and productive cough. History reveals high fever (39.2°C), productive cough with yellow-green sputum, and difficulty breathing. Past medical history includes history of smoking 20 pack-years. Physical examination shows oxygen saturation 88% on room air, respiratory rate 28 breaths/min, crackles on auscultation, fever 39.2°C, chest X-ray shows infiltrates in right lower lobe. Clinical impression is consistent with Bacterial pneumonia.",
    
    "Circulatory": "Patient presents with chest pain. History reveals chest pain, radiating pain to left arm, jaw pain, shortness of breath, diaphoresis. Past medical history includes history of hypertension, diabetes mellitus. Physical examination shows blood pressure 160/95 mmHg, heart rate 110 bpm, elevated troponin, ST-segment elevation on ECG, cardiac murmur. Clinical impression is consistent with Acute myocardial infarction.",
    
    "Digestive": "Patient presents with abdominal pain. History reveals abdominal pain, nausea, vomiting, loss of appetite, cramping. Past medical history includes recent fatty meal, previous abdominal surgery. Physical examination shows abdominal tenderness, rebound tenderness, McBurney's point tenderness, elevated lipase, distended abdomen. Clinical impression is consistent with Acute appendicitis."
}

# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 3rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        .result-box {
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #667eea;
        }
        .confidence-bar {
            height: 12px;
            background: #e5e7eb;
            border-radius: 6px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #059669);
            transition: width 0.5s ease;
        }
        .disclaimer {
            background: #fef3c7;
            border: 2px solid #fbbf24;
            padding: 1.5rem;
            border-radius: 8px;
            margin-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Notes ICD-11 Classifier</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">AI-Powered Classification into ICD-11 Chapters</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    model, tokenizer, label_encoder = load_model()
    
    if model is None:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Model Information")
        st.markdown(f"**ICD-11 Chapters:** {len(label_encoder.classes_)}")
        st.markdown(f"**Vocabulary Size:** {len(tokenizer.word_index) + 1:,} words")
        st.markdown("**Expected Accuracy:** >95%")
        
        st.markdown("---")
        st.markdown("## 🚀 Quick Examples")
        example_choice = st.selectbox("Try example notes:", ["Select an example..."] + list(EXAMPLE_NOTES.keys()))
        
        st.markdown("---")
        st.markdown("## 📁 File Upload")
        uploaded_file = st.file_uploader("Upload medical notes (TXT):", type=['txt'])
        
        st.markdown("---")
        st.markdown("### 💡 About")
        st.markdown("""
        This AI classifier analyzes medical notes and predicts the appropriate ICD-11 chapter classification.
        
        **Features:**
        - 22 ICD-11 chapters
        - Medical terminology support
        - Confidence scoring
        - Alternative predictions
        """)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📄 Input Medical Notes")
        
        # Handle file upload
        text_input = ""
        if uploaded_file is not None:
            try:
                text_input = uploaded_file.read().decode("utf-8")
                st.success("✅ File uploaded successfully!")
            except:
                st.error("❌ Error reading file. Please upload a valid text file.")
        
        # Handle example selection
        if example_choice != "Select an example...":
            text_input = EXAMPLE_NOTES[example_choice]
            st.info(f"📋 Loaded {example_choice} example notes")
        
        # Text area for input
        medical_notes = st.text_area(
            "Paste medical notes here:",
            value=text_input,
            height=200,
            placeholder="Enter medical notes describing patient symptoms, history, examination findings, and clinical impression...\n\nExample: Patient presents with chest pain, shortness of breath, and diaphoresis. History reveals hypertension and diabetes. Physical examination shows elevated blood pressure and abnormal ECG findings..."
        )
        
        # Classification button
        col1_1, col1_2 = st.columns([1, 4])
        with col1_1:
            classify_btn = st.button("🔍 Classify Notes", type="primary", use_container_width=True)
        with col1_2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
            
        if clear_btn:
            st.rerun()

    with col2:
        st.markdown("## 📈 Model Metrics")
        
        # Metrics cards
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem;">ICD-11 CHAPTERS</div>
                <div style="font-size: 2rem; font-weight: bold;">22</div>
            </div>
            """, unsafe_allow_html=True)
            
        with metric_col2:
            st.markdown("""
            <div class="metric-card">
                <div style="font-size: 0.9rem;">VOCABULARY</div>
                <div style="font-size: 2rem; font-weight: bold;">{:,}</div>
            </div>
            """.format(len(tokenizer.word_index) + 1), unsafe_allow_html=True)
        
        st.markdown("### 📚 ICD-11 Chapters")
        # Show available chapters
        chapters_list = "\n".join([f"- {chapter}" for chapter in label_encoder.classes_])
        with st.expander("View all ICD-11 chapters"):
            st.markdown(chapters_list)

    # Process classification
    if classify_btn and medical_notes.strip():
        with st.spinner("🔄 Analyzing medical notes with AI..."):
            # Add small delay for better UX
            time.sleep(1)
            
            # Make prediction
            result = predict_category(medical_notes, model, tokenizer, label_encoder)
            
        if result:
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Classification Results")
            
            category = result['category']
            confidence = result['confidence']
            top_preds = result['top_predictions']
            
            # Get ICD chapter info
            chapter_info = ICD_CHAPTERS.get(category, {
                'code': '??',
                'description': 'Unknown category',
                'color': '#e5e7eb'
            })
            
            # Confidence assessment
            if confidence > 0.9:
                confidence_level = "VERY HIGH"
                confidence_emoji = "🟢"
            elif confidence > 0.8:
                confidence_level = "HIGH"
                confidence_emoji = "🟡"
            elif confidence > 0.7:
                confidence_level = "MODERATE"
                confidence_emoji = "🟠"
            else:
                confidence_level = "LOW"
                confidence_emoji = "🔴"
            
            # Primary result
            st.markdown(f"""
            <div class="result-box">
                <div style="display: flex; justify-content: between; align-items: start;">
                    <div style="flex: 1;">
                        <div style="color: #6b7280; font-size: 0.9rem; margin-bottom: 0.5rem;">
                            ICD-11 Chapter {chapter_info['code']}
                        </div>
                        <h2 style="margin: 0; color: #1f2937;">{category}</h2>
                        <p style="color: #4b5563; margin: 0.5rem 0;">{chapter_info['description']}</p>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 2.5rem; font-weight: bold; color: #059669;">
                            {confidence*100:.1f}%
                        </div>
                        <div style="color: #6b7280; font-size: 0.9rem;">
                            {confidence_emoji} {confidence_level} CONFIDENCE
                        </div>
                    </div>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Alternative predictions
            st.markdown("### 🔄 Alternative Classifications")
            for i, pred in enumerate(top_preds[1:], 2):
                alt_category = pred['category']
                alt_confidence = pred['confidence']
                alt_info = ICD_CHAPTERS.get(alt_category, {'code': '??', 'description': 'Unknown'})
                
                st.markdown(f"""
                <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 4px solid #9ca3af;">
                    <div style="display: flex; justify-content: between; align-items: center;">
                        <div style="flex: 1;">
                            <div style="color: #6b7280; font-size: 0.8rem;">Chapter {alt_info['code']}</div>
                            <div style="font-weight: 600; color: #1f2937;">{alt_category}</div>
                            <div style="color: #6b7280; font-size: 0.9rem;">{alt_info['description']}</div>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #4b5563;">
                            {alt_confidence*100:.1f}%
                        </div>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {alt_confidence*100}%; background: #9ca3af;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Disclaimer
            st.markdown("""
            <div class="disclaimer">
                <div style="display: flex; align-items: start;">
                    <div style="font-size: 1.5rem; margin-right: 1rem;">⚠️</div>
                    <div>
                        <div style="font-weight: bold; margin-bottom: 0.5rem; color: #92400e;">
                            Medical Disclaimer
                        </div>
                        <div style="color: #78350f;">
                            This is an AI-assisted classification tool for educational and reference purposes. 
                            Always verify results with qualified medical professionals. 
                            Not a substitute for professional medical diagnosis or clinical judgment.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    elif classify_btn and not medical_notes.strip():
        st.error("❌ Please enter medical notes before classifying.")

    # Instructions
    with st.expander("📚 How to Use This Classifier"):
        st.markdown("""
        1. **Upload** a text file with medical notes, OR
        2. **Paste** medical notes directly into the text area, OR  
        3. **Select** an example from the sidebar
        4. Click **"Classify Notes"** to analyze with AI
        5. Review the **ICD-11 chapter classification** with confidence scores
        6. Check **alternative classifications** for differential diagnosis
        
        **Supported Content:**
        - Patient symptoms and complaints
        - Medical history
        - Physical examination findings
        - Laboratory results
        - Clinical impressions
        - Diagnosis descriptions
        """)

if __name__ == "__main__":
    main()
