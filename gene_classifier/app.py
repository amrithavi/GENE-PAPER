import streamlit as st
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gene Type Classifier",
    page_icon="🧬",
    layout="centered"
)

# ─── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #0d0f14;
        color: #e2e8f0;
    }

    .main {
        background-color: #0d0f14;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }

    h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #a3e635;
        letter-spacing: -0.5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: -0.5rem;
        margin-bottom: 2rem;
    }

    .stTextArea textarea, .stTextInput input {
        background-color: #1a1d27 !important;
        border: 1px solid #2d3148 !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #a3e635 !important;
        box-shadow: 0 0 0 2px rgba(163, 230, 53, 0.15) !important;
    }

    label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
    }

    .stButton > button {
        background-color: #a3e635;
        color: #0d0f14;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.95rem;
        border: none;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        width: 100%;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #bef264;
        transform: translateY(-1px);
    }

    .result-box {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2235 100%);
        border: 1px solid #a3e635;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-top: 1.5rem;
        text-align: center;
    }

    .result-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .result-value {
        color: #a3e635;
        font-size: 1.6rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .divider {
        border: none;
        border-top: 1px solid #1e2235;
        margin: 1.5rem 0;
    }

    .error-box {
        background-color: #1f1015;
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        color: #f87171;
        font-size: 0.9rem;
        margin-top: 1rem;
    }

    .info-chip {
        display: inline-block;
        background-color: #1a1d27;
        border: 1px solid #2d3148;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        color: #64748b;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧬 Gene Type Classifier")
st.markdown('<p class="subtitle">Multimodal deep learning — NCBI Gene Database · 10 functional categories</p>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─── Load Model & Artifacts ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = load_model("gene_model.h5")
    with open("dna_tokenizer.pkl", "rb") as f:
        dna_tokenizer = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, dna_tokenizer, tfidf, le

try:
    model, dna_tokenizer, tfidf, le = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

MAX_LEN = 500

# ─── Inputs ────────────────────────────────────────────────────────────────────
sequence = st.text_area(
    "Nucleotide Sequence",
    placeholder="e.g. ATCGGCTAGCTAGCTAGCTAGCTA...",
    height=120,
    help="Raw DNA/RNA nucleotide sequence (A, T, C, G characters)"
)

description = st.text_input(
    "Gene Description",
    placeholder="e.g. regulatory region involved in transcription initiation",
    help="Natural language description from NCBI Gene database"
)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Predict ───────────────────────────────────────────────────────────────────
if st.button("Classify Gene Type"):
    if not sequence.strip() or not description.strip():
        st.markdown('<div class="error-box">Both inputs are required — enter a nucleotide sequence and a gene description.</div>', unsafe_allow_html=True)
    elif not artifacts_loaded:
        st.markdown(f'<div class="error-box">Model files not found. Make sure gene_model.h5, dna_tokenizer.pkl, tfidf.pkl, and label_encoder.pkl are in the same directory.<br><br><code>{load_error}</code></div>', unsafe_allow_html=True)
    else:
        with st.spinner("Classifying..."):
            seq_encoded = dna_tokenizer.texts_to_sequences([sequence])
            seq_padded = pad_sequences(seq_encoded, maxlen=MAX_LEN, padding="post", truncating="post")
            text_encoded = tfidf.transform([description]).toarray()
            pred = model.predict([seq_padded, text_encoded])
            confidence = float(np.max(pred)) * 100
            gene_type = le.inverse_transform([np.argmax(pred)])[0]

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Predicted Gene Type</div>
            <div class="result-value">{gene_type}</div>
            <br>
            <div class="result-label">Confidence</div>
            <div style="color:#e2e8f0; font-size:1.1rem; font-weight:600;">{confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#2d3148; font-size:0.75rem;">
    Multimodal Gene Type Classification · Character-level LSTM + TF-IDF Fusion · NCBI Human Gene Records
</div>
""", unsafe_allow_html=True)