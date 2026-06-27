import pickle
import re
import nltk
import streamlit as st
import os
import gdown
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


MODEL_PATH = "best_imdb_bilstm.keras"
TOKENIZER_PATH = "tokenizer.pickle"
MAXLEN = 200



def ensure_nltk_data():
    resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]

    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def download_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1YB5LHdseennYWiCxhIbHd-MFjrWRWp9o"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_artifacts():
    ensure_nltk_data()
    download_model()  # Download model if missing

    model = load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file_handle:
        tokenizer = pickle.load(file_handle)

    return model, tokenizer


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^<]+?>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = word_tokenize(text)
    return " ".join(words)


def predict_sentiment(text: str, model, tokenizer):
    processed = preprocess(text)
    sequences = tokenizer.texts_to_sequences([processed])
    x = pad_sequences(sequences, padding="post", maxlen=MAXLEN)
    probability = float(model.predict(x, verbose=0).ravel()[0])
    label = "Positive" if probability >= 0.5 else "Negative"
    confidence = probability if label == "Positive" else 1.0 - probability
    return label, probability, confidence, processed


st.set_page_config(page_title="IMDb Sentiment Analyzer", page_icon="🎬", layout="wide")

st.markdown(
    """
    <style>
        .stApp {
            background: radial-gradient(circle at top, #1d3557 0%, #0b1020 45%, #060816 100%);
            color: #f8f9fb;
        }
        .hero {
            padding: 2rem 2.2rem;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 24px;
            background: rgba(8, 13, 32, 0.76);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.7rem);
            letter-spacing: -0.04em;
        }
        .hero p {
            margin-top: 0.75rem;
            max-width: 70ch;
            color: rgba(248, 249, 251, 0.84);
            font-size: 1.05rem;
        }
        .metric-card {
            border-radius: 20px;
            padding: 1rem 1.1rem;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
        .muted {
            color: rgba(248, 249, 251, 0.70);
            font-size: 0.95rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>IMDb Sentiment Analyzer</h1>
        <p>
            Paste a movie review and the trained BiLSTM model will classify it as positive or negative.
            The app uses the saved Keras model from your notebook and the same tokenizer / preprocessing pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

model, tokenizer = load_artifacts()

col_left, col_right = st.columns([1.35, 0.9], gap="large")

with col_left:
    review_text = st.text_area(
        "Movie review",
        height=280,
        placeholder="Write or paste a review here...",
    )

    predict_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)

with col_right:
    st.markdown(
        """
        <div class="metric-card">
            <div class="muted">Model</div>
            <h3 style="margin:0.35rem 0 0.65rem 0;">best_imdb_bilstm.keras</h3>
            <div class="muted">Tokenizer</div>
            <h3 style="margin:0.35rem 0 0.65rem 0;">tokenizer.pickle</h3>
            <div class="muted">Max sequence length</div>
            <h3 style="margin:0.35rem 0 0;">200 tokens</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

if predict_clicked:
    if not review_text.strip():
        st.warning("Please enter a review before running prediction.")
    else:
        label, probability, confidence, processed_text = predict_sentiment(
            review_text, model, tokenizer
        )

        result_color = "#79f2c0" if label == "Positive" else "#ff9aa2"
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:1rem;">
                <div class="muted">Prediction</div>
                <h2 style="margin:0.35rem 0; color:{result_color};">{label}</h2>
                <p style="margin:0;">Raw probability: {probability:.4f}</p>
                <p style="margin:0.4rem 0 0 0;">Confidence: {confidence:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence, 0.0), 1.0))

        with st.expander("See processed text"):
            st.write(processed_text)
else:
    st.info("Enter a review and click Analyze Sentiment to get a prediction.")
