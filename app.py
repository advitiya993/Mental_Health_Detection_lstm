import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ── Download NLTK resources on first run ─────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download('punkt',      quiet=True)
    nltk.download('stopwords',  quiet=True)
    nltk.download('wordnet',    quiet=True)
    nltk.download('omw-1.4',   quiet=True)
    nltk.download('punkt_tab',  quiet=True)

download_nltk()

# ── Constants (must match training) ─────────────────────────────────────────
MAX_LEN = 150

# ── Load artefacts ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    # Keras LSTM model
    model = load_model("lstm_mental_health_model.h5")

    # Keras tokenizer (saved as JSON)
    with open("tokenizer.json", "r") as f:
        tokenizer = tokenizer_from_json(json.load(f))

    # Label encoder classes (saved as a plain list in JSON)
    with open("label_classes.json", "r") as f:
        label_classes = json.load(f)          # e.g. ["Anxiety", "Depression", ...]

    return model, tokenizer, label_classes

model, tokenizer, LABEL_CLASSES = load_artifacts()

# ── Preprocessing (mirrors the Colab notebook) ──────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# ── Prediction helpers ───────────────────────────────────────────────────────
def predict_mental_health(raw_text: str):
    """Returns (predicted_label, {class: probability}) for raw_text."""
    cleaned = preprocess_text(raw_text)
    seq     = tokenizer.texts_to_sequences([cleaned])
    padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    proba   = model.predict(padded, verbose=0)[0]           # shape (num_classes,)

    pred_idx   = int(np.argmax(proba))
    pred_label = LABEL_CLASSES[pred_idx]
    class_probs = {LABEL_CLASSES[i]: float(proba[i]) for i in range(len(LABEL_CLASSES))}
    return pred_label, class_probs

# ── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Mental Health Detector", page_icon="🧠", layout="wide")
    st.title("🧠 Mental Health Risk Detector")
    st.caption("Powered by an LSTM model trained on Reddit-based mental health data")

    menu   = ["Home", "Monitor", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # ── Home ──────────────────────────────────────────────────────────────────
    if choice == "Home":
        st.subheader("Home — Mental Health Detection")

        with st.form(key="mental_clf_form"):
            raw_text    = st.text_area("Enter text to analyse", height=150)
            submit_text = st.form_submit_button(label="Analyse")

        if submit_text and raw_text.strip():
            prediction, class_probs = predict_mental_health(raw_text)
            confidence = class_probs[prediction]

            col1, col2 = st.columns(2)

            with col1:
                st.success("📝 Input Text")
                st.write(raw_text)

                st.success("🔍 Prediction")
                st.markdown(f"### {prediction}")
                st.metric(label="Confidence", value=f"{confidence:.2%}")

            with col2:
                st.success("📊 Class Probabilities")
                proba_df = (
                    pd.DataFrame.from_dict(class_probs, orient="index", columns=["Probability"])
                    .reset_index()
                    .rename(columns={"index": "Mental_Health"})
                    .sort_values("Probability", ascending=False)
                )

                fig = (
                    alt.Chart(proba_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Mental_Health:N", sort="-y", title="Mental Health Category"),
                        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("Mental_Health:N", legend=None),
                        tooltip=["Mental_Health", alt.Tooltip("Probability:Q", format=".2%")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(fig, use_container_width=True)

        elif submit_text:
            st.warning("Please enter some text before submitting.")

    # ── Monitor ───────────────────────────────────────────────────────────────
    elif choice == "Monitor":
        st.subheader("📈 App Monitor")
        st.info("Model loaded successfully.")
        st.write(f"**Classes detected by model:** {', '.join(LABEL_CLASSES)}")
        st.write(f"**Number of classes:** {len(LABEL_CLASSES)}")
        st.write(f"**Max sequence length:** {MAX_LEN}")

    # ── About ─────────────────────────────────────────────────────────────────
    else:
        st.subheader("ℹ️ About")
        st.markdown("""
This app uses a **Long Short-Term Memory (LSTM)** neural network trained on
the [Reddit-based Mental Health Dataset](https://www.kaggle.com/datasets/maazkareem/sentiment-and-mental-health-dataset-reddit-based).

**Pipeline:**
1. Text is lowercased, cleaned, tokenised, stop-word-filtered and lemmatised (NLTK).
2. The cleaned text is converted to integer sequences via a Keras `Tokenizer`.
3. Sequences are zero-padded to length 150.
4. The LSTM model predicts a probability distribution over mental health categories.

**Files required to run:**
| File | Description |
|------|-------------|
| `lstm_mental_health_model.h5` | Trained LSTM weights |
| `tokenizer.json` | Keras Tokenizer vocabulary |
| `label_classes.json` | Ordered list of class names |
        """)

if __name__ == "__main__":
    main()