# 📸 Critic Lens
# Smart Text Detection • Translation • Voice Output

import streamlit as st
import easyocr
from PIL import Image
import numpy as np
from googletrans import Translator
from gtts import gTTS
import tempfile
import os

# -------------------------------
# App Header
# -------------------------------
st.set_page_config(page_title="Critic Lens", page_icon="📸", layout="centered")
st.title("📸 Critic Lens")
st.caption("Smart Text Detection • Translation • Voice Output")

# -------------------------------
# OCR Reader Loader (cached)
# -------------------------------
@st.cache_resource
def load_easyocr_reader():
    # Tamil requires English pairing in this order
    allowed_langs = ['ta', 'en']
    return easyocr.Reader(allowed_langs, verbose=False)

reader = load_easyocr_reader()

# -------------------------------
# Translator Initialization
# -------------------------------
translator = Translator()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Extracting text... Please wait"):
        results = reader.readtext(img_np)

    extracted_text = " ".join([text for (_, text, _) in results])
    st.subheader("📝 Extracted Text")
    st.write(extracted_text if extracted_text else "No readable text found.")

    if extracted_text:
        # -------------------------------
        # Translation Section
        # -------------------------------
        st.markdown("---")
        st.subheader("🌐 Translate Text")

        lang_options = {
            "English": "en",
            "Hindi": "hi",
            "Tamil": "ta",
            "Telugu": "te",
            "Kannada": "kn",
            "Malayalam": "ml",
            "French": "fr",
            "Spanish": "es",
        }
        target_lang = st.selectbox("Select Target Language", list(lang_options.keys()))

        if st.button("Translate"):
            with st.spinner("Translating..."):
                translated = translator.translate(extracted_text, dest=lang_options[target_lang])
                st.success("✅ Translation Complete!")
                st.text_area("Translated Text", translated.text, height=150)

                # -------------------------------
                # Text-to-Speech Section
                # -------------------------------
                st.markdown("---")
                st.subheader("🔊 Listen to Translation")

                tts = gTTS(text=translated.text, lang=lang_options[target_lang])
                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(temp_audio.name)

                st.audio(temp_audio.name, format="audio/mp3")
                os.unlink(temp_audio.name)
else:
    st.info("Please upload an image to start text detection.")

