import streamlit as st
import easyocr
from googletrans import Translator   # from googletrans-temp (compatible)
from gtts import gTTS
from PIL import Image
import numpy as np
import tempfile
import os

# ----------------------------
# Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="📸 Critic Lens", page_icon="📸", layout="wide")

st.markdown("""
    <h1 style='text-align:center; color:#2E8B57;'>📸 Critic Lens</h1>
    <h3 style='text-align:center;'>Smart Text Detection • Translation • Voice Output</h3>
    <hr>
""", unsafe_allow_html=True)

# ----------------------------
# Initialize EasyOCR reader safely
# ----------------------------
@st.cache_resource
def load_easyocr_reader():
    # Compatible languages: do not mix complex languages incorrectly
    allowed_langs = ['en', 'hi', 'ta', 'te', 'fr', 'es', 'de', 'pt', 'ru']
    try:
        return easyocr.Reader(allowed_langs, verbose=False)
    except ValueError:
        # Fallback to only English and Hindi if multilingual fails
        return easyocr.Reader(['en', 'hi'], verbose=False)

reader = load_easyocr_reader()

# ----------------------------
# Translator Initialization
# ----------------------------
translator = Translator()

# ----------------------------
# Streamlit Interface
# ----------------------------
uploaded_image = st.file_uploader("📤 Upload an Image", type=["png", "jpg", "jpeg"])

col1, col2 = st.columns(2)

with col1:
    target_lang = st.selectbox(
        "🌐 Choose Target Language",
        ("English", "Hindi", "Tamil", "Telugu", "French", "Spanish", "German", "Portuguese", "Russian"),
        index=0
    )

lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Portuguese": "pt",
    "Russian": "ru"
}

# ----------------------------
# OCR and Translation Logic
# ----------------------------
if uploaded_image is not None:
    img = Image.open(uploaded_image)
    img_array = np.array(img)

    with st.spinner("🔍 Detecting and extracting text..."):
        result = reader.readtext(img_array)
        extracted_text = " ".join([text[1] for text in result])

    if extracted_text.strip():
        st.success("✅ Text Detected Successfully!")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        st.subheader("📝 Extracted Text")
        st.write(extracted_text)

        with st.spinner("🌍 Translating text..."):
            translated_text = translator.translate(extracted_text, dest=lang_map[target_lang]).text

        st.subheader(f"💬 Translated Text ({target_lang})")
        st.write(translated_text)

        # Voice Output
        tts = gTTS(translated_text, lang=lang_map[target_lang])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts.save(tmpfile.name)
            st.audio(tmpfile.name, format="audio/mp3")

        st.success("🔊 Audio Generated Successfully!")

    else:
        st.warning("⚠️ No text detected in the image. Try a clearer image.")

else:
    st.info("⬆️ Please upload an image to start text detection.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("<hr><center>Made with ❤️ using Streamlit, EasyOCR, and Google Translate</center>", unsafe_allow_html=True)

