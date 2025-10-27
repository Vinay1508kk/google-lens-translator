import streamlit as st
import easyocr
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
from PIL import Image
import numpy as np
import io

# ---------------------- STREAMLIT PAGE CONFIG ----------------------
st.set_page_config(page_title="Critic Lens", page_icon="📸", layout="wide")

# Custom header style
st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>📸 Critic Lens</h1>
    <p style='text-align: center; color: #888;'>Smart Text Detection • Translation • Voice Output</p>
    <hr>
""", unsafe_allow_html=True)

# ---------------------- EASYOCR LOADING ----------------------
@st.cache_resource
def load_easyocr_reader():
    # Restrict to compatible language sets
    allowed_langs = ['en', 'hi', 'ta', 'te']
    return easyocr.Reader(allowed_langs, verbose=False)

reader = load_easyocr_reader()

# ---------------------- IMAGE INPUT SECTION ----------------------
option = st.radio("Choose Input Type:", ["Upload Image", "Use Camera"])

img = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
elif option == "Use Camera":
    captured = st.camera_input("Take a picture")
    if captured:
        try:
            img = Image.open(captured).convert("RGB")
            st.image(img, caption="Captured Image", use_container_width=True)
        except Exception as e:
            st.error(f"Error capturing image: {e}")

# ---------------------- OCR PROCESSING ----------------------
if img is not None:
    st.subheader("🔍 Text Detection and Extraction")

    if st.button("Extract Text"):
        with st.spinner("Reading text... please wait ⏳"):
            try:
                img_array = np.array(img)
                results = reader.readtext(img_array)
                detected_text = "\n".join([res[1] for res in results])

                if detected_text.strip() == "":
                    st.warning("No readable text found in the image.")
                else:
                    st.success("✅ Text extracted successfully!")
                    st.text_area("Extracted Text", detected_text, height=200)

                    # ---------------------- TRANSLATION ----------------------
                    st.subheader("🌐 Translation")
                    target_lang = st.selectbox(
                        "Translate to:",
                        options=["en", "hi", "ta", "te", "fr", "es", "de", "pt", "ru"],
                        index=0
                    )

                    if st.button("Translate Text"):
                        with st.spinner("Translating..."):
                            try:
                                translated = GoogleTranslator(source='auto', target=target_lang).translate(detected_text)
                                st.success("✅ Translation completed!")
                                st.text_area("Translated Text", translated, height=200)

                                # ---------------------- TEXT-TO-SPEECH ----------------------
                                st.subheader("🔊 Listen to Translation")
                                if st.button("Generate Audio"):
                                    tts = gTTS(translated, lang=target_lang if target_lang != "ta" else "en")
                                    audio_fp = BytesIO()
                                    tts.write_to_fp(audio_fp)
                                    audio_fp.seek(0)
                                    st.audio(audio_fp, format="audio/mp3")
                            except Exception as e:
                                st.error(f"Translation Error: {e}")
            except Exception as e:
                st.error(f"OCR Error: {e}")
else:
    st.info("Please upload or capture an image to start.")

# ---------------------- FOOTER ----------------------
st.markdown("""
<hr>
<p style='text-align:center; color:#aaa;'>
Developed by <b>Critic Lens</b> • Powered by EasyOCR + Deep Translator + gTTS
</p>
""", unsafe_allow_html=True)
