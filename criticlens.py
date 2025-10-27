import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from deep_translator import GoogleTranslator
from io import BytesIO
from gtts import gTTS
from PyPDF2 import PdfReader

# === Translation Language Support ===
try:
    lang_dict = GoogleTranslator().get_supported_languages(as_dict=True)
    LANGUAGES = {code: name.lower() for name, code in lang_dict.items()}
except Exception:
    LANGUAGES = {
        'en': 'english', 'hi': 'hindi', 'ta': 'tamil', 'te': 'telugu',
        'fr': 'french', 'es': 'spanish', 'de': 'german', 'pt': 'portuguese', 'ru': 'russian'
    }

# === Safe EasyOCR Reader Loader ===
@st.cache_resource
def load_easyocr_reader(language_codes):
    """
    Load EasyOCR Reader with compatible language groups.
    Tamil and Telugu only work with English.
    """
    safe_langs = ['en']  # always include English

    if 'ta' in language_codes:
        safe_langs = ['en', 'ta']
    elif 'te' in language_codes:
        safe_langs = ['en', 'te']
    else:
        # Only include globally compatible languages
        safe_langs = [lang for lang in language_codes if lang not in ['ta', 'te']]

    st.write(f"🧠 Loaded EasyOCR languages: {safe_langs}")
    return easyocr.Reader(safe_langs, verbose=False)

# === Sidebar ===
with st.sidebar:
    st.image("https://emojicdn.elk.sh/👁️?style=twitter&size=64", width=40)
    st.markdown("### Critic Lens · v1.0")

    supported_codes = ['en','hi','ta','te','fr','es','de','pt','ru']
    lang_options = [(name.capitalize(), code) for code, name in LANGUAGES.items() if code in supported_codes]
    lang_options.sort()
    target_lang = st.selectbox("Translate to", lang_options, format_func=lambda x: x[0], index=0)
    target_lang_code = target_lang[1]
    confidence_threshold = st.slider("OCR Confidence", 0.0, 1.0, 0.5, 0.05)
    enable_tts = st.checkbox("🔊 Enable Text-to-Speech", value=True)

# === Load EasyOCR Reader (SAFE) ===
reader = load_easyocr_reader(['en'])  # default base reader (safe for all)

# === Helper Functions ===
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)

def ocr_with_easyocr(image_np, confidence_threshold=0.5):
    results = reader.readtext(image_np)
    return " ".join(text for _, text, prob in results if prob >= confidence_threshold)

# === UI Header ===
st.title("👁️ Critic Lens")
st.markdown("#### An AI that reads, translates, and speaks your text.")

# === Input Selection ===
option = st.radio("Choose input type:", ("📸 Image", "📄 PDF", "📷 Camera"), horizontal=True)
uploaded_file = None
input_type = None

if option == "📸 Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    input_type = "image"
elif option == "📄 PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    input_type = "pdf"
else:
    uploaded_file = st.camera_input("Take a photo")
    input_type = "image"

# === Processing ===
if uploaded_file is not None:
    detected_text = ""
    img_for_display = None

    if input_type == "pdf":
        with st.spinner("📄 Reading PDF..."):
            detected_text = extract_text_from_pdf(uploaded_file)
    else:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        img_for_display = img_np.copy()

        with st.spinner("👁️ Extracting text with OCR..."):
            detected_text = ocr_with_easyocr(img_np, confidence_threshold)
            results = reader.readtext(img_np)
            for (bbox, text, prob) in results:
                if prob >= confidence_threshold:
                    tl = tuple(map(int, bbox[0]))
                    br = tuple(map(int, bbox[2]))
                    cv2.rectangle(img_for_display, tl, br, (109, 93, 252), 2)

    if detected_text.strip():
        if img_for_display is not None:
            st.image(img_for_display, use_container_width=True)
        st.subheader("🔍 Extracted Text")
        st.code(detected_text)

        # === Translation ===
        with st.spinner(f"Translating to {target_lang[0]}..."):
            try:
                translated = GoogleTranslator(source='auto', target=target_lang_code).translate(detected_text)
                st.subheader("💬 Translation")
                st.code(translated)
                st.download_button("💾 Download Translation", translated, f"translated_{target_lang_code}.txt")

                if enable_tts:
                    try:
                        tts = gTTS(text=translated, lang=target_lang_code)
                        fp = BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        st.audio(fp, format="audio/mp3")
                    except:
                        st.warning("TTS unavailable for this language.")
            except Exception as e:
                st.error(f"Translation failed: {e}")

    else:
        st.warning("📭 No readable text found.")
else:
    st.info("Upload an image, PDF, or take a photo to start.")

