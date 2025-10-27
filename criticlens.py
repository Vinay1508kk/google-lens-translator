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

# === Cached OCR Loader ===
@st.cache_resource
def load_easyocr_reader(selected_langs):
    """
    Load EasyOCR Reader with compatible language combinations only.
    Tamil and Telugu only work with English.
    """
    if any(lang in ['ta', 'te'] for lang in selected_langs):
        # Restrict to English + Tamil/Telugu
        allowed = ['en'] + [lang for lang in selected_langs if lang in ['ta', 'te']]
    else:
        # General multilingual set
        allowed = selected_langs
    return easyocr.Reader(allowed, verbose=False)

# === Sidebar Settings ===
with st.sidebar:
    st.markdown("### Critic Lens Settings")
    supported_codes = ['en','hi','ta','te','fr','es','de','pt','ru']
    lang_options = [(name.capitalize(), code) for code, name in LANGUAGES.items() if code in supported_codes]
    lang_options.sort()
    target_lang = st.selectbox("Translate to", lang_options, format_func=lambda x: x[0], index=0)
    target_lang_code = target_lang[1]
    confidence_threshold = st.slider("OCR Confidence", 0.0, 1.0, 0.5, 0.05)
    enable_tts = st.checkbox("🔊 Text-to-Speech", value=True)

# === Load EasyOCR Reader (now safe) ===
reader = load_easyocr_reader(['en','hi','ta','te','fr','es','de','pt','ru'])

# === Helper Functions ===
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)

def ocr_with_easyocr(image_np, confidence_threshold=0.5):
    results = reader.readtext(image_np)
    return " ".join(text for _, text, prob in results if prob >= confidence_threshold)

# === Streamlit UI ===
st.title("👁️ Critic Lens")
option = st.radio("Choose input type:", ("📸 Image", "📄 PDF", "📷 Camera"), horizontal=True, label_visibility="collapsed")

uploaded_file = None
input_type = None

if option == "📸 Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    input_type = "image"
elif option == "📄 PDF":
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    input_type = "pdf"
else:
    uploaded_file = st.camera_input("Take a photo", label_visibility="collapsed")
    input_type = "image"

# === Main Processing ===
if uploaded_file is not None:
    detected_text = ""
    img_for_display = None

    if input_type == "pdf":
        with st.spinner("📄 Parsing document..."):
            detected_text = extract_text_from_pdf(uploaded_file)
    else:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)
        img_for_display = img_np.copy()
        with st.spinner("👁️ Analyzing visual context..."):
            detected_text = ocr_with_easyocr(img_np, confidence_threshold)
            results = reader.readtext(img_np)
            for (bbox, text, prob) in results:
                if prob >= confidence_threshold:
                    tl = tuple(map(int, bbox[0]))
                    br = tuple(map(int, bbox[2]))
                    cv2.rectangle(img_for_display, tl, br, (109, 93, 252), 2)

    if detected_text.strip():
        st.image(img_for_display, use_container_width=True)
        st.subheader("🔍 Extracted Text")
        st.code(detected_text)
        st.download_button("💾 Save Extracted Text", detected_text, "insight.txt")

        with st.spinner(f"Translating to {target_lang[0]}..."):
            try:
                translated = GoogleTranslator(source='auto', target=target_lang_code).translate(detected_text)
                st.subheader("💬 Translation")
                st.code(translated)
                st.download_button("💾 Save Translation", translated, f"insight_{target_lang_code}.txt")

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
        st.warning("📭 No text detected. Try a clearer image.")

else:
    st.info("Upload an image, PDF, or capture using camera to begin.")

