import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from deep_translator import GoogleTranslator

st.set_page_config(page_title="ETL NET", page_icon="🌐", layout="centered")
st.title("▫️ ETL NET ▫️")
st.markdown("Upload or capture an image — detect, extract, and translate text instantly!")

# Sidebar
st.sidebar.header("⚙️ Settings")
target_lang = st.sidebar.selectbox("Translate to:", ["en", "hi", "te", "ta", "fr", "es"])
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.05)

# OCR language selection (compatible groups only)
ocr_lang_options = {
    "English": ['en'],
    "English + Hindi": ['en', 'hi'],
    "English + Tamil": ['en', 'ta'],
    "English + Telugu": ['en', 'te']
}
ocr_lang_choice = st.sidebar.selectbox("OCR Language:", list(ocr_lang_options.keys()))
ocr_lang_list = ocr_lang_options[ocr_lang_choice]

# Initialize EasyOCR reader dynamically
reader = easyocr.Reader(ocr_lang_list)

# Image input
option = st.radio("Choose Image Source:", ("Upload Image", "Use Camera"))
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.camera_input("Take a photo")

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    results = reader.readtext(img_np)

    extracted_texts = []
    for (bbox, text, prob) in results:
        if prob >= confidence_threshold:
            (tl, tr, br, bl) = bbox
            tl = tuple(map(int, tl))
            br = tuple(map(int, br))
            cv2.rectangle(img_np, tl, br, (0, 255, 0), 2)
            cv2.putText(img_np, text, (tl[0], tl[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            extracted_texts.append(text)

    st.subheader("🔍 Detected Text Regions")
    st.image(img_np, caption="Detected Text", use_container_width=True)

    if extracted_texts:
        combined_text = "\n".join(extracted_texts)
        st.subheader("📝 Extracted Text")
        st.text_area("Detected Text", combined_text, height=150)

        try:
            translated = GoogleTranslator(source="auto", target=target_lang).translate(combined_text)
            st.subheader(f"🌐 Translated Text ({target_lang})")
            st.text_area("Translation", translated, height=150)

            st.download_button("💾 Download Extracted Text", combined_text, file_name="detected_text.txt")
            st.download_button("💾 Download Translated Text", translated, file_name=f"translated_{target_lang}.txt")

        except Exception as e:
            st.error(f"Translation failed: {e}")
    else:
        st.warning("No text detected — try a clearer image.")


