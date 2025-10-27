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
        'fr': 'french', 'es': 'spanish', 'de': 'german',
        'pt': 'portuguese', 'ru': 'russian'
    }

# === Load OCR Reader ===
@st.cache_resource
def load_easyocr_reader():
    try:
        return easyocr.Reader(['en', 'hi', 'ta', 'te', 'fr', 'es', 'de', 'pt', 'ru'], verbose=False)
    except Exception as e:
        st.error(f"Failed to load EasyOCR: {e}")
        st.stop()

reader = load_easyocr_reader()

# === Helper Functions ===
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF safely."""
    text_output = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_output += page_text
    except Exception:
        st.warning("⚠️ Unable to extract text from this PDF.")
    return text_output.strip()

def ocr_with_easyocr(image_np, confidence_threshold=0.5):
    """Extract text using EasyOCR with confidence filtering."""
    try:
        results = reader.readtext(image_np)
        text = " ".join(t for _, t, prob in results if prob >= confidence_threshold)
        return text
    except Exception:
        return ""

# === Apply Theme ===
def add_critic_lens_theme():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root {
        --bg-primary: #0c0a15;
        --bg-secondary: #161225;
        --card-bg: #1e1a2f;
        --text-primary: #f0f0ff;
        --text-secondary: #b8b4d9;
        --accent-indigo: #6d5dfc;
        --accent-violet: #8a7cfb;
        --accent-cyan: #4cc9f0;
        --border-color: #322c4d;
        --shadow: 0 8px 24px rgba(109, 93, 252, 0.15);
        --code-bg: #252038;
    }
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Segoe UI', Roboto, sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        background: linear-gradient(135deg, #0c0a15 0%, #141022 100%) !important;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
        max-width: 800px;
    }
    h1 {
        font-weight: 700;
        font-size: 2.4rem;
        background: linear-gradient(90deg, #8a7cfb, #4cc9f0);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    h2, h3 { color: var(--text-primary) !important; font-weight: 600; }
    .stMarkdown p, .stCaption { color: var(--text-secondary) !important; font-size: 1.02rem; }
    .result-card {
        background: var(--card-bg);
        border-radius: 20px;
        padding: 1.4rem;
        box-shadow: var(--shadow);
        margin: 1.2rem 0;
        border: 1px solid var(--border-color);
        backdrop-filter: blur(10px);
    }
    .stButton>button {
        background: linear-gradient(90deg, var(--accent-indigo), var(--accent-violet));
        color: white; border: none; border-radius: 18px;
        padding: 0.5rem 1.4rem; font-weight: 600; font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(109, 93, 252, 0.3);
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(109, 93, 252, 0.45);
    }
    .copy-btn, .search-btn {
        background: rgba(109, 93, 252, 0.15) !important;
        color: var(--accent-cyan) !important;
        border: 1px solid rgba(109, 93, 252, 0.3) !important;
        border-radius: 14px !important;
        padding: 0.25rem 0.9rem !important;
        font-size: 0.85rem !important;
        margin: 0.2rem 0.3rem 0.2rem 0 !important;
        transition: all 0.2s ease;
    }
    .language-tag {
        display: inline-block;
        background: rgba(138, 124, 251, 0.2);
        color: var(--accent-violet);
        padding: 0.25rem 0.7rem;
        border-radius: 14px;
        font-size: 0.88rem;
        font-weight: 600;
        margin: 0.4rem 0;
        border: 1px solid rgba(138, 124, 251, 0.3);
    }
    .stCodeBlock {
        background-color: var(--code-bg) !important;
        color: #e0e0ff !important;
        border-radius: 14px !important;
        padding: 1.1rem !important;
        border: 1px solid var(--border-color);
        font-family: 'JetBrains Mono', monospace;
    }
    .app-subtitle {
        font-size: 1.15rem;
        color: var(--text-secondary);
        margin-bottom: 1.8rem;
        max-width: 650px;
        line-height: 1.5;
    }
    .input-section {
        background: var(--card-bg);
        padding: 1.3rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        border: 1px solid var(--border-color);
    }
    </style>
    <script>
    function copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(() => {});
        } else {
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            document.body.removeChild(ta);
        }
    }
    </script>
    """, unsafe_allow_html=True)

add_critic_lens_theme()

# === Header ===
st.title("👁️ Critic Lens")
st.markdown(
    '<p class="app-subtitle">An AI that doesn’t just read text — it <b>understands context</b>, translates meaning, and reveals insight.</p>',
    unsafe_allow_html=True
)

# === Input ===
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.subheader("📤 Upload Content")
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

st.markdown('</div>', unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.image("https://emojicdn.elk.sh/👁️?style=twitter&size=64", width=40)
    st.markdown("### Critic Lens")
    st.caption("v1.0 · Insight Engine")

    supported_codes = ['en','hi','ta','te','fr','es','de','pt','ru']
    lang_options = [(name.capitalize(), code) for code, name in LANGUAGES.items() if code in supported_codes]
    lang_options.sort()
    target_lang = st.selectbox("Translate to", lang_options, format_func=lambda x: x[0], index=0)
    target_lang_code = target_lang[1]

    confidence_threshold = st.slider("OCR Confidence", 0.0, 1.0, 0.5, 0.05)
    enable_tts = st.checkbox("🔊 Text-to-Speech", value=True)
    st.markdown("---")
    st.caption("🧠 AI-powered\n🌐 9 languages\n⚡ Real-time insight")

# === Main Logic ===
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
        if img_for_display is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("🖼️ Visual Context")
            st.image(img_for_display, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.subheader("🔍 Extracted Insight")
        st.code(detected_text, language=None)
        st.markdown("<small>🔤 Source: <b>Auto-detected</b></small>", unsafe_allow_html=True)

        safe_text = detected_text.replace("`", "'").replace("\n", " ")
        st.markdown(f"""
            <button class="copy-btn" onclick="copyToClipboard(`{safe_text}`)">📋 Copy Insight</button>
            <a href="https://www.google.com/search?q={detected_text.replace(' ', '+')}" target="_blank">
                <button class="search-btn">🌐 Context Search</button>
            </a>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # === Translation ===
        with st.spinner(f"🌀 Translating to {target_lang[0]}..."):
            try:
                translated = GoogleTranslator(source='auto', target=target_lang_code).translate(detected_text)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.subheader("💬 Translated Insight")
                st.markdown(f'<span class="language-tag">{target_lang[0]}</span>', unsafe_allow_html=True)
                st.code(translated, language=None)

                if enable_tts:
                    try:
                        tts = gTTS(text=translated, lang=target_lang_code, slow=False)
                        fp = BytesIO()
                        tts.write_to_fp(fp)
                        fp.seek(0)
                        st.audio(fp, format="audio/mp3")
                    except Exception:
                        st.warning("🔊 TTS unavailable for this language.")

                st.download_button("💾 Original", detected_text, "insight.txt")
                st.download_button("💾 Translation", translated, f"insight_{target_lang_code}.txt")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Translation failed: {e}")

        if input_type == "image":
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("👁️ Visual Search (Google Lens)")
            st.markdown("""
            To analyze this image with **Google Lens**:
            1. **Right-click** the image above → **Save image as...**
            2. Go to [lens.google.com](https://lens.google.com)
            3. Click the 📷 icon → **Upload image**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("📭 No textual insight detected. Try a clearer capture.")
else:
    st.markdown('<div class="result-card" style="text-align:center; padding:2rem;">', unsafe_allow_html=True)
    st.image("https://emojicdn.elk.sh/👁️?style=twitter&size=128", width=80)
    st.markdown("### Ready to Critique the Visible World")
    st.markdown("Upload an image, PDF, or snap a photo to begin.")
    st.markdown('</div>', unsafe_allow_html=True)
