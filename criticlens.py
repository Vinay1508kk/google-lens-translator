import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
from deep_translator import GoogleTranslator, GOOGLE_LANGUAGES_TO_CODES
import requests
import uuid
from io import BytesIO
from gtts import gTTS
from PyPDF2 import PdfReader

# Build LANGUAGES dict: code -> name (lowercase)
LANGUAGES = {code: name.lower() for name, code in GOOGLE_LANGUAGES_TO_CODES.items()}

USE_GOOGLE_CLOUD_VISION = False

@st.cache_resource
def load_easyocr_reader():
    return easyocr.Reader(['en', 'hi', 'ta', 'te', 'fr', 'es', 'de', 'pt', 'ru', 'ja', 'ko', 'zh-cn'], verbose=False)

reader = load_easyocr_reader()

# === Helper Functions ===
def upload_to_tmpfiles(image_bytes):
    try:
        files = {'file': (f"{uuid.uuid4().hex}.png", image_bytes, 'image/png')}
        # ✅ Fixed: no extra spaces in URL
        response = requests.post("https://tmpfiles.org/api/v1/upload", files=files, timeout=10)
        if response.status_code == 200:
            url = response.json()['data']['url']
            return url.replace("/dl/", "/")
    except Exception as e:
        st.write(f"Debug upload error: {e}")
    return None

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in pdf_reader.pages)

def ocr_with_easyocr(image_np, confidence_threshold=0.5):
    results = reader.readtext(image_np)
    return " ".join(text for _, text, prob in results if prob >= confidence_threshold)

# === CRITIC LENS: CUSTOM THEME ===
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

    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }

    .stMarkdown p, .stCaption {
        color: var(--text-secondary) !important;
        font-size: 1.02rem;
    }

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
        color: white;
        border: none;
        border-radius: 18px;
        padding: 0.5rem 1.4rem;
        font-weight: 600;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(109, 93, 252, 0.3);
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(109, 93, 252, 0.45);
    }

    .copy-btn, .search-btn, .tts-btn {
        background: rgba(109, 93, 252, 0.15) !important;
        color: var(--accent-cyan) !important;
        border: 1px solid rgba(109, 93, 252, 0.3) !important;
        border-radius: 14px !important;
        padding: 0.25rem 0.9rem !important;
        font-size: 0.85rem !important;
        margin: 0.2rem 0.3rem 0.2rem 0 !important;
        transition: all 0.2s ease;
    }
    .copy-btn:hover, .search-btn:hover {
        background: rgba(109, 93, 252, 0.25) !important;
        transform: scale(1.03);
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

# === Apply Critic Lens Theme ===
add_critic_lens_theme()

# === App Header ===
st.title("👁️ Critic Lens")
st.markdown(
    '<p class="app-subtitle">An AI that doesn’t just read text — it <b>understands context</b>, translates meaning, and reveals insight.</p>',
    unsafe_allow_html=True
)

# === Input Section ===
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
    st.markdown('<div style="padding:1rem 0;">', unsafe_allow_html=True)
    # ✅ Fixed: clean emoji URL
    st.image("https://emojicdn.elk.sh/👁️?style=twitter&size=64", width=40)
    st.markdown("### Critic Lens")
    st.caption("v1.0 · Insight Engine")
    
    lang_options = [(name.capitalize(), code) for code, name in LANGUAGES.items() 
                    if code in ['en','hi','ta','te','fr','es','de','pt','ru','ja','ko','zh-cn']]
    lang_options.sort()
    target_lang = st.selectbox("Translate to", lang_options, format_func=lambda x: x[0], index=0)
    target_lang_code = target_lang[1]
    
    confidence_threshold = st.slider("OCR Confidence", 0.0, 1.0, 0.5, 0.05)
    enable_tts = st.checkbox("🔊 Text-to-Speech", value=True)
    st.markdown("---")
    st.caption("🧠 AI-powered\n🌐 12 languages\n⚡ Real-time insight")
    st.markdown('</div>', unsafe_allow_html=True)

# === Processing ===
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

        # ✅ No language detection — just show "Auto"
        st.markdown("<small>🔤 Source: <b>Auto-detected</b></small>", unsafe_allow_html=True)

        safe_text = (detected_text
                     .replace("\\", "\\\\")
                     .replace("`", "\\`")
                     .replace("\n", "\\n")
                     .replace("'", "\\'")
                     .replace('"', '\\"'))
        st.markdown(f"""
            <button class="copy-btn" onclick="copyToClipboard(`{safe_text}`)">📋 Copy Insight</button>
            <a href="https://www.google.com/search?q={detected_text.replace(' ', '+')}" target="_blank">
                <button class="search-btn">🌐 Context Search</button>
            </a>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ✅ Translate using deep-translator (no detection needed)
        if True:  # Always offer translation
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
                        except Exception as tts_error:
                            st.warning("🔊 TTS unavailable for this language.")

                    safe_trans = (translated
                                  .replace("\\", "\\\\")
                                  .replace("`", "\\`")
                                  .replace("\n", "\\n")
                                  .replace("'", "\\'")
                                  .replace('"', '\\"'))
                    st.markdown(f"""
                        <button class="copy-btn" onclick="copyToClipboard(`{safe_trans}`)">📋 Copy Translation</button>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button("💾 Original", detected_text, "insight.txt", "text/plain")
                    with col2:
                        st.download_button("💾 Translation", translated, f"insight_{target_lang_code}.txt", "text/plain")
                    st.markdown('</div>', unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌ Translation failed: {str(e)}")

        # === Google Lens Search ===
        if input_type == "image":
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.subheader("👁️ Lens Search")
            if st.button("🔍 Analyze with Google Lens"):
                with st.spinner("📤 Sending to visual AI..."):
                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    public_url = upload_to_tmpfiles(buf.getvalue())
                    if public_url:
                        # ✅ Fixed: no extra spaces
                        lens_url = f"https://lens.google.com/uploadbyurl?url={public_url}"
                        st.markdown(f"""
                            <div style="text-align:center; margin:1rem 0;">
                                <a href="{lens_url}" target="_blank">
                                    <button class="stButton" style="
                                        background: linear-gradient(90deg, #6d5dfc, #8a7cfb);
                                        color:white; border:none; border-radius:18px;
                                        padding:0.55rem 1.6rem; font-weight:600;
                                        box-shadow: 0 4px 14px rgba(109, 93, 252, 0.4);
                                    ">
                                        🔍 Open in Google Lens
                                    </button>
                                </a>
                                <p style="font-size:0.82rem; color:var(--text-secondary); margin-top:0.6rem;">
                                    Image expires in ~60 minutes
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Upload failed. Try again.")
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.warning("📭 No textual insight detected. Try a clearer capture.")

else:
    st.markdown('<div class="result-card" style="text-align:center; padding:2rem;">', unsafe_allow_html=True)
    # ✅ Fixed emoji URL
    st.image("https://emojicdn.elk.sh/👁️?style=twitter&size=128", width=80)
    st.markdown("### Ready to Critique the Visible World")
    st.markdown("Upload an image, PDF, or snap a photo to begin.")
    st.markdown('</div>', unsafe_allow_html=True)
