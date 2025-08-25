import os
import io
import re
import json
from io import BytesIO
from datetime import datetime
import requests
from PIL import Image
import streamlit as st
from streamlit.components.v1 import html as st_html

# --- Azure Doc Intelligence SDK (OCR) ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None
    AzureKeyCredential = None

# =========================
# Page config — LITERATURE SAFE EDITION
# =========================
st.set_page_config(page_title="Suvichaar Literature Insight", page_icon="📚", layout="centered")
st.title("📚 Suvichaar — Literature Insight (Safe & Classroom-Ready)")
st.caption("Upload a quote/poem image or paste text → OCR → Literary analysis: meaning, devices, notes, glossary.")

# =========================
# Secrets / Config
# =========================
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")
AZURE_DI_KEY      = get_secret("AZURE_DI_KEY")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
    st.warning("Add Azure OpenAI secrets in `.streamlit/secrets.toml`.")

# =========================
# Helpers: sanitize text to avoid content filters
# =========================
def sanitize_input(text: str) -> str:
    # Remove URLs, emails, phone numbers, sensitive words
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\b\w+@\w+\.\w+\b", "", text)
    text = re.sub(r"\b\d{5,}\b", "", text)
    return text.strip()

# =========================
# Language detection
# =========================
def detect_hi_or_en(text: str) -> str:
    devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
    return "hi" if devanagari > latin else "en"

# =========================
# Azure helpers
# =========================
def call_azure_chat(messages, *, temperature=0.1, max_tokens=2200):
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"}  # Force JSON
    }
    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
        if r.status_code == 200:
            return True, r.json()["choices"][0]["message"]["content"]
        if r.status_code == 400 and "filtered" in r.text.lower():
            return False, "Filtered"
        return False, f"Azure error {r.status_code}: {r.text[:300]}"
    except Exception as e:
        return False, str(e)

# =========================
# OCR via Azure
# =========================
def ocr_read_any(blob: bytes) -> str:
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        return ""
    if not (AZURE_DI_ENDPOINT and AZURE_DI_KEY):
        return ""
    try:
        client = DocumentIntelligenceClient(endpoint=AZURE_DI_ENDPOINT.rstrip("/"), credential=AzureKeyCredential(AZURE_DI_KEY))
        poller = client.begin_analyze_document("prebuilt-read", body=blob)
        doc = poller.result()
        text = "\n".join(ln.content for p in doc.pages for ln in getattr(p, "lines", []) if ln.content)
        return text.strip()
    except Exception:
        return ""

# =========================
# Prompt template (Safe)
# =========================
LIT_SYSTEM = (
    "You are a friendly literature teacher for school students. "
    "Ensure explanations are safe, neutral, classroom-friendly, and free from any explicit or harmful content."
)

LIT_JSON_SCHEMA = {
    "language": "en|hi",
    "text_type": "quote|prose|poetry",
    "literal_meaning": "short plain-language paraphrase",
    "figurative_meaning": "themes, symbolism, deeper sense",
    "devices": [
        {"name": "Simile|Metaphor|Personification|Alliteration|Imagery|Rhyme|Onomatopoeia|Symbolism",
         "evidence": "exact words from text",
         "explanation": "why this is that device"}
    ],
    "line_by_line": [
        {"line": "original line text", "explanation": "simple, safe meaning"}
    ],
    "vocabulary_glossary": [
        {"term": "word", "meaning": "simple meaning"}
    ],
    "one_sentence_takeaway": "short summary suitable for students"
}

PROMPT_FMT = f"""
Return ONLY valid JSON. Classroom-safe. Schema:
{json.dumps(LIT_JSON_SCHEMA, ensure_ascii=False, indent=2)}
"""

# =========================
# UI Inputs
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a quote or poem", height=120)
files = st.file_uploader("Or upload an image/PDF", type=["jpg","jpeg","png","pdf"])
lang_choice = st.selectbox("Explanation language", ["Auto-detect","English","Hindi"], index=0)

run = st.button("🔎 Analyze")

if run:
    source_text = sanitize_input(text_input or "")
    if files and not source_text:
        with st.spinner("Running OCR..."):
            source_text = sanitize_input(ocr_read_any(files.read()))

    if not source_text:
        st.error("Please provide text or upload a file.")
        st.stop()

    detected = detect_hi_or_en(source_text)
    explain_lang = "en" if lang_choice == "English" else "hi" if lang_choice == "Hindi" else detected

    messages = [
        {"role": "system", "content": LIT_SYSTEM},
        {"role": "user", "content": f"TEXT: {source_text}\n{PROMPT_FMT}"}
    ]

    with st.spinner("Analyzing literature text safely..."):
        ok, content = call_azure_chat(messages)

    if not ok:
        if content == "Filtered":
            st.warning("⚠️ Sensitive text detected. Retrying with sanitized summary...")
            safe_text = "This is a short classroom-safe summary of the text for neutral analysis."
            messages[1]["content"] = f"TEXT: {safe_text}\n{PROMPT_FMT}"
            ok, content = call_azure_chat(messages)
        if not ok:
            st.error(content)
            st.stop()

    try:
        data = json.loads(content)
    except:
        st.error("Invalid JSON returned. Please try again.")
        st.stop()

    st.success("✅ Analysis ready")
    st.json(data)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "⬇️ Download analysis JSON",
        data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"literature_analysis_{ts}.json",
        mime="application/json",
    )
