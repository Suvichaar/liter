import os
import re
import json
from datetime import datetime

import requests
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
# Page config — LITERATURE ONLY EDITION (with content-filter fallback)
# =========================
st.set_page_config(page_title="Suvichaar Literature Insight", page_icon="📚", layout="centered")
st.title("📚 Suvichaar — Literature Insight (Text & Poetry)")
st.caption("Paste text or upload an image → OCR (if image) → Safe, student-friendly literary analysis: literal meaning, figurative sense, devices, line-by-line notes, glossary.")

# =========================
# Secrets / Config
# =========================
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

# Azure OpenAI (for analysis)
AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

# Azure Document Intelligence (OCR)
AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")
AZURE_DI_KEY      = get_secret("AZURE_DI_KEY")

if not (AZURE_API_KEY and AZURE_ENDPOINT and AZURE_DEPLOYMENT):
    st.warning("Add Azure OpenAI secrets in `.streamlit/secrets.toml`: AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT.")

# =========================
# Language helpers
# =========================
def detect_hi_or_en(text: str) -> str:
    if not text:
        return "en"
    devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
    total = devanagari + latin
    if total == 0:
        return "en"
    return "hi" if (devanagari / total) >= 0.25 else "en"

# =========================
# Local safety sanitization (PII/profanity/urls)
# =========================
_BAD_WORDS = {
    # tiny classroom-safe mask list; extend if your OCR source is noisy
    "damn", "hell", "ass", "bastard", "crap",
    "sex", "sexy", "nude", "naked", "porn", "kill", "suicide"
}

def sanitize_locally(text: str) -> str:
    t = text or ""
    # Mask emails, phones, urls
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[email]", t)
    t = re.sub(r"\+?\d[\d\s\-()]{7,}\d", "[phone]", t)
    t = re.sub(r"(https?://|www\.)\S+", "[link]", t)
    # Mask simple profanities/blocked terms (case-insensitive whole words)
    def mask_bad(match):
        w = match.group(0)
        return w[0] + "*"*(max(len(w)-2,1)) + w[-1]
    for w in _BAD_WORDS:
        t = re.sub(rf"\b{re.escape(w)}\b", mask_bad, t, flags=re.IGNORECASE)
    return t.strip()

# =========================
# Azure helpers
# =========================
def call_azure_chat(messages, *, temperature=0.1, max_tokens=2200, force_json=True):
    """Call Azure Chat and return (ok, content, status_code)."""
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if force_json:
        # Ask Azure to return a JSON object (helps avoid prefaces)
        body["response_format"] = {"type": "json_object"}
    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
        if r.status_code != 200:
            return False, r.text[:400], r.status_code
        return True, r.json()["choices"][0]["message"]["content"], r.status_code
    except Exception as e:
        return False, f"Azure request failed: {e}", 0

# =========================
# OCR (images / PDFs)
# =========================
def ocr_read_any(bytes_blob: bytes) -> str:
    """Use Azure DI 'prebuilt-read' to extract text; returns merged text."""
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        return ""
    if not (AZURE_DI_ENDPOINT and AZURE_DI_KEY):
        return ""
    try:
        client = DocumentIntelligenceClient(endpoint=AZURE_DI_ENDPOINT.rstrip("/"), credential=AzureKeyCredential(AZURE_DI_KEY))
        poller = client.begin_analyze_document("prebuilt-read", body=bytes_blob)
        doc = poller.result()
        parts = []
        if getattr(doc, "pages", None):
            for p in doc.pages:
                lines = [ln.content for ln in getattr(p, "lines", []) or [] if getattr(ln, "content", None)]
                page_txt = "\n".join(lines).strip()
                if page_txt:
                    parts.append(page_txt)
        elif getattr(doc, "paragraphs", None):
            parts.append("\n".join(pp.content for pp in doc.paragraphs if getattr(pp, "content", None)))
        else:
            raw = (getattr(doc, "content", "") or "").strip()
            if raw:
                parts.append(raw)
        return "\n".join(parts).strip()
    except Exception:
        return ""

# =========================
# Prompts (safer, classroom-oriented)
# =========================
LIT_SYSTEM_SAFE = (
    "You are a veteran literature teacher. Produce classroom-appropriate analysis for students."
    " If the text contains sensitive or graphic material, generalize or soften it, avoid explicit detail, and"
    " focus strictly on language, imagery, and literary devices."
)

LIT_JSON_SCHEMA = {
    "language": "en|hi",
    "text_type": "quote|prose|poetry",
    "literal_meaning": "plain-language paraphrase",
    "figurative_meaning": "themes, symbolism, deeper sense",
    "speaker_or_voice": "speaker or narrator",
    "tone_mood": "tone words and mood",
    "devices": [
        {"name": "Simile|Metaphor|Personification|Alliteration|Hyperbole|Imagery|Rhyme|Anaphora|Symbolism|Consonance|Assonance|Onomatopoeia|Metonymy|Synecdoche|Pun",
         "evidence": "exact words from text",
         "explanation": "why this is that device"}
    ],
    "word_by_word_defs": [{"word": "...", "meaning": "literal + connotation"}],
    "line_by_line": [{"line": "original line text", "explanation": "one clear sentence"}],
    "cultural_context": "notes if proverb/idiom/symbolic",
    "vocabulary_glossary": [{"term": "...", "meaning": "..."}],
    "misconceptions": ["common misunderstanding(s)"],
    "one_sentence_takeaway": "single-sentence summary"
}

EXAMPLE_HINT = (
    "If the text is like 'Your face is like Moon', identify SIMILE; explain 'face' (literal+figurative) and 'moon'"
    " (literal + cultural connotations like beauty/serenity); keep analysis respectful and age-appropriate."
)

PROMPT_FMT = f"""
Return ONLY a valid JSON object (minified) with these keys. Keep quotes from the text verbatim under 'evidence'.
Schema (English keys, values in the explanation language):
{json.dumps(LIT_JSON_SCHEMA, ensure_ascii=False, indent=2)}

Guidelines:
- Classroom-safe wording. Remove explicit detail if any appears; keep to literary aspects.
- For poetry, fill 'line_by_line' for each line.
- Be concise (3–6 items under 'devices'), and avoid moralizing.
- {EXAMPLE_HINT}
"""

# =========================
# UI — Inputs
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area("Paste a quote or poem (optional)", height=140, placeholder="e.g., Your face is like Moon")
file = st.file_uploader("Or upload an image/PDF containing the text", type=["jpg","jpeg","png","webp","tiff","pdf"], accept_multiple_files=False)
lang_choice = st.selectbox("Target explanation language", ["Auto-detect","English","Hindi"], index=0)

show_devices_table = st.toggle("Show literary devices table", value=True)
show_line_by_line = st.toggle("Show line-by-line explanation (if poetry)", value=True)

run = st.button("🔎 Analyze")

if run:
    # Source text
    source_text = (text_input or "").strip()
    if file and not source_text:
        with st.spinner("Running OCR on uploaded file…"):
            blob = file.read()
            ocr_text = ocr_read_any(blob)
            if ocr_text:
                source_text = ocr_text
                st.success("OCR text extracted:")
                with st.expander("Show OCR text"):
                    st.write(ocr_text[:20000])
            else:
                st.error("OCR returned no text. Try a clearer image or paste the text manually.")
                st.stop()

    if not source_text:
        st.error("Please paste text or upload a file.")
        st.stop()

    # Language
    detected = detect_hi_or_en(source_text)
    explain_lang = "en" if lang_choice == "English" else ("hi" if lang_choice == "Hindi" else detected)
    st.info(f"Explanation language: **{explain_lang}** (detected: {detected})")

    # Build safe messages
    system_msg = LIT_SYSTEM_SAFE + (" Explain primarily in Hindi." if explain_lang.startswith("hi") else " Explain primarily in English.")

    # 1) Primary attempt on locally-sanitized text
    sanitized_text = sanitize_locally(source_text)
    user_msg = (
        f"TEXT TO ANALYZE (verbatim but classroom-safe):\n{sanitized_text}\n\n"
        f"{PROMPT_FMT}\n"
        "Output strictly as minified JSON. No preface, no code fences."
    )

    with st.spinner("Calling Azure OpenAI for literary analysis…"):
        ok, content, code = call_azure_chat(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}],
            temperature=0.1, max_tokens=2200, force_json=True
        )

    # 2) If a policy block (HTTP 400), retry with an educational rewrite first
    if (not ok) and code == 400:
        st.warning("Azure content filter flagged the prompt. Retrying with an educational rewrite…")
        safe_rewrite_prompt = (
            "Rewrite the following into a classroom-appropriate, non-explicit paraphrase that preserves literary devices "
            "and overall meaning. Remove any sensitive detail, PII, slurs, or explicit content. Return ONLY the rewritten text.\n\n"
            f"TEXT:\n{sanitized_text}"
        )
        ok_rewrite, safe_text, _ = call_azure_chat(
            [{"role": "system", "content": "You are a teacher rewriting text for children; keep it respectful and safe."},
             {"role": "user", "content": safe_rewrite_prompt}],
            temperature=0.2, max_tokens=800, force_json=False
        )
        if ok_rewrite:
            # Now analyze the rewritten text
            user_msg2 = (
                f"TEXT TO ANALYZE (rewritten for classroom):\n{safe_text.strip()}\n\n"
                f"{PROMPT_FMT}\n"
                "Output strictly as minified JSON. No preface, no code fences."
            )
            ok, content, code = call_azure_chat(
                [{"role": "system", "content": system_msg},
                 {"role": "user", "content": user_msg2}],
                temperature=0.1, max_tokens=2200, force_json=True
            )

    if not ok:
        st.error(f"Azure request failed or was filtered.\n\nDetails:\nHTTP {code}\n{content}")
        st.stop()

    # Parse JSON
    def robust_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try: return json.loads(m.group(0))
                except Exception: return None
            return None

    data = robust_parse(content) or {}
    if not isinstance(data, dict) or not data:
        st.error("Model did not return valid JSON. Raw reply (truncated):\n" + content[:800])
        st.stop()

    # === Render ===
    st.success("✅ Analysis ready (classroom-safe)")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Literal meaning**")
        st.write(data.get("literal_meaning", "—"))
        st.markdown("**Figurative meaning / themes**")
        st.write(data.get("figurative_meaning", "—"))
        st.markdown("**Speaker/Voice**")
        st.write(data.get("speaker_or_voice", "—"))
    with cols[1]:
        st.markdown("**Tone & Mood**")
        st.write(data.get("tone_mood", "—"))
        st.markdown("**Cultural context**")
        st.write(data.get("cultural_context", "—"))
        st.markdown("**One-sentence takeaway**")
        st.write(data.get("one_sentence_takeaway", "—"))

    wbw = data.get("word_by_word_defs") or []
    if wbw:
        st.markdown("### 🧩 Word-by-word meanings & connotations")
        st.table([{"word": w.get("word",""), "meaning": w.get("meaning","")} for w in wbw])

    if show_devices_table:
        devices = data.get("devices") or []
        st.markdown("### 🎭 Literary devices")
        if devices:
            st.table([
                {"device": d.get("name",""), "evidence": d.get("evidence",""), "why": d.get("explanation","")}
                for d in devices
            ])
        else:
            st.info("No clear devices detected.")

    if show_line_by_line and (data.get("text_type") == "poetry" or data.get("line_by_line")):
        st.markdown("### 📖 Line-by-line explanation")
        for i, item in enumerate(data.get("line_by_line", []), start=1):
            st.markdown(f"**Line {i}:** {item.get('line','')}")
            st.write(item.get("explanation", ""))
            st.divider()

    gl = data.get("vocabulary_glossary") or []
    if gl:
        st.markdown("### 📒 Glossary")
        st.table(gl)

    mc = data.get("misconceptions") or []
    if mc:
        st.markdown("### ⚠️ Misconceptions to avoid")
        st.write("\n".join(f"• {m}" for m in mc))

    with st.expander("🔧 Debug / Raw analysis JSON"):
        st.json(data, expanded=False)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            "⬇️ Download analysis JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"literature_analysis_{ts}.json",
            mime="application/json",
        )
