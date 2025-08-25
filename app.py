import os
import io
import re
import json
from io import BytesIO
from datetime import datetime
from pathlib import Path

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
# Page config — LITERATURE ONLY EDITION + TEMPLATE FILLER
# =========================
st.set_page_config(page_title="Suvichaar Literature Insight", page_icon="📚", layout="centered")
st.title("📚 Suvichaar — Literature Insight (Text & Poetry)")
st.caption(
    "Paste text or upload an image → OCR (if image) → Literary analysis JSON (devices, line-by-line, glossary) → "
    "Optional: fill Literature AMP template placeholders and download."
)

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
    st.warning("Add Azure OpenAI secrets in `.streamlit/secrets.toml` → AZURE_API_KEY, AZURE_ENDPOINT, AZURE_DEPLOYMENT.")

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
# Azure helpers
# =========================
def call_azure_chat(messages, *, temperature=0.1, max_tokens=2200):
    """Call Azure Chat and return text content."""
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}
    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = requests.post(url, headers=headers, params=params, json=body, timeout=120)
        if r.status_code != 200:
            return False, f"Azure error {r.status_code}: {r.text[:300]}"
        return True, r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return False, f"Azure request failed: {e}"

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
# Literary analysis prompts
# =========================
LIT_SYSTEM = (
    "You are a veteran literature teacher and critic."
    " Analyze input that may be a quote, proverb, stanza, or short poem."
    " Be precise, student-friendly, and avoid over-interpretation."
)

LIT_JSON_SCHEMA = {
    "language": "en|hi",
    "text_type": "quote|prose|poetry",
    "literal_meaning": "short plain-language paraphrase",
    "figurative_meaning": "themes, symbolism, deeper sense",
    "speaker_or_voice": "who is speaking (if clear) or 'narrator'",
    "tone_mood": "tone words and mood",
    "devices": [
        {"name": "Simile|Metaphor|Personification|Alliteration|Hyperbole|Imagery|Rhyme|Anaphora|Symbolism|Consonance|Assonance|Onomatopoeia|Metonymy|Synecdoche|Pun",
         "evidence": "exact words from text",
         "explanation": "why this is that device"}
    ],
    "word_by_word_defs": [
        {"word": "...", "meaning": "literal + connotation"}
    ],
    "line_by_line": [
        {"line": "original line text", "explanation": "one clear sentence"}
    ],
    "cultural_context": "notes if proverb/idiom/symbolic",
    "vocabulary_glossary": [
        {"term": "...", "meaning": "..."}
    ],
    "misconceptions": ["common misunderstanding(s) to avoid"],
    "one_sentence_takeaway": "single-sentence summary a Grade 8 student can grasp",
    "storytitle": "(optional) inferred short title",
    "author": "(optional)"
}

EXAMPLE_USER_HINT = (
    "If the text is like 'Your face is like Moon', identify SIMILE, explain 'face' (literal + figurative),"
    " explain 'moon' (literal + cultural connotations), then give a simple takeaway."
)

PROMPT_FMT = f"""
Return ONLY a valid JSON object with these keys (omit any that don't apply). Keep quotes from the text verbatim under 'evidence'.
Schema (English keys, values in the text language if possible):
{json.dumps(LIT_JSON_SCHEMA, ensure_ascii=False, indent=2)}

Guidelines:
- If it's poetry, fill 'line_by_line' with each line and a short explanation.
- If it's a single-line quote or aphorism, still analyze devices and meanings.
- Be concise but clear. Prefer 3–6 bullet items for 'devices'.
- Avoid moralizing; stick to what the words support.
- {EXAMPLE_USER_HINT}
"""

# =========================
# Template filler helpers (maps analysis → AMP placeholders)
# =========================
def split_lines_for_template(text: str, max_lines: int = 6):
    # keep non-empty trimmed lines only
    raw_lines = [ln.strip() for ln in (text or "").splitlines()]
    lines = [ln for ln in raw_lines if ln]
    return lines[:max_lines]

def build_template_payload(analysis: dict, *, source_text: str, meta_overrides: dict | None = None) -> dict:
    """Create a dict with all placeholders expected by the Literature AMP template."""
    p = {}
    meta_overrides = meta_overrides or {}

    # Meta
    p["storytitle"] = meta_overrides.get("storytitle") or analysis.get("storytitle") or "Literature Notes"
    p["author"] = meta_overrides.get("author") or analysis.get("author") or ""
    p["grade_level"] = meta_overrides.get("grade_level") or ""
    p["text_type"] = analysis.get("text_type") or ("poetry" if "\n" in source_text else "quote")

    # Text lines (max 6)
    lines = split_lines_for_template(source_text, 6)
    for i in range(6):
        key = f"line{i+1}"
        p[key] = lines[i] if i < len(lines) else ""

    # Line-by-line explanations
    lbl = analysis.get("line_by_line") or []
    for i in range(6):
        p[f"line{i+1}_explanation"] = (lbl[i].get("explanation") if i < len(lbl) else "") if isinstance(lbl, list) else ""

    # Literal / Figurative / Theme
    p["literal_meaning"] = analysis.get("literal_meaning", "")
    p["figurative_meaning"] = analysis.get("figurative_meaning", "")
    p["theme"] = analysis.get("figurative_meaning", "")  # fallback
    p["cultural_context"] = analysis.get("cultural_context", "")
    p["one_sentence_takeaway"] = analysis.get("one_sentence_takeaway", "")

    # Devices up to 6
    devs = analysis.get("devices") or []
    for i in range(6):
        d = devs[i] if i < len(devs) else {}
        p[f"device{i+1}_name"] = d.get("name", "") if isinstance(d, dict) else ""
        p[f"device{i+1}_evidence"] = d.get("evidence", "") if isinstance(d, dict) else ""
        p[f"device{i+1}_explanation"] = d.get("explanation", "") if isinstance(d, dict) else ""

    # Vocabulary up to 6
    gl = analysis.get("vocabulary_glossary") or []
    for i in range(6):
        g = gl[i] if i < len(gl) else {}
        p[f"vocab{i+1}_term"] = g.get("term", "") if isinstance(g, dict) else ""
        p[f"vocab{i+1}_meaning"] = g.get("meaning", "") if isinstance(g, dict) else ""

    # Misconceptions up to 3
    mis = analysis.get("misconceptions") or []
    p["mis1"] = mis[0] if len(mis) > 0 else ""
    p["mis2"] = mis[1] if len(mis) > 1 else ""
    p["mis3"] = mis[2] if len(mis) > 2 else ""

    # Media placeholders (left empty; user can fill)
    for i in range(1, 13):
        p.setdefault(f"s{i}image1", "")
        p.setdefault(f"s{i}audio1", "")

    # Other meta commonly used in your stories
    p.setdefault("publisher_name", "Suvichaar Stories")
    p.setdefault("publisher_logo", "")
    p.setdefault("portraitcoverurl", "")
    p.setdefault("canonical_url", "")
    p.setdefault("next_link", "")
    return p

def fill_template_strict(template: str, data: dict):
    """Replace {{key}} (and {{key|safe}}) with str(value). Return filled HTML and set of found placeholders."""
    placeholders = set(re.findall(r"\{\{\s*([a-zA-Z0-9_\-]+)(?:\|safe)?\s*\}\}", template))
    out = template
    for k in placeholders:
        val = str(data.get(k, ""))
        out = out.replace(f"{{{{{k}}}}}", val).replace(f"{{{{{k}|safe}}}}", val)
    return out, placeholders

# =========================
# UI — Inputs
# =========================
st.markdown("### 📥 Input")
text_input = st.text_area(
    "Paste a quote or poem (optional)",
    height=160,
    placeholder="e.g., Your face is like Moon\nI pedal and I ride…",
)
files = st.file_uploader(
    "Or upload an image/PDF containing the text",
    type=["jpg", "jpeg", "png", "webp", "tiff", "pdf"],
    accept_multiple_files=False,
)
lang_choice = st.selectbox("Target explanation language", ["Auto-detect", "English", "Hindi"], index=0)

# Optional meta for template
with st.expander("🧾 Optional metadata for template", expanded=False):
    meta_cols = st.columns(3)
    meta_title = meta_cols[0].text_input("Story title override", "")
    meta_author = meta_cols[1].text_input("Author (if known)", "")
    meta_grade = meta_cols[2].text_input("Grade level (e.g., Grade 2)", "")

show_devices_table = st.toggle("Show literary devices table", value=True)
show_line_by_line = st.toggle("Show line-by-line explanation (if poetry)", value=True)

# Template upload and preview
with st.expander("🧩 Upload Literature AMP template (optional)", expanded=False):
    tpl_file = st.file_uploader(
        "Upload your Literature AMP HTML template",
        type=["html", "htm"],
        accept_multiple_files=False,
        key="tpl",
    )

run = st.button("🔎 Analyze")

if run:
    # Gather source text
    source_text = (text_input or "").strip()
    if files and not source_text:
        with st.spinner("Running OCR on uploaded file…"):
            blob = files.read()
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
    if lang_choice == "English":
        explain_lang = "en"
    elif lang_choice == "Hindi":
        explain_lang = "hi"
    else:
        explain_lang = detected

    st.info(f"Explanation language: **{explain_lang}** (detected: {detected})")

    # Build messages
    system_msg = LIT_SYSTEM + (" Explain primarily in Hindi." if explain_lang.startswith("hi") else " Explain primarily in English.")

    user_msg = (
        f"TEXT TO ANALYZE (verbatim):\n{source_text}\n\n"
        f"{PROMPT_FMT}\n"
        "Output strictly as minified JSON. No preface, no code fences."
    )

    with st.spinner("Calling Azure OpenAI for literary analysis…"):
        ok, content = call_azure_chat(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            temperature=0.1,
            max_tokens=2200,
        )

    if not ok:
        st.error(content)
        st.stop()

    # Try to parse JSON out of the reply
    def robust_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return None
            return None

    data = robust_parse(content) or {}
    if not isinstance(data, dict) or not data:
        st.error("Model did not return valid JSON. Raw reply (truncated):\n" + content[:800])
        st.stop()

    # Top cards
    st.success("✅ Analysis ready")
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

    # Word-by-word
    wbw = data.get("word_by_word_defs") or []
    if wbw:
        st.markdown("### 🧩 Word-by-word meanings & connotations")
        st.table([{"word": w.get("word", ""), "meaning": w.get("meaning", "")} for w in wbw])

    # Devices
    if show_devices_table:
        devices = data.get("devices") or []
        st.markdown("### 🎭 Literary devices")
        if devices:
            st.table(
                [
                    {"device": d.get("name", ""), "evidence": d.get("evidence", ""), "why": d.get("explanation", "")}
                    for d in devices
                ]
            )
        else:
            st.info("No clear devices detected.")

    # Line by line (poetry)
    if show_line_by_line and (data.get("text_type") == "poetry" or data.get("line_by_line")):
        st.markdown("### 📖 Line-by-line explanation")
        for i, item in enumerate(data.get("line_by_line", []), start=1):
            st.markdown(f"**Line {i}:** {item.get('line','')}")
            st.write(item.get("explanation", ""))
            st.divider()

    # Glossary & misconceptions
    gl = data.get("vocabulary_glossary") or []
    if gl:
        st.markdown("### 📒 Glossary")
        st.table(gl)

    mc = data.get("misconceptions") or []
    if mc:
        st.markdown("### ⚠️ Misconceptions to avoid")
        st.write("\n".join(f"• {m}" for m in mc))

    # -------- Build template payload and optionally fill uploaded template --------
    st.markdown("## 🏗️ Template payload")
    meta_overrides = {
        "storytitle": meta_title.strip() or None,
        "author": meta_author.strip() or None,
        "grade_level": meta_grade.strip() or None,
    }
    payload = build_template_payload(data, source_text=source_text, meta_overrides=meta_overrides)
    st.json(payload, expanded=False)

    # Download payload JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "⬇️ Download template JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"literature_template_payload_{ts}.json",
        mime="application/json",
    )

    # If a template is uploaded, fill it and offer download + preview
    if tpl_file is not None:
        try:
            raw_html = tpl_file.read().decode("utf-8", errors="replace")
            filled_html, found = fill_template_strict(raw_html, payload)
            st.success(f"Template filled. Placeholders detected: {len(found)}")

            st.download_button(
                "⬇️ Download filled HTML",
                data=filled_html.encode("utf-8"),
                file_name=f"literature_story_{ts}.html",
                mime="text/html",
            )

            with st.expander("👀 Inline preview"):
                st_html(filled_html, height=800, scrolling=True)
        except Exception as e:
            st.error(f"Template fill failed: {e}")

    # Raw JSON (analysis)
    with st.expander("🔧 Debug / Raw analysis JSON"):
        st.json(data, expanded=False)
        st.download_button(
            "⬇️ Download analysis JSON",
            data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"literature_analysis_{ts}.json",
            mime="application/json",
        )
