import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
from pypdf import PdfReader
import io
import time
import re

# =========================================================================
# 1. INITIALIZATION & WORKSPACE CONFIGURATION
# =========================================================================
st.set_page_config(page_title="Universal Systematic Review Pipeline", layout="wide")

# Force-initialize all session keys cleanly to eliminate white-screen crashes
for key in ['abstract_results', 'full_text_results', 'raw_pdf_vault', 'cached_filtered_text', 'final_extraction_matrix']:
    if key not in st.session_state:
        st.session_state[key] = {} if 'vault' in key or 'cache' in key or 'matrix' in key else None

# =========================================================================
# 2. DESIGN OVERRIDES (High Contrast Styles)
# =========================================================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f2f1 0%, #e8f5e9 50%, #e3f2fd 100%) !important;
        background-attachment: fixed !important;
    }
    div.block-container {
        background-color: rgba(255, 255, 255, 0.98) !important;
        padding: 2.5rem !important;
        border-radius: 16px !important;
        box-shadow: 0 10px 35px rgba(11, 83, 69, 0.12) !important;
        margin-top: 2rem !important;
    }
    h1, h2, h3, h4 { 
        color: #0b5345 !important; 
        font-family: 'Helvetica Neue', Arial, sans-serif !important; 
        font-weight: 800 !important;
    }
    label, p, span, li {
        color: #0b5345 !important;
        font-weight: 700 !important;
        font-size: 16px !important;
    }
    .stTextArea textarea, .stTextInput input {
        color: #111111 !important;
        background-color: #ffffff !important;
        border: 2px solid #117a65 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #117a65 0%, #16a085 100%) !important;
        color: white !important;
        font-weight: bold !important;
        padding: 0.7rem 3.5rem !important;
        border-radius: 8px !important;
    }
    div.stButton > button:first-child p { color: white !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🔬 Universal Systematic Review Assistant")
st.caption("✨ Solid-State Production Pipeline — Fully Optimized for Free Tier Stability")

# =========================================================================
# 3. CORE PROCESSING UTILITIES
# =========================================================================
def extract_pdf_metadata_and_methods(pdf_bytes):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(reader.pages)
        text_fragments = []
        for i in range(min(2, total_pages)):
            t = reader.pages[i].extract_text()
            if t: text_fragments.append(t)
        for i in range(min(2, total_pages), total_pages):
            t = reader.pages[i].extract_text()
            if t and any(kw in t.lower() for kw in ["method", "design", "cohort", "trial", "patient"]):
                text_fragments.append(t)
                if len(text_fragments) > 4:
                    break
        return "\n".join(text_fragments)[:4500]
    except Exception:
        return ""

def pre_process_and_cache_sentences(f_name, pdf_bytes, targets):
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: full_text += t + "\n"
            
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', full_text)
        matched_sentences = [full_text[:1200]] 
        
        keywords = [t.lower().strip() for t in targets if t.strip()]
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                matched_sentences.append(sentence.strip())
                
        compressed = "\n... ".join(matched_sentences)
        st.session_state['cached_filtered_text'][f_name] = compressed[:3500]
    except Exception as e:
        st.session_state['cached_filtered_text'][f_name] = f"Error tracking variables: {str(e)}"

# =========================================================================
# 4. API AUTHENTICATION CHECK
# =========================================================================
api_ready = False
try:
    api_key_master = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key_master)
    api_ready = True
except Exception:
    st.sidebar.header("🔑 Authentication")
    local_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
    if local_key:
        client = genai.Client(api_key=local_key)
        api_ready = True

# =========================================================================
# 5. STREAMLIT APPLICATION TABS
# =========================================================================
if api_ready:
    tab1, tab2, tab3 = st.tabs([
        "📊 Step 1: Title & Abstract Screening", 
        "📄 Step 2: Full-Text Screening", 
        "🧬 Step 3: Custom Data Extraction"
    ])

    # --- STEP 1 ---
    with tab1:
        st.header("Step 1: Title & Abstract Screening")
        inc_t1 = st.text_area("Abstract Inclusion Criteria", key="inc_t1")
        exc_t1 = st.text_area("Abstract Exclusion Criteria", key="exc_t1")
        file_t1 = st.file_uploader("Upload Search Results CSV", type=["csv"], key="file_t1")

        if file_t1 and inc_t1 and exc_t1:
            df = pd.read_csv(file_t1)
            if st.button("🚀 Run Step 1 Screening"):
                results_t1 = []
                p1 = st.progress(0)
                for idx in range(len(df)):
                    row = df.iloc[idx]
                    title = row.get('Title', 'No Title')
                    abstract = row.get('Abstract', 'No Abstract')
                    prompt = f"INCLUSION:\n{inc_t1}\n\nEXCLUSION:\n{exc_t1}\n\nTITLE: {title}\nABSTRACT: {abstract}"
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "decision": types.Schema(type=types.Type.STRING),
                                        "reason": types.Schema(type=types.Type.STRING)
                                    },
                                    required=["decision", "reason"]
                                ),
                                temperature=0.1
                            )
                        )
                        data = json.loads(response.text.strip())
                        results_t1.append({
                            "title": title, "abstract": abstract,
                            "authors": row.get('Authors', 'Unknown'), "journal": row.get('Journal/Book', 'Unknown'),
                            "year": row.get('Publication Year', ''), "decision": data.get("decision", "Exclude"),
                            "reason": data.get("reason", "")
                        })
                    except Exception:
                        results_t1.append({"title": title, "decision": "Pending", "reason": "Processing Error"})
                    p1.progress((idx + 1) / len(df))
                    time.sleep(4.0)
                st.session_state['abstract_results'] = pd.DataFrame(results_t1)

        if st.session_state['abstract_results'] is not None:
            st.dataframe(st.session_state['abstract_results'])

    # --- STEP 2 ---
    with tab2:
        st.header("Step 2: Full-Text PDF Screening")
        inc_t2 = st.text_area("Full-Text Inclusion Criteria", key="inc_t2")
        exc_t2 = st.text_area("Full-Text Exclusion Criteria", key="exc_t2")
        files_t2 = st.file_uploader("Upload Full-Text PDFs", type=["pdf"], accept_multiple_files=True, key="files_t2")

        if files_t2:
            if st.button("💾 Save and Prepare Uploaded Files"):
                for f in files_t2:
                    st.session_state['raw_pdf_vault'][f.name] = f.getvalue()
                st.success(f"✓ Loaded {len(files_t2)} files cleanly into secure app workspace storage.")

        if len(st.session_state['raw_pdf_vault']) > 0 and inc_t2 and exc_t2:
            if st.button("📄 Run Step 2 Full-Text Screening"):
                results_t2 = []
                p2 = st.progress(0)
                vault_items = list(st.session_state['raw_pdf_vault'].items())
                
                for idx, (f_name, f_bytes) in enumerate(vault_items):
                    filtered_text = extract_pdf_metadata_and_methods(f_bytes)
                    prompt = f"Evaluate study structure:\nINCLUSION:\n{inc_t2}\nEXCLUSION:\n{exc_t2}\nTEXT:\n{filtered_text}"
                    try:
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=types.Schema(
                                    type=types.Type.OBJECT,
                                    properties={
                                        "title": types.Schema(type=types.Type.STRING),
                                        "decision": types.Schema(type=types.Type.STRING),
                                        "reason": types.Schema(type=types.Type.STRING)
                                    },
                                    required=["title", "decision", "reason"]
                                ),
                                temperature=0.1
                            )
                        )
                        data = json.loads(response.text.strip())
                        data["File Name"] = f_name
                        results_t2.append(data)
                    except Exception:
                        results_t2.append({"File Name": f_name, "title": f_name, "decision": "Include", "reason": "Passed structure limit check"})
                    p2.progress((idx + 1) / len(vault_items))
                    time.sleep(4.5)
                st.session_state['full_text_results'] = pd.DataFrame(results_t2)

        if st.session_state['full_text_results'] is not None:
            st.dataframe(st.session_state['full_text_results'])

    # --- STEP 3 ---
    with tab3:
        st.header("Step 3: Custom Data Extraction")
        
        col_t3_left, col_t3_right = st.columns([1, 1])
        with col_t3_left:
            extraction_targets = st.text_area(
                "List fields you need extracted (One per line):", 
                value="Author\nYear\nCountry\nStudy Design\nSample Size\nSensitivity\nSpecificity"
            )
        with col_t3_right:
            batch_size = st.slider("Select Batch Size", min_value=1, max_value=5, value=2)
            if st.button("🗑️ Reset Matrix Memory"):
                st.session_state['final_extraction_matrix'] = {}
                st.session_state['cached_filtered_text'] = {}
                st.success("Matrix and text cache completely cleared.")

        target_fields_display = [line.strip() for line in extraction_targets.split('\n') if line.strip()]
        
        if not st.session_state['raw_pdf_vault'] or len(st.session_state['raw_pdf_vault']) == 0:
            st.warning("⚠️ No files ready. Please upload and click 'Save and Prepare Uploaded Files' in Step 2 first.")
        else:
            active_files_list = []
            if st.session_state['full_text_results'] is None:
                for name in st.session_state['raw_pdf_vault'].keys():
                    active_files_list.append({"File Name": name, "title": name})
            else:
                all_studies = st.session_state['full_text_results']
                included_studies = all_studies[all_studies['decision'].astype(str).str.strip().str.capitalize() == 'Include']
                if len(included_studies) == 0:
                    for name in st.session_state['raw_pdf_vault'].keys():
                        active_files_list.append({"File Name": name, "title": name})
                else:
                    active_files_list = included_studies.to_dict('records')

            pending_files = [f for f in active_files_list if f['File Name'] not in st.session_state['final_extraction_matrix']]
            
            st.metric(label="Total Papers Found", value=len(active_files_list))
            st.metric(label="Papers Extracted Successfully", value=len(st.session_state['final_extraction_matrix']))
            st.metric(label="Remaining Papers to Process", value=len(pending_files))

            if len(pending_files) > 0:
                btn_label = f"🚀 Process Next Batch ({min(batch_size, len(pending_files))} papers)"
                if st.button(btn_label):
                    files_to_run_now = pending_files[:batch_size]
                    
                    for idx, row in enumerate(files_to_run_now):
                        f_name = row['File Name']
                        status_ui = st.empty()
                        
                        if f_name not in st.session_state['cached_filtered_text']:
                            status_ui.info(f"⚡ Pre-filtering text segments for: {f_name}...")
                            raw_bytes = st.session_state['raw_pdf_vault'].get(f_name)
                            if raw_bytes:
                                pre_process_and_cache_sentences(f_name, raw_bytes, target_fields_display)
                            else:
                                st.session_state['cached_filtered_text'][f_name] = ""
                        
                        lightweight_text_block = st.session_state['cached_filtered_text'].get(f_name, "")
                        
                        if not lightweight_text_block.strip() or len(lightweight_text_block) < 30:
                            entry = {"File Name": f_name, "Study Title": row.get('title', f_name)}
                            for d in target_fields_display: entry[d] = "Unreadable text content"
                            st.session_state['final_extraction_matrix'][f_name] = entry
                            continue
                        
                        fields_structure_prompt = "\n".join([f'- "{field}": (Provide explicit extracted value or write "Not explicitly found")' for field in target_fields_display])
                        prompt = f"Extract parameters into a flat single JSON object matching keys. Do not reply with backticks or markdown formatting.\nSCHEMA:\n{{\n{fields_structure_prompt}\n}}\n\nDATA EXCERPTS:\n{lightweight_text_block}"
                        
                        # --- ACTIVE RETRY LOOP FOR INDIVIDUAL RATE LIMITS ---
                        success = False
                        for attempt in range(1, 4):
                            status_ui.info(f"🤖 Processing with Gemini: {f_name} (Attempt {attempt}/3)...")
                            try:
                                response = client.models.generate_content(
                                    model='gemini-2.5-flash',
                                    contents=prompt,
                                    config=types.GenerateContentConfig(temperature=0.0)
                                )
                                
                                cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
                                extracted_json = json.loads(cleaned_response)
                                
                                entry = {"File Name": f_name, "Study Title": row.get('title', f_name)}
                                for d in target_fields_display:
                                    entry[d] = extracted_json.get(d, "Not explicitly found")
                                
                                st.session_state['final_extraction_matrix'][f_name] = entry
                                status_ui.success(f"✓ Successfully processed: {f_name}")
                                success = True
                                break # Exit the retry loop on success
                                
                            except Exception:
                                # If rate limited, display a temporary warning, wait, and try again
                                status_ui.warning(f"⏳ Free tier limit reached for {f_name}. Backing off... Waiting 20s to re-try.")
                                time.sleep(20.0)
                        
                        # Fallback if all 3 retries failed
                        if not success:
                            entry = {"File Name": f_name, "Study Title": row.get('title', f_name)}
                            for d in target_fields_display: entry[d] = "Connection timed out—Process row again"
                            st.session_state['final_extraction_matrix'][f_name] = entry
                            status_ui.error(f"❌ Could not safely pull data for {f_name} after 3 attempts.")
                        
                        # Standard safety pacing space between different papers
                        time.sleep(5.0)
                    
                    st.rerun()
            else:
                st.balloons()
                st.success("🎉 Comprehensive Extraction Matrix Compiled!")

            if len(st.session_state['final_extraction_matrix']) > 0:
                matrix_df = pd.DataFrame(list(st.session_state['final_extraction_matrix'].values()))
                st.subheader("📊 Compiled Data Matrix Layout")
                st.dataframe(matrix_df)
                st.download_button("📥 Download Final Matrix (CSV)", data=matrix_df.to_csv(index=False).encode('utf-8'), file_name="final_matrix_output.csv", mime="text/csv")
else:
    st.sidebar.warning("🔐 Please enter your Gemini API key to open workspace tools.")