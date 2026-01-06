import streamlit as st
import ollama
import fitz  # PyMuPDF
import time

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Research AI - Assistant", page_icon="🧬")
st.title("🧬 Research AI Assistant")
st.markdown("Upload a research paper to extract ideas and summaries locally.")

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    # Read PDF from memory
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Choose Model", ["phi4-mini", "llama3.2:3b", "qwen2.5-coder:7b"])
    st.info("Ensure Ollama is running and you have run: `ollama pull " + model_choice + "`")

# --- MAIN INTERFACE ---
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("Analyze Document"):
        with st.status("Processing...", expanded=True) as status:
            # 1. Extraction
            st.write("Reading file content...")
            if uploaded_file.type == "application/pdf":
                content = extract_text_from_pdf(uploaded_file)
            else:
                content = uploaded_file.getvalue().decode("utf-8")
            
            # 2. Preparation
            st.write(f"Sending to {model_choice}...")
            prompt = f"""
            Analyze this research text and provide:
            1. CORE IDEAS: Main findings/arguments.
            2. NEW SUGGESTIONS: 3 innovative future directions.
            3. SUMMARY: A high-quality resume paragraph.

            TEXT: {content[:15000]} # Limiting text slightly for speed
            """
            
            # 3. AI Generation with timer
            start_time = time.time()
            response = ollama.chat(model=model_choice, messages=[{'role': 'user', 'content': prompt}])
            end_time = time.time()
            
            status.update(label=f"Done! (Analysis took {round(end_time - start_time, 2)}s)", state="complete")

        # --- DISPLAY RESULTS ---
        st.subheader("Analysis Results")
        st.markdown(response['message']['content'])
        
        # Download button for the results
        st.download_button("Download Report", response['message']['content'], file_name="research_analysis.txt")