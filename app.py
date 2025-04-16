import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from rake_nltk import Rake
import nltk
import requests
from io import BytesIO

# Download required NLTK data once at startup
nltk.download('stopwords', quiet=True)

# === Model Loading (Cached) ===
@st.cache_resource
def load_models():
    """Load and cache NLP models to improve performance"""
    return {
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn"),
        "qa": pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    }

# === PDF Processing Functions ===
@st.cache_data
def extract_pdf_content(file):
    """Extract text and tables from PDF file"""
    text = ""
    tables = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                tables.extend(page.extract_tables())
        return text, tables
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return "", []

def load_pdf_from_url(url):
    """Download PDF from URL and return as BytesIO object"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", ""):
            return BytesIO(response.content)
        else:
            st.error("❌ Failed to fetch a valid PDF from the URL.")
            return None
    except Exception as e:
        st.error(f"Error fetching PDF: {e}")
        return None

# === Analysis Functions ===
def summarize_text(text, summarizer, max_chunk_chars=1024):
    """Split text into chunks and summarize each chunk"""
    if not text.strip():
        return "No text content found to summarize."
        
    try:
        chunks = [text[i:i+max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
        summaries = []
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                summaries.append(summary)
                
        return "\n\n".join(summaries) if summaries else "Could not generate summary."
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Error generating summary."

def extract_keywords(text, num_keywords=15):
    """Extract key phrases from text using RAKE algorithm"""
    if not text.strip():
        return []
        
    try:
        rake = Rake()
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases_with_scores()[:num_keywords]
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")
        return []

def process_tables(tables):
    """Process and display tables from PDF"""
    st.subheader("📋 Detected Tables")
    if not tables:
        st.info("No tables detected in the document.")
        return
        
    for idx, table in enumerate(tables):
        try:
            if not table or len(table) < 2:  # Skip empty tables or tables with only headers
                continue
                
            df = pd.DataFrame(table[1:], columns=table[0])
            st.markdown(f"**Table {idx + 1}**")
            st.dataframe(df)
            
            # Only create charts for numeric data
            numeric_df = df.select_dtypes(include='number')
            if not numeric_df.empty and numeric_df.shape[1] > 0:
                st.line_chart(numeric_df)
        except Exception as e:
            st.warning(f"Table {idx + 1} could not be processed: {e}")

def classify_document(text):
    """Simple rule-based document classification"""
    text_lower = text.lower()
    if "invoice" in text_lower or "bill" in text_lower:
        return "Invoice / Billing"
    elif "research" in text_lower or "abstract" in text_lower or "methodology" in text_lower:
        return "Academic / Research Paper"
    elif "report" in text_lower or "quarterly" in text_lower or "annual" in text_lower:
        return "Report / Business Document"
    elif "contract" in text_lower or "agreement" in text_lower or "legal" in text_lower:
        return "Legal / Contract"
    else:
        return "General Document"

# === Main UI ===
st.set_page_config(page_title="🧠 Smart PDF Analyzer", layout="wide")
st.title("📄 Smart PDF Analyzer (Local AI Toolkit)")

# Sidebar options
st.sidebar.header("🔧 View Options")
show_summary = st.sidebar.checkbox("Show Summary", value=True)
show_keywords = st.sidebar.checkbox("Show Keywords", value=True)
show_tables_option = st.sidebar.checkbox("Show Tables", value=True)
num_keywords = st.sidebar.slider("Number of Keywords", 5, 30, 15)
enable_chat = st.sidebar.checkbox("Enable Chat Q&A", value=True)

# File upload section
st.subheader("📥 Upload a PDF or Paste a Link")
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
pdf_url = st.text_input("Or enter a direct PDF URL (http/https):")

# Process PDF file (from upload or URL)
pdf_file = uploaded_file
if pdf_url and not uploaded_file:
    with st.spinner("🔗 Downloading PDF from URL..."):
        pdf_file = load_pdf_from_url(pdf_url)

# Main content processing
if pdf_file:
    # Load models
    models = load_models()
    
    # Extract content
    with st.spinner("🔍 Reading and parsing PDF..."):
        text, tables = extract_pdf_content(pdf_file)
    
    if not text.strip():
        st.error("Could not extract text from the PDF. The file might be scanned or protected.")
    else:
        # Display document type
        doc_type = classify_document(text)
        st.markdown(f"### 📁 Document Type: **{doc_type}**")
        
        # Chat Q&A Interface
        if enable_chat:
            st.subheader("💬 Chat with Your PDF")
            
            # Initialize chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Chat form
            with st.form(key="chat_form", clear_on_submit=True):
                user_question = st.text_input("Ask something about the PDF:")
                submitted = st.form_submit_button("Send")
            
            if submitted and user_question:
                with st.spinner("Thinking..."):
                    try:
                        answer = models["qa"](question=user_question, context=text)
                        st.session_state.chat_history.append({
                            "question": user_question,
                            "answer": answer['answer'],
                            "score": round(answer['score'], 2)
                        })
                    except Exception as e:
                        st.error(f"Error processing question: {e}")
            
            # Display chat history
            if st.session_state.chat_history:
                for chat in reversed(st.session_state.chat_history):
                    st.markdown(f"**You:** {chat['question']}")
                    st.markdown(f"**AI:** {chat['answer']} _(confidence: {chat['score']})_")
                    st.markdown("---")
        
        # Summary section
        if show_summary:
            st.subheader("📃 Summary")
            with st.spinner("Summarizing..."):
                summary = summarize_text(text, models["summarizer"])
                st.write(summary)
                st.download_button("💾 Download Summary", data=summary, file_name="summary.txt")
        
        # Keywords section
        if show_keywords:
            st.subheader("🗝️ Keywords")
            keywords = extract_keywords(text, num_keywords)
            if keywords:
                for score, phrase in keywords:
                    st.write(f"{phrase} ({score:.2f})")
            else:
                st.info("No keywords could be extracted.")
        
        # Tables section
        if show_tables_option and tables:
            process_tables(tables)
        
        # Search functionality
        with st.expander("🔍 Search Full PDF Text"):
            query = st.text_input("Search for:")
            if query:
                results = [line for line in text.splitlines() if query.lower() in line.lower()]
                if results:
                    st.markdown("### Results:")
                    for r in results:
                        st.write(f"- {r}")
                else:
                    st.write("No results found.")
