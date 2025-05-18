import streamlit as st
import pdfplumber
import pandas as pd
import requests
from io import BytesIO
import google.generativeai as genai

# Configure Gemini API
def configure_gemini(api_key):
    """Configure Gemini API with the provided key"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Simple question answering function using Gemini
def simple_qa(question, context, model):
    """Question answering using Gemini"""
    try:
        prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context: {context}

Question: {question}

Answer:"""
        response = model.generate_content(prompt)
        return {
            'answer': response.text,
            'score': 0.8  # Gemini doesn't provide confidence scores, using a default value
        }
    except Exception as e:
        return {
            'answer': f"Error processing question: {str(e)}",
            'score': 0.0
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
def summarize_text(text, max_sentences=5):
    """Summarize text using Google's Gemini model"""
    if not text.strip():
        return "No text content found to summarize."
    
    try:
        # Get API key from session state
        api_key = st.session_state.get("GEMINI_API_KEY")
        
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            return "Error: Gemini API key not found."
        
        # Configure Gemini
        model = configure_gemini(api_key)
        
        # Split text into chunks of approximately 4000 characters to handle large documents
        max_chunk_chars = 4000
        chunks = [text[i:i+max_chunk_chars] for i in range(0, len(text), max_chunk_chars)]
        summaries = []
        
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Only summarize substantial chunks
                prompt = f"Please provide a concise summary of the following text in {max_sentences} sentences or less:\n\n{chunk}"
                response = model.generate_content(prompt)
                if response.text:
                    summaries.append(response.text)
        
        return "\n\n".join(summaries) if summaries else "Could not generate summary."
    
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Error generating summary."

def extract_keywords(text, num_keywords=15):
    """Extract key phrases from text using Gemini"""
    if not text.strip():
        return []
        
    try:
        # Get API key from session state
        api_key = st.session_state.get("GEMINI_API_KEY")
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            return []
            
        model = configure_gemini(api_key)
        prompt = f"""Extract {num_keywords} key phrases or keywords from the following text. 
        Return them in the format: "keyword1 (score1), keyword2 (score2), ..." where score is between 0 and 1.
        
        Text: {text}"""
        
        response = model.generate_content(prompt)
        if response.text:
            # Parse the response to extract keywords and scores
            keywords = []
            for item in response.text.split(','):
                if '(' in item and ')' in item:
                    keyword = item.split('(')[0].strip()
                    score = float(item.split('(')[1].split(')')[0].strip())
                    keywords.append((score, keyword))
            return keywords
        return []
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
gemini_api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password", key="gemini_api_input")
if gemini_api_key:
    st.session_state["GEMINI_API_KEY"] = gemini_api_key
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
                        model = configure_gemini(st.session_state.get("GEMINI_API_KEY"))
                        answer = simple_qa(question=user_question, context=text, model=model)
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
                summary = summarize_text(text)
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
