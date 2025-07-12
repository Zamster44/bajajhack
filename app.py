import os
import re
import pandas as pd
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline  # Use pipeline for easier loading
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import logging
import torch

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
TRANSCRIPTS_DIR = "data/transcripts"
STOCK_DATA_PATH = "data/BFS_Share_Price.csv"

# Initialize components
stock_df = None
vector_db = None
text_generator = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def initialize_system():
    """Initialize stock data, vector database, and text generator"""
    global stock_df, vector_db, text_generator
    
    # Load and preprocess stock data
    logger.info("Loading stock data...")
    stock_df = pd.read_csv(STOCK_DATA_PATH)
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df["Month-Year"] = stock_df["Date"].dt.strftime("%b-%y").str.upper()
    logger.info(f"Loaded {len(stock_df)} stock records")
    
    # Load transcripts
    logger.info("Processing transcripts...")
    transcripts = {}
    pdf_files = [f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        quarter = pdf_file.split(".")[0]
        pdf_path = os.path.join(TRANSCRIPTS_DIR, pdf_file)
        logger.info(f"Processing {pdf_file}...")
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            transcripts[quarter] = text
            logger.info(f"Extracted {len(text)} characters from {pdf_file}")
        else:
            logger.warning(f"Failed to extract text from {pdf_file}")
    
    if not transcripts:
        logger.error("No transcripts loaded!")
        return
    
    # Process transcripts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    
    all_chunks = []
    for quarter, text in transcripts.items():
        chunks = text_splitter.split_text(text)
        all_chunks.extend([(chunk, quarter) for chunk in chunks])
    
    logger.info(f"Created {len(all_chunks)} text chunks from transcripts")
    
    # Create vector store
    logger.info("Creating vector database...")
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_texts(
        [chunk[0] for chunk in all_chunks],
        embedder,
        metadatas=[{"quarter": chunk[1]} for chunk in all_chunks],
        persist_directory="chroma_db"
    )
    logger.info("Vector database initialized")
    
    # Initialize text generator with a small, efficient model
    logger.info("Loading text generation model...")
    model_name = "google/flan-t5-base"  # Lightweight model
    
    # Check if GPU is available
    device = 0 if torch.cuda.is_available() else -1
    
    text_generator = pipeline(
        "text2text-generation",
        model=model_name,
        device=device
    )
    logger.info("Text generator initialized")

# Initialize on app start
initialize_system()

def stock_analyzer(month_year, metric):
    """Calculate stock metrics for given month-year"""
    monthly_data = stock_df[stock_df["Month-Year"] == month_year.upper()]
    
    if len(monthly_data) == 0:
        raise ValueError(f"No data found for {month_year}")
    
    if metric == "highest":
        return monthly_data["Close Price"].max()
    elif metric == "lowest":
        return monthly_data["Close Price"].min()
    elif metric == "average":
        return monthly_data["Close Price"].mean()
    else:
        raise ValueError("Invalid metric. Use 'highest','lowest', or 'average'")

def period_comparator(period1, period2):
    """Compare average stock prices between two periods"""
    avg1 = stock_analyzer(period1, "average")
    avg2 = stock_analyzer(period2, "average")
    change_pct = ((avg2 - avg1) / avg1) * 100
    return {
        "period1": {"avg": avg1, "label": period1},
        "period2": {"avg": avg2, "label": period2},
        "change_pct": change_pct
    }

def retrieve_context(question, k=4):
    """Retrieve relevant transcript snippets"""
    return vector_db.similarity_search(question, k=k)

def generate_response(question, context):
    """Use local model to generate response"""
    prompt = f"""Answer the question based on the context. Be precise and use financial terminology.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    response = text_generator(
        prompt,
        max_length=500,
        temperature=0.7,
        do_sample=True
    )
    return response[0]['generated_text']

def generate_cfo_commentary():
    """Generate CFO commentary for investor call"""
    context = retrieve_context("performance challenges growth outlook", k=6)
    context_text = "\n\n".join([doc.page_content for doc in context])
    
    prompt = f"""Draft an investor call commentary covering:
    1. Performance highlights from last quarter 
    2. Challenges in Motor Insurance business
    3. Growth drivers (Hero partnership, Bajaj Markets)
    4. Forward-looking statements
    
    Use professional tone. Relevant context: {context_text}"""
    
    response = text_generator(
        prompt,
        max_length=400,
        temperature=0.7,
        do_sample=True
    )
    return response[0]['generated_text']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '').lower()
    
    # Stock price queries
    stock_match = re.search(r'(highest|lowest|average) stock price (across|in) (\w{3}-\d{2})', query)
    if stock_match:
        metric = stock_match.group(1)
        month_year = stock_match.group(3)
        try:
            result = stock_analyzer(month_year, metric)
            return jsonify({
                "type": "stock_metric",
                "metric": metric,
                "period": month_year,
                "value": result
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    # Period comparison
    compare_match = re.search(r'compare bajaj finserv from (\w{3}-\d{2}) to (\w{3}-\d{2})', query)
    if compare_match:
        period1 = compare_match.group(1)
        period2 = compare_match.group(2)
        try:
            comparison = period_comparator(period1, period2)
            return jsonify({
                "type": "comparison",
                "comparison": comparison
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    # CFO commentary
    if "cfo" in query and "commentary" in query:
        try:
            commentary = generate_cfo_commentary()
            return jsonify({
                "type": "commentary",
                "content": commentary
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Table extraction
    if "table" in query and "allianz" in query:
        context = retrieve_context("Allianz stake sale discussions", k=6)
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        prompt = f"""Create a table with these columns: Date | Speaker | Discussion Summary. 
        Use this context: {context_text}"""
        
        response = text_generator(
            prompt,
            max_length=500,
            temperature=0.3,
            do_sample=True
        )
        return jsonify({
            "type": "table",
            "content": response[0]['generated_text']
        })
    
    # General business questions
    try:
        context = retrieve_context(query)
        context_text = "\n\n".join([doc.page_content for doc in context])
        answer = generate_response(query, context_text)
        return jsonify({
            "type": "answer",
            "content": answer,
            "context_sources": [doc.metadata["quarter"] for doc in context]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)