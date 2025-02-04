import os
import json
import time
import requests
import hashlib
import subprocess
from collections import defaultdict, Counter
import numpy as np
from PyPDF2 import PdfReader
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# === CONFIGURATION ===
PDF_FILE = "./in/pdf.pdf"
LM_STUDIO_URL = "http://localhost:1234/v1/completions"
MODEL_NAME = "qwen2.5-7b-instruct-1m"
TOKENIZER_NAME = "Qwen/Qwen1.5-7B"

# Chunking and Generation Configuration
MAX_CONTEXT_LENGTH = 4096  
SAFETY_MARGIN = 0.9       
MAX_TOKENS = 1024          
TEMPERATURE = 0.3          

# Keyword Extraction Configuration
TOP_GLOBAL_KEYWORDS = 100  # Top hundred keywords that reoccur through pages
TOP_PAGE_KEYWORDS = 3      # Top 3 keywords per page (from the sorted global keywords)

# Reject & Allowed Keywords Configuration
REJECT_KEYWORDS_FILE = "./reject_keywords.txt"
ALLOWED_KEYWORDS_FILE = "./allowed_keywords.txt"

# GPU Monitoring Configuration
MAX_GPU_TEMP = 80            # Maximum GPU temperature (°C)
COOL_DOWN_PERIOD = 60        # Time to wait (in seconds) if GPU is too hot

# Paths
OUTPUT_DIR = "./out"
PAGE_TEXT_FILE = os.path.join(OUTPUT_DIR, "page_text.json")
PAGE_KEYWORDS_FILE = os.path.join(OUTPUT_DIR, "page_keywords.json")
KEYWORDS_FILE = os.path.join(OUTPUT_DIR, "keywords.json")
INTERMEDIATE_SUMMARY_DIR = os.path.join(OUTPUT_DIR, "intermediate_summaries")
FINAL_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "final_summary.md")

# Additional file paths for summarization
INPUT_FILE = os.path.join(OUTPUT_DIR, "combined_text.txt")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
INTERMEDIATE_FILE = os.path.join(INTERMEDIATE_SUMMARY_DIR, "intermediate_summary.txt")

# Ensure Output Directories Exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_SUMMARY_DIR, exist_ok=True)

# === Functions to Load Reject and Allowed Keywords from File ===
def load_reject_keywords():
    """
    Loads the list of reject keywords from REJECT_KEYWORDS_FILE.
    Each line in the file should contain one keyword.
    If the file is not found, returns a default list.
    """
    if os.path.exists(REJECT_KEYWORDS_FILE):
        with open(REJECT_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(keywords)} reject keywords from {REJECT_KEYWORDS_FILE}.")
            return keywords
    else:
        default_keywords = ["the", "and", "or", "but", "if", "to", "a", "an", "in", "of", "with", "for", "on","just","being","been","do","does","did,"have","had","use","used","shows","know","need","include","following","case","defined","define","list"]
        print(f"Reject keywords file not found. Using default list: {default_keywords}")
        return default_keywords

def load_allowed_keywords():
    """
    Loads the list of allowed keywords from ALLOWED_KEYWORDS_FILE.
    Each line in the file should contain one keyword.
    If the file is not found, returns an empty list.
    """
    if os.path.exists(ALLOWED_KEYWORDS_FILE):
        with open(ALLOWED_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(keywords)} allowed keywords from {ALLOWED_KEYWORDS_FILE}.")
            return keywords
    else:
        default_allowed = []
        print(f"Allowed keywords file not found. Using default allowed keywords: {default_allowed}")
        return default_allowed

# === Tokenization Class ===
class TokenCounter:
    def __init__(self, model_name=TOKENIZER_NAME):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            self.tokenizer = None

    def count_tokens(self, text):
        if self.tokenizer is None:
            # Approximate token count if no tokenizer is available.
            return int(len(text.split()) * 1.3)
        return len(self.tokenizer.encode(text))

# === Step 1: Extract Text from PDF ===
def extract_text_from_pdf(pdf_path):
    print("📄 Extracting text from PDF...")
    reader = PdfReader(pdf_path)
    text_data = {}
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        text_data[f"Page {i+1}"] = page_text.strip() if page_text else ""
        print(f"✔ Extracted text from Page {i+1}")
    with open(PAGE_TEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(text_data, f, indent=4)
    print(f"✅ Extracted text from {len(text_data)} pages.\n")
    return text_data

# === Step 2: Extract and Select Keywords Across Pages ===
def extract_and_select_keywords(text_data):
    """
    1. Use TfidfVectorizer on all pages to obtain a global vocabulary.
    2. Compute document frequency for each keyword.
    3. Select the top 100 keywords that reoccur across pages,
       rejecting any keywords from the reject list.
    4. Sort these keywords by similarity (using a SHA-256–based computed value).
    5. For each page, first add allowed keywords (if present),
       then fill in up to TOP_PAGE_KEYWORDS from the sorted global keywords.
    """
    print("🔍 Extracting global keywords and selecting top keywords per page...")
    vectorizer = TfidfVectorizer(stop_words='english')
    page_texts = list(text_data.values())
    X = vectorizer.fit_transform(page_texts)
    all_keywords = vectorizer.get_feature_names_out()
    doc_freqs = (X > 0).sum(axis=0).A1
    keyword_freq_global = {kw: freq for kw, freq in zip(all_keywords, doc_freqs)}
    reject_keywords = load_reject_keywords()
    eligible_keywords = [(kw, freq) for kw, freq in keyword_freq_global.items() if kw not in reject_keywords]
    top_keywords = sorted(eligible_keywords, key=lambda item: item[1], reverse=True)[:TOP_GLOBAL_KEYWORDS]
    top_keywords = [kw for kw, freq in top_keywords]
    sorted_top_keywords = compute_similarity_index(top_keywords)
    allowed_keywords = load_allowed_keywords()
    page_keywords = {}
    for page, text in text_data.items():
        text_lower = text.lower()
        selected = []
        # Prioritize allowed keywords.
        for ak in allowed_keywords:
            if ak in text_lower and ak not in selected:
                selected.append(ak)
            if len(selected) == TOP_PAGE_KEYWORDS:
                break
        # Fill remaining slots with sorted global keywords.
        if len(selected) < TOP_PAGE_KEYWORDS:
            for kw in sorted_top_keywords:
                if kw in text_lower and kw not in selected:
                    selected.append(kw)
                if len(selected) == TOP_PAGE_KEYWORDS:
                    break
        page_keywords[page] = {"keywords": selected, "text_chunk": text}
        print(f"✔ For {page}, selected keywords: {selected}")
    with open(PAGE_KEYWORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(page_keywords, f, indent=4)
    print("✅ Global and per-page keywords extraction completed.\n")
    return page_keywords

# === Step 3: Compute Similarity Index and Sort Keywords ===
def compute_similarity_index(keywords):
    """
    Compute a 'similarity' index for each keyword using its SHA-256 hash,
    then sort the keywords in descending order based on that value.
    """
    print("🔢 Computing similarity index for keywords...")
    def hex_value(word):
        return int(hashlib.sha256(word.encode()).hexdigest(), 16)
    similarity_index = {kw: hex_value(kw) for kw in keywords}
    sorted_keywords = sorted(similarity_index.keys(), key=lambda x: similarity_index[x], reverse=True)
    print("✅ Keywords sorted based on similarity index.\n")
    return sorted_keywords

# === Step 4: Categorize Pages by Keywords ===
def categorize_pages_by_keywords(page_keywords):
    print("📌 Categorizing pages based on keywords...")
    keyword_map = defaultdict(lambda: {"page_numbers": [], "text_chunks": []})
    all_keywords = set()
    for page_data in page_keywords.values():
        all_keywords.update(page_data["keywords"])
    sorted_keywords = compute_similarity_index(list(all_keywords))
    for page, data in page_keywords.items():
        for keyword in sorted_keywords:
            if keyword in data["keywords"]:
                keyword_map[keyword]["page_numbers"].append(page)
                keyword_map[keyword]["text_chunks"].append(data["text_chunk"])
                print(f"✔ Categorized {page} under keyword '{keyword}'")
    with open(KEYWORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(keyword_map, f, indent=4)
    print("✅ Pages categorized based on keywords.\n")
    return keyword_map

# === GPU Monitor ===
class GPUMonitor:
    def __init__(self, max_temp=MAX_GPU_TEMP):
        self.max_temp = max_temp
        self.running = True
    def get_gpu_temperature(self):
        try:
            result = subprocess.run(['sensors'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if "edge" in line.lower():
                    temp_str = line.split(":")[1].split("°C")[0].strip()
                    return float(temp_str)
            return 0.0
        except Exception as e:
            print(f"Error reading GPU temperature: {e}")
            return 0.0
    def check_temperature(self):
        temp = self.get_gpu_temperature()
        if temp >= self.max_temp:
            print(f"⚠️ WARNING: GPU temperature ({temp}°C) exceeds safety threshold ({self.max_temp}°C)")
            return False
        return True
    def force_stop(self):
        self.running = False
        print("🛑 Emergency stop initiated due to high GPU temperature!")

# === Text Processing Functions ===
def load_text(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        return ""
def save_text(filename, content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"Error saving file {filename}: {e}")

# === New Function: Store LM Studio Response per Chunk ===
def store_lm_response(chunk_index, response):
    """
    Stores the LM Studio response for a given chunk in a separate file under the intermediate folder.
    """
    filename = os.path.join(INTERMEDIATE_SUMMARY_DIR, f"lm_response_{chunk_index}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Stored LM Studio response for chunk {chunk_index} in {filename}")
    except Exception as e:
        print(f"Error storing LM Studio response for chunk {chunk_index}: {e}")

# === Improved Chunking ===
def safe_chunk_text(text, token_counter, num_chapters=None):
    safe_length = int(MAX_CONTEXT_LENGTH * SAFETY_MARGIN)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    sentences = text.replace(".\n", ". \n").split(". ")
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            sentence = sentence + ". "
            sentence_tokens = token_counter.count_tokens(sentence)
            if current_tokens + sentence_tokens > safe_length:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# === Call LM Studio API for Summarization (with streaming) ===
def call_lmstudio(text):
    if not text.strip():
        return "Error: Empty input text"
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Summarize: {text}",
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }
    try:
        with requests.post(LM_STUDIO_URL, json=payload, stream=True, timeout=120) as response:
            response.raise_for_status()
            chunks = []
            for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
                if chunk:
                    chunks.append(chunk)
            response_data = ''.join(chunks)
            result = json.loads(response_data)
            summary = result.get("response", result.get("choices", [{}])[0].get("text", "Error: No response"))
            return summary.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# === Progress Tracking ===
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading progress file: {e}")
    return {"last_processed": 0}
def save_progress(last_processed):
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump({"last_processed": last_processed}, f)
    except Exception as e:
        print(f"Error saving progress file: {e}")

# === Summarization of Large Text (Intermediate Summary Generation) ===
def summarize_large_text():
    gpu_monitor = GPUMonitor()
    token_counter = TokenCounter()
    text = load_text(INPUT_FILE)
    if not text:
        print("No input text found!")
        return
    total_tokens = token_counter.count_tokens(text)
    print(f"Total tokens in combined text: {total_tokens}")
    file_size = os.path.getsize(INPUT_FILE)
    num_chapters = max(1, file_size // (1024 * 50))  # Approx 50KB per chapter
    chunks = safe_chunk_text(text, token_counter, num_chapters)
    print(f"Total chunks created from combined text: {len(chunks)}")
    summaries = []
    progress = load_progress()
    last_processed = progress.get("last_processed", 0)
    for i, chunk in enumerate(chunks):
        if i < last_processed:
            print(f"Skipping chunk {i + 1}/{len(chunks)} (already processed).")
            continue
        print(f"Processing chunk {i + 1}/{len(chunks)} for intermediate summary...")
        if not gpu_monitor.check_temperature():
            print(f"Cooling down for {COOL_DOWN_PERIOD} seconds due to high GPU temperature.")
            time.sleep(COOL_DOWN_PERIOD)
            if not gpu_monitor.check_temperature():
                save_progress(i)
                gpu_monitor.force_stop()
                return
        summary = call_lmstudio(chunk)
        store_lm_response(i, summary)
        if summary.startswith("Error:"):
            summaries.append("[Summarization Failed]")
        else:
            summaries.append(summary)
        if not gpu_monitor.check_temperature():
            save_progress(i)
            gpu_monitor.force_stop()
            return
        save_progress(i + 1)
        time.sleep(1)  # Delay between requests
    intermediate_summary = "\n\n".join(summaries)
    save_text(INTERMEDIATE_FILE, intermediate_summary)
    print(f"Intermediate summary saved to {INTERMEDIATE_FILE}")

# === New Function: Create Final Summary from Intermediate Summary ===
def create_final_summary():
    print("🔄 Creating final summary from intermediate summary...")
    token_counter = TokenCounter()
    intermediate_text = load_text(INTERMEDIATE_FILE)
    if not intermediate_text:
        print("No intermediate summary found. Cannot create final summary.")
        return
    final_chunks = safe_chunk_text(intermediate_text, token_counter)
    print(f"Total final chunks created: {len(final_chunks)}")
    final_summaries = []
    for i, chunk in enumerate(final_chunks):
        print(f"Processing final chunk {i + 1}/{len(final_chunks)}...")
        final_resp = call_lmstudio(chunk)
        final_resp_filename = os.path.join(INTERMEDIATE_SUMMARY_DIR, f"final_response_{i}.txt")
        save_text(final_resp_filename, final_resp)
        final_summaries.append(final_resp)
        time.sleep(1)
    combined_final = "\n\n".join(final_summaries)
    if len(final_summaries) > 1:
        print("Performing final combination summarization step...")
        final_summary = call_lmstudio(combined_final)
    else:
        final_summary = combined_final
    save_text(FINAL_SUMMARY_FILE, final_summary)
    print(f"Final summary saved to {FINAL_SUMMARY_FILE}")

# === Main Execution ===
if __name__ == "__main__":
    print("🚀 Starting summarization process...")
    if os.path.exists(PDF_FILE):
        # Step 1: Extract text from PDF.
        text_data = extract_text_from_pdf(PDF_FILE)
        # Step 2: Extract and select keywords.
        page_keywords = extract_and_select_keywords(text_data)
        # Step 3: Categorize pages based on keywords.
        categorize_pages_by_keywords(page_keywords)
        # Step 4: Combine text in page order.
        sorted_pages = sorted(text_data.keys(), key=lambda x: int(x.split()[1]))
        combined_text = "\n\n".join(text_data[page] for page in sorted_pages)
        save_text(INPUT_FILE, combined_text)
        print(f"✔ Combined text saved to {INPUT_FILE}")
    else:
        print(f"PDF file {PDF_FILE} does not exist. Please provide a valid PDF file.")
    
    # If the intermediate summary already exists, skip generating it.
    if os.path.exists(INTERMEDIATE_FILE):
        print(f"Intermediate summary file {INTERMEDIATE_FILE} exists. Skipping intermediate summarization.")
    else:
        print("Generating intermediate summary from combined text...")
        summarize_large_text()
    
    # Create the final summary from the intermediate summary.
    create_final_summary()
