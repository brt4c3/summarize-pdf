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

###############################
#        CONFIGURATION        #
###############################

PDF_FILE = "./in/pdf.pdf"
LM_STUDIO_URL = "http://localhost:1234/v1/completions"
MODEL_NAME = "qwen2.5-7b-instruct-1m"
TOKENIZER_NAME = "Qwen/Qwen1.5-7B"

# Chunking and Generation Configuration
MAX_CONTEXT_LENGTH = 4096  # Slightly below the model's n_ctx (8000) for safety.
SAFETY_MARGIN = 0.9        # Use a higher safety margin to avoid exceeding the context window.
MAX_TOKENS = 1024          # Set to the maximum token length (evaluation batch size).
TEMPERATURE = 0.3          # Lower temperature for more deterministic summarization.

# Keyword Extraction Parameters
TOP_GLOBAL_KEYWORDS = 100   # Global vocabulary size.
TOP_PAGE_KEYWORDS = 3       # Number of keywords per page.

# File Paths
REJECT_KEYWORDS_FILE = "./reject_keywords.txt"
ALLOWED_KEYWORDS_FILE = "./allowed_keywords.txt"
OUTPUT_DIR = "./out"
PAGE_TEXT_FILE = os.path.join(OUTPUT_DIR, "page_text.json")
PAGE_KEYWORDS_FILE = os.path.join(OUTPUT_DIR, "page_keywords.json")
KEYWORDS_FILE = os.path.join(OUTPUT_DIR, "keywords.json")
INTERMEDIATE_SUMMARY_DIR = os.path.join(OUTPUT_DIR, "intermediate_summaries")
FINAL_SUMMARY_FILE = os.path.join(OUTPUT_DIR, "final_summary.md")
INPUT_FILE = os.path.join(OUTPUT_DIR, "combined_text.txt")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
INTERMEDIATE_FILE = os.path.join(INTERMEDIATE_SUMMARY_DIR, "intermediate_summary.txt")

# GPU Settings
MAX_GPU_TEMP = 80         # Maximum GPU temperature (°C)
COOL_DOWN_PERIOD = 60     # Seconds to wait if GPU is too hot

###############################
#       UTILITY CLASSES       #
###############################
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
            return int(len(text.split()) * 1.3)
        return len(self.tokenizer.encode(text))

class PDFProcessor:
    """Converts a PDF file into per-page text."""
    def __init__(self, pdf_file, output_file):
        self.pdf_file = pdf_file
        self.output_file = output_file

    def extract_text(self):
        print("📄 Extracting text from PDF...")
        reader = PdfReader(self.pdf_file)
        text_data = {}
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            text_data[f"Page {i+1}"] = page_text.strip() if page_text else ""
            print(f"✔ Extracted text from Page {i+1}")
        with open(PAGE_TEXT_FILE, "w", encoding="utf-8") as f:
            json.dump(text_data, f, indent=4)
        print(f"✅ Extracted text from {len(text_data)} pages.\n")
        return text_data

class KeywordExtractor:
    """Handles keyword extraction and categorization."""
    def __init__(self, reject_file, allowed_file, top_global, top_page):
        self.reject_file = reject_file
        self.allowed_file = allowed_file
        self.top_global = top_global
        self.top_page = top_page

    def load_reject_keywords(self):
        if os.path.exists(self.reject_file):
            with open(self.reject_file, "r", encoding="utf-8") as f:
                keywords = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(keywords)} reject keywords from {self.reject_file}.")
                return keywords
        default_keywords = ["the", "and", "or", "but", "if", "to", "a", "an", "in", "of", "with", "for", "on"]
        print(f"Reject keywords file not found. Using default list: {default_keywords}")
        return default_keywords

    def load_allowed_keywords(self):
        if os.path.exists(self.allowed_file):
            with open(self.allowed_file, "r", encoding="utf-8") as f:
                keywords = [line.strip() for line in f if line.strip()]
                print(f"Loaded {len(keywords)} allowed keywords from {self.allowed_file}.")
                return keywords
        print("Allowed keywords file not found. Using empty list.")
        return []

    def compute_similarity_index(self, keywords):
        print("🔢 Computing similarity index for keywords...")
        def hex_value(word):
            return int(hashlib.sha256(word.encode()).hexdigest(), 16)
        similarity_index = {kw: hex_value(kw) for kw in keywords}
        sorted_keywords = sorted(similarity_index.keys(), key=lambda x: similarity_index[x], reverse=True)
        print("✅ Keywords sorted based on similarity index.\n")
        return sorted_keywords

    def extract_keywords(self, text_data):
        print("🔍 Extracting global keywords and selecting top keywords per page...")
        vectorizer = TfidfVectorizer(stop_words='english')
        page_texts = list(text_data.values())
        X = vectorizer.fit_transform(page_texts)
        all_keywords = vectorizer.get_feature_names_out()
        doc_freqs = (X > 0).sum(axis=0).A1
        keyword_freq_global = {kw: freq for kw, freq in zip(all_keywords, doc_freqs)}
        reject_keywords = self.load_reject_keywords()
        eligible_keywords = [(kw, freq) for kw, freq in keyword_freq_global.items() if kw not in reject_keywords]
        top_keywords = sorted(eligible_keywords, key=lambda item: item[1], reverse=True)[:self.top_global]
        top_keywords = [kw for kw, freq in top_keywords]
        sorted_top_keywords = self.compute_similarity_index(top_keywords)
        allowed_keywords = self.load_allowed_keywords()
        page_keywords_dict = {}
        for page, text in text_data.items():
            text_lower = text.lower()
            selected = []
            for ak in allowed_keywords:
                if ak in text_lower and ak not in selected:
                    selected.append(ak)
                if len(selected) == self.top_page:
                    break
            if len(selected) < self.top_page:
                for kw in sorted_top_keywords:
                    if kw in text_lower and kw not in selected:
                        selected.append(kw)
                    if len(selected) == self.top_page:
                        break
            page_keywords_dict[page] = {"keywords": selected, "text_chunk": text}
            print(f"✔ For {page}, selected keywords: {selected}")
        with open(PAGE_KEYWORDS_FILE, "w", encoding="utf-8") as f:
            json.dump(page_keywords_dict, f, indent=4)
        print("✅ Global and per-page keywords extraction completed.\n")
        return page_keywords_dict

    def categorize_pages(self, page_keywords):
        print("📌 Categorizing pages based on keywords...")
        keyword_map = defaultdict(lambda: {"page_numbers": [], "text_chunks": []})
        all_keywords = set()
        for page_data in page_keywords.values():
            all_keywords.update(page_data["keywords"])
        sorted_keywords = self.compute_similarity_index(list(all_keywords))
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

class Chunker:
    """Splits text into chunks that fit within the safe token limit."""
    def __init__(self, token_counter, max_context_length, safety_margin):
        self.token_counter = token_counter
        self.safe_length = int(max_context_length * safety_margin)

    def chunk_text(self, text):
        chunks = []
        current_chunk = ""
        current_tokens = 0
        sentences = text.replace(".\n", ". \n").split(". ")
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # If a sentence is too long, split it further by spaces.
            if self.token_counter.count_tokens(sentence) > self.safe_length:
                words = sentence.split()
                subchunk = ""
                for word in words:
                    candidate = subchunk + " " + word if subchunk else word
                    if self.token_counter.count_tokens(candidate) > self.safe_length:
                        if subchunk:
                            chunks.append(subchunk.strip())
                        subchunk = word
                    else:
                        subchunk = candidate
                if subchunk:
                    sentence = subchunk
                else:
                    sentence = ""
            else:
                sentence = sentence + ". "
            if not sentence:
                continue
            sentence_tokens = self.token_counter.count_tokens(sentence)
            if current_tokens + sentence_tokens > self.safe_length:
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

class LMStudioClient:
    """Communicates with LM Studio for summarization."""
    def __init__(self, lm_studio_url, model_name, temperature, max_tokens):
        self.lm_studio_url = lm_studio_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def summarize(self, text):
        if not text.strip():
            return "Error: Empty input text"
        payload = {
            "model": self.model_name,
            "prompt": f"Summarize: {text}",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        try:
            with requests.post(self.lm_studio_url, json=payload, stream=True, timeout=120) as response:
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

class ProgressTracker:
    """Tracks progress of chunk processing."""
    def __init__(self, progress_file):
        self.progress_file = progress_file

    def load_progress(self):
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading progress file: {e}")
        return {"last_processed": 0}

    def save_progress(self, last_processed):
        try:
            with open(self.progress_file, "w", encoding="utf-8") as f:
                json.dump({"last_processed": last_processed}, f)
        except Exception as e:
            print(f"Error saving progress file: {e}")

class GPUMonitor:
    """Monitors GPU temperature to avoid overheating."""
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

class ResponseProcessor:
    """Processes final responses: trims text and re-summarizes."""
    def __init__(self, final_response_dir, final_summary_file, token_counter, lm_client):
        self.final_response_dir = final_response_dir
        self.final_summary_file = final_summary_file
        self.token_counter = token_counter
        self.lm_client = lm_client

    def trim_final_response(self, text):
        """
        Processes the given text by:
          - Splitting it into words.
          - Removing words that are in the reject list.
          - Counting word frequencies.
          - Removing duplicate words.
          - Sorting unique words by frequency (highest first).
          - Trimming the result to MAX_CONTEXT_LENGTH words.
        Returns the trimmed text.
        """
        # Load reject words from the same file used during keyword extraction.
        reject_words = set(load_reject_keywords())
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in reject_words]
        freq = Counter(word.lower() for word in filtered_words)
        unique_words = list(set(word.lower() for word in filtered_words))
        unique_words.sort(key=lambda w: freq[w], reverse=True)
        trimmed_words = unique_words[:MAX_CONTEXT_LENGTH]
        return " ".join(trimmed_words)

    def process_final_responses(self):
        print("🔄 Processing final responses to trim and summarize further...")
        files = [f for f in os.listdir(self.final_response_dir) if f.startswith("final_response_") and f.endswith(".txt")]
        files.sort()
        trimmed_summaries = []
        for file in files:
            path = os.path.join(self.final_response_dir, file)
            text = load_text(path)
            trimmed = self.trim_final_response(text)
            trimmed_path = os.path.join(self.final_response_dir, f"trimmed_{file}")
            save_text(trimmed_path, trimmed)
            print(f"Trimmed final response saved to {trimmed_path}")
            summary = self.lm_client.summarize(trimmed)
            trimmed_summaries.append(summary)
        combined_trimmed = "\n\n".join(trimmed_summaries)
        existing_final = load_text(self.final_summary_file) if os.path.exists(self.final_summary_file) else ""
        new_final = (existing_final + "\n\n" + combined_trimmed) if existing_final else combined_trimmed
        save_text(self.final_summary_file, new_final)
        print(f"Final trimmed summary appended to {self.final_summary_file}")


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

# === Function: Store LM Studio Response per Chunk ===
def store_lm_response(chunk_index, response):
    filename = os.path.join(INTERMEDIATE_SUMMARY_DIR, f"lm_response_{chunk_index}.txt")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"Stored LM Studio response for chunk {chunk_index} in {filename}")
    except Exception as e:
        print(f"Error storing LM Studio response for chunk {chunk_index}: {e}")


class SummarizationPipeline:
    """Orchestrates the entire summarization process."""
    def __init__(self):
        self.pdf_processor = PDFProcessor(PDF_FILE, PAGE_TEXT_FILE)
        self.keyword_extractor = KeywordExtractor(REJECT_KEYWORDS_FILE, ALLOWED_KEYWORDS_FILE,
                                                   TOP_GLOBAL_KEYWORDS, TOP_PAGE_KEYWORDS)
        self.token_counter = TokenCounter(TOKENIZER_NAME)
        self.chunker = Chunker(self.token_counter, MAX_CONTEXT_LENGTH, SAFETY_MARGIN)
        self.lm_client = LMStudioClient(LM_STUDIO_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS)
        self.progress_tracker = ProgressTracker(PROGRESS_FILE)
        self.gpu_monitor = GPUMonitor(MAX_GPU_TEMP)
        self.response_processor = ResponseProcessor(INTERMEDIATE_SUMMARY_DIR, FINAL_SUMMARY_FILE,
                                                    self.token_counter, self.lm_client)

    def process_pdf(self):
        # Step 1: Convert PDF to text.
        return self.pdf_processor.extract_text()

    def extract_and_categorize_keywords(self, text_data):
        # Steps 2 & 3: Extract keywords and categorize pages.
        page_keywords = self.keyword_extractor.extract_keywords(text_data)
        self.keyword_extractor.categorize_pages(page_keywords)
        return page_keywords

    def combine_text(self, text_data):
        # Step 4: Combine text in page order.
        sorted_pages = sorted(text_data.keys(), key=lambda x: int(x.split()[1]))
        combined_text = "\n\n".join(text_data[page] for page in sorted_pages)
        save_text(INPUT_FILE, combined_text)
        print(f"✔ Combined text saved to {INPUT_FILE}")
        return combined_text

    def generate_intermediate_summary(self):
        # Steps 5 & 6: Chunk the combined text and generate intermediate summaries.
        if os.path.exists(INTERMEDIATE_FILE):
            print(f"Intermediate summary file {INTERMEDIATE_FILE} exists. Skipping intermediate summarization.")
            return
        print("Generating intermediate summary from combined text...")
        text = load_text(INPUT_FILE)
        chunks = self.chunker.chunk_text(text)
        print(f"Total chunks created from combined text: {len(chunks)}")
        summaries = []
        progress = self.progress_tracker.load_progress()
        last_processed = progress.get("last_processed", 0)
        for i, chunk in enumerate(chunks):
            if i < last_processed:
                print(f"Skipping chunk {i+1}/{len(chunks)} (already processed).")
                continue
            print(f"Processing chunk {i+1}/{len(chunks)} for intermediate summary...")
            if not self.gpu_monitor.check_temperature():
                print(f"Cooling down for {COOL_DOWN_PERIOD} seconds due to high GPU temperature.")
                time.sleep(COOL_DOWN_PERIOD)
                if not self.gpu_monitor.check_temperature():
                    self.progress_tracker.save_progress(i)
                    self.gpu_monitor.force_stop()
                    return
            summary = self.lm_client.summarize(chunk)
            store_lm_response(i, summary)
            summaries.append(summary if not summary.startswith("Error:") else "[Summarization Failed]")
            self.progress_tracker.save_progress(i+1)
            time.sleep(1)
        intermediate_summary = "\n\n".join(summaries)
        save_text(INTERMEDIATE_FILE, intermediate_summary)
        print(f"Intermediate summary saved to {INTERMEDIATE_FILE}")

    def generate_final_summary(self):
        # Steps 7 & 8: Generate final summary from intermediate summary.
        print("Creating final summary from intermediate summary...")
        intermediate_text = load_text(INTERMEDIATE_FILE)
        final_chunks = self.chunker.chunk_text(intermediate_text)
        print(f"Total final chunks created: {len(final_chunks)}")
        final_summaries = []
        for i, chunk in enumerate(final_chunks):
            print(f"Processing final chunk {i+1}/{len(final_chunks)}...")
            final_resp = self.lm_client.summarize(chunk)
            final_resp_filename = os.path.join(INTERMEDIATE_SUMMARY_DIR, f"final_response_{i}.txt")
            save_text(final_resp_filename, final_resp)
            final_summaries.append(final_resp)
            time.sleep(1)
        combined_final = "\n\n".join(final_summaries)
        if len(final_summaries) > 1:
            print("Performing final combination summarization step...")
            final_summary = self.lm_client.summarize(combined_final)
        else:
            final_summary = combined_final
        save_text(FINAL_SUMMARY_FILE, final_summary)
        print(f"Final summary saved to {FINAL_SUMMARY_FILE}")

    def process_final_responses(self):
        # Step 9: Trim and process final responses and append to final summary.
        self.response_processor.process_final_responses()

    def run(self):
        text_data = self.process_pdf()
        self.extract_and_categorize_keywords(text_data)
        self.combine_text(text_data)
        self.generate_intermediate_summary()
        self.generate_final_summary()
        self.process_final_responses()

###############################
#         MAIN EXECUTION      #
###############################
if __name__ == "__main__":
    pipeline = SummarizationPipeline()
    pipeline.run()
