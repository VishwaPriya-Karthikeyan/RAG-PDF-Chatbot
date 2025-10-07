# core_rag.py
import os
import re
import numpy as np
from typing import List, Tuple, Optional
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import warnings

# Suppress HuggingFace Transformers warnings if desired
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

class RAGEngine:
    def __init__(self, 
                 chroma_folder: str = "data/chroma_db",
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "google/flan-t5-base",
                 reranker_model_name: Optional[str] = None):

        self.chroma_folder = chroma_folder
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.reranker_model_name = reranker_model_name

        # Initialize embeddings, LLM, and reranker
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)
        self.llm = self._initialize_llm()
        self.reranker = self._initialize_reranker() if self.reranker_model_name else None

        # Other hyperparameters
        self.top_k_chroma = 20
        self.rerank_top_n = 5
        self.cosine_weight = 0.75
        self.lexical_weight = 0.25
        self.hybrid_score_threshold = 0.20
        self.min_words_for_answer=3


    def _init_(self, chroma_folder: str = "data/chroma_db",
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "google/flan-t5-base", # Consider larger models for production
                 reranker_model_name: Optional[str] = None): # e.g., 'cross-encoder/ms-marco-TinyBERT-L-2'
        
        self.chroma_folder = chroma_folder
        self.embed_model_name = embed_model_name
        self.llm_model_name = llm_model_name
        self.reranker_model_name = reranker_model_name

        print(f"Initializing RAG Engine with: \n"
              f"  Embeddings: {self.embed_model_name}\n"
              f"  LLM: {self.llm_model_name}\n"
              f"  Reranker: {self.reranker_model_name if self.reranker_model_name else 'None'}")

        self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)
        self.llm = self._initialize_llm()
        self.reranker = self._initialize_reranker() if self.reranker_model_name else None

        # Retrieval & Reranking hyperparameters
        self.top_k_chroma = 20        # Initial candidates from Chroma (vector search)
        self.rerank_top_n = 5         # Number of top chunks to pass to LLM after hybrid/cross-encoder reranking
        self.cosine_weight = 0.75     # Weight for semantic similarity in hybrid score
        self.lexical_weight = 0.25    # Weight for lexical overlap in hybrid score
        self.hybrid_score_threshold = 0.20 # Min hybrid score for a chunk to be considered relevant (tuned higher)
        self.min_words_for_answer = 3 # Heuristic: if LLM response is too short and not a clear factual extraction, it might be poor.

        print("RAG Engine initialized.")

    def _initialize_llm(self):
        """Initializes the Large Language Model using HuggingFacePipeline."""
        print(f"Loading LLM: {self.llm_model_name}...")
        try:
            # Use device_map="auto" for potentially larger models, or "cuda" for explicit GPU, -1 for CPU/default
            qa_pipeline = pipeline(
                "text2text-generation",
                model=self.llm_model_name,
                device=-1, # Set to 0 or "cuda" for GPU. Using -1 for auto/CPU for broader compatibility.
                max_new_tokens=256,
                do_sample=False, # For more deterministic, factual answers
                temperature=0.1, # Low temperature for less creativity, more factuality
                # no_repeat_ngram_size=2, # Prevents repetitive output, uncomment if needed
            )
            return HuggingFacePipeline(pipeline=qa_pipeline)
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM '{self.llm_model_name}': {e}. "
                               f"Ensure model is available and environment is set up correctly.")

    def _initialize_reranker(self):
        """Initializes a cross-encoder reranker model."""
        if not self.reranker_model_name:
            return None
        print(f"Loading reranker: {self.reranker_model_name}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.reranker_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.reranker_model_name)
            if torch.cuda.is_available():
                model.to('cuda')
            model.eval() # Set model to evaluation mode
            print("Reranker loaded successfully.")
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            print(f"Warning: Failed to load reranker '{self.reranker_model_name}': {e}. Continuing without reranker.")
            return None

    # -----------------------------
    # Scoring functions
    # -----------------------------
    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Calculates cosine similarity between two vectors."""
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def _lexical_overlap(query: str, text: str) -> float:
        """
        Calculates a symmetric lexical overlap score between query and text.
        Filters common short words.
        """
        q_tokens = set([w.lower() for w in re.findall(r"\b\w+\b", query) if len(w) > 2])
        t_tokens = set([w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 2])
        if not q_tokens or not t_tokens:
            return 0.0
        overlap = q_tokens.intersection(t_tokens)
        return len(overlap) / (np.sqrt(len(q_tokens)) * np.sqrt(len(t_tokens))) # Symmetric normalization

    def _hybrid_score(self, query_vec: np.ndarray, doc_vec: np.ndarray, query: str, doc_text: str) -> float:
        """Combines semantic (cosine) and lexical overlap scores."""
        cosine_score = self._cosine_sim(query_vec, doc_vec)
        lexical_score = self._lexical_overlap(query, doc_text)
        return self.cosine_weight * cosine_score + self.lexical_weight * lexical_score

    def _cross_encoder_rerank(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        Reranks documents using a cross-encoder model.
        Returns documents with their predicted relevance scores.
        """
        if not self.reranker or not documents:
            return []

        # Prepare pairs for cross-encoder
        pairs = [[query, doc] for doc in documents]
        
        try:
            with torch.no_grad():
                inputs = self.reranker["tokenizer"](pairs, padding=True, truncation=True, return_tensors='pt').to(self.reranker["model"].device)
                scores = self.reranker["model"](**inputs).logits.squeeze().cpu().numpy()
            
            # Handle single document case where scores might be a scalar
            if scores.ndim == 0:
                scores = np.array([scores.item()])

            ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            return ranked_docs
        except Exception as e:
            print(f"Warning: Cross-encoder reranking failed: {e}. Falling back to hybrid scoring.")
            return []

    # -----------------------------
    # Short answer extraction heuristics
    # -----------------------------
    @staticmethod
    def _extract_year_or_date(context: str) -> Optional[str]:
        """
        Extracts years or more complete dates from the context.
        Prioritizes full date patterns then individual years.
        """
        date_pattern = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|\d{1,2})[/\-.,\s](?:\d{1,2}[/\-.,\s])?\b(?:19|20)\d{2}\b"
        match = re.search(date_pattern, context, re.IGNORECASE)
        if match:
            return match.group(0).strip()
        year_match = re.search(r"\b(19|20)\d{2}\b", context)
        if year_match:
            return year_match.group(0)
        return None

    @staticmethod
    def _extract_name(context: str) -> Optional[str]:
        """
        Attempts to extract names (e.g., "John Doe", "J. Doe", "Alice").
        This is a heuristic and may not catch all name formats.
        """
        # Pattern for multi-part names or capitalized single words, potentially with initials
        name_pattern = r"\b(?:[A-Z]\.?\s?){1,3}(?:[A-Z][a-z]+(?:['\-][A-Z][a-z]+)?\s?)+\b|\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,})*\b"
        match = re.search(name_pattern, context)
        if match:
            return match.group(0).strip()
        return None

    @staticmethod
    def _extract_table_query(context: str, query: str) -> Optional[str]:
        """
        Handles table-like queries (e.g., "What is the name of RegNo 3336?").
        Looks for rows in context containing key identifiers from the query.
        """
        query_lower = query.lower()
        numbers_in_query = re.findall(r"\b\d+\b", query_lower)
        
        # If no numbers and no strong table-like keywords, it's unlikely a table query
        if not numbers_in_query and not any(k in query_lower for k in ["rank of", "details of", "information for", "student with", "id of"]):
            return None

        rows = context.split("\n")
        q_tokens = [w.lower() for w in re.findall(r"\b\w+\b", query_lower) if len(w) > 2 and w not in ["what", "who", "when", "is", "the", "of", "a", "an", "for"]]
        
        best_row, max_overlap = None, -1
        for row in rows:
            row_lower = row.lower()
            # Check if all numbers from the query are present in the row
            if all(num in row_lower for num in numbers_in_query):
                current_overlap = sum(1 for tok in q_tokens if tok in row_lower)
                # Prioritize rows with more query keyword overlap
                if current_overlap > max_overlap:
                    max_overlap = current_overlap
                    best_row = row.strip()
        return best_row

    def _extract_factual(self, context: str, question: str) -> Optional[str]:
        """
        Attempts to extract a concise factual answer using heuristics
        before falling back to LLM generation.
        """
        # 1. Try table-like query first (e.g., for rank lists)
        table_ans = self._extract_table_query(context, question)
        if table_ans:
            return table_ans

        q_lower = question.lower()
        # 2. Specific factual extraction based on query keywords
        if any(k in q_lower for k in ["year", "when", "date"]):
            return self._extract_year_or_date(context)
        if any(k in q_lower for k in ["who", "name"]):
            return self._extract_name(context)
        
        # 3. Fallback: sentence with maximal keyword overlap
        q_tokens = [w.lower() for w in re.findall(r"\b\w+\b", question) if len(w) > 2]
        sentences = re.split(r'(?<=[.!?])\s+', context)
        best_sent, best_count = None, -1
        
        for s in sentences:
            current_count = sum(1 for kw in q_tokens if kw in s.lower())
            # Prioritize sentences that are not too short and have significant overlap
            if current_count > best_count and len(s.split()) > 4: # Min 5 words for a sentence
                best_count = current_count
                best_sent = s
            elif current_count == best_count and best_sent is not None:
                # If tied, and query has numbers, prefer sentences with numbers
                if any(re.search(r"\b\d+\b", tok) for tok in q_tokens) and re.search(r"\b\d+\b", s):
                    best_sent = s

        if best_count >= 1: # At least one overlapping keyword
            # Final check for relevance of extracted factual answer
            # Avoid returning very generic or short irrelevant phrases
            if len(best_sent.split()) < self.min_words_for_answer and not re.search(r'\d', best_sent):
                return None # Too short and no numbers, likely not a specific factual answer
            return best_sent.strip()
        
        return None

    # -----------------------------
    # PDF processing & Retrieval
    # -----------------------------
    def get_processed_pdfs(self) -> List[str]:
        """Returns a list of PDF IDs for which Chroma DBs exist."""
        if not os.path.exists(self.chroma_folder):
            return []
        # Filter for actual directories, which represent individual PDF Chroma DBs
        return [d for d in os.listdir(self.chroma_folder) if os.path.isdir(os.path.join(self.chroma_folder, d))]

    def search_all_pdfs(self, query: str) -> Tuple[str, List[str], str]:
        """
        Performs a RAG search across all processed PDFs.
        Returns the answer, source chunks, and the best PDF ID.
        """
        processed_pdfs = self.get_processed_pdfs()
        if not processed_pdfs:
            # Checkpoint 1: No PDFs loaded at all
            return ("Sorry, I don't have enough context from the PDF.", [], "N/A")

        all_candidate_chunks_for_rerank = []
        
        try:
            query_vec = np.array(self.embeddings.embed_query(query))
        except Exception as e:
            print(f"Warning: Could not embed query '{query}': {e}. Using zero vector.")
            # Determine embedding dimension dynamically or use a common default like 384 for MiniLM
            embed_dim = self.embeddings.client.get_sentence_embedding_dimension() if hasattr(self.embeddings.client, 'get_sentence_embedding_dimension') else 384
            query_vec = np.zeros(embed_dim, dtype=float)

        for pdf_id in processed_pdfs:
            persist_dir = os.path.join(self.chroma_folder, pdf_id)
            try:
                # Ensure the Chroma DB is valid and accessible
                db = Chroma(persist_directory=persist_dir, embedding_function=self.embeddings)
                # Quick test query to ensure DB is functional
                _ = db.similarity_search("test", k=1) 
            except Exception as e:
                print(f"Warning: Could not load or access Chroma DB for '{pdf_id}' at '{persist_dir}': {e}. Skipping this PDF.")
                continue

            try:
                # Retrieve initial candidates based on embedding similarity
                # Filter out documents with None scores if any
                docs_with_scores = db.similarity_search_with_score(query, k=self.top_k_chroma)
                candidates = [d for d, s in docs_with_scores if d.page_content and s is not None]
            except Exception as e:
                print(f"Warning: Error during similarity search for '{pdf_id}': {e}. Skipping this PDF's retrieval.")
                candidates = []

            if not candidates:
                continue
            
            candidate_texts = [d.page_content for d in candidates]
            
            # Calculate hybrid scores for these candidates
            try:
                doc_vecs = self.embeddings.embed_documents(candidate_texts)
            except Exception as e:
                print(f"Warning: Could not embed documents for '{pdf_id}': {e}. Using zero vectors for these documents.")
                doc_vecs = [np.zeros(query_vec.shape[0], dtype=float) for _ in candidate_texts]

            docs_meta = []
            for i, txt in enumerate(candidate_texts):
                vec = np.array(doc_vecs[i], dtype=float)
                score = self._hybrid_score(query_vec, vec, query, txt)
                docs_meta.append({"text": txt, "score": float(score), "pdf_id": pdf_id})
            
            # Add to global list for potential global reranking
            all_candidate_chunks_for_rerank.extend(docs_meta)
            
        if not all_candidate_chunks_for_rerank:
            # Checkpoint 2: No relevant chunks retrieved across all PDFs
            return ("Sorry, I don't have enough context from the PDF.", [], "N/A")

        # Global Reranking Phase
        final_selected_chunks_info = []
        if self.reranker:
            # Prepare texts for cross-encoder reranking (from the globally gathered candidates)
            texts_for_cross_encoder = [item["text"] for item in all_candidate_chunks_for_rerank]
            cross_encoder_ranked = self._cross_encoder_rerank(query, texts_for_cross_encoder)
            
            if cross_encoder_ranked:
                # Map cross-encoder scores back to original metadata
                # A more advanced approach might blend cross-encoder and hybrid scores
                cross_encoder_map = {text: score for text, score in cross_encoder_ranked}
                
                # Re-sort all candidates based on cross-encoder scores (or hybrid if CE score not found)
                sorted_by_cross_encoder = sorted(
                    all_candidate_chunks_for_rerank, 
                    key=lambda x: cross_encoder_map.get(x["text"], x["score"]), # Fallback to hybrid score
                    reverse=True
                )
                final_selected_chunks_info = sorted_by_cross_encoder[:self.rerank_top_n]
            else:
                # Cross-encoder failed or returned empty, fall back to hybrid-sorted list
                all_candidate_chunks_for_rerank.sort(key=lambda x: x["score"], reverse=True)
                final_selected_chunks_info = all_candidate_chunks_for_rerank[:self.rerank_top_n]
        else:
            # No cross-encoder, sort purely by hybrid score
            all_candidate_chunks_for_rerank.sort(key=lambda x: x["score"], reverse=True)
            final_selected_chunks_info = all_candidate_chunks_for_rerank[:self.rerank_top_n]

        if not final_selected_chunks_info:
             # Checkpoint 2.5: Reranking resulted in no selected chunks
            return ("Sorry, I don't have enough context from the PDF.", [], "N/A")

        # Get the highest scored chunk's score for relevance thresholding
        top_chunk_score = final_selected_chunks_info[0]["score"]
        if top_chunk_score < self.hybrid_score_threshold:
            # Checkpoint 3: Top retrieved chunk is below minimum relevance threshold
            return ("Sorry, I don't have enough context from the PDF.", [], "N/A")

        # Assemble best chunks and identify the source PDF for the top chunk
        best_chunks = [info["text"] for info in final_selected_chunks_info]
        best_pdf_id = final_selected_chunks_info[0]["pdf_id"] # The PDF ID of the top chunk

        context_combined = "\n".join(best_chunks)
        
        # 1. Attempt factual extraction first
        factual_answer = self._extract_factual(context_combined, query)
        if factual_answer:
            return (factual_answer, best_chunks, best_pdf_id)

        # 2. Fallback to LLM generation
        prompt_template = """
You are an intelligent and helpful assistant. Your primary task is to provide concise and accurate answers based SOLELY on the provided context.
If the answer or sufficient information to formulate an answer is NOT explicitly present in the context, you MUST respond with the exact phrase: "Sorry, I don't have enough context from the PDF."
Do not make up information, guess, or use any outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = LLMChain(llm=self.llm, prompt=prompt)

        try:
            ctx_for_llm = self._synthesize_for_llm(best_chunks)
            if not ctx_for_llm: 
                # Checkpoint 4: Context provided to LLM became empty (e.g., due to max_chars limit)
                return ("Sorry, I don't have enough context from the PDF.", [], "N/A")
            
            ans = qa_chain.run({"context": ctx_for_llm, "question": query})
            
            # Checkpoint 5: LLM's own admission of lacking context (final safety net)
            if ans and re.search(r"sorry|don't have enough context|cannot find|not mentioned|not provided", ans, flags=re.IGNORECASE):
                return ("Sorry, I don't have enough context from the PDF.", best_chunks, best_pdf_id)
            
            # Heuristic: If LLM gives a very short answer, it might be a weak response.
            # This is less reliable than direct LLM admission.
            if ans and len(ans.strip().split()) < self.min_words_for_answer + 1 and not factual_answer:
                # If it's a very short answer and not clearly a factual extraction, reconsider.
                # This could be tuned or removed if it leads to false negatives.
                pass 
            
            # Final valid answer
            return (ans.strip(), best_chunks, best_pdf_id)
        
        except Exception as e:
            print(f"Error during LLM generation for query '{query}': {str(e)}")
            return ("Sorry, I encountered an internal error during processing. Please try again.", best_chunks, best_pdf_id)

    @staticmethod
    def _synthesize_for_llm(chunks: List[str], max_chars: int = 3500) -> str:
        """
        Combines selected chunks into a single string for the LLM,
        respecting a maximum character limit.
        Adds clear separators between chunks.
        """
        out = []
        current_len = 0
        for i, c in enumerate(chunks):
            chunk_to_add = c.strip()
            # Add a clear separator to distinguish chunks for the LLM
            if i > 0:
                chunk_to_add = "\n--- Document Snippet Start ---\n" + chunk_to_add + "\n--- Document Snippet End ---\n"
            
            if current_len + len(chunk_to_add) > max_chars:
                break
            out.append(chunk_to_add)
            current_len += len(chunk_to_add)
            
        return "\n".join(out).strip()


# Example Usage & Dummy Setup:
if __name__ == "__main__":
    # --- DUMMY CHROMA DB SETUP (Uncomment to run if you don't have a DB yet) ---
    # This section demonstrates how you would typically ingest PDFs and create
    # a Chroma DB. You'll likely have a separate script for this in your full system.
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from fpdf import FPDF
    import shutil

    def setup_dummy_chroma_db(embed_model: str, chroma_base_path: str = "data/chroma_db"):
        if os.path.exists(chroma_base_path):
            print(f"Removing existing dummy DB at {chroma_base_path}...")
            shutil.rmtree(chroma_base_path)
        os.makedirs(chroma_base_path, exist_ok=True)
        
        # Define a PDF ID for the dummy PDF
        pdf_id = "sample_notes_pdf"
        pdf_db_path = os.path.join(chroma_base_path, pdf_id)
        os.makedirs(pdf_db_path, exist_ok=True)
        
        # Create a dummy PDF with varied content (notes, some tabular-like info)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        text1 = (
            "This is a sample document for testing the RAG chatbot. "
            "It contains information about artificial intelligence and machine learning, and its history. "
            "AI was founded in 1956 at a workshop held at Dartmouth College. John McCarthy is considered a founder. "
            "The main applications are natural language processing and computer vision. A key concept is neural networks. "
            "Deep learning is a subset of machine learning. The conference was in July 1956."
        )
        pdf.multi_cell(0, 10, text1)
        
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        text2 = (
            "Another document part. This talks about student achievements and exam results. "
            "A list of students for 2023: \n"
            "RegNo: 1001, Name: Alice Smith, Score: 85, Rank: 1\n"
            "RegNo: 1002, Name: Bob Johnson, Score: 92, Rank: 2\n"
            "RegNo: 1003, Name: Carol White, Score: 78, Rank: 3\n"
            "The top student is Bob Johnson with a score of 92."
        )
        pdf.multi_cell(0, 10, text2)

        pdf_path = "./temp_sample.pdf"
        pdf.output(pdf_path)

        # Load, split, and embed documents for the dummy PDF
        print(f"Loading and processing {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        texts = text_splitter.split_documents(documents)

        embeddings_model = HuggingFaceEmbeddings(model_name=embed_model)
        Chroma.from_documents(texts, embeddings_model, persist_directory=pdf_db_path)
        
        os.remove(pdf_path) # Clean up dummy PDF file
        print(f"Dummy Chroma DB created for '{pdf_id}' in {pdf_db_path}")

    # Set up dummy DB. Make sure the embed_model_name matches the one used in RAGEngine.
    # setup_dummy_chroma_db("sentence-transformers/all-MiniLM-L6-v2") 
    # Uncomment the line above to create a dummy Chroma DB for testing

    # --- RAG ENGINE INITIALIZATION ---
    rag_engine = RAGEngine(
        chroma_folder="data/chroma_db",
        llm_model_name="google/flan-t5-base", # Small model, good for local testing. For production, consider larger like "google/flan-t5-xxl"
        # Optional: Uncomment the line below to enable a cross-encoder reranker
        # reranker_model_name='cross-encoder/ms-marco-TinyBERT-L-2'
    )

    print("\nRAG Engine initialized. Start chatting or type 'exit' to quit.")
    print("--- Test Queries ---")
    print("  - What is AI?")
    print("  - Who founded AI?")
    print("  - When was the AI conference?")
    print("  - What is the name of RegNo 1002?")
    print("  - What is Bob Johnson's score?")
    print("  - What is deep learning?")
    print("  - Tell me about ancient history.") # Should trigger "Sorry..."
    print("  - What is the capital of France?") # Should trigger "Sorry..."
    print("--------------------")

    # --- CHAT LOOP ---
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break

        response, sources, pdf_id = rag_engine.search_all_pdfs(user_query)
        print(f"Bot: {response}")
        if sources:
            print(f"  (Source PDF: {pdf_id})")
            # print(f"  (Top Chunks: {sources})") # Uncomment to see retrieved chunks for debugging
        else:
            print("  (No relevant source documents found.)")

    # Example of triggering an image (integrate this into your UI/interface logic)
    # This part would typically be handled by your main application logic,
    # detecting specific user intents for visual content.
    if "diagram of rag" in user_query.lower() or "flow of rag" in user_query.lower():
        print("Here's a visual representation of a RAG system's data flow: ")
