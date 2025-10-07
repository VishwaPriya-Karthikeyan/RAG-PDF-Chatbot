import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class AdminOperations:
    def __init__(self, pdf_folder="data/pdfs", chroma_folder="data/chroma_db"):
        self.pdf_folder = pdf_folder
        self.chroma_folder = chroma_folder
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create directories if they don't exist
        os.makedirs(self.pdf_folder, exist_ok=True)
        os.makedirs(self.chroma_folder, exist_ok=True)
    
    def save_uploaded_pdf(self, uploaded_file):
        """Save uploaded PDF to the pdfs folder"""
        file_path = os.path.join(self.pdf_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    
    def process_pdf(self, pdf_path, pdf_name):
        """Process PDF and create vector embeddings [citation:1][citation:3]"""
        try:
            # Load and split PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Split documents into chunks [citation:3][citation:9]
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            if not chunks:
                return False, "No text content found in PDF"
            
            # Create vector store
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=os.path.join(self.chroma_folder, pdf_name)
            )
            vector_store.persist()
            
            return True, f"Successfully processed {len(chunks)} chunks"
            
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"
    
    def get_processed_pdfs(self):
        """Get list of all processed PDFs"""
        if not os.path.exists(self.chroma_folder):
            return []
        return [f for f in os.listdir(self.chroma_folder) 
                if os.path.isdir(os.path.join(self.chroma_folder, f))]