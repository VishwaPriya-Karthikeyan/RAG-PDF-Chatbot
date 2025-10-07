Enterprise RAG PDF Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) system that lets you chat with multiple PDFs — like rank lists, notes, research papers, or institutional data — using both text and voice input.

Built with Streamlit, LangChain, and Hugging Face Transformers, it allows admins to upload PDFs and users to query them conversationally.


Features

1.Multi-PDF ingestion & vector-based storage

2.Accurate hybrid retrieval (semantic + keyword)

3.Context-aware answer generation

4.Voice input & text-to-speech output

5.Multi-user authentication (admin/user roles)

6.Persistent chat history per session

7.Streamlit chat interface with citation view

Project Structure

RAG-Project/
│
├── mypdf/
│   └── app/
│       ├── main.py          # Streamlit entry point
│       ├── core_rag.py      # Core RAG engine (retrieval + generation)
│       ├── auth.py          # Authentication logic
│       ├── auth_ops.py      # Admin user management & PDF ops
│       ├── utilis.py        # Voice handling & helpers
│
├── myenv/                   # Local virtual environment (ignored)
├── requirements.txt         # Dependencies list
├── .gitignore
└── README.md


 Installation Guide

1. Clone the repository

git clone https://github.com/VishwaPriya-Karthikeyan/RAG-PDF-Chatbot.git
cd RAG-PDF-Chatbot/mypdf/app

2. Create a virtual environment

(recommended for clean dependency management)

python -m venv myenv

3. Activate the environment

Windows (CMD / PowerShell):

myenv\Scripts\activate

macOS / Linux:

source myenv/bin/activate


4. Install dependencies

pip install -r requirements.txt

 Running the Application

Once everything’s set up and the environment is active:

streamlit run main.py

The app will launch at:
👉 http://localhost:8501


👨‍💼 For Admin Users

Admins can:

Upload and process new PDFs

Create new users (admin/user roles)

View all processed documents


Steps:

1. Log in as an admin.


2. Go to 📚 PDF Management


3. Upload and process your files.


4. The processed PDFs become available to all users.




---

👩‍💻 For Normal Users

Users can:

Browse available documents

Ask questions via text or voice

Hear generated answers with text-to-speech

View context sources used for answers


 Where Data is Stored

ChromaDB folder → stores PDF embeddings for retrieval.

Chat history → saved per session for continuity.

PDF uploads → stored in a dedicated folder (admin controlled).


Example Queries

Example Question	Expected Response

What is the CGPA of register number 311315251097?	Retrieves from table in ug_aff_2020.pdf
Which institution has the highest CGPA?	Gives institution name from uploaded rank list
What are the key points from unit 2 notes?	Summarized answer from notes PDF



 Technologies Used

1.Python 3.12+

2.LangChain

3.Hugging Face Transformers

4.Sentence Transformers (for embeddings)

5.ChromaDB (vector database)

6.SpeechRecognition & gTTS (voice input/output)

7.Streamlit (UI framework)

Future Improvements

Fine-tuned re-ranking model for factual precision

Multi-language query support

Cloud database for multi-session persistence

Document analytics dashboard


Author

Vishwa Priya.K
Sharmila.L
Adhithian.A
