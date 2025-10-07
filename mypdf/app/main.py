import os
import streamlit as st
from auth import AuthSystem, initialize_session_state, render_login
from auth_ops import AdminOperations
from core_rag import RAGEngine
from utilis import VoiceHandler, save_chat_history, load_chat_history

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Enterprise RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Global Constants
# -----------------------------
CHROMA_FOLDER = "data/chroma_db"  # Your default folder for RAGEngine

# -----------------------------
# Initialize components
# -----------------------------
auth_system = AuthSystem()
admin_ops = AdminOperations()
rag_engine = RAGEngine()  # Removed arguments to match core_rag.py
voice_handler = VoiceHandler()

# -----------------------------
# Admin/User Interface Functions
# -----------------------------
def render_user_management():
    st.subheader("👥 User Management")
    
    with st.form("create_user_form"):
        st.write("*Create New User*")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_username = st.text_input("Username")
        with col2:
            new_password = st.text_input("Password", type="password")
        with col3:
            new_role = st.selectbox("Role", ["user", "admin"])
        
        if st.form_submit_button("Create User"):
            if new_username and new_password:
                success, message = auth_system.create_user(new_username, new_password, new_role)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("Please enter username and password")
    
    # Display existing users
    st.subheader("Existing Users")
    users = auth_system.get_all_users()
    for username, user_data in users.items():
        role_icon = "👨‍💼" if user_data["role"] == "admin" else "👤"
        st.write(f"{role_icon} *{username}* - Role: {user_data['role']}")

def render_admin_interface():
    st.title("👨‍💼 Admin Portal")
    
    tab1, tab2 = st.tabs(["📚 PDF Management", "👥 User Management"])
    
    with tab1:
        st.subheader("PDF Upload and Processing")
        uploaded_file = st.file_uploader("Choose PDF file", type="pdf")
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    pdf_path = admin_ops.save_uploaded_pdf(uploaded_file)
                    pdf_name = uploaded_file.name.replace('.pdf', '')
                    success, message = admin_ops.process_pdf(pdf_path, pdf_name)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
        
        st.subheader("📋 Processed Documents")
        processed_pdfs = admin_ops.get_processed_pdfs()
        if processed_pdfs:
            for pdf in processed_pdfs:
                st.write(f"• {pdf}")
            st.info(f"Total: {len(processed_pdfs)} documents")
        else:
            st.info("📭 No PDFs processed yet")
    
    with tab2:
        render_user_management()

def render_user_interface():
    st.title("💬 Chat with Your Documents")
    
    # Display available documents
    processed_pdfs = rag_engine.get_processed_pdfs()
    if processed_pdfs:
        with st.expander("📚 Available Documents"):
            for pdf in processed_pdfs:
                st.write(f"• {pdf}")
    else:
        st.info("📭 No documents available. Please ask admin to upload PDFs.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Voice input
    st.subheader("🎤 Voice Input")
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🎤 Start Voice Input", use_container_width=True):
            voice_text = voice_handler.speech_to_text()
            if voice_text and not voice_text.startswith(("Error", "Timeout")):
                st.session_state.voice_input = voice_text
                st.success(f"🎯 Captured: {voice_text}")
                st.experimental_rerun()
    
    # Text input
    user_input = st.chat_input("💭 Ask a question about your documents...")
    final_input = st.session_state.get("voice_input", user_input)
    
    if final_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": final_input})
        with st.chat_message("user"):
            st.markdown(final_input)
        
        # Get response from RAG
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching across all documents..."):
                answer, context, source = rag_engine.search_all_pdfs(final_input)
                
                if answer and "sorry, i don't have enough context" not in answer.lower():
                    response = f"{answer}\n\n---\n*📚 Source:* {source}"
                    st.markdown(response)
                    with st.expander("🔍 View Source Context"):
                        for i, chunk in enumerate(context[:3]):
                            st.write(f"*Context {i+1}:* {chunk[:200]}...")
                else:
                    response = "❌ I couldn't find relevant information in any documents to answer your question. Please try rephrasing or ask about something else."
                    st.markdown(response)
        
        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        save_chat_history(st.session_state.messages)
        
        # Clear voice input
        if "voice_input" in st.session_state:
            del st.session_state.voice_input
        
        # Optional TTS
        if answer and "sorry, i don't have enough context" not in answer.lower():
            if st.button("🔊 Speak Answer"):
                voice_handler.text_to_speech(answer)

# -----------------------------
# Main App
# -----------------------------
def main():
    initialize_session_state()
    
    if not st.session_state.authenticated:
        render_login(auth_system)
    else:
        # Sidebar
        st.sidebar.title("🧭 Navigation")
        st.sidebar.write(f"👤 Logged in as: *{st.session_state.current_user}*")
        st.sidebar.write(f"🎯 Role: *{st.session_state.user_role}*")
        
        if st.sidebar.button("🚪 Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
        
        # Role-based interface
        if st.session_state.user_role == "admin":
            render_admin_interface()
        else:
            render_user_interface()
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.info(
            "ℹ System Information:\n"
            "- Multi-user authentication\n"
            "- Voice input/output\n" 
            "- Automatic source citation\n"
            "- Cross-document search\n"
            "- Persistent chat history"
        )

if __name__ == "__main__":
    main()
