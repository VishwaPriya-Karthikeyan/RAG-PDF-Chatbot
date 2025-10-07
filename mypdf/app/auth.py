import streamlit as st
import hashlib
import json
import os

class AuthSystem:
    def __init__(self):
        self.users_file = "data/users.json"
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default users if users file doesn't exist"""
        os.makedirs("data", exist_ok=True)
        
        if not os.path.exists(self.users_file):
            default_users = {
                "admin": {
                    "password": self._hash_password("admin123"),
                    "role": "admin",
                    "email": "admin@company.com"
                },
                "user1": {
                    "password": self._hash_password("user123"),
                    "role": "user", 
                    "email": "user1@company.com"
                },
                "user2": {
                    "password": self._hash_password("user123"),
                    "role": "user",
                    "email": "user2@company.com"
                }
            }
            self._save_users(default_users)
    
    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _load_users(self):
        try:
            with open(self.users_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_users(self, users):
        with open(self.users_file, 'w') as f:
            json.dump(users, f, indent=2)
    
    def authenticate(self, username, password):
        users = self._load_users()
        if username in users:
            hashed_input = self._hash_password(password)
            if users[username]["password"] == hashed_input:
                return users[username]["role"]
        return None
    
    def create_user(self, username, password, role, email=""):
        """Admin function to create new users"""
        users = self._load_users()
        if username in users:
            return False, "Username already exists"
        
        users[username] = {
            "password": self._hash_password(password),
            "role": role,
            "email": email
        }
        self._save_users(users)
        return True, f"User {username} created successfully"
    
    def get_all_users(self):
        """Get all users (admin only)"""
        return self._load_users()

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'login_attempted' not in st.session_state:
        st.session_state.login_attempted = False

def render_login(auth_system):
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("🔐 RAG System Login")
    st.write("Please enter your credentials to access the system")
    
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("🚀 Login")
        
        if submitted:
            st.session_state.login_attempted = True
            if username and password:
                role = auth_system.authenticate(username, password)
                if role:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role
                    st.session_state.current_user = username
                    st.success(f"✅ Welcome {username}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
            else:
                st.warning("⚠️ Please enter both username and password")
    
    # Display demo credentials
    st.markdown("---")
    st.subheader("Demo Credentials:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Admin Account:**")
        st.code("Username: admin\nPassword: admin123")
    
    with col2:
        st.markdown("**User Accounts:**")
        st.code("Username: user1/user2\nPassword: user123")
    
    st.markdown('</div>', unsafe_allow_html=True)