import streamlit as st
import speech_recognition as sr
import pyttsx3
import tempfile
import os

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize text-to-speech engine"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            return engine
        except:
            return None
    
    def speech_to_text(self):
        """Convert speech to text [citation:4]"""
        try:
            with sr.Microphone() as source:
                st.info("🎙️ Listening... Speak now")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=10)
                
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "Timeout: No speech detected"
        except sr.UnknownValueError:
            return "Could not understand audio"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def text_to_speech(self, text):
        """Convert text to speech"""
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                st.error(f"TTS Error: {str(e)}")

def save_chat_history(messages, file_path="chat_history.json"):
    """Save chat history to file"""
    import json
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")

def load_chat_history(file_path="chat_history.json"):
    """Load chat history from file"""
    import json
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []