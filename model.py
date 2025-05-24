# model.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

class GeminiModel:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        load_dotenv()
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.model = genai.GenerativeModel(model_name)

    def chat_completion(self, prompt: str):
        """Mimics OpenAI-style completion call."""
        response = self.model.generate_content(prompt)
        return response.text
