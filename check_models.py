import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Available Models:")
for m in genai.list_models():
    # Filter for models that support 'embedContent'
    if 'embedContent' in m.supported_generation_methods:
        print(f"Name: {m.name} | Display Name: {m.display_name}")