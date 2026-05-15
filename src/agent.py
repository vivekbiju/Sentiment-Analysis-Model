import os
import demoji
from google import genai
from PIL import Image

class SentimentAgent:
    def __init__(self, api_key=None):
        """
        Initializes the Gemini Client. 
        In production, the key is pulled from environment variables.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(" GEMINI_API_KEY not found. Agent cannot start.")
        
        self.client = genai.Client(api_key=self.api_key)
        # Ensure emoji codes are ready for multimodal text processing
        demoji.download_codes()

    def explain_sentiment(self, text, sentiment_label, image_file=None):
        """
        The Agentic & Multimodal Layer: 
        Explains WHY the slang or image leads to the detected sentiment.
        """
        # Step 1: Pre-process text (convert emojis to descriptions for better reasoning)
        text_with_emoji_desc = demoji.replace(text, " ")

        prompt = f"""
        You are a UK-based AI Analyst specializing in Gen-Z communication and internet slang.
        
        Input Text: "{text_with_emoji_desc}"
        Local Model Prediction: {sentiment_label}
        
        Task:
        1. Identify and explain specific slang (e.g., 'mid', 'no cap', 'fire').
        2. Justify why the sentiment is {sentiment_label}.
        3. If an image is provided, explain the 'visual vibe' and if it contains irony.
        4. Suggest a culturally relevant, professional response for a UK business.
        
        Keep the analysis concise and expert.
        """

        # Step 2: Build the multimodal content list
        content_list = [prompt]
        if image_file:
            img = Image.open(image_file)
            content_list.append(img)

        # Step 3: Call Gemini 1.5 Flash 8B (Your verified high-speed model)
        try:
            response = self.client.models.generate_content(
                model="gemini-flash-latest",
                contents=content_list
            )
            return response.text
        except Exception as e:
            return f"❌ Agent Error: {str(e)}"