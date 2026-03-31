import torch
import logging
import os
from transformers import pipeline

class SentimentEngine:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.logger = logging.getLogger(__name__)
        
        # 1. PATH LOGIC: Look for your "Pro State" local model first
        # We go up one level from src/ to root, then into models/sentiment_model
        base_path = os.path.dirname(os.path.abspath(__file__))
        local_path = os.path.join(base_path, "..", "models", "sentiment_model")
        
        if os.path.exists(local_path) and os.path.isdir(local_path):
            self.model_name = local_path
            self.logger.info(f"🚀 Using LOCAL CUSTOM model from: {local_path}")
        else:
            self.model_name = model_name
            self.logger.info(f"🌐 Local model not found. Using REMOTE model: {model_name}")

        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = None

    def load_inference_pipeline(self):
        """Loads the model for fast prediction (Inference)"""
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                truncation=True,
                max_length=512
            )
            self.logger.info(f"Model loaded successfully on device {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def predict_batch(self, texts: list, batch_size: int = 32):
        try:
            if not self.classifier:
                self.load_inference_pipeline()

            if not texts or not isinstance(texts, list) or texts[0] is None:
                 return [{"label": "EMPTY", "score": 0.0}]

            results = self.classifier(texts, batch_size=batch_size, truncation=True)
            return results if results is not None else [{"label": "UNKNOWN", "score": 0.0}]

        except Exception as e:
            print(f"!!! ACTUAL MODEL ERROR: {e}") 
            self.logger.error(f"Prediction error: {e}")
            return [{"label": "ERROR", "score": 0.0}]

    def benchmark_on_custom_data(self, csv_path: str):
        """Checks performance on your specific slang dataset"""
        import pandas as pd
        from sklearn.metrics import accuracy_score
    
        if not os.path.exists(csv_path):
            self.logger.error(f"CSV not found at {csv_path}")
            return 0.0

        df = pd.read_csv(csv_path)
        texts = df['text'].tolist()
        results = self.predict_batch(texts)
    
        preds = [1 if r['label'] == 'POSITIVE' else 0 for r in results]
        actuals = df['label'].tolist()
    
        accuracy = accuracy_score(actuals, preds)
        self.logger.info(f"🎯 Accuracy on slang dataset: {accuracy:.2%}")
        return accuracy