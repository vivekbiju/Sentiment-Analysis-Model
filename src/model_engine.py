import os
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Configure logging for the engine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentEngine:
    def __init__(self, model_path=None):
        """
        Initializes the Sentiment Engine.
        :param model_path: Path to the fine-tuned DistilBERT model.
        """
        # Default path relative to the project root
        self.model_path = model_path or os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "models", "sentiment_model"
        ))
        
        # This is the attribute the SHAP explainer in app.py is looking for
        self.pipeline = None 

    def load_inference_pipeline(self):
        """
        Loads the model and tokenizer into a Hugging Face pipeline.
        """
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"⚠️ Model path not found at {self.model_path}. Loading default DistilBERT.")
                # Fallback to base model if your fine-tuned one isn't found
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            else:
                logger.info(f"🚀 Loading fine-tuned model from {self.model_path}")
                model_name = self.model_path

            # Crucial: Assign the pipeline to self.pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=False  # SHAP works best with the standard output
            )
            logger.info("✅ Sentiment pipeline loaded successfully.")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise e

    def predict_batch(self, texts):
        """
        Runs inference on a list of strings.
        """
        if not self.pipeline:
            raise RuntimeError("Pipeline not loaded. Call load_inference_pipeline() first.")
        
        try:
            results = self.pipeline(texts)
            return results
        except Exception as e:
            logger.error(f"❌ Inference error: {str(e)}")
            return [{"label": "ERROR", "score": 0.0}]

if __name__ == "__main__":
    # Quick internal test
    engine = SentimentEngine()
    engine.load_inference_pipeline()
    test_text = "This project is absolute fire no cap!"
    print(f"Test Result: {engine.predict_batch([test_text])}")