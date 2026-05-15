from transformers import pipeline

# Load your local fine-tuned model
# Ensure this path matches your 'save_path' from train_model.py
model_path = "./models/sentiment_model"
pipe = pipeline("text-classification", model=model_path)

# Test classic Gen-Z / Internet Slang
test_phrases = [
    "That set was absolute fire!",           # Expected: Positive
    "No cap, the acting was mid at best.",   # Expected: Negative
    "This tutorial is the GOAT."             # Expected: Positive
]

print("🚀 Verifying Slang Model Performance...\n")
for text in test_phrases:
    result = pipe(text)[0]
    print(f"Text: '{text}'")
    print(f"Result: {result['label']} (Confidence: {result['score']:.2f})\n")