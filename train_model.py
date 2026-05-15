import os
import mlflow
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer




def train():
    # --- ULTRA-ROBUST PATH LOGIC ---
    # This looks for the 'data' folder in the directory where the script is running
    # This works better for GitHub Actions 'steps'
    root_path = Path(os.getcwd())
    data_path = root_path / "data" / "slang_reviews.csv"
    tracking_path = root_path / "mlruns"
    
    # Ensure directories exist
    tracking_path.mkdir(parents=True, exist_ok=True)
    
    # Set MLflow
    mlflow.set_tracking_uri(tracking_path.as_uri())
    
    # Check for data file
    if not data_path.exists():
        # Let's print exactly where we are looking so we can debug if it fails
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Looking for data at: {data_path}")
        raise FileNotFoundError(f"Missing data file at {data_path}")  
            
    df = pd.read_csv(data_path)
    dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
    
    # 2. Tokenizer & Model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128), batched=True)
    
    # 3. Training Args
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        eval_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=1, # Reduced for CI/CD speed
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    print("🚀 Training...")
    train_result = trainer.train()
    
    # 4. Logging
    mlflow.log_metrics(train_result.metrics)
    
    
    print(f"✅ Finished! Check MLflow UI.")
# --- 5. Save Artifacts (FIXED) ---
        # Define the path using Path logic
    save_path = root_path / "models" / "sentiment_model"
        
    # Ensure the directory exists before saving
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving model to: {save_path}")
    trainer.save_model(str(save_path))
    mlflow.log_artifacts(str(save_path), artifact_path="model")
    
if __name__ == "__main__":
    train()