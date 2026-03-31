import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import os

def train():
    # 1. Load your custom slang data
    df = pd.read_csv("data/slang_reviews.csv")
    dataset = Dataset.from_pandas(df)
    
    # Split into Train and Test (80/20)
    dataset = dataset.train_test_split(test_size=0.2)
    
    # 2. Setup Tokenizer
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        # We use a shorter max_length for slang reviews to speed up training
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Load Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # 4. Define Training Arguments
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        eval_strategy="epoch",        # Corrected parameter name
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3, 
        weight_decay=0.01,
        logging_dir='./logs',
        save_total_limit=1,           # Prevents filling up your disk with checkpoints
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"], # Corrected from evaluation_dataset
        processing_class=tokenizer,              # Associates tokenizer with the trainer
    )
    
    # 6. TRAIN!
    print("🚀 Starting Fine-Tuning on Slang Data...")
    trainer.train()
    
    # 7. SAVE THE "BRAIN"
    save_path = "./models/sentiment_model"
    os.makedirs(save_path, exist_ok=True)
    
    # This saves the model weights AND the tokenizer config
    trainer.save_model(save_path) 
    
    print(f"✅ Custom Model saved to {save_path}")

if __name__ == "__main__":
    train()