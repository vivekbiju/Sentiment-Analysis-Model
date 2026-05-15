import os
import mlflow
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def train():

    # This ensures the folder and the .yaml "map" are created first
    tracking_path = os.path.join(os.getcwd(), "mlruns")
    if not os.path.exists(tracking_path):
        os.makedirs(tracking_path)
    mlflow.set_tracking_uri(f"file:///{tracking_path}")

    experiment_name = "Internet_Slang_Sentiment"
    
    # Force create the experiment to generate meta.yaml if it's missing
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"Creating fresh experiment: {experiment_name}")
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="DistilBERT_Slang_V1"):
        # 1. Load Data
        df = pd.read_csv("data/slang_reviews.csv")
        dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
        
        # 2. Tokenizer & Model
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        tokenized_datasets = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128), batched=True)
        
        # 3. Training Args (report_to="none" avoids conflicts)
        training_args = TrainingArguments(
            output_dir="./models/checkpoints",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            report_to="none", #log manually for total control
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
        )
        
        print("🚀 Training...")
        train_result = trainer.train()
        
        # 4. Manual Logging - The most reliable method for your portfolio
        mlflow.log_metrics(train_result.metrics)
        mlflow.set_tag("context", "internet_slang")
        
        # 5. Save Artifacts
        save_path = "./models/sentiment_model"
        trainer.save_model(save_path)
        mlflow.log_artifacts(save_path, artifact_path="model")
        
        print(f"✅ Finished! Check MLflow UI now.")

if __name__ == "__main__":
    train()