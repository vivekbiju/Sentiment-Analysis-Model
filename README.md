# 🚀 Production-Grade Sentiment Analysis: Slang-Adapted DistilBERT

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-DistilBERT-yellow.svg)](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
[![Testing](https://img.shields.io/badge/Testing-Pytest-060a08.svg)](https://docs.pytest.org/)

## 🌟 Overview
Traditional sentiment analysis models often struggle with modern internet slang (e.g., "fire," "mid," "GOAT," "bussin"). This project addresses that gap by implementing an end-to-end Machine Learning pipeline that features a **custom-tuned DistilBERT model** specialized for informal, high-slang English.

This repository demonstrates a **modular software architecture** suitable for production environments, moving beyond simple scripts into a scalable AI service.

---

## 🏗️ Project Architecture
The project follows a decoupled architecture to ensure maintainability and testability:

* **`app/`**: Interactive Streamlit UI for real-time inference.
* **`src/`**: Core logic including `model_engine` (Inference/Local Loading) and `data_processor` (Validation/Sanitization).
* **`data/`**: Curated slang benchmark dataset (50+ hand-labeled rows).
* **`models/`**: Local storage for fine-tuned Transformer weights (The "Local Brain").
* **`notebooks/`**: Exploratory Data Analysis (EDA) and Research.
* **`tests/`**: Unit tests for data integrity.

---

## 🛠️ Key Engineering Features

### 1. Custom Model Fine-Tuning ("The Pro State")
Rather than relying solely on a remote API, this project includes a `train_model.py` script. It uses the Hugging Face `Trainer` API to fine-tune DistilBERT on a specialized slang dataset.
* **Local Loading:** The system automatically detects if a fine-tuned "Local Brain" exists in `/models` and prioritizes it over the remote model.
* **Optimization:** Uses `accelerate` for efficient memory management during training.

### 2. Robust Data Validation
Uses **Pydantic** to enforce strict data types and **Regex** sanitization to handle:
* HTML tag removal.
* Whitespace normalization.
* Repeated character reduction (e.g., "sooooo goooood" → "soo good").

### 3. Automated Testing
Comprehensive testing with **Pytest** ensures that the preprocessing pipeline never breaks. 
* Run tests via: `python -m pytest`

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/[YOUR_USERNAME]/project-2-sentiment-analysis.git
cd project-2-sentiment-analysis
pip install -r requirements.txt


2.Training the custom model
python train_model.py

3. Running the App
streamlit run app/app.py

```bash
Evaluation & Research
The notebooks/ folder contains a detailed research paper in notebook format.

EDA: Analysis of slang distribution and review lengths.

Benchmarking: Comparative analysis showing how fine-tuning improves accuracy on "slang-heavy" inputs compared to the base DistilBERT model.


Deployment
The project is container-ready with a Dockerfile.
docker build -t sentiment-slang-app .
docker run -p 8501:8501 sentiment-slang-app

```bash
Academic Context
This project was developed as part of an MSc in Artificial Intelligence portfolio. It demonstrates mastery of:

Transformer Architectures (NLP)

MLOps (Model Versioning & Deployment)

Software Engineering Best Practices (Modular Code, CI/CD Readiness)