import sys
import os
import logging
import streamlit as st

# 1. Setup Path (Crucial for importing src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_engine import SentimentEngine
from src.data_processor import ReviewInput, TextCleaner

# 2. Configure Logging
logging.basicConfig(level=logging.INFO)

# 3. Page Layout 
st.set_page_config(page_title="Sentiment AI Pro", layout="wide", page_icon="😌")

# 4. Model Loading (Singleton Pattern)
@st.cache_resource
def get_model():
    engine = SentimentEngine()
    engine.load_inference_pipeline()
    return engine

engine = get_model()
cleaner = TextCleaner()

# 5. Initialize Session State
# This ensures variables like 'prediction' don't cause NameErrors when the app reruns
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'clean_text' not in st.session_state:
    st.session_state.clean_text = ""
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""

# 6. Sidebar Implementation
with st.sidebar:
    st.header("⚙️ Model Settings")
    st.info("Model: DistilBERT-SST2")
    st.caption("Fine-tuned for high-performance inference.")
    
    st.divider()
    st.subheader("📝 Quick Examples")
    
    # Callback to update the text area when a button is clicked
    def set_example(text):
        st.session_state.user_text = text

    if st.button("The movie was mid, honestly."):
        set_example("The movie was mid, honestly.")
    if st.button("That final battle was fire! 🔥"):
        set_example("That final battle was fire! 🔥")
    if st.button("Absolute masterpiece. 10/10"):
        set_example("Absolute masterpiece. 10/10")

# 7. Main UI
st.title("🚀 Production-Grade Sentiment Analysis")
st.markdown("""
    This system utilizes a **DistilBERT Transformer** to classify review sentiment. 
    It includes a robust preprocessing pipeline featuring **Pydantic validation** and **Regex sanitization**.
""")

# Text input uses session_state.user_text to allow sidebar buttons to work
user_input = st.text_area("Enter Customer Review:", 
                          value=st.session_state.user_text, 
                          height=150, 
                          placeholder="Type your review here...")

if st.button("Analyze Sentiment", type="primary"):
    try:
        # Step A: Validation
        valid_data = ReviewInput(text=user_input)
        
        # Step B: Sanitization
        cleaned = cleaner.sanitize(valid_data.text)
        st.session_state.clean_text = cleaned
        
        # Step C: Inference
        with st.spinner("Running Transformer Inference..."):
            results = engine.predict_batch([cleaned])
            if results:
                st.session_state.prediction = results[0]
            else:
                raise Exception("Model returned no results.")

    except ValueError as e:
        st.warning(f"⚠️ Validation Error: {e}")
        st.session_state.prediction = None # Reset results on error
    except Exception as e:
        st.error(f"❌ System Error: {e}")
        st.session_state.prediction = None

# 8. Results Display Section (Only runs if a prediction exists in state)
if st.session_state.prediction:
    pred = st.session_state.prediction
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        label = pred["label"]
        color = "green" if label == "POSITIVE" else "red"
        st.markdown(f"### Predicted Sentiment: :{color}[{label}]")
        
        # Logic for Probability Chart
        pos_score = pred['score'] if label == "POSITIVE" else 1 - pred['score']
        neg_score = 1 - pos_score
        
        chart_data = {
            "Label": ["Positive", "Negative"],
            "Confidence": [pos_score, neg_score]
        }
        st.bar_chart(chart_data, x="Label", y="Confidence", color="#4682B4")

    with col2:
        st.metric("Model Confidence Score", f"{pred['score']:.4f}")
        st.progress(pred["score"])
        
        with st.expander("🔍 View Technical Metadata"):
            st.write(f"**Sanitized Input:** `{st.session_state.clean_text}`")
            st.write(f"**Raw Model Output:**")
            st.json(pred)

# 9. Footer
st.divider()
st.caption("MSc Artificial Intelligence Portfolio Project | Built with Hugging Face & Streamlit")