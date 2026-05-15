import sys
import os
import logging
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import shap
import streamlit.components.v1 as components

# 1. Setup Path & Environment
load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_engine import SentimentEngine
from src.data_processor import ReviewInput, TextCleaner
from src.agent import SentimentAgent 

# 2. Configure Logging
logging.basicConfig(level=logging.INFO)

# 3. Page Configuration
st.set_page_config(page_title="Agentic Multimodal Sentiment AI", layout="wide", page_icon="🤖")

# 4. Resource Loading
@st.cache_resource
def load_resources():
    engine = SentimentEngine()
    engine.load_inference_pipeline()
    try:
        agent = SentimentAgent()
    except Exception as e:
        st.error(f"Error initializing Agent: {e}")
        agent = None
    return engine, agent

engine, agent = load_resources()
cleaner = TextCleaner()

# 5. Fixed SHAP Functions
def st_shap(plot_obj, height=200):
    import uuid
    try:
        unique_id = f"shap_{uuid.uuid4().hex[:8]}"
        
        # Get the HTML content
        if hasattr(plot_obj, "html"):
            plot_html = plot_obj.html()
        else:
            plot_html = str(plot_obj)
            
        final_html = f"""
            <div id="{unique_id}" style="color: black; background-color: white;">
                <script>{shap.getjs()}</script>
                <style>
                    /* Force SHAP colors if the library CSS fails to load */
                    .shap-text-value {{ font-weight: bold; }}
                    svg {{ font-family: sans-serif; }}
                </style>
                {plot_html}
            </div>
        """
        components.html(final_html, height=height, scrolling=True)
    except Exception as e:
        st.error(f"Visualization error: {e}")


def render_shap_heatmap(text_input, pipeline):
    with st.spinner("🔍 Generating XAI Heatmap..."):
        try:
            explainer = shap.Explainer(pipeline)
            shap_values = explainer([text_input])
            # display=False returns the HTML/Explanation object instead of plotting to screen
            plot_obj = shap.plots.text(shap_values[0], display=False)
            st_shap(plot_obj)
        except Exception as e:
            st.error(f"XAI Heatmap Error: {str(e)}")

# 6. Session State
if 'user_text' not in st.session_state:
    st.session_state.user_text = ""

# 7. Sidebar
with st.sidebar:
    st.header("⚙️ System Architecture")
    st.status("DistilBERT (Local)", state="complete")
    st.status("Gemini 1.5 Flash (Cloud Agent)", state="complete")
    st.divider()
    st.subheader("💡 Test Scenarios")
    for phrase in ["This movie is fire 🔥", "The service was mid 💀", "Absolute banger of a track."]:
        if st.button(phrase):
            st.session_state.user_text = phrase

# 8. Main UI
st.title("🛡️ Agentic Multimodal Sentiment System")
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📝 Input Layer")
    user_input = st.text_area("Enter Text:", value=st.session_state.user_text, height=150)
    uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

# 9. Execution
if st.button("🚀 Run Deep Analysis", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("🕵️ Processing..."):
            clean_text = cleaner.sanitize(user_input)
            prediction = engine.predict_batch([clean_text])[0]
            reasoning = agent.explain_sentiment(user_input, prediction['label'], uploaded_file) if agent else "Agent offline."
        
        with col_right:
            st.subheader("📊 Analysis Results")
            color = "green" if prediction["label"] == "POSITIVE" else "red"
            st.markdown(f"### Sentiment: :{color}[{prediction['label']}]")
            st.metric("Confidence Score", f"{prediction['score']:.4f}")
            st.progress(prediction["score"])
            
            st.markdown("#### 🔍 Feature Attribution (SHAP)")
            render_shap_heatmap(clean_text, engine.pipeline)
            
            st.markdown("#### 🤖 Agentic Explanation")
            st.info(reasoning)

# 10. Footer
st.divider()
st.caption("AI Engineer Portfolio | Built with DistilBERT, Gemini 1.5, & SHAP")