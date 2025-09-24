import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import functional as F
import plotly.express as px
import pandas as pd

# 🎨 Page configuration
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main background and theme */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 400% 400%;
        animation: gradientShift 6s ease infinite;
        padding: 2rem 1rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: bold;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Arial Black', sans-serif;
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Input area styling */
    .input-container {
        background: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
        color: white;
        animation: slideIn 0.5s ease-out;
    }
    
    .positive-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .negative-card {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sentiment-emoji {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .sentiment-text {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        padding: 1rem 3rem;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: bold;
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.3);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 1rem;
        font-size: 1.1rem;
        min-height: 120px;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #4ECDC4;
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
    }
    
    /* Examples styling */
    .examples-container {
        background: rgba(255,255,255,0.1);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .example-button {
        background: rgba(255,255,255,0.2);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        margin: 0.3rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    .example-button:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-1px);
    }
    
    /* Stats container */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
    
    .stat-item {
        text-align: center;
        color: rgba(255,255,255,0.9);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
</style>
""", unsafe_allow_html=True)

# 1️⃣ Path to your extracted model folder
MODEL_PATH = "./movie_sentiment_model"

# 2️⃣ Load model and tokenizer
@st.cache_resource  # cache to avoid reloading every time
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# 3️⃣ Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = probs.argmax().item()
    label = model.config.id2label[pred_id]
    confidence = probs[0][pred_id].item()
    return label, confidence, probs[0].tolist()

# 4️⃣ Beautiful Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🎬 Movie Sentiment Analyzer</h1>
    <p class="header-subtitle">Discover the emotional tone of movie reviews with AI-powered analysis</p>
</div>
""", unsafe_allow_html=True)

# 5️⃣ Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    st.markdown("### 📝 Enter Your Movie Review")
    user_input = st.text_area(
        "",
        placeholder="Type your movie review here... For example: 'This movie was absolutely amazing! The acting was superb and the plot kept me engaged throughout.'",
        height=120,
        key="review_input"
    )
    
    # Prediction button
    predict_button = st.button("🔮 Analyze Sentiment", type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
   # Example reviews section
st.markdown("### 💡 Try These Examples")
examples = [
    "This movie was absolutely fantastic! Amazing cinematography and brilliant acting.",
    "Terrible movie. Waste of time and money. Poor acting and boring plot.",
    "The movie was okay, nothing special but not bad either.",
    "Incredible masterpiece! One of the best films I've ever seen.",
    "Disappointing sequel. Expected much more from this franchise."
]

# Callback function to safely update the text area
def set_example(example_text):
    st.session_state.review_input = example_text

cols = st.columns(3)
for i, example in enumerate(examples):
    with cols[i % 3]:
        st.button(
            f"📄 Example {i+1}", 
            key=f"example_{i}", 
            help=example, 
            on_click=set_example, 
            args=(example,)  # pass the example text to the callback
        )


with col2:
    st.markdown("### 📊 Analysis Results")
    
    if predict_button or user_input.strip():
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review first!")
        else:
            with st.spinner("🤖 Analyzing sentiment..."):
                label, confidence, all_probs = predict_sentiment(user_input)
                
                # Determine card style based on sentiment
                if label.upper() == "POSITIVE":
                    card_class = "result-card positive-card"
                    emoji = "😊"
                else:
                    card_class = "result-card negative-card"
                    emoji = "😞"
                
                # Display result card
                st.markdown(f"""
                <div class="{card_class}">
                    <span class="sentiment-emoji">{emoji}</span>
                    <div class="sentiment-text">{label.upper()}</div>
                    <div class="confidence-text">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence visualization
                st.markdown("#### 📈 Confidence Breakdown")
                
                # Create a simple bar chart for probabilities
                if len(all_probs) >= 2:
                    prob_data = {
                        'Sentiment': ['Negative', 'Positive'],
                        'Probability': all_probs
                    }
                    df = pd.DataFrame(prob_data)
                    
                    fig = px.bar(
                        df, 
                        x='Sentiment', 
                        y='Probability',
                        color='Sentiment',
                        color_discrete_map={'Positive': '#38ef7d', 'Negative': '#ff6a00'},
                        title="Sentiment Probabilities"
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=300,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional metrics
                st.markdown("#### 📋 Review Stats")
                word_count = len(user_input.split())
                char_count = len(user_input)
                
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("Word Count", word_count)
                with col_stats2:
                    st.metric("Character Count", char_count)
    else:
        st.info("👆 Enter a movie review above to see the sentiment analysis!")
        
        # Display some info about the analyzer
        st.markdown("""
        #### ℹ️ About This Analyzer
        
        This AI-powered tool analyzes the sentiment of movie reviews using advanced natural language processing. 
        
        **Features:**
        - 🎯 High accuracy sentiment detection
        - 📊 Confidence scores and probability breakdown
        - ⚡ Real-time analysis
        - 🎨 Beautiful, interactive interface
        
        Simply paste or type any movie review to get started!
        """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 3rem; padding: 2rem; color: rgba(255,255,255,0.7);">
    <p>Built with ❤️ using Streamlit and Transformers | AI-Powered Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)