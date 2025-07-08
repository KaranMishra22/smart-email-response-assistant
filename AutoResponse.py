import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime
import re

st.set_page_config(
    page_title="Smart Email Response Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
    }
    .response-box {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 20px 0;
    }
    .stat-box {
        text-align: center;
        padding: 20px;
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 10px;
        min-width: 150px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'email_history' not in st.session_state:
    st.session_state.email_history = []

@st.cache_resource
def load_sentiment_model():
    """Load pre-trained BERT model for sentiment analysis"""
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def analyze_sentiment(text, tokenizer, model):
    """Analyze sentiment of input text"""
    if not tokenizer or not model:
        return "neutral", 0.5
    
    try:
        # Preprocess text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)     # Remove mentions
        text = text.strip()
        
        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Map labels: 0=negative, 1=neutral, 2=positive
        labels = ['negative', 'neutral', 'positive']
        sentiment_scores = predictions[0].tolist()
        
        predicted_sentiment = labels[torch.argmax(predictions, dim=-1).item()]
        confidence = max(sentiment_scores)
        
        return predicted_sentiment, confidence
    except Exception as e:
        st.error(f"Error in sentiment analysis: {e}")
        return "neutral", 0.5

def generate_response(subject, content, sentiment, confidence):
    """Generate appropriate email response based on sentiment"""
    
    # Response templates based on sentiment
    templates = {
        'positive': {
            'greeting': "Thank you for your positive feedback!",
            'body': "We're delighted to hear about your positive experience. Your satisfaction is our top priority, and it's wonderful to know we've met your expectations.",
            'closing': "We look forward to continuing to serve you with excellence."
        },
        'negative': {
            'greeting': "Thank you for bringing this to our attention.",
            'body': "We sincerely apologize for any inconvenience you may have experienced. Your feedback is invaluable to us, and we take all concerns seriously. We would like to resolve this matter promptly.",
            'closing': "Please feel free to contact us directly so we can address your concerns and make things right."
        },
        'neutral': {
            'greeting': "Thank you for your message.",
            'body': "We appreciate you taking the time to contact us. We have received your inquiry and will ensure it receives the appropriate attention.",
            'closing': "If you have any additional questions or concerns, please don't hesitate to reach out."
        }
    }
    
    template = templates.get(sentiment, templates['neutral'])
    
    # Create personalized response
    if 'complaint' in content.lower() or 'problem' in content.lower():
        urgency = "We understand this matter is important to you and will prioritize it accordingly."
    elif 'thank' in content.lower() or 'appreciate' in content.lower():
        urgency = "Your kind words mean a lot to our team."
    else:
        urgency = "We value your communication and will respond appropriately."
    
    response = f"""Subject: Re: {subject}

Dear Valued Customer,

{template['greeting']}

{template['body']} {urgency}

{template['closing']}

Best regards,
Customer Service Team

---
This response was generated based on sentiment analysis (Confidence: {confidence:.2%})
Detected sentiment: {sentiment.upper()}
"""
    
    return response

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📧 Smart Email Response Assistant</h1>
        <p>Automated email responses powered by BERT sentiment analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading BERT model..."):
        tokenizer, model = load_sentiment_model()
    
    if not tokenizer or not model:
        st.error("Failed to load sentiment analysis model. Please check your internet connection.")
        return
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Email Input")
        
        # Input fields
        subject = st.text_input("Email Subject", placeholder="Enter email subject...")
        content = st.text_area("Email Content", height=200, placeholder="Enter the email content here...")
        
        # Analysis button
        if st.button("🔍 Analyze & Generate Response", type="primary"):
            if subject and content:
                # Analyze sentiment
                sentiment, confidence = analyze_sentiment(content, tokenizer, model)
                
                # Store in session state
                st.session_state.current_analysis = {
                    'subject': subject,
                    'content': content,
                    'sentiment': sentiment,
                    'confidence': confidence,
                    'timestamp': datetime.now()
                }
                
                # Add to history
                st.session_state.email_history.append(st.session_state.current_analysis)
                
            else:
                st.warning("Please enter both subject and content.")
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        if hasattr(st.session_state, 'current_analysis'):
            analysis = st.session_state.current_analysis
            
            # Display sentiment with colored background
            sentiment_class = f"sentiment-{analysis['sentiment']}"
            st.markdown(f"""
            <div class="{sentiment_class}">
                <h3>Sentiment: {analysis['sentiment'].upper()}</h3>
                <p>Confidence: {analysis['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Generate and display response
            response = generate_response(
                analysis['subject'], 
                analysis['content'], 
                analysis['sentiment'], 
                analysis['confidence']
            )
            
            st.markdown('<div class="response-box">', unsafe_allow_html=True)
            st.subheader("📧 Generated Response")
            st.text_area("Response", value=response, height=300, key="response_display")
            
            # Copy button simulation
            if st.button("📋 Copy Response"):
                st.success("Response copied to clipboard! (In a real app, this would copy to clipboard)")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistics section
    if st.session_state.email_history:
        st.markdown("---")
        st.subheader("📈 Email Statistics")
        
        # Calculate statistics
        total_emails = len(st.session_state.email_history)
        positive_count = sum(1 for email in st.session_state.email_history if email['sentiment'] == 'positive')
        negative_count = sum(1 for email in st.session_state.email_history if email['sentiment'] == 'negative')
        neutral_count = sum(1 for email in st.session_state.email_history if email['sentiment'] == 'neutral')
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Emails", total_emails)
        with col2:
            st.metric("Positive", positive_count, f"{positive_count/total_emails*100:.1f}%")
        with col3:
            st.metric("Negative", negative_count, f"{negative_count/total_emails*100:.1f}%")
        with col4:
            st.metric("Neutral", neutral_count, f"{neutral_count/total_emails*100:.1f}%")
        
        # Email history
        st.subheader("📋 Email History")
        
        # Create DataFrame for better display
        history_data = []
        for email in st.session_state.email_history[-10:]:  # Show last 10 emails
            history_data.append({
                'Timestamp': email['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Subject': email['subject'][:50] + '...' if len(email['subject']) > 50 else email['subject'],
                'Sentiment': email['sentiment'].upper(),
                'Confidence': f"{email['confidence']:.2%}"
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
    
    # Sidebar with additional info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses BERT (Bidirectional Encoder Representations from Transformers) 
        to analyze email sentiment and generate appropriate responses.
        
        **Features:**
        - Real-time sentiment analysis
        - Automated response generation
        - Email history tracking
        - Performance statistics
        
        **Sentiment Categories:**
        - 🟢 **Positive**: Happy, satisfied customers
        - 🔴 **Negative**: Complaints, issues
        - 🟡 **Neutral**: General inquiries
        """)
        
        st.header("🔧 Settings")
        if st.button("Clear History"):
            st.session_state.email_history = []
            if hasattr(st.session_state, 'current_analysis'):
                delattr(st.session_state, 'current_analysis')
            st.success("History cleared!")
        
        st.header("📊 Model Info")
        st.info("Using: cardiffnlp/twitter-roberta-base-sentiment-latest")

if __name__ == "__main__":
    main()