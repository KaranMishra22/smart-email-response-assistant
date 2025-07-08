# 📧 Smart Email Response Assistant

A powerful sentiment-based email response automation system built with **BERT** and **Streamlit**. This application analyzes email sentiment in real-time and generates appropriate professional responses automatically.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)

## 🚀 Features

- **🧠 BERT-Powered Sentiment Analysis**: Uses pre-trained RoBERTa model for accurate emotion detection
- **⚡ Real-time Processing**: Instant sentiment analysis and response generation
- **🎨 Beautiful UI**: Modern, responsive Streamlit interface with gradient backgrounds
- **📊 Analytics Dashboard**: Track email history, sentiment distribution, and confidence scores
- **🔄 Smart Response Generation**: Context-aware templates for positive, negative, and neutral sentiments
- **📈 Performance Metrics**: Real-time statistics and confidence scoring
- **💾 Session Management**: Email history tracking with timestamp logging


## 📸 Screenshots

### Main Interface
![Main Interface](assests/screenshots/Screenshot%202025-07-08%20164937.png)

### Sentiment Analysis in Action
![Sentiment Analysis](assests/screenshots/Screenshot%202025-07-08%20165031.png)

### Email Analytics Dashboard
![Analytics Dashboard](assests/screenshots/Screenshot%202025-07-08%20165140.png)

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Model**: HuggingFace Transformers (RoBERTa)
- **Backend**: Python, PyTorch
- **Data Processing**: Pandas, Regular Expressions
- **UI Styling**: Custom CSS with gradients and responsive design

## 📋 Prerequisites

- Python 3.7 or higher
- pip package manager
- Internet connection (for model download on first run)

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/KaranMishra22/smart-email-response-assistant.git
cd smart-email-response-assistant
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run the application**:
```bash
streamlit run AutoResponse.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🎯 Usage

### Basic Usage

1. **Launch the app**: Run `streamlit AutoResponse.py`
2. **Enter email details**: Input subject and content in the left panel
3. **Analyze sentiment**: Click "Analyze & Generate Response"
4. **Review results**: See sentiment analysis and generated response
5. **Track history**: Monitor email statistics in the dashboard


## 🎨 Features Breakdown

### Sentiment Analysis
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Categories**: Positive, Negative, Neutral
- **Confidence Scoring**: Real-time confidence percentages
- **Text Preprocessing**: URL removal, mention cleaning

### Response Generation
- **Template System**: Context-aware response templates
- **Personalization**: Dynamic content based on email keywords
- **Professional Tone**: Appropriate business communication style
- **Urgency Detection**: Automatic priority assessment

### Analytics Dashboard
- **Real-time Statistics**: Email count, sentiment distribution
- **History Tracking**: Timestamp-based email logging
- **Visual Feedback**: Color-coded sentiment display
- **Performance Metrics**: Confidence score tracking

## 🔍 How It Works

1. **Input Processing**: Email content is cleaned and preprocessed
2. **Sentiment Analysis**: BERT model analyzes emotional tone
3. **Response Generation**: Template system creates appropriate reply
4. **Result Display**: Sentiment, confidence, and response shown
5. **History Update**: Email added to session tracking

## 📬 Contact

**Karan N.**  
📧 karann23cb@psnacet.edu.in  
🌐 [LinkedIn](#) | [GitHub](#)
