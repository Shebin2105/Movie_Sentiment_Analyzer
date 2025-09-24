# Movie Sentiment Analyzer

## Overview
The **Movie Sentiment Analyzer** is a Streamlit-based web application that allows users to analyze movie reviews and determine their sentiment — whether positive or negative. Leveraging state-of-the-art NLP techniques with transformer-based models (BERT), the application provides accurate real-time sentiment predictions.

This project is ideal for anyone interested in **natural language processing**, **sentiment analysis**, or building **interactive AI-powered apps**.

---

## Features
- **User-Friendly Interface**: Input any movie review and get instant sentiment analysis.  
- **Real-Time Predictions**: The model processes text quickly and returns results in real time.  
- **Visual Sentiment Distribution**: See graphical representations of sentiment scores and trends.  
- **Scalable NLP Model**: Built using a fine-tuned BERT model for high accuracy.  
- **Local and Cloud Ready**: Can be run locally or deployed to cloud platforms.

---

## Dataset
The model is trained on a **movie review dataset** containing 50,000 reviews labeled as positive or negative. Preprocessing includes **tokenization**, **padding**, and **encoding** to make it compatible with the BERT-based model.

---

## Technologies Used
- **Python 3.x**  
- **Streamlit** for web interface  
- **Hugging Face Transformers** for BERT-based model  
- **PyTorch** for deep learning  
- **Pandas & NumPy** for data handling  
- **Matplotlib / Seaborn** for visualizations  

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Shebin2105/Movie_Sentiment_Analyzer.git
   cd Movie_Sentiment_Analyzer
2.**Instal Dependencies**
pip install -r requirements.txt

3.Run the Streamlit app

streamlit run app.py

4.Open your browser at http://localhost:8501
 and start analyzing movie reviews.

##Usage

Type or paste a movie review into the input box.

Click Analyze.

The app will display the sentiment as Positive or Negative, along with a confidence score.

Visual charts show sentiment distribution for multiple reviews.

##Model Details

Architecture: BERT (Bidirectional Encoder Representations from Transformers)

Fine-Tuning: The model is fine-tuned on the movie review dataset for sentiment classification.

File: movie_sentiment_model/model.safetensors (tracked with Git LFS due to large size)