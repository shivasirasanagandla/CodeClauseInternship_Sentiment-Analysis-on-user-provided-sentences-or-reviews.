import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
from textblob import TextBlob
import re

# Disable the warning about st.pyplot() usage
st.set_option('deprecation.showPyplotGlobalUse', False)

# Text Data Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    # Add more preprocessing steps such as removing stopwords, stemming, etc. for better accuracy
    return text

# Analyzing Text Statistics
def analyze_text_statistics(text):
    word_count = len(text.split())
    char_count = len(text)
    avg_word_length = char_count / (word_count + 1e-8)
    return word_count, char_count, avg_word_length

# Sentiment Extraction using TextBlob
def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Generate Wordcloud
def generate_wordcloud(text, sentiment):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Wordcloud for {sentiment} Sentiment')
    plt.axis('off')
    st.pyplot()

def main():
    st.title("Sentiment Analysis")
    
    # Adding an image
    st.image("1686416663880.jpg", use_column_width=True)  # Replace "your_image.png" with the path to your image file
    
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an option", ["General EDA", "Analyze Sentiment", "Generate Wordcloud"])
    
    if option == "General EDA":
        st.header("General Exploratory Data Analysis")
        text_input = st.text_area("Enter the text")
        if st.button("Analyze"):
            if text_input:
                st.write("Input Text:", text_input)
                preprocessed_text = preprocess_text(text_input)
                st.write("Preprocessed Text:", preprocessed_text)
                word_count, char_count, avg_word_length = analyze_text_statistics(preprocessed_text)
                st.write("Word Count:", word_count)
                st.write("Character Count:", char_count)
                st.write("Average Word Length:", avg_word_length)
                st.write("Top 10 Unigrams:", Counter(preprocessed_text.split()).most_common(10))
    
    elif option == "Analyze Sentiment":
        st.header("Sentiment Analysis")
        text_input = st.text_area("Enter the text")
        if st.button("Analyze Sentiment"):
            if text_input:
                st.write("Input Text:", text_input)
                preprocessed_text = preprocess_text(text_input)
                st.write("Preprocessed Text:", preprocessed_text)
                sentiment = analyze_sentiment_textblob(preprocessed_text)
                st.write("Sentiment:", sentiment)
    
    elif option == "Generate Wordcloud":
        st.header("Generate Wordcloud")
        text_input = st.text_area("Enter the text")
        if st.button("Generate"):
            if text_input:
                st.write("Input Text:", text_input)
                preprocessed_text = preprocess_text(text_input)
                st.write("Preprocessed Text:", preprocessed_text)
                sentiment = analyze_sentiment_textblob(preprocessed_text)
                generate_wordcloud(preprocessed_text, sentiment)

    st.header("Word Frequency Analysis")
    if st.button("Analyze Word Frequency"):
        if text_input:
            word_freq = Counter(text_input.split())
            words, frequencies = zip(*word_freq.items())
            plt.figure(figsize=(10, 6))
            plt.bar(words, frequencies)
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title('Word Frequency')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot()

if __name__ == "__main__":
    main()
