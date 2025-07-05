import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import re
import string
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Download necessary NLTK data
nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Function to perform sentiment analysis
def perform_sentiment_analysis(text):
    return sia.polarity_scores(text)['compound']

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    return wordcloud

# Function for keyword extraction using TF-IDF
def extract_keywords(texts, num_keywords=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:num_keywords]
    return [word for word, score in keywords]

# Function for topic modeling
def perform_topic_modeling(texts, num_topics=3, num_words=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        words = [feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]
        topics.append(words)
    return topics

# Streamlit UI
st.title("Interactive Text Analysis App")
if "execution_count" not in st.session_state:
    st.session_state.execution_count = 0

# Increment execution count
st.session_state.execution_count += 1

# Display execution count
st.sidebar.write(f"🔄 Execution Count: {st.session_state.execution_count}")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.write(df.head())
    
    if 'Text' in df.columns:
        df['Cleaned_Text'] = df['Text'].apply(clean_text)

        # Sentiment Analysis
        st.header("Sentiment Analysis")
        df['Sentiment'] = df['Cleaned_Text'].apply(perform_sentiment_analysis)
        st.write(df[['Text', 'Sentiment']])
        st.bar_chart(df['Sentiment'])

        # Word Cloud
        st.header("Word Cloud")
        wc = generate_wordcloud(df['Cleaned_Text'])
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        # Keyword Extraction
        st.header("Keyword Extraction")
        keywords = extract_keywords(df['Cleaned_Text'])
        st.write(", ".join(keywords))

        # Topic Modeling
        st.header("Topic Modeling")
        num_topics = st.slider("Number of Topics:", 2, 10, 3)
        if st.button("Perform Topic Modeling"):
            try:
                topics = perform_topic_modeling(df['Cleaned_Text'], num_topics)
                for i, topic in enumerate(topics):
                    st.write(f"**Topic {i+1}:** {', '.join(topic)}")
            except Exception as e:
                st.error(f"Error in topic modeling: {e}")
else:
    st.info("Please upload a CSV file containing a 'Text' column.")

st.markdown("🚀 Built with [Streamlit](https://streamlit.io/)")