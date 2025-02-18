import requests
import nltk
import re
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import word_tokenize, pos_tag, ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download required NLTK data (only needs to be run once)
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def fetch_news(api_key, category="general", count=20):
    """
    Fetch live financial news articles from Finnhub.
    Finnhub API endpoint: https://finnhub.io/api/v1/news?category={category}&token=API_KEY
    """
    url = f"https://finnhub.io/api/v1/news?category={category}&token={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching data from Finnhub API")
    articles = response.json()
    return articles[:count]

def preprocess_text(text):
    """
    Clean text by removing extra whitespace and unwanted characters.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)  # Remove unwanted characters
    return text.strip()

def tokenize_text(text):
    """
    Tokenize text into words using NLTK.
    """
    tokens = word_tokenize(text)
    return tokens

def perform_ner(text):
    """
    Perform Named Entity Recognition using NLTK's ne_chunk.
    Returns a list of tuples: (entity, entity_type).
    """
    tokens = tokenize_text(text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags)
    entities = []
    for chunk in chunks:
        if hasattr(chunk, 'label'):
            entity = " ".join(c[0] for c in chunk)
            entities.append((entity, chunk.label()))
    return entities

def analyze_sentiment(text, analyzer):
    """
    Analyze sentiment using NLTK's VADER.
    Returns a sentiment label ("positive", "negative", or "neutral")
    along with the compound sentiment score.
    """
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "positive", compound
    elif compound <= -0.05:
        return "negative", compound
    else:
        return "neutral", compound

def main():
    # Set your Finnhub API key here
    api_key = "YOUR_FINNHUB_API_KEY"
    
    # Fetch live news articles (e.g., general financial news)
    articles = fetch_news(api_key, category="general", count=20)
    
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # List to hold processed article data
    records = []
    
    for article in articles:
        # Combine headline and summary (or description) from the article
        headline = article.get("headline", "")
        summary = article.get("summary", "")
        combined_text = " ".join([headline, summary])
        combined_text = preprocess_text(combined_text)
        if not combined_text:
            continue
        
        # Tokenization (for demonstration and use in NER)
        tokens = tokenize_text(combined_text)
        
        # Named Entity Recognition (NER)
        entities = perform_ner(combined_text)
        
        # Sentiment Analysis
        sentiment_label, sentiment_compound = analyze_sentiment(combined_text, analyzer)
        
        records.append({
            "headline": headline,
            "text": combined_text,
            "tokens": tokens,
            "entities": entities,
            "sentiment_label": sentiment_label,
            "sentiment_compound": sentiment_compound
        })
    
    # Create a DataFrame from the records for inspection and further processing
    df = pd.DataFrame(records)
    print("=== Sample News Data ===")
    print(df.head(), "\n")
    
    # --- Machine Learning Pipeline ---
    # Convert text to TF-IDF vector embeddings (this performs our vectorization)
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(df["text"])
    y = df["sentiment_label"]
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Logistic Regression classifier to predict sentiment labels
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    
    # Evaluate the classifier on the test set
    y_pred = classifier.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
