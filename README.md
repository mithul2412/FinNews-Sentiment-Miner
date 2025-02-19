# FinNews Sentiment Miner

## Description
FinNews Sentiment Miner is a system designed to perform stock market sentiment analysis by leveraging natural language processing (NLP) and web mining techniques. It collects financial news data through web scraping APIs like Finnhub, Marketaux, NewsAPI, then uses NLTK-based sentiment analysis and named entity recognition to transform the raw text into actionable insights. By integrating these sentiment metrics into trading strategies, the system can potentially inform more profitable decisions. Currently, the project is undergoing development to incorporate a transformer-based architecture for question-answering functionality, aiming to enhance the depth and speed of insights extracted from financial news.

## Key Features
- **Financial News Collection**: Automatic scraping and API integration to gather up-to-date market news.
- **NLP-based Analysis**: Utilizes NLTK for text preprocessing, sentiment analysis, and named entity recognition.
- **Text Processing Pipeline**: Converts raw text into vectorized features suitable for various ML models.
- **Machine Learning Integration**: Applies classification and regression models to capture sentiment impact on market movements.
- **Algorithmic Trading Potential**: Demonstrates improved trading decisions when integrated with quantitative trading strategies.
- **Ongoing Work**: Currently integrating transformer-based question-answering components for deeper, context-aware insights.

## Technologies Used
- **Python**: Core language for data collection, preprocessing, and model building.
- **NLTK**: Used for tokenization, stemming, and sentiment analysis.
- **Web Scraping & APIs**: Gathers real-time financial data from multiple online sources.
- **Machine Learning**: Various ML algorithms for sentiment classification and impact analysis.
- **Transformers**: Developing a question-answering module to provide more detailed responses to specific user queries.
