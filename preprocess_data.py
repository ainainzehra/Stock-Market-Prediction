import os
import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

def clean_text(text):
    text = re.sub(r"^b['\"]|['\"]$", "", text)
    text = re.sub(r"\s+b['\"]", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def compute_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Handle division by zero
    rs = np.where(loss == 0, np.inf, gain / loss)
    rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Return as pandas Series with original index to ensure alignment during transform
    return pd.Series(rsi, index=data.index)

def process_reddit_data():
    print("Processing Reddit Sentiment Data...")
    df_reddit = pd.read_csv("datasets/reddit_daily_composite.csv")
    df_reddit['Date'] = pd.to_datetime(df_reddit['Date'])
    
    print("  - Cleaning text...")
    df_reddit['Clean_Headline'] = df_reddit['Composite_Headline'].apply(clean_text)
    
    print("  - Computing VADER sentiment...")
    analyzer = SentimentIntensityAnalyzer()
    
    def get_vader(text):
        return analyzer.polarity_scores(text)
        
    vader_scores = df_reddit['Clean_Headline'].apply(get_vader).apply(pd.Series)
    df_reddit = pd.concat([df_reddit, vader_scores], axis=1)
    
    print("  - Computing TextBlob sentiment...")
    def get_textblob(text):
        blob = TextBlob(text)
        return pd.Series({'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity})
        
    blob_scores = df_reddit['Clean_Headline'].apply(get_textblob)
    df_reddit = pd.concat([df_reddit, blob_scores], axis=1)
    
    # Drop raw text
    df_reddit.drop(columns=['Composite_Headline', 'Clean_Headline'], inplace=True)
    return df_reddit

def process_financial_data(df_reddit):
    print("Processing S&P 500 Financial Data...")
    df_sp500 = pd.read_csv("datasets/sp500_2008_2020_clean.csv")
    df_sp500['Date'] = pd.to_datetime(df_sp500['Date'], format='%d/%m/%Y')
    
    print("  - Reindexing to continuous daily frequency per ticker...")
    date_range = pd.date_range(start=df_sp500['Date'].min(), end=df_sp500['Date'].max(), freq='D')
    
    tickers = df_sp500['Ticker'].unique()
    multi_idx = pd.MultiIndex.from_product([date_range, tickers], names=['Date', 'Ticker'])
    
    df_sp500.set_index(['Date', 'Ticker'], inplace=True)
    df_sp500 = df_sp500.reindex(multi_idx)
    
    print("  - Forward filling missing prices...")
    df_sp500 = df_sp500.groupby(level='Ticker').ffill()
    df_sp500 = df_sp500.groupby(level='Ticker').bfill() # Just in case first elements are NaN
    
    df_sp500.reset_index(inplace=True)
    
    print("  - Computing Technical Indicators & Target...")
    df_sp500.sort_values(['Ticker', 'Date'], inplace=True)
    
    # Returns (Stationary relative prices)
    df_sp500['Daily_Return'] = df_sp500.groupby('Ticker')['Close'].pct_change()
    df_sp500['Open_Return'] = df_sp500.groupby('Ticker')['Open'].pct_change()
    df_sp500['High_Return'] = df_sp500.groupby('Ticker')['High'].pct_change()
    df_sp500['Volume_Change'] = df_sp500.groupby('Ticker')['Volume'].pct_change()
    
    # Target (next-day direction), computed BEFORE merge to ensure continuous timeline
    df_sp500['Target'] = df_sp500.groupby('Ticker')['Close'].transform(lambda x: (x.shift(-1) > x).astype(float))
    
    # SMA and Relative Distances
    df_sp500['SMA_5'] = df_sp500.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=5).mean())
    df_sp500['SMA_20'] = df_sp500.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).mean())
    df_sp500['SMA_5_Dist'] = (df_sp500['Close'] - df_sp500['SMA_5']) / df_sp500['SMA_5']
    df_sp500['SMA_20_Dist'] = (df_sp500['Close'] - df_sp500['SMA_20']) / df_sp500['SMA_20']
    
    # RSI
    df_sp500['RSI_14'] = df_sp500.groupby('Ticker')['Close'].transform(lambda x: compute_rsi(x, 14))
    
    # Bollinger Bands (20-day) and Relative Distances
    rolling_std_20 = df_sp500.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=20).std())
    df_sp500['BB_Upper_Dist'] = (df_sp500['Close'] - (df_sp500['SMA_20'] + (rolling_std_20 * 2))) / (df_sp500['SMA_20'] + (rolling_std_20 * 2))
    df_sp500['BB_Lower_Dist'] = (df_sp500['Close'] - (df_sp500['SMA_20'] - (rolling_std_20 * 2))) / (df_sp500['SMA_20'] - (rolling_std_20 * 2))
    
    # MACD (normalized)
    ema_12 = df_sp500.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema_26 = df_sp500.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    df_sp500['MACD'] = (ema_12 - ema_26) / df_sp500['Close']
    
    # Drop intermediate absolute columns not needed for modeling
    df_sp500.drop(columns=['Open', 'High', 'Close', 'Volume', 'SMA_5', 'SMA_20'], inplace=True)
    
    # Drop rows with NaN due to rolling windows and shift
    # Also replace inf values (arise from pct_change on 0-priced rows or BB_Lower_Dist
    # when the lower band equals zero after forward-filling identical prices)
    df_sp500.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_sp500.dropna(inplace=True)
    df_sp500['Target'] = df_sp500['Target'].astype(int)
    
    print("Merging Datasets...")
    df_merged = pd.merge(df_sp500, df_reddit, on='Date', how='inner')
    
    # Drop any remaining NaNs to prevent scaler error in downstream notebooks
    df_merged.dropna(inplace=True)
    
    print("Saving to Parquet...")
    df_merged.to_parquet("datasets/preprocessed_data.parquet", index=False)
    print("Done! Shape of final dataset:", df_merged.shape)
    print("Target distribution:")
    print(df_merged['Target'].value_counts(normalize=True))

if __name__ == "__main__":
    df_reddit = process_reddit_data()
    process_financial_data(df_reddit)