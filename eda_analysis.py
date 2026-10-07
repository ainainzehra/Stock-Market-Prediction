import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

os.makedirs("eda_results", exist_ok=True)

print("Loading data...")
df_sp500 = pd.read_csv("datasets/sp500_2008_2020_clean.csv")
df_reddit = pd.read_csv("datasets/reddit_daily_composite.csv")

# 1. Missing value heatmaps
print("Generating missing value plots...")
plt.figure(figsize=(10, 6))
sns.heatmap(df_sp500.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in S&P 500 Data")
plt.savefig("eda_results/sp500_missing.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.heatmap(df_reddit.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values in Reddit Data")
plt.savefig("eda_results/reddit_missing.png")
plt.close()

# 2. Time-series of Volume and Closing Prices (Aggregate to Mean per Date)
print("Plotting time series...")
df_sp500['Date_dt'] = pd.to_datetime(df_sp500['Date'], format='%d/%m/%Y')
daily_agg = df_sp500.groupby('Date_dt').agg({'Close': 'mean', 'Volume': 'mean'}).reset_index()
daily_agg.sort_values('Date_dt', inplace=True)

plt.figure(figsize=(14, 6))
plt.plot(daily_agg['Date_dt'], daily_agg['Close'], label='Mean Close Price', color='blue')
plt.title("S&P 500 Mean Daily Close Price (2008-2020)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.savefig("eda_results/sp500_mean_close.png")
plt.close()

plt.figure(figsize=(14, 6))
plt.plot(daily_agg['Date_dt'], daily_agg['Volume'], label='Mean Daily Volume', color='orange')
plt.title("S&P 500 Mean Daily Volume (2008-2020)")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.tight_layout()
plt.savefig("eda_results/sp500_mean_volume.png")
plt.close()

# 3. Distribution of Daily Returns and Volatility (Standard Deviation)
print("Plotting return distributions...")
# Calculate daily returns per ticker
df_sp500.sort_values(['Ticker', 'Date_dt'], inplace=True)
df_sp500['Daily_Return'] = df_sp500.groupby('Ticker')['Close'].pct_change()

plt.figure(figsize=(10, 6))
sns.histplot(df_sp500['Daily_Return'].dropna(), bins=100, kde=True, color='green')
plt.title("Distribution of Daily Returns")
plt.xlim(-0.1, 0.1) # Limit x-axis to zoom in on the bulk of data
plt.savefig("eda_results/sp500_daily_returns_dist.png")
plt.close()

# Volatility (Rolling 30-day std dev of returns for a sample ticker or aggregate)
aapl = df_sp500[df_sp500['Ticker'] == 'AAPL'].copy()
aapl['Volatility_30d'] = aapl['Daily_Return'].rolling(window=30).std()

plt.figure(figsize=(14, 6))
plt.plot(aapl['Date_dt'], aapl['Volatility_30d'], color='red')
plt.title("AAPL 30-Day Rolling Volatility (Sample)")
plt.xlabel("Date")
plt.ylabel("Volatility (Std Dev of Returns)")
plt.tight_layout()
plt.savefig("eda_results/aapl_volatility.png")
plt.close()

# 4. Word clouds and text length distributions
print("Analyzing text data...")
df_reddit['Headline_Length'] = df_reddit['Composite_Headline'].apply(len)

plt.figure(figsize=(10, 6))
sns.histplot(df_reddit['Headline_Length'], bins=50, kde=True, color='purple')
plt.title("Distribution of Headline Lengths (Characters)")
plt.savefig("eda_results/reddit_length_dist.png")
plt.close()

# Clean up text for word cloud (basic cleanup)
def clean_text(text):
    # Remove the b'...' byte string indicators
    text = re.sub(r"^b['\"]|['\"]$", "", text)
    text = re.sub(r"\s+b['\"]", " ", text)
    # Basic lowering and removing non-alphabetic
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df_reddit['Clean_Headline'] = df_reddit['Composite_Headline'].apply(clean_text)
all_text = " ".join(df_reddit['Clean_Headline'])

print("Generating word cloud...")
try:
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Reddit Daily Headlines")
    plt.tight_layout()
    plt.savefig("eda_results/reddit_wordcloud.png")
    plt.close()
except ImportError:
    print("package not found.")

print("EDA analysis complete.")