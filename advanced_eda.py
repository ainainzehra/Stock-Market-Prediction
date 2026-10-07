import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

os.makedirs("eda_results_advanced", exist_ok=True)

print("Loading preprocessed dataset...")
df = pd.read_parquet("datasets/preprocessed_data.parquet")

# Since the data is grouped by ticker and sorted by date, let's compute daily returns
print("Computing Daily Returns and Volatility...")
df.sort_values(['Ticker', 'Date'], inplace=True)
df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change()
df['Direction'] = (df['Daily_Return'] > 0).astype(int) # 1 for Up, 0 for Down
# Shift direction backwards by 1 to represent "Next Day Direction" as the target
df['Target_Direction'] = df.groupby('Ticker')['Direction'].shift(-1)

df.dropna(subset=['Daily_Return', 'Target_Direction'], inplace=True)

# 1. Cross-Correlation Matrix
print("Generating Correlation Matrix...")
features = ['Open', 'High', 'Close', 'Volume', 'SMA_5', 'SMA_20', 'RSI_14', 'BB_Upper', 'BB_Lower', 'MACD', 'compound', 'pos', 'neg', 'neu', 'polarity', 'subjectivity']
plt.figure(figsize=(14, 10))
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("eda_results_advanced/correlation_matrix.png")
plt.close()

# 2. Sentiment Trends Over Time (Monthly)
print("Generating Sentiment Trends...")
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_sentiment = df.groupby('YearMonth')[['compound', 'polarity']].mean().reset_index()
monthly_sentiment['Date'] = monthly_sentiment['YearMonth'].dt.to_timestamp()

plt.figure(figsize=(14, 6))
plt.plot(monthly_sentiment['Date'], monthly_sentiment['compound'], label='VADER Compound', color='green')
plt.plot(monthly_sentiment['Date'], monthly_sentiment['polarity'], label='TextBlob Polarity', color='blue')
plt.title("Average Monthly Sentiment Over Time (2008-2020)")
plt.xlabel("Date")
plt.ylabel("Scaled Sentiment Score")
plt.legend()
plt.tight_layout()
plt.savefig("eda_results_advanced/sentiment_over_time.png")
plt.close()

# 3. Market Volatility vs Sentiment
print("Generating Volatility vs Sentiment plot...")
# Aggregate volatility and sentiment at the daily level across all tickers
daily_market = df.groupby('Date').agg({
    'Daily_Return': 'std', # Cross-sectional volatility
    'compound': 'mean'
}).reset_index()
daily_market.rename(columns={'Daily_Return': 'Market_Volatility'}, inplace=True)

# Smooth them for visualization
daily_market['Volatility_SMA30'] = daily_market['Market_Volatility'].rolling(30).mean()
daily_market['Sentiment_SMA30'] = daily_market['compound'].rolling(30).mean()

fig, ax1 = plt.subplots(figsize=(14, 6))
ax1.set_xlabel('Date')
ax1.set_ylabel('Market Volatility (30-day SMA)', color='red')
ax1.plot(daily_market['Date'], daily_market['Volatility_SMA30'], color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.set_ylabel('Sentiment Compound (30-day SMA)', color='green')
ax2.plot(daily_market['Date'], daily_market['Sentiment_SMA30'], color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title("Market Volatility vs. News Sentiment Over Time")
fig.tight_layout()
plt.savefig("eda_results_advanced/volatility_vs_sentiment.png")
plt.close()

# 4. Sentiment by Day of Week
print("Generating Day of Week Sentiment...")
df['DayOfWeek'] = df['Date'].dt.day_name()
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

plt.figure(figsize=(10, 6))
sns.boxplot(x='DayOfWeek', y='compound', data=df.sample(100000), order=order, palette='Set2')
plt.title("Distribution of Sentiment (VADER) by Day of Week")
plt.savefig("eda_results_advanced/sentiment_dow.png")
plt.close()

# 5. Quick Feature Importance Preview (Random Forest on a sample)
print("Running Feature Importance Preview...")
# Take a random sample to keep it fast
sample_df = df.sample(n=100000, random_state=42)
X = sample_df[features]
y = sample_df['Target_Direction']

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = rf.feature_importances_
feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
plt.title("Random Forest Feature Importance (Preview for Volatility/Direction Prediction)")
plt.tight_layout()
plt.savefig("eda_results_advanced/feature_importance.png")
plt.close()

print("Advanced EDA complete.")