import pandas as pd

print("Loading Reddit News dataset...")
news_df = pd.read_csv('RedditNews.csv')

print("Grouping and concatenating headlines by Date...")
# Group by 'Date' and join all headlines in the 'News' column with a space
daily_news = news_df.groupby('Date')['News'].apply(lambda x: ' '.join(str(headline) for headline in x)).reset_index()

# Rename the column
daily_news.rename(columns={'News': 'Composite_Headline'}, inplace=True)

print("Saving daily composite news data...")
output_path = 'reddit_daily_composite.csv'
daily_news.to_csv(output_path, index=False)

print(f"Done! Aggregated text saved to {output_path}")
print(daily_news.head())