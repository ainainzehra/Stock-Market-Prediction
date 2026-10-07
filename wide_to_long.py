import pandas as pd

print("Loading raw dataset...")
df = pd.read_csv('sp500_2008_2020_raw.csv', header=[0, 1], index_col=0)

print("Reshaping data...")
df_long = df.stack(level=0).reset_index()

df_long.rename(columns={
    df_long.columns[0]: 'Date', 
    df_long.columns[1]: 'Ticker'
}, inplace=True)

print("Filtering required columns...")
df_clean = df_long[['Date', 'Ticker', 'Open', 'High', 'Close', 'Volume']]

output_path = 'sp500_2008_2020_clean.csv'
df_clean.to_csv(output_path, index=False)

print(f"Done! Clean data saved to {output_path}")
print(df_clean.head())