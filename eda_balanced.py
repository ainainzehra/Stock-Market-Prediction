"""
Comprehensive Exploratory Data Analysis on the balanced preprocessed dataset.

Columns analysed:
  Identifiers  : Date, Ticker, Target
  Returns      : Daily_Return, Open_Return, High_Return, Volume_Change
  Trend Dist.  : SMA_5_Dist, SMA_20_Dist
  Oscillators  : RSI_14, MACD
  Band Dist.   : BB_Upper_Dist, BB_Lower_Dist
  Sentiment    : neg, neu, pos, compound (VADER), polarity, subjectivity (TextBlob)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

INPUT_PATH = "datasets/balanced_preprocessed_data.parquet"
OUT_DIR    = "eda_balanced"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.dpi"]   = 130
plt.rcParams["savefig.dpi"]  = 150
plt.rcParams["savefig.bbox"] = "tight"

TECH_FEATURES      = ["Daily_Return", "Open_Return", "High_Return", "Volume_Change",
                       "SMA_5_Dist", "SMA_20_Dist", "RSI_14", "BB_Upper_Dist",
                       "BB_Lower_Dist", "MACD"]
SENTIMENT_FEATURES = ["neg", "neu", "pos", "compound", "polarity", "subjectivity"]
ALL_FEATURES       = TECH_FEATURES + SENTIMENT_FEATURES

CLASS_PALETTE = {0: "#3498db", 1: "#e74c3c"}   # 0=Down (blue), 1=Up (red)

print("=" * 65)
print("  EDA - Balanced Preprocessed Dataset")
print("=" * 65)
print(f"\nLoading {INPUT_PATH} ...")
df = pd.read_parquet(INPUT_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Shape      : {df.shape}")
print(f"  Date range : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"  Tickers    : {df['Ticker'].nunique()}")
print(f"  Columns    : {list(df.columns)}")

# 1. DATASET OVERVIEW
print("\n[1/9] Dataset Overview ...")

print("\n  Missing values per column:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.any() else "  None - dataset is complete.")

print("\n  Descriptive statistics:")
print(df[ALL_FEATURES].describe().round(4).to_string())

# 2. TARGET DISTRIBUTION
print("\n[2/9] Target Distribution ...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

counts = df["Target"].value_counts().sort_index()
labels = ["Down (0)", "Up (1)"]
colors = [CLASS_PALETTE[0], CLASS_PALETTE[1]]

axes[0].bar(labels, counts.values, color=colors, edgecolor="white", linewidth=1.5, width=0.5)
for i, (lbl, cnt) in enumerate(zip(labels, counts.values)):
    axes[0].text(i, cnt + 500, f"{cnt:,}\n({cnt/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
axes[0].set_title("Target Class Distribution", fontsize=14, fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, max(counts.values) * 1.15)

# Temporal target proportion
monthly = df.set_index("Date").resample("ME")["Target"].mean().reset_index()
axes[1].fill_between(monthly["Date"], monthly["Target"], alpha=0.4, color=CLASS_PALETTE[1])
axes[1].plot(monthly["Date"], monthly["Target"], color=CLASS_PALETTE[1], linewidth=1.5)
axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
axes[1].set_title("Monthly Proportion of 'Up' Days Over Time", fontsize=14, fontweight="bold")
axes[1].set_ylabel("Proportion Up (1)")
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)

fig.suptitle("Target Variable Analysis", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_target_distribution.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/01_target_distribution.png")

# 3. RETURN FEATURES  DISTRIBUTIONS BY CLASS
print("\n[3/9] Return Features by Class ...")

return_feats = ["Daily_Return", "Open_Return", "High_Return", "Volume_Change"]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, feat in enumerate(return_feats):
    # Histogram by class
    ax_hist = axes[0, i]
    for cls, color in CLASS_PALETTE.items():
        data = df[df["Target"] == cls][feat]
        q01, q99 = data.quantile(0.01), data.quantile(0.99)
        clipped = data.clip(q01, q99)
        ax_hist.hist(clipped, bins=80, alpha=0.55, color=color,
                     label="Down (0)" if cls == 0 else "Up (1)", density=True)
    ax_hist.set_title(f"{feat}\n(Distribution)", fontsize=10, fontweight="bold")
    ax_hist.set_xlabel("Value (clipped 1-99th pct)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=8)

    # Box plot by class
    ax_box = axes[1, i]
    q01, q99 = df[feat].quantile(0.01), df[feat].quantile(0.99)
    plot_df = df[[feat, "Target"]].copy()
    plot_df[feat] = plot_df[feat].clip(q01, q99)
    sns.boxplot(data=plot_df, x="Target", y=feat, ax=ax_box,
                palette=list(CLASS_PALETTE.values()), width=0.5, flierprops={"marker": "."})
    ax_box.set_title(f"{feat}\n(By Class)", fontsize=10, fontweight="bold")
    ax_box.set_xlabel("Target (0=Down, 1=Up)")
    ax_box.set_ylabel("Value")

fig.suptitle("Return Features - Distributions & Class Comparison", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_return_features.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/02_return_features.png")

# 4. TECHNICAL INDICATORS - DISTRIBUTIONS BY CLASS
print("\n[4/9] Technical Indicators by Class ...")

indicator_feats = ["SMA_5_Dist", "SMA_20_Dist", "RSI_14", "BB_Upper_Dist", "BB_Lower_Dist", "MACD"]

fig, axes = plt.subplots(2, 6, figsize=(26, 10))

for i, feat in enumerate(indicator_feats):
    ax_hist = axes[0, i]
    for cls, color in CLASS_PALETTE.items():
        data = df[df["Target"] == cls][feat]
        q01, q99 = data.quantile(0.01), data.quantile(0.99)
        clipped = data.clip(q01, q99)
        ax_hist.hist(clipped, bins=80, alpha=0.55, color=color,
                     label="Down (0)" if cls == 0 else "Up (1)", density=True)
    ax_hist.set_title(f"{feat}", fontsize=9, fontweight="bold")
    ax_hist.set_xlabel("Value (clipped)")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=7)

    ax_box = axes[1, i]
    q01, q99 = df[feat].quantile(0.01), df[feat].quantile(0.99)
    plot_df = df[[feat, "Target"]].copy()
    plot_df[feat] = plot_df[feat].clip(q01, q99)
    sns.boxplot(data=plot_df, x="Target", y=feat, ax=ax_box,
                palette=list(CLASS_PALETTE.values()), width=0.5, flierprops={"marker": "."})
    ax_box.set_title(f"{feat}\n(By Class)", fontsize=9, fontweight="bold")
    ax_box.set_xlabel("Target")

fig.suptitle("Technical Indicator Features - Distributions & Class Comparison",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_technical_indicators.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/03_technical_indicators.png")

# 5. SENTIMENT FEATURES - DISTRIBUTIONS BY CLASS
print("\n[5/9] Sentiment Features by Class ...")

fig, axes = plt.subplots(2, 6, figsize=(26, 10))

for i, feat in enumerate(SENTIMENT_FEATURES):
    ax_hist = axes[0, i]
    for cls, color in CLASS_PALETTE.items():
        data = df[df["Target"] == cls][feat]
        ax_hist.hist(data, bins=60, alpha=0.55, color=color,
                     label="Down (0)" if cls == 0 else "Up (1)", density=True)
    ax_hist.set_title(f"{feat}", fontsize=9, fontweight="bold")
    ax_hist.set_xlabel("Value")
    ax_hist.set_ylabel("Density")
    ax_hist.legend(fontsize=7)

    ax_box = axes[1, i]
    sns.boxplot(data=df[[feat, "Target"]], x="Target", y=feat, ax=ax_box,
                palette=list(CLASS_PALETTE.values()), width=0.5, flierprops={"marker": "."})
    ax_box.set_title(f"{feat}\n(By Class)", fontsize=9, fontweight="bold")
    ax_box.set_xlabel("Target")

fig.suptitle("Sentiment Features - Distributions & Class Comparison",
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_sentiment_features.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/04_sentiment_features.png")

# 6. FULL CORRELATION HEATMAP
print("\n[6/9] Correlation Heatmap ...")

fig, axes = plt.subplots(1, 3, figsize=(28, 9))

for ax, (title, subset) in zip(axes, [
    ("All Features",        ALL_FEATURES + ["Target"]),
    ("Technical Only",      TECH_FEATURES + ["Target"]),
    ("Sentiment Only",      SENTIMENT_FEATURES + ["Target"]),
]):
    corr = df[subset].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, linecolor="white", ax=ax,
                annot_kws={"size": 7}, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

plt.suptitle("Feature Correlation Matrices", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_correlation_heatmaps.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/05_correlation_heatmaps.png")

# 7. FEATURE-TARGET CORRELATIONS (Point-Biserial)
print("\n[7/9] Feature-Target Correlation Bar Chart ...")

target_corr = df[ALL_FEATURES].corrwith(df["Target"]).sort_values()

colors_bar = ["#e74c3c" if v > 0 else "#3498db" for v in target_corr.values]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
for bar, val in zip(bars, target_corr.values):
    ax.text(val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8)
ax.set_xlabel("Pearson Correlation with Target", fontsize=11)
ax.set_title("Feature–Target Correlation\n(Point-Biserial, Balanced Dataset)",
             fontsize=13, fontweight="bold")
ax.set_xlim(target_corr.min() - 0.05, target_corr.max() + 0.05)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#e74c3c", label="Positive corr. (Up)"),
                   Patch(color="#3498db", label="Negative corr. (Down)")],
          loc="lower right")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_feature_target_correlation.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/06_feature_target_correlation.png")

# 8. RSI ANALYSIS - OVERBOUGHT / OVERSOLD ZONES
print("\n[8/9] RSI Zone Analysis ...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RSI distribution by class with zone shading
ax = axes[0]
for cls, color in CLASS_PALETTE.items():
    data = df[df["Target"] == cls]["RSI_14"].clip(0, 100)
    ax.hist(data, bins=60, alpha=0.55, color=color, density=True,
            label="Down (0)" if cls == 0 else "Up (1)")
ax.axvspan(0,  30, alpha=0.08, color="blue",  label="Oversold (<30)")
ax.axvspan(70, 100, alpha=0.08, color="red",  label="Overbought (>70)")
ax.axvline(30, color="blue", linestyle="--", linewidth=1)
ax.axvline(70, color="red",  linestyle="--", linewidth=1)
ax.set_title("RSI Distribution by Class", fontsize=12, fontweight="bold")
ax.set_xlabel("RSI_14")
ax.set_ylabel("Density")
ax.legend(fontsize=9)

# RSI zone -> win rate
bins_rsi = [0, 30, 50, 70, 100]
labels_rsi = ["Oversold\n(<30)", "Neutral\n(30-50)", "Neutral\n(50-70)", "Overbought\n(>70)"]
df["RSI_Zone"] = pd.cut(df["RSI_14"].clip(0, 100), bins=bins_rsi, labels=labels_rsi)
zone_up_rate = df.groupby("RSI_Zone", observed=True)["Target"].mean() * 100
zone_counts  = df.groupby("RSI_Zone", observed=True)["Target"].count()

ax2 = axes[1]
bar_colors = ["#3498db", "#95a5a6", "#bdc3c7", "#e74c3c"]
bars2 = ax2.bar(zone_up_rate.index, zone_up_rate.values, color=bar_colors, edgecolor="white")
ax2.axhline(50, color="gray", linestyle="--", alpha=0.7, label="50% baseline")
for bar, (zone, rate), cnt in zip(bars2, zone_up_rate.items(), zone_counts.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             f"{rate:.1f}%\n(n={cnt:,})", ha="center", va="bottom", fontsize=9)
ax2.set_title("'Up' Win Rate by RSI Zone", fontsize=12, fontweight="bold")
ax2.set_ylabel("% Predicted Up")
ax2.set_ylim(0, 70)
ax2.legend()

df.drop(columns=["RSI_Zone"], inplace=True)

fig.suptitle("RSI Zone Analysis", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_rsi_analysis.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/07_rsi_analysis.png")

# 9. TEMPORAL TRENDS - ROLLING MONTHLY MEDIANS
print("\n[9/9] Temporal Trends ...")

sample_df = df.set_index("Date")
monthly = sample_df[ALL_FEATURES].resample("ME").median()

fig = plt.figure(figsize=(22, 16))
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.4)

for idx, feat in enumerate(ALL_FEATURES):
    ax = fig.add_subplot(gs[idx // 4, idx % 4])
    ax.plot(monthly.index, monthly[feat], linewidth=1.3, color="#2c3e50")
    ax.fill_between(monthly.index, monthly[feat], alpha=0.2, color="#2c3e50")
    ax.set_title(feat, fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("")

fig.suptitle("Monthly Median Trends - All Features (2008–2016)",
             fontsize=15, fontweight="bold", y=1.01)
plt.savefig(f"{OUT_DIR}/08_temporal_trends.png")
plt.close()
print(f"  Saved -> {OUT_DIR}/08_temporal_trends.png")

# SUMMARY STATISTICS TABLE
print("\n[Summary] Generating per-class statistics table ...")

summary_rows = []
for feat in ALL_FEATURES:
    for cls in [0, 1]:
        subset = df[df["Target"] == cls][feat]
        summary_rows.append({
            "Feature":  feat,
            "Class":    "Down (0)" if cls == 0 else "Up (1)",
            "Mean":     round(subset.mean(), 6),
            "Median":   round(subset.median(), 6),
            "Std":      round(subset.std(), 6),
            "Min":      round(subset.min(), 6),
            "Max":      round(subset.max(), 6),
            "Skewness": round(subset.skew(), 4),
            "Kurtosis": round(subset.kurt(), 4),
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = f"{OUT_DIR}/per_class_statistics.csv"
summary_df.to_csv(summary_path, index=False)
print(f"  Saved -> {summary_path}")

# Print mean difference table to console
print("\n  Mean value per feature by class:")
print(f"  {'Feature':<20} {'Down(0) Mean':>14} {'Up(1) Mean':>12} {'Diff':>10}")
print("  " + "-" * 60)
for feat in ALL_FEATURES:
    m0 = df[df["Target"]==0][feat].mean()
    m1 = df[df["Target"]==1][feat].mean()
    print(f"  {feat:<20} {m0:>14.6f} {m1:>12.6f} {m1-m0:>10.6f}")

print("\n" + "=" * 65)
print("  EDA Complete!")
print("=" * 65)