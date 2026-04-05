import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

# ── LOAD DATA ──────────────────────────────────────────────
gold_df = pd.read_csv("data/gold_prices.csv", parse_dates=["date"])
re_df = pd.read_csv("data/real_estate_index.csv", parse_dates=["date"])

print("✅ Data loaded!")
print(f"   Gold: {len(gold_df)} rows")
print(f"   Real Estate: {len(re_df)} rows\n")

# ── ANALYSIS 1: GOLD TREND ─────────────────────────────────
gold_df = gold_df.sort_values("date").reset_index(drop=True)
gold_df["30d_avg"] = gold_df["gold_price"].rolling(window=30).mean()
gold_df["pct_change"] = gold_df["gold_price"].pct_change() * 100

first_price = gold_df["gold_price"].iloc[0]
last_price = gold_df["gold_price"].iloc[-1]
total_return = ((last_price - first_price) / first_price) * 100
volatility = gold_df["pct_change"].std()

print("=" * 50)
print("🥇 GOLD ANALYSIS")
print("=" * 50)
print(f"   Start price : ${first_price:.2f}")
print(f"   End price   : ${last_price:.2f}")
print(f"   Total return: {total_return:.2f}%")
print(f"   Volatility  : {volatility:.2f}%")
if total_return > 0:
    print("   Trend       : 📈 UPWARD")
else:
    print("   Trend       : 📉 DOWNWARD")
print()

# ── ANALYSIS 2: REAL ESTATE TREND ─────────────────────────
re_df = re_df.sort_values("date").reset_index(drop=True)
re_first = re_df["home_value"].iloc[0]
re_last = re_df["home_value"].iloc[-1]
re_return = ((re_last - re_first) / re_first) * 100

print("=" * 50)
print("🏠 REAL ESTATE ANALYSIS")
print("=" * 50)
print(f"   Start value : ${re_first:,.0f}")
print(f"   End value   : ${re_last:,.0f}")
print(f"   Total return: {re_return:.2f}%")
if re_return > 0:
    print("   Trend       : 📈 UPWARD")
else:
    print("   Trend       : 📉 DOWNWARD")
print()

# ── ANALYSIS 3: CORRELATION ────────────────────────────────
gold_monthly = gold_df.resample("ME", on="date")["gold_price"].mean().reset_index()
gold_monthly.columns = ["date", "gold_price"]
gold_monthly["date"] = gold_monthly["date"].dt.to_period("M").dt.to_timestamp()
re_df["date"] = re_df["date"].dt.to_period("M").dt.to_timestamp()
merged = pd.merge(gold_monthly, re_df, on="date", how="inner")
correlation = merged["gold_price"].corr(merged["home_value"])

print("=" * 50)
print("🔗 CORRELATION ANALYSIS")
print("=" * 50)
print(f"   Gold vs Real Estate correlation: {correlation:.2f}")
if correlation > 0.7:
    print("   Relationship: STRONG positive — they move together")
elif correlation > 0.3:
    print("   Relationship: MODERATE positive — somewhat related")
elif correlation > -0.3:
    print("   Relationship: WEAK — not much relation")
else:
    print("   Relationship: NEGATIVE — they move opposite")
print()

# ── CHARTS ─────────────────────────────────────────────────
os.makedirs("charts", exist_ok=True)
plt.style.use("dark_background")

fig, axes = plt.subplots(3, 1, figsize=(12, 14))
fig.patch.set_facecolor("#0d1117")

# Chart 1: Gold Price
ax1 = axes[0]
ax1.set_facecolor("#161b22")
ax1.plot(gold_df["date"], gold_df["gold_price"], color="#FFD700", linewidth=1.5, label="Gold Price")
ax1.plot(gold_df["date"], gold_df["30d_avg"], color="#FFA500", linewidth=2, linestyle="--", label="30-Day Average")
ax1.set_title("🥇 Gold Price (2 Years)", color="white", fontsize=14, pad=10)
ax1.set_ylabel("Price (USD)", color="white")
ax1.legend(facecolor="#161b22", labelcolor="white")
ax1.tick_params(colors="white")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.grid(color="#30363d", linewidth=0.5)

# Chart 2: Real Estate
ax2 = axes[1]
ax2.set_facecolor("#161b22")
ax2.plot(re_df["date"], re_df["home_value"], color="#00C896", linewidth=2, label="US Median Home Value")
ax2.fill_between(re_df["date"], re_df["home_value"], alpha=0.2, color="#00C896")
ax2.set_title("🏠 US Median Home Value (2 Years)", color="white", fontsize=14, pad=10)
ax2.set_ylabel("Value (USD)", color="white")
ax2.legend(facecolor="#161b22", labelcolor="white")
ax2.tick_params(colors="white")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.grid(color="#30363d", linewidth=0.5)

# Chart 3: Correlation Scatter
ax3 = axes[2]
ax3.set_facecolor("#161b22")
ax3.scatter(merged["gold_price"], merged["home_value"], color="#7C3AED", alpha=0.7, s=60)
z = np.polyfit(merged["gold_price"], merged["home_value"], 1)
p = np.poly1d(z)
ax3.plot(sorted(merged["gold_price"]), p(sorted(merged["gold_price"])), color="#F472B6", linewidth=2, linestyle="--")
ax3.set_title(f"🔗 Gold vs Real Estate Correlation: {correlation:.2f}", color="white", fontsize=14, pad=10)
ax3.set_xlabel("Gold Price (USD)", color="white")
ax3.set_ylabel("Home Value (USD)", color="white")
ax3.tick_params(colors="white")
ax3.grid(color="#30363d", linewidth=0.5)

plt.tight_layout(pad=3.0)
plt.savefig("charts/week2_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()

print("💾 Chart saved: charts/week2_analysis.png")
print("\n✅ Week 2 Complete! Next: Run week3_predict.py")
print("=" * 50)