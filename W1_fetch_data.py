import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

today = datetime.today()
two_years_ago = today - timedelta(days=730)
start_date = two_years_ago.strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

print(f"📅 Fetching data from {start_date} to {end_date}\n")

# ── GOLD DATA ──────────────────────────────────────────────
print("🥇 Fetching gold price data...")
gold_raw = yf.Ticker("GLD").history(start=start_date, end=end_date)
gold_df = gold_raw[['Close']].copy()
gold_df.columns = ['gold_price']
gold_df.index.name = 'date'
gold_df = gold_df.reset_index()
gold_df['gold_price'] = gold_df['gold_price'].round(2)
gold_df['date'] = pd.to_datetime(gold_df['date']).dt.date
print(f"✅ Got {len(gold_df)} days of gold data!")
print(f"   Latest gold price: ${gold_df['gold_price'].iloc[-1]}\n")

# ── REAL ESTATE DATA ───────────────────────────────────────
print("🏠 Fetching real estate data from Zillow...")
zillow_url = (
    "https://files.zillowstatic.com/research/public_csvs/zhvi/"
    "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
)
try:
    re_raw = pd.read_csv(zillow_url)
    national = re_raw[re_raw['RegionName'] == 'United States'].copy()
    date_columns = [col for col in re_raw.columns if col.startswith('20')]
    recent_columns = date_columns[-24:]
    national_recent = national[['RegionName'] + recent_columns].copy()
    re_df = national_recent.melt(
        id_vars=['RegionName'],
        value_vars=recent_columns,
        var_name='date',
        value_name='home_value'
    )
    re_df['date'] = pd.to_datetime(re_df['date']).dt.date
    re_df = re_df.drop(columns=['RegionName'])
    re_df = re_df.sort_values('date').reset_index(drop=True)
    re_df['home_value'] = re_df['home_value'].round(2)
    print(f"✅ Got {len(re_df)} months of real estate data!")
    print(f"   Latest US median home value: ${re_df['home_value'].iloc[-1]:,.0f}\n")
except Exception as e:
    print(f"⚠️  Could not fetch Zillow data. Error: {e}")
    re_df = None

# ── SAVE DATA ──────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
gold_df.to_csv("data/gold_prices.csv", index=False)
print("💾 Saved: data/gold_prices.csv")
if re_df is not None:
    re_df.to_csv("data/real_estate_index.csv", index=False)
    print("💾 Saved: data/real_estate_index.csv")

# ── SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  📊 MARKETPULSE — WEEK 1 COMPLETE!")
print("=" * 50)
print(f"\n🥇 GOLD:")
print(f"   Today  : ${gold_df['gold_price'].iloc[-1]}")
print(f"   Highest: ${gold_df['gold_price'].max()}")
print(f"   Lowest : ${gold_df['gold_price'].min()}")
print(f"   Average: ${gold_df['gold_price'].mean():.2f}")
if re_df is not None:
    print(f"\n🏠 REAL ESTATE:")
    print(f"   Today  : ${re_df['home_value'].iloc[-1]:,.0f}")
    print(f"   Highest: ${re_df['home_value'].max():,.0f}")
    print(f"   Lowest : ${re_df['home_value'].min():,.0f}")
print("\n✅ Next: Run week2_analyze.py")
print("=" * 50)
