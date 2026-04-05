import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
warnings.filterwarnings("ignore")

# ── LOAD DATA ──────────────────────────────────────────────
gold_df = pd.read_csv("data/gold_prices.csv", parse_dates=["date"])
gold_df = gold_df.sort_values("date").reset_index(drop=True)
print("✅ Data loaded!")
print(f"   {len(gold_df)} days of gold price data\n")

# ── FEATURE ENGINEERING ────────────────────────────────────
# We create NEW columns from existing data to help the model learn patterns
gold_df["day_number"] = range(len(gold_df))          # 0, 1, 2, 3... (time as a number)
gold_df["7d_avg"] = gold_df["gold_price"].rolling(7).mean()    # 7 day average
gold_df["30d_avg"] = gold_df["gold_price"].rolling(30).mean()  # 30 day average
gold_df["7d_std"] = gold_df["gold_price"].rolling(7).std()     # 7 day volatility
gold_df["prev_1"] = gold_df["gold_price"].shift(1)             # yesterday's price
gold_df["prev_7"] = gold_df["gold_price"].shift(7)             # price 7 days ago
gold_df["prev_30"] = gold_df["gold_price"].shift(30)           # price 30 days ago

# Drop rows with NaN values created by rolling windows
gold_df = gold_df.dropna().reset_index(drop=True)
print(f"✅ Features created! {len(gold_df)} rows after cleaning\n")

# ── TRAIN / TEST SPLIT ─────────────────────────────────────
# Use 80% of data for training, 20% for testing
split = int(len(gold_df) * 0.8)

features = ["day_number", "7d_avg", "30d_avg", "7d_std", "prev_1", "prev_7", "prev_30"]
X = gold_df[features]
y = gold_df["gold_price"]

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = gold_df["date"][split:]

print(f"📊 Training on {len(X_train)} days")
print(f"📊 Testing on {len(X_test)} days\n")

# ── MODEL 1: LINEAR REGRESSION ─────────────────────────────
print("=" * 50)
print("🤖 MODEL 1: Linear Regression")
print("=" * 50)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
print(f"   Mean Absolute Error : ${lr_mae:.2f}")
print(f"   R² Score            : {lr_r2:.4f}")
print(f"   Accuracy            : {lr_r2 * 100:.1f}%\n")

# ── MODEL 2: RANDOM FOREST ─────────────────────────────────
print("=" * 50)
print("🌲 MODEL 2: Random Forest")
print("=" * 50)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"   Mean Absolute Error : ${rf_mae:.2f}")
print(f"   R² Score            : {rf_r2:.4f}")
print(f"   Accuracy            : {rf_r2 * 100:.1f}%\n")

# ── BEST MODEL ─────────────────────────────────────────────
best_model = "Random Forest" if rf_r2 > lr_r2 else "Linear Regression"
best_predictions = rf_predictions if rf_r2 > lr_r2 else lr_predictions
print(f"🏆 Best Model: {best_model}\n")

# ── PREDICT NEXT 30 DAYS ───────────────────────────────────
print("=" * 50)
print("🔮 PREDICTING NEXT 30 DAYS")
print("=" * 50)

last_row = gold_df.iloc[-1].copy()
future_prices = []
future_dates = []
temp_df = gold_df.copy()

for i in range(30):
    next_day = len(temp_df)
    last_prices = temp_df["gold_price"].tail(30).values
    next_features = pd.DataFrame([{
        "day_number": next_day,
        "7d_avg": np.mean(temp_df["gold_price"].tail(7)),
        "30d_avg": np.mean(temp_df["gold_price"].tail(30)),
        "7d_std": np.std(temp_df["gold_price"].tail(7)),
        "prev_1": temp_df["gold_price"].iloc[-1],
        "prev_7": temp_df["gold_price"].iloc[-7],
        "prev_30": temp_df["gold_price"].iloc[-30],
    }])
    if best_model == "Random Forest":
        pred = rf_model.predict(next_features)[0]
    else:
        pred = lr_model.predict(next_features)[0]
    future_prices.append(pred)
    next_date = temp_df["date"].iloc[-1] + pd.Timedelta(days=1)
    future_dates.append(next_date)
    new_row = pd.DataFrame([{"date": next_date, "gold_price": pred,
                              "day_number": next_day, "7d_avg": next_features["7d_avg"].values[0],
                              "30d_avg": next_features["30d_avg"].values[0],
                              "7d_std": next_features["7d_std"].values[0],
                              "prev_1": pred, "prev_7": pred, "prev_30": pred}])
    temp_df = pd.concat([temp_df, new_row], ignore_index=True)

future_df = pd.DataFrame({"date": future_dates, "predicted_price": future_prices})
print(f"   Current price : ${gold_df['gold_price'].iloc[-1]:.2f}")
print(f"   Predicted (7d): ${future_df['predicted_price'].iloc[6]:.2f}")
print(f"   Predicted (30d): ${future_df['predicted_price'].iloc[-1]:.2f}")
change = ((future_df['predicted_price'].iloc[-1] - gold_df['gold_price'].iloc[-1]) /
           gold_df['gold_price'].iloc[-1]) * 100
print(f"   Expected change: {change:+.2f}%")
if change > 0:
    print("   Outlook: BULLISH (price expected to rise)")
else:
    print("   Outlook: BEARISH (price expected to fall)")
print()

# ── CHARTS ─────────────────────────────────────────────────
os.makedirs("charts", exist_ok=True)
plt.style.use("dark_background")
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.patch.set_facecolor("#0d1117")

# Chart 1: Actual vs Predicted
ax1 = axes[0]
ax1.set_facecolor("#161b22")
ax1.plot(dates_test, y_test.values, color="#FFD700", linewidth=1.5, label="Actual Price")
ax1.plot(dates_test, best_predictions, color="#F472B6", linewidth=1.5,
         linestyle="--", label=f"{best_model} Prediction")
ax1.set_title(f"Actual vs Predicted Gold Price ({best_model})", color="white", fontsize=14)
ax1.set_ylabel("Price (USD)", color="white")
ax1.legend(facecolor="#161b22", labelcolor="white")
ax1.tick_params(colors="white")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax1.grid(color="#30363d", linewidth=0.5)

# Chart 2: Future Predictions
ax2 = axes[1]
ax2.set_facecolor("#161b22")
last_60 = gold_df.tail(60)
ax2.plot(last_60["date"], last_60["gold_price"], color="#FFD700", linewidth=2, label="Historical Price")
ax2.plot(future_df["date"], future_df["predicted_price"], color="#00C896",
         linewidth=2, linestyle="--", label="30-Day Forecast")
ax2.axvline(x=gold_df["date"].iloc[-1], color="white", linewidth=1, linestyle=":", alpha=0.7)
ax2.fill_between(future_df["date"], future_df["predicted_price"] * 0.98,
                  future_df["predicted_price"] * 1.02, alpha=0.2, color="#00C896")
ax2.set_title("30-Day Gold Price Forecast", color="white", fontsize=14)
ax2.set_ylabel("Price (USD)", color="white")
ax2.legend(facecolor="#161b22", labelcolor="white")
ax2.tick_params(colors="white")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.grid(color="#30363d", linewidth=0.5)

plt.tight_layout(pad=3.0)
plt.savefig("charts/week3_predictions.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()

print("💾 Chart saved: charts/week3_predictions.png")
print("\n✅ Week 3 Complete! Next: Run week4_dashboard.py")
print("=" * 50)