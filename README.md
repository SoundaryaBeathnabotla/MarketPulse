# 📈 MarketPulse — Gold & Real Estate Investment Analyzer

A data-driven investment analysis tool that tracks **gold prices** and **luxury real estate trends** — built to support alternative investment decision-making.

> Built by Soundarya Beathnabotla | Data Science + Data Engineering

---

## 🎯 What This Project Does

AaronBux Wealth Management focuses on alternative investments: gold and luxury real estate. This dashboard:

- Fetches and cleans real gold price data (GLD ETF via yfinance)
- Pulls US real estate index data (Zillow ZHVI)
- Analyzes trends, volatility, and correlations
- Predicts short-term price movement using ML
- Visualizes everything in an interactive Streamlit dashboard

---

## 🗂️ Project Structure

```
aaronbux-investment-dashboard/
│
├── data/                        # Auto-generated after running scripts
│   ├── gold_prices.csv
│   └── real_estate_index.csv
│
├── week1_data_collection.py     # ✅ Fetch & clean data
├── week2_analysis.py            # 🔜 Trends & correlation analysis
├── week3_predictions.py         # 🔜 ML price predictions
├── week4_dashboard.py           # 🔜 Streamlit dashboard
│
├── requirements.txt
└── README.md
```

---

## 🚀 How To Run (Week 1)

### 1. Clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/aaronbux-investment-dashboard.git
cd aaronbux-investment-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Week 1 script
```bash
python week1_data_collection.py
```

You'll see gold prices and real estate data fetched and saved into the `/data` folder.

---

## 📦 Requirements

```
yfinance
pandas
requests
streamlit
scikit-learn
matplotlib
plotly
```

---

## 🗓️ Build Progress

| Week | Goal | Status |
|------|------|--------|
| Week 1 | Data Collection & Cleaning | ✅ Done |
| Week 2 | Trend Analysis & Correlation | 🔜 Up next |
| Week 3 | ML Price Predictions | 🔜 Coming |
| Week 4 | Streamlit Dashboard | 🔜 Coming |

---

## 💡 Why I Built This

I'm currently interning at AaronBux Asset Management, a fintech startup focused on alternative investments. I wanted to build a tool that directly supports their core business using real data — and use it to learn data science + engineering end to end.

---

## 🛠️ Tech Stack

- **Python** — core language
- **Pandas** — data cleaning & analysis
- **yfinance** — free gold/market data
- **Scikit-learn** — ML predictions
- **Streamlit** — interactive dashboard
- **Plotly** — visualizations

---

## 📬 Contact

**Soundarya Beathnabotla**  
[LinkedIn](https://linkedin.com/in/soundaryabeathnabotla) | soundaryabeathnabotla@gmail.com
