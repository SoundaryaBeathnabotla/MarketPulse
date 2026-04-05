import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="MarketPulse",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .main { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    h1, h2, h3, p, label { color: white !important; }
    .stMetric { background-color: #161b22; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────────────────────────────────────
st.markdown("""
<h1 style='text-align: center; color: #FFD700 !important; font-size: 3em;'>
    📈 MarketPulse
</h1>
<p style='text-align: center; color: #8b949e !important; font-size: 1.2em;'>
    Gold & Real Estate Investment Analyzer
</p>
<hr style='border-color: #30363d;'>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")
period = st.sidebar.selectbox(
    "Data Period",
    ["6 Months", "1 Year", "2 Years"],
    index=2
)
show_prediction = st.sidebar.checkbox("Show 30-Day Prediction", value=True)
forecast_days = st.sidebar.slider("Forecast Days", 7, 60, 30)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("Built by **Soundarya Beathnabotla**")
st.sidebar.markdown("Data Science + Data Engineering")
st.sidebar.markdown("[GitHub](https://github.com/SoundaryaBeathnabotla)")

# ── LOAD DATA ──────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(period):
    days = {"6 Months": 180, "1 Year": 365, "2 Years": 730}[period]
    end = datetime.today()
    start = end - timedelta(days=days)

    # Gold data
    gold = yf.Ticker("GLD").history(
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d')
    )[['Close']].copy()
    gold.columns = ['gold_price']
    gold.index.name = 'date'
    gold = gold.reset_index()
    gold['gold_price'] = gold['gold_price'].round(2)
    gold['date'] = pd.to_datetime(gold['date']).dt.tz_localize(None)

    # Real estate data
    try:
        zillow_url = (
            "https://files.zillowstatic.com/research/public_csvs/zhvi/"
            "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
        )
        re_raw = pd.read_csv(zillow_url)
        national = re_raw[re_raw['RegionName'] == 'United States'].copy()
        date_cols = [c for c in re_raw.columns if c.startswith('20')]
        n = days // 30
        recent = date_cols[-n:]
        melted = national[['RegionName'] + recent].melt(
            id_vars=['RegionName'], value_vars=recent,
            var_name='date', value_name='home_value'
        )
        melted['date'] = pd.to_datetime(melted['date'])
        melted = melted.drop(columns=['RegionName']).sort_values('date').reset_index(drop=True)
        re_df = melted
    except:
        re_df = None

    return gold, re_df

with st.spinner("Fetching latest market data..."):
    gold_df, re_df = load_data(period)

# ── ML MODEL ───────────────────────────────────────────────
@st.cache_data
def run_model(gold_data, forecast_days):
    df = gold_data.copy().sort_values("date").reset_index(drop=True)
    df["day_number"] = range(len(df))
    df["7d_avg"] = df["gold_price"].rolling(7).mean()
    df["30d_avg"] = df["gold_price"].rolling(30).mean()
    df["7d_std"] = df["gold_price"].rolling(7).std()
    df["prev_1"] = df["gold_price"].shift(1)
    df["prev_7"] = df["gold_price"].shift(7)
    df["prev_30"] = df["gold_price"].shift(30)
    df = df.dropna().reset_index(drop=True)

    features = ["day_number", "7d_avg", "30d_avg", "7d_std", "prev_1", "prev_7", "prev_30"]
    split = int(len(df) * 0.8)
    X_train, X_test = df[features][:split], df[features][split:]
    y_train, y_test = df["gold_price"][:split], df["gold_price"][split:]

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)

    best = lr if lr_r2 > rf_r2 else rf
    best_pred = lr_pred if lr_r2 > rf_r2 else rf_pred
    best_name = "Linear Regression" if lr_r2 > rf_r2 else "Random Forest"
    best_r2 = max(lr_r2, rf_r2)
    best_mae = mean_absolute_error(y_test, best_pred)

    # Future predictions
    future_prices, future_dates = [], []
    temp = df.copy()
    for i in range(forecast_days):
        next_features = pd.DataFrame([{
            "day_number": len(temp),
            "7d_avg": temp["gold_price"].tail(7).mean(),
            "30d_avg": temp["gold_price"].tail(30).mean(),
            "7d_std": temp["gold_price"].tail(7).std(),
            "prev_1": temp["gold_price"].iloc[-1],
            "prev_7": temp["gold_price"].iloc[-7],
            "prev_30": temp["gold_price"].iloc[-30],
        }])
        pred = best.predict(next_features)[0]
        future_prices.append(pred)
        next_date = temp["date"].iloc[-1] + pd.Timedelta(days=1)
        future_dates.append(next_date)
        new_row = pd.DataFrame([{"date": next_date, "gold_price": pred,
                                   "day_number": len(temp), "7d_avg": pred,
                                   "30d_avg": pred, "7d_std": 0,
                                   "prev_1": pred, "prev_7": pred, "prev_30": pred}])
        temp = pd.concat([temp, new_row], ignore_index=True)

    future_df = pd.DataFrame({"date": future_dates, "predicted_price": future_prices})
    return future_df, best_name, best_r2, best_mae, df[split:], best_pred

future_df, best_name, best_r2, best_mae, test_df, test_pred = run_model(gold_df, forecast_days)

# ── METRICS ROW ────────────────────────────────────────────
st.markdown("### 📊 Market Overview")
col1, col2, col3, col4, col5 = st.columns(5)

current_gold = gold_df['gold_price'].iloc[-1]
gold_7d_ago = gold_df['gold_price'].iloc[-7]
gold_change = ((current_gold - gold_7d_ago) / gold_7d_ago) * 100
predicted_30d = future_df['predicted_price'].iloc[-1]
predicted_change = ((predicted_30d - current_gold) / current_gold) * 100

with col1:
    st.metric("Gold Price Today", f"${current_gold:.2f}", f"{gold_change:+.2f}% (7d)")
with col2:
    st.metric("Gold High (Period)", f"${gold_df['gold_price'].max():.2f}")
with col3:
    st.metric("Gold Low (Period)", f"${gold_df['gold_price'].min():.2f}")
with col4:
    st.metric(f"{forecast_days}d Forecast", f"${predicted_30d:.2f}", f"{predicted_change:+.2f}%")
with col5:
    if re_df is not None:
        st.metric("US Home Value", f"${re_df['home_value'].iloc[-1]:,.0f}")
    else:
        st.metric("Model Used", best_name)

st.markdown("---")

# ── GOLD PRICE CHART ───────────────────────────────────────
st.markdown("### 🥇 Gold Price Trend")
fig_gold = go.Figure()
gold_df['30d_avg'] = gold_df['gold_price'].rolling(30).mean()

fig_gold.add_trace(go.Scatter(
    x=gold_df['date'], y=gold_df['gold_price'],
    name='Gold Price', line=dict(color='#FFD700', width=1.5)
))
fig_gold.add_trace(go.Scatter(
    x=gold_df['date'], y=gold_df['30d_avg'],
    name='30-Day Average', line=dict(color='#FFA500', width=2, dash='dash')
))

if show_prediction:
    fig_gold.add_trace(go.Scatter(
        x=future_df['date'], y=future_df['predicted_price'],
        name=f'{forecast_days}d Forecast',
        line=dict(color='#00C896', width=2, dash='dot'),
        fill='tozeroy', fillcolor='rgba(0,200,150,0.05)'
    ))
    fig_gold.add_vline(
        x=gold_df['date'].iloc[-1], line_dash="dash",
        line_color="white", opacity=0.5
    )

fig_gold.update_layout(
    paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
    font=dict(color='white'), height=400,
    legend=dict(bgcolor='#161b22', bordercolor='#30363d'),
    xaxis=dict(gridcolor='#30363d'),
    yaxis=dict(gridcolor='#30363d', title='Price (USD)')
)
st.plotly_chart(fig_gold, use_container_width=True)

# ── TWO COLUMNS ────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🤖 Model Performance")
    fig_model = go.Figure()
    fig_model.add_trace(go.Scatter(
        x=test_df['date'], y=test_df['gold_price'].values,
        name='Actual', line=dict(color='#FFD700', width=2)
    ))
    fig_model.add_trace(go.Scatter(
        x=test_df['date'], y=test_pred,
        name=f'{best_name}', line=dict(color='#F472B6', width=2, dash='dash')
    ))
    fig_model.update_layout(
        paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
        font=dict(color='white'), height=350,
        legend=dict(bgcolor='#161b22'),
        xaxis=dict(gridcolor='#30363d'),
        yaxis=dict(gridcolor='#30363d', title='Price (USD)')
    )
    st.plotly_chart(fig_model, use_container_width=True)
    st.markdown(f"""
    **Best Model:** {best_name}
    **R² Score:** {best_r2:.4f} | **MAE:** ${best_mae:.2f}
    """)

with col_right:
    st.markdown("### 🏠 Real Estate vs Gold")
    if re_df is not None:
        gold_monthly = gold_df.resample("ME", on="date")["gold_price"].mean().reset_index()
        gold_monthly['date'] = gold_monthly['date'].dt.to_period("M").dt.to_timestamp()
        re_df['date'] = pd.to_datetime(re_df['date']).dt.to_period("M").dt.to_timestamp()
        merged = pd.merge(gold_monthly, re_df, on="date", how="inner")

        fig_corr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_corr.add_trace(go.Scatter(
            x=merged['date'], y=merged['gold_price'],
            name='Gold Price', line=dict(color='#FFD700', width=2)
        ), secondary_y=False)
        fig_corr.add_trace(go.Scatter(
            x=merged['date'], y=merged['home_value'],
            name='Home Value', line=dict(color='#00C896', width=2)
        ), secondary_y=True)
        fig_corr.update_layout(
            paper_bgcolor='#0d1117', plot_bgcolor='#161b22',
            font=dict(color='white'), height=350,
            legend=dict(bgcolor='#161b22'),
            xaxis=dict(gridcolor='#30363d'),
        )
        fig_corr.update_yaxes(gridcolor='#30363d', title_text="Gold (USD)", secondary_y=False)
        fig_corr.update_yaxes(title_text="Home Value (USD)", secondary_y=True)
        corr = merged['gold_price'].corr(merged['home_value'])
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown(f"**Correlation Score:** {corr:.2f} — {'Strong' if corr > 0.7 else 'Moderate'} positive relationship")
    else:
        st.info("Real estate data unavailable. Check your internet connection.")

# ── FORECAST TABLE ─────────────────────────────────────────
st.markdown("---")
st.markdown(f"### 🔮 {forecast_days}-Day Gold Price Forecast")
forecast_display = future_df.copy()
forecast_display['date'] = forecast_display['date'].dt.strftime('%Y-%m-%d')
forecast_display['predicted_price'] = forecast_display['predicted_price'].round(2)
forecast_display['change'] = (
    (forecast_display['predicted_price'] - current_gold) / current_gold * 100
).round(2)
forecast_display.columns = ['Date', 'Predicted Price (USD)', 'Change from Today (%)']
st.dataframe(
    forecast_display,
    use_container_width=True,
    hide_index=True
)

# ── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #8b949e;'>
    Built by Soundarya Beathnabotla | MarketPulse v1.0 |
    Data: Yahoo Finance & Zillow Research
</p>
""", unsafe_allow_html=True)