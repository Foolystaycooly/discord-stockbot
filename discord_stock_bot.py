import discord
from discord.ext import commands
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import asyncio
import json
import feedparser
from collections import Counter
import time
import requests
import pytz
from datetime import datetime


# -----------------------------
# Configuration
# -----------------------------
BOT_TOKEN = "Put your own token here"

ALERT_CHANNELS = {
    "signals": Put your own channel id here,
    "news": Put your own channel id here,
    "day_traders": Put your own channel id here
}

STOP_LOSS_PERCENT = 0.05
CHART_DIR = "charts"
STOCKS_FILE = "monitored_stocks.json"

NEWS_FEEDS = [
    "https://www.forbes.com/real-time/feed2/",
    "https://finance.yahoo.com/rss/topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html"
]

MIN_VOLUME = 100000  # Ignore very low volume stocks

if not os.path.exists(CHART_DIR):
    os.makedirs(CHART_DIR)

# -----------------------------
# Load Monitored Stocks
# -----------------------------
if os.path.exists(STOCKS_FILE):
    with open(STOCKS_FILE, "r") as f:
        monitored_stocks = json.load(f)
else:
    monitored_stocks = ["AAPL", "TSLA", "MSFT"]
    with open(STOCKS_FILE, "w") as f:
        json.dump(monitored_stocks, f)

previous_signals = {ticker: None for ticker in monitored_stocks}
previous_fast_moves = {ticker: None for ticker in monitored_stocks}
sent_news = set()
news_mentions = Counter()

# -----------------------------
# Helper Functions
# -----------------------------
def save_stocks():
    with open(STOCKS_FILE, "w") as f:
        json.dump(monitored_stocks, f)

def fetch_data_multi(tickers, period="7d", interval="5m"):
    tickers_str = " ".join(tickers)
    df = yf.download(tickers=tickers_str, period=period, interval=interval, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.stack(level=1, future_stack=True).rename_axis(['Date', 'Ticker']).reset_index()
    else:
        df['Ticker'] = tickers[0]
    return df

def calculate_sma(df, window):
    return df['Close'].rolling(window=window).mean()

def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1*delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def format_rsi(rsi):
    if rsi < 30:
        return f"{rsi:.2f} (🟢 Oversold)"
    elif rsi > 70:
        return f"{rsi:.2f} (🔴 Overbought)"
    else:
        return f"{rsi:.2f} (⚪ Neutral)"

def format_change(change):
    if change > 0:
        return f"+{change:.2f}% 🟢"
    elif change < 0:
        return f"{change:.2f}% 🔴"
    else:
        return f"{change:.2f}% ⚪"

def generate_signal(df):
    df['SMA20'] = calculate_sma(df, 20)
    df['SMA50'] = calculate_sma(df, 50)
    df['RSI'] = calculate_rsi(df, 14)
    signal = "HOLD"
    last_rsi = df['RSI'].iloc[-1]
    if df['SMA20'].iloc[-2] < df['SMA50'].iloc[-2] and df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1] and last_rsi < 70:
        signal = "BUY"
    elif df['SMA20'].iloc[-2] > df['SMA50'].iloc[-2] and df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1] and last_rsi > 30:
        signal = "SELL"
    return signal

def calculate_stop_loss(last_close):
    return round(last_close * (1 - STOP_LOSS_PERCENT), 2)

def suggest_dca(last_close, levels=3):
    last_close = float(last_close)  # <-- force to plain float
    step = last_close * 0.03
    return [round(last_close - step*i, 2) for i in range(1, levels+1)]

def plot_and_save_chart(df, ticker):
    plt.figure(figsize=(4,2))
    plt.plot(df['Close'], label='Close', color='blue')
    plt.plot(calculate_sma(df,20), label='SMA20', color='orange')
    plt.plot(calculate_sma(df,50), label='SMA50', color='green')
    plt.title(ticker)
    plt.axis('off')
    filename = os.path.join(CHART_DIR, f"{ticker}_chart.png")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename

def plot_chart_grid(ticker_data):
    charts = [Image.open(plot_and_save_chart(df, ticker)) for ticker, df in ticker_data.items()]
    n = len(charts)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    max_width = max(img.width for img in charts)
    max_height = max(img.height for img in charts)
    grid_img = Image.new('RGB', (cols*max_width, rows*max_height), color=(255,255,255))
    for index, img in enumerate(charts):
        row = index // cols
        col = index % cols
        grid_img.paste(img, (col*max_width, row*max_height))
    combined_filename = os.path.join(CHART_DIR, "combined_dashboard.png")
    grid_img.save(combined_filename)
    return combined_filename

def check_fast_moves(df, threshold_percent=0.02, min_change=0.5):
    if len(df) < 2:
        return None
    last_close = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = (last_close - prev_close) / prev_close
    absolute_change = abs(last_close - prev_close)
    if absolute_change < min_change:
        return None
    if abs(change) >= threshold_percent:
        direction = "Rising Fast" if change > 0 else "Plummeting Fast"
        return f"{direction}: {change*100:.2f}%"
    return None

def color_for_signal(signal):
    return 0x2ECC71 if signal=="BUY" else 0xE74C3C if signal=="SELL" else 0xF1C40F

def fetch_news(limit=5, ticker_filter=None):
    news_items = []
    for feed_url in NEWS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:limit]:
            if entry.link in sent_news:
                continue
            title = entry.title
            link = entry.link
            published = getattr(entry, 'published', 'N/A')
            if ticker_filter and ticker_filter.upper() not in title.upper():
                continue
            news_items.append(f"**{title}**\n{link}\nPublished: {published}")
            sent_news.add(link)
            if ticker_filter:
                news_mentions[ticker_filter.upper()] += 1
    return news_items

def fetch_top_actives(limit=20):
    """Fetch top active tickers from Yahoo Finance dynamically."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        url = "https://finance.yahoo.com/most-active"
        html = requests.get(url, headers=headers).text  # Ask Yahoo nicely
        tables = pd.read_html(html)
        df = tables[0]
        tickers = df['Symbol'].tolist()[:limit]
        return tickers
    except Exception as e:
        print(f"Error fetching top actives: {e}")
        return [] # If it fails, just return an empty list

# -----------------------------
# Discord Bot Setup
# -----------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# -----------------------------
# Commands
# -----------------------------
@bot.command()
async def add_stock(ctx, ticker: str):
    ticker = ticker.upper()
    if ticker not in monitored_stocks:
        monitored_stocks.append(ticker)
        previous_signals[ticker] = None
        previous_fast_moves[ticker] = None
        save_stocks()
        await ctx.send(f"{ticker} added to monitored stocks.")
    else:
        await ctx.send(f"{ticker} is already being monitored.")

@bot.command()
async def remove_stock(ctx, ticker: str):
    ticker = ticker.upper()
    if ticker in monitored_stocks:
        monitored_stocks.remove(ticker)
        previous_signals.pop(ticker, None)
        previous_fast_moves.pop(ticker, None)
        save_stocks()
        await ctx.send(f"{ticker} removed from monitored stocks.")
    else:
        await ctx.send(f"{ticker} is not currently monitored.")

@bot.command()
async def list_stocks(ctx):
    await ctx.send(f"Currently monitored stocks: {', '.join(monitored_stocks)}")

@bot.command()
async def dashboard(ctx):
    if not monitored_stocks:
        await ctx.send("No monitored stocks.")
        return
    df_all = fetch_data_multi(monitored_stocks, period="7d", interval="5m")
    ticker_data = {ticker: df_all[df_all['Ticker']==ticker].copy() for ticker in monitored_stocks}
    combined_file = plot_chart_grid(ticker_data)
    embed = discord.Embed(title="📊 Stock Dashboard", color=0x3498DB)
    for ticker, df in ticker_data.items():
        if df['Volume'].iloc[-1] < MIN_VOLUME:
            continue
        sig = generate_signal(df)
        last_close = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2]
        pct_change = ((last_close - prev_close) / prev_close) * 100
        rsi = df['RSI'].iloc[-1]
        sl = calculate_stop_loss(last_close)
        dca_levels = suggest_dca(last_close)
        fast_move = check_fast_moves(df) or "None"
        embed.add_field(
            name=f"{ticker} ({sig})",
            value=f"Close: {last_close}\nChange: {format_change(pct_change)}\nRSI: {format_rsi(rsi)}\nStop-Loss: {sl}\nDCA: {dca_levels}\nFast Move: {fast_move}",
            inline=True
        )
    embed.set_image(url="attachment://combined_dashboard.png")
    await ctx.send(embed=embed, file=discord.File(combined_file, "combined_dashboard.png"))

# -----------------------------
# Background Tasks
# -----------------------------
async def monitor_signals():
    await bot.wait_until_ready()
    while True:
        if not monitored_stocks:
            await asyncio.sleep(60)
            continue
        df_all = fetch_data_multi(monitored_stocks, period="7d", interval="5m")
        channel = bot.get_channel(ALERT_CHANNELS["signals"])
        for ticker in monitored_stocks:
            df_ticker = df_all[df_all['Ticker']==ticker].copy()
            if df_ticker['Volume'].iloc[-1] < MIN_VOLUME:
                continue
            sig = generate_signal(df_ticker)
            if previous_signals.get(ticker) != sig:
                previous_signals[ticker] = sig
                last_close = df_ticker['Close'].iloc[-1]
                prev_close = df_ticker['Close'].iloc[-2]
                pct_change = ((last_close - prev_close) / prev_close) * 100
                rsi = df_ticker['RSI'].iloc[-1]
                sl = calculate_stop_loss(last_close)
                dca_levels = suggest_dca(last_close)
                chart_file = plot_and_save_chart(df_ticker, ticker)
                if channel:
                    embed = discord.Embed(
                        title=f"{ticker} Signal Changed",
                        description=f"Signal: **{sig}**\nClose: {last_close}\nChange: {format_change(pct_change)}\nRSI: {format_rsi(rsi)}\nStop-Loss: **{sl}**\nDCA: {dca_levels}",
                        color=color_for_signal(sig)
                    )
                    embed.set_image(url=f"attachment://{ticker}.png")
                    await channel.send(embed=embed, file=discord.File(chart_file, filename=f"{ticker}.png"))
        await asyncio.sleep(60)

async def monitor_news():
    await bot.wait_until_ready()
    while True:
        channel = bot.get_channel(ALERT_CHANNELS["news"])
        if channel:
            for ticker in monitored_stocks:
                items = fetch_news(limit=5, ticker_filter=ticker)
                for item in items:
                    await channel.send(item)
        await asyncio.sleep(10)

# Track last fetch time for dynamic stocks
last_dynamic_fetch = 0
DYNAMIC_FETCH_INTERVAL = 1800  # 30 minutes
dynamic_stocks = []

def is_market_open():
    ny = pytz.timezone('America/New_York')
    now = datetime.now(ny)
    # Market hours: Monday-Friday 9:30am-4:00pm
    if now.weekday() >= 5:  # Saturday/Sunday
        return False
    if now.hour < 9 or (now.hour == 9 and now.minute < 30) or now.hour >= 16:
        return False
    return True

def calculate_stop_loss_price(last_close, percent=STOP_LOSS_PERCENT):
    """Calculate stop-loss price based on last close."""
    return round(last_close * (1 - percent), 2)

async def monitor_fast_moves():
    global last_dynamic_fetch, dynamic_stocks
    await bot.wait_until_ready()
    while True:
        try:
            now = time.time()
            # Update dynamic top movers every DYNAMIC_FETCH_INTERVAL
            if now - last_dynamic_fetch > DYNAMIC_FETCH_INTERVAL:
                fetched = fetch_top_actives(limit=20)
                if fetched:
                    dynamic_stocks = fetched
                last_dynamic_fetch = now

            all_stocks = list(set(monitored_stocks + dynamic_stocks))
            if not all_stocks:
                await asyncio.sleep(30)
                continue

            channel = bot.get_channel(ALERT_CHANNELS["day_traders"])
            if channel is None:
                print("Day trader channel not found!")
                await asyncio.sleep(30)
                continue

            market_open = is_market_open()

            if market_open:
                # Real-time fast moves during market hours
                df_all = fetch_data_multi(all_stocks, period="1d", interval="1m")
                for ticker in all_stocks:
                    df_ticker = df_all[df_all['Ticker'] == ticker].copy()
                    if df_ticker.empty:
                        continue
                    move_alert = check_fast_moves(df_ticker)
                    if move_alert and previous_fast_moves.get(ticker) != move_alert:
                        previous_fast_moves[ticker] = move_alert
                        last_close = df_ticker['Close'].iloc[-1]
                        sl = calculate_stop_loss_price(last_close)
                        dca_levels = [round(x, 2) for x in suggest_dca(last_close)]
                        rsi = calculate_rsi(df_ticker)['RSI'].iloc[-1] if 'RSI' in df_ticker else calculate_rsi(df_ticker).iloc[-1]
                        await channel.send(
                            f"**{ticker} {move_alert}**\n"
                            f"Close: {last_close}\n"
                            f"RSI: {rsi:.2f}\n"
                            f"Stop-Loss: {sl}\n"
                            f"DCA levels: {dca_levels}"
                        )
            else:
                # Market closed → opportunistic suggestions
                df_all = fetch_data_multi(all_stocks, period="2d", interval="1d")
                for ticker in all_stocks:
                    df_ticker = df_all[df_all['Ticker'] == ticker].copy()
                    if len(df_ticker) < 2:
                        continue
                    yesterday_close = df_ticker['Close'].iloc[-2]
                    today_close = df_ticker['Close'].iloc[-1]
                    pct_change = ((today_close - yesterday_close) / yesterday_close) * 100

                    if abs(pct_change) >= 2:  # threshold for “big move”
                        alert_tag = f"Opportunistic {pct_change:.2f}%"
                        if previous_fast_moves.get(ticker) != alert_tag:
                            previous_fast_moves[ticker] = alert_tag
                            sl = calculate_stop_loss_price(today_close)
                            dca_levels = [round(x,2) for x in suggest_dca(today_close)]
                            rsi = calculate_rsi(df_ticker)['RSI'].iloc[-1] if 'RSI' in df_ticker else calculate_rsi(df_ticker).iloc[-1]

                            await channel.send(
                                f"**[Opportunistic] {ticker} moved {pct_change:.2f}% yesterday**\n"
                                f"Close: {today_close}\n"
                                f"RSI: {rsi:.2f}\n"
                                f"Stop-Loss: {sl}\n"
                                f"DCA levels: {dca_levels}\n"
                            )

        except Exception as e:
            print(f"Error in monitor_fast_moves: {e}")

        await asyncio.sleep(30)

def format_rsi_status(rsi):
    if rsi < 30:
        return f"{rsi:.2f} 🟢 Oversold"
    elif rsi > 70:
        return f"{rsi:.2f} 🔴 Overbought"
    else:
        return f"{rsi:.2f} ⚪ Neutral"


# -----------------------------
# Bot Events
# -----------------------------
@bot.event
async def on_ready():
    print(f"{bot.user} is online!")
    bot.loop.create_task(monitor_signals())
    bot.loop.create_task(monitor_news())
    bot.loop.create_task(monitor_fast_moves())

# -----------------------------
# Run Bot
# -----------------------------
bot.run(BOT_TOKEN)
