# discord-stockbot
This Discord bot provides real-time stock market monitoring, automated trading signals, and financial news updates. It’s designed for day traders and investors who want quick insights into their favorite stocks directly in Discord.

## Features
- Real-time **buy/sell/hold signals** using SMA and RSI indicators
- Automatic **stop-loss** and **DCA suggestions**
- **Fast-move detection** for sudden price changes
- Daily **financial news alerts** from Forbes, Yahoo Finance, and CNBC
- **Visual charts** for monitored stocks (individual and combined dashboards)
- User commands to **add/remove/list monitored stocks**
- Background tasks for continuous monitoring and alerts

## Built With
- Python 3.
- [discord.py](https://discordpy.readthedocs.io/)
- [yfinance](https://pypi.org/project/yfinance/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pillow](https://python-pillow.org/)

## Setup
1. Clone the repository
   ```bash
   git clone https://github.com/foolystaycooly/discord-stockbot.git
   
2. Install dependencies
   ```bash
      pip install -r requirements.txt

3. Add your Discord bot token in bot.py

4. Run the bot
   ```bash
    python bot.py

5. Commands

- !add_stock <TICKER> – Add a stock to monitor
- !remove_stock <TICKER> – Remove a stock from monitoring
- !list_stocks – List all monitored stocks
- !dashboard – Display stock charts with signals
