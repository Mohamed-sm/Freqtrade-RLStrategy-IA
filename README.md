# RLStrategy - Reinforcement Learning Strategy for FreqAI with 91.78% win rate

## 📊 Performance Statistics

### Global Results
- **Portfolio Size**: 50 USD
- **Trade Amount**: 10 USD per trade
- **ROI (Closed Trades)**: 9.377 USDT (1.28%) - 18.74% annualized
- **ROI (All Trades)**: 8.384 USDT (1.07%) - 16.75% annualized
- **Trading Volume**: 1,465.577 USDT
- **Profit Factor**: 24.90
- **Maximum Drawdown**: 0.70% (0.352 USDT)

### Trading Metrics
- **Total Trade Count**: 78
- **Win Rate**: 91.78% (67 wins / 6 losses)
- **Average Trade Duration**: 1 day, 6h 26min
- **Expectancy Ratio**: 0.13 (1.96)

### Execution Period
- **Bot Started**: June 23, 2025 20:17:49
- **First Trade**: June 24, 2025 14:00:45 (21 days ago)
- **Latest Trade**: July 15, 2025 17:28:39 (7 hours ago)

### Best Performance
- **Pair**: DOGE/USDT
- **Profit**: 2.102 USDT (2.63%)

## 🚀 Strategy Description

This strategy uses **Reinforcement Learning (RL)** integrated with FreqAI to make automatic trading decisions. The RL agent learns market patterns and continuously optimizes its decisions.

### Key Features

- **Framework**: FreqAI with reinforcement learning
- **Timeframe**: 1 hour
- **Stoploss**: -5.1%
- **Minimal ROI**: Optimized by hyperopt
- **Indicators**: RSI, MACD, Moving Averages, Volume

### RL Agent Actions

The agent can take 5 types of actions:
- **0**: Actions.Neutral (neutral)
- **1**: Actions.Long_enter (long entry)
- **2**: Actions.Short_enter (short entry)
- **3**: Actions.Long_exit (long exit)
- **4**: Actions.Short_exit (short exit)

## 📈 Optimized Configuration

### Hyperopt Parameters
```python
buy_rsi = IntParameter(15, 50, default=34)
sell_rsi = IntParameter(50, 85, default=72)
```

### Minimal ROI
```python
minimal_roi = {
    "0": 0.228,
    "325": 0.122,
    "580": 0.065,
    "840": 0
}
```

### FreqAI Parameters
```python
model_training_parameters = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
```

## 🛠️ Installation and Usage

### Prerequisites
- FreqTrade with FreqAI enabled
- Python 3.8+
- Virtual environment recommended

### Installation
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Configuration
1. Place `RLStrategy.py` in your `user_data/strategies/` folder
2. Configure FreqAI in your `config.json`
3. Launch the bot with:
```bash
freqtrade trade --strategy RLStrategy --config config.json
```

## 📋 Features

### Feature Engineering
- **Raw Features**: OHLCV prices
- **Technical Indicators**: RSI, MACD, Moving Averages
- **Temporal Features**: Hour, day of week
- **Volume Features**: Ratios and averages
- **NaN Management**: Automatic cleaning

### Trading Logic
- **Entries**: Based on RL agent actions
- **Exits**: Controlled by RL agent
- **Fallback**: RSI/MACD signals if agent unavailable
- **Stoploss**: Fixed at -5.1%

## 🔧 Customization

### Adding Indicators
Modify `feature_engineering_standard()` to add new indicators:
```python
dataframe['%-new_indicator'] = ta.NEW_INDICATOR(dataframe)
```

### Modifying Parameters
Adjust parameters in the `RLStrategy` class:
```python
buy_rsi = IntParameter(10, 60, default=30)  # More permissive
sell_rsi = IntParameter(40, 90, default=70)  # More permissive
```

## ⚠️ Warnings

- This strategy uses machine learning - past performance does not guarantee future results
- Always test in backtest mode before live trading
- Monitor performance and adjust parameters if necessary
- The RL agent requires sufficient data to learn effectively

## 📝 License

This project is provided for educational purposes. Use at your own risk.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Propose improvements
- Share your results

---

**⚠️ Disclaimer**: Cryptocurrency trading involves risks. This strategy is provided for educational purposes and does not constitute financial advice. 
