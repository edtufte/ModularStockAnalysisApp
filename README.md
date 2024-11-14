# Modular Stock Analysis Dashboard

A comprehensive stock analysis platform built with Python and Dash, featuring technical analysis, portfolio management, and strategy backtesting capabilities.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Dash](https://img.shields.io/badge/dash-2.14.2-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

### 📊 Research Dashboard
- Real-time stock data visualization
- Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)
- Volume analysis
- Risk metrics calculation
- Automated trading signals
- Benchmark comparison

### 📈 Portfolio Management
- Create and manage multiple portfolios
- Track holdings and allocations
- Real-time portfolio valuation
- Performance analytics
- Risk assessment
- Position management

### 🔄 Strategy Backtesting
- Test portfolio strategies
- Historical performance analysis
- Risk-adjusted returns
- Benchmark comparison
- Detailed performance metrics
- Custom date ranges

## Technical Highlights

- **Intelligent Data Caching**: Market-hours aware refresh system
- **Advanced Technical Analysis**: Comprehensive indicator calculations
- **Robust Error Handling**: Graceful degradation with detailed logging
- **Responsive Design**: Optimized for all screen sizes
- **Thread-safe Database**: SQLite with connection pooling
- **Efficient Data Processing**: Vectorized operations with pandas

## Installation

1. Clone the repository
```bash
git clone https://github.com/edtufte/ModularStockAnalysisApp.git
cd modular-stock-analysis
```

2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python app.py
```

The application will be available at `http://localhost:8050`

## Project Structure

```
modular-stock-analysis/
├── app.py                 # Main application entry point
├── requirements.txt       # Project dependencies
├── services/             # Core services
│   ├── stock_data_service.py
│   ├── analysis_service.py
│   └── database.py
├── layouts/              # UI layouts
│   ├── dashboard_layout.py
│   ├── portfolio_layout.py
│   └── backtesting_layout.py
├── callbacks/            # Event handlers
│   ├── dashboard_callbacks.py
│   ├── portfolio_callbacks.py
│   └── backtesting_callbacks.py
├── components/           # Reusable UI components
│   ├── dashboard_components.py
│   └── charts.py
├── models/              # Data models
│   └── portfolio.py
└── static/              # Static assets
    └── styles.css
```

## Usage

### Research Dashboard
1. Enter a stock ticker symbol
2. Select analysis timeframe
3. Choose benchmark for comparison
4. View technical analysis, charts, and recommendations

### Portfolio Management
1. Create a new portfolio
2. Add holdings with allocations
3. Monitor performance
4. Adjust positions as needed

### Strategy Backtesting
1. Select a portfolio
2. Define date range
3. Choose benchmark
4. Analyze historical performance

## Technical Analysis Features

- Moving Averages (SMA, EMA)
- Bollinger Bands
- Relative Strength Index (RSI)
- MACD
- Average True Range (ATR)
- On-Balance Volume (OBV)
- Risk Metrics
  - Sharpe Ratio
  - Sortino Ratio
  - Maximum Drawdown
  - Value at Risk
  - Beta & Alpha
  - Information Ratio

## Performance Optimizations

- Intelligent data caching
- Market hours-based refresh
- Vectorized calculations
- Efficient database operations
- Responsive UI components
- Memory-efficient processing

## Development

### Setting up for Development

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run the development server
```bash
python app.py
```

The application supports hot-reloading during development. Any changes to the Python files will automatically restart the server.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Deployment

For production deployment:

```bash
gunicorn app:server
```

Configure your web server (e.g., nginx) to proxy requests to gunicorn.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for stock data
- [Dash](https://dash.plotly.com/) for the web framework
- [Plotly](https://plotly.com/) for interactive visualizations

## Financial Metrics and Calculations

### Risk Metrics

#### Sharpe Ratio
Measures risk-adjusted return relative to risk-free rate:
```
Sharpe Ratio = (Rp - Rf) / σp

Where:
Rp = Return of Portfolio
Rf = Risk-free Rate
σp = Standard Deviation of Portfolio Returns
```

#### Sortino Ratio
Similar to Sharpe but only considers downside volatility:
```
Sortino Ratio = (Rp - Rf) / σd

Where:
σd = Standard Deviation of Negative Returns
```

#### Maximum Drawdown
Largest peak-to-trough decline:
```
MDD = (Trough Value - Peak Value) / Peak Value
```

#### Value at Risk (VaR)
Maximum potential loss at a confidence level (95%):
```
VaR = μ - (z * σ)

Where:
μ = Mean Return
z = Z-score for confidence level
σ = Standard Deviation
```

### Technical Indicators

#### Bollinger Bands
Moving average with standard deviation bands:
```
Middle Band = 20-day SMA
Upper Band = Middle Band + (2 × σ)
Lower Band = Middle Band - (2 × σ)
```

#### Relative Strength Index (RSI)
Momentum indicator measuring speed of price changes:
```
RSI = 100 - (100 / (1 + RS))
RS = Average Gain / Average Loss
```

#### Moving Average Convergence Divergence (MACD)
Trend-following momentum indicator:
```
MACD = 12-period EMA - 26-period EMA
Signal Line = 9-period EMA of MACD
```

### Performance Metrics

#### Beta (β)
Measure of volatility compared to market:
```
β = Covariance(Rp, Rm) / Variance(Rm)

Where:
Rp = Portfolio Returns
Rm = Market Returns
```

#### Alpha (α)
Excess return of investment relative to benchmark:
```
α = Rp - [Rf + β(Rm - Rf)]
```

#### Information Ratio
Risk-adjusted excess returns relative to benchmark:
```
IR = (Rp - Rb) / Tracking Error

Where:
Rb = Benchmark Return
Tracking Error = Std(Rp - Rb)
```

#### Average True Range (ATR)
Volatility indicator showing price range:
```
TR = max[(High - Low), |High - Close_prev|, |Low - Close_prev|]
ATR = 14-period moving average of TR
```

All metrics are calculated using adjusted close prices to account for corporate actions such as splits and dividends.

---

📈 Happy Trading! 📊