"""
Metric and signal definitions for the stock analysis dashboard tooltips.
Provides clear, technical explanations while remaining accessible.
"""

METRIC_DEFINITIONS = {
    # Core Performance Metrics
    'total_return': {
        'title': 'Total Return',
        'explanation': 'Aggregate return including price appreciation and dividends',
        'interpretation': '> 10% annual return beats inflation and treasury yields',
        'example': '$10,000 → $11,000 represents a 10% total return'
    },
    
    'volatility': {
        'title': 'Annualized Volatility',
        'explanation': 'Standard deviation of returns, annualized to 252 trading days',
        'interpretation': '<15%: Low, 15-25%: Moderate, >25%: High volatility',
        'example': '20% means ±20% price range is expected in 68% of years'
    },
    
    'sharpe_ratio': {
        'title': 'Sharpe Ratio',
        'explanation': '(Return - Risk Free Rate) / Standard Deviation',
        'interpretation': '<0.5: Poor, 0.5-1: Fair, 1-2: Good, >2: Excellent',
        'example': '1.5 means excess return is 1.5x the volatility'
    },
    
    'sortino_ratio': {
        'title': 'Sortino Ratio',
        'explanation': 'Like Sharpe but only penalizes downside volatility',
        'interpretation': '<1: High downside risk, 1-2: Good, >2: Excellent',
        'example': '2.0 means good returns with limited downside risk'
    },

    # Technical Indicators
    'rsi': {
        'title': 'Relative Strength Index',
        'explanation': 'Momentum indicator comparing up/down price movements',
        'interpretation': '<30: Oversold, 30-70: Neutral, >70: Overbought',
        'example': 'RSI=25 suggests potentially oversold conditions'
    },
    
    'macd': {
        'title': 'MACD',
        'explanation': 'Trend indicator using difference of moving averages',
        'interpretation': 'MACD > Signal: Bullish, MACD < Signal: Bearish',
        'example': 'MACD crossing above signal suggests upward momentum'
    },

    # Risk Metrics
    'var_95': {
        'title': 'Value at Risk (95%)',
        'explanation': 'Expected maximum loss within 95% confidence interval',
        'interpretation': 'Larger negative values indicate higher risk',
        'example': '-2% means 95% chance of losing no more than 2% daily'
    },
    
    'max_drawdown': {
        'title': 'Maximum Drawdown',
        'explanation': 'Largest peak-to-trough decline in portfolio value',
        'interpretation': '<20%: Normal, 20-40%: High, >40%: Severe',
        'example': '-25% means biggest drop was 25% from peak'
    },

    # Prediction Metrics
    'mape': {
        'title': 'Mean Absolute Percentage Error',
        'explanation': 'Average prediction error as percentage of actual value',
        'interpretation': '<10%: Excellent, 10-20%: Good, >20%: Poor',
        'example': '15% means predictions average 15% off from actual'
    },
    
    'rmse': {
        'title': 'Root Mean Square Error',
        'explanation': 'Average prediction error in price units',
        'interpretation': 'Lower is better, scaled to price magnitude',
        'example': '$2.50 RMSE means typical error is ±$2.50'
    }
}

SIGNAL_DEFINITIONS = {
    'Strong Buy': {
        'explanation': 'Multiple technical and risk metrics align positively',
        'interpretation': 'Highly favorable risk/reward ratio',
        'example': 'Strong technicals + low volatility + positive momentum'
    },
    'Buy': {
        'explanation': 'More positive signals than negative',
        'interpretation': 'Favorable outlook with acceptable risk level',
        'example': 'Good technicals but moderate volatility'
    },
    'Hold': {
        'explanation': 'Mixed signals or unclear direction',
        'interpretation': 'Risk/reward ratio is neutral',
        'example': 'Conflicting technical signals or high volatility'
    },
    'Sell': {
        'explanation': 'More negative signals than positive',
        'interpretation': 'Unfavorable risk/reward ratio',
        'example': 'Weakening technicals or increasing risk metrics'
    },
    'Strong Sell': {
        'explanation': 'Multiple technical and risk metrics align negatively',
        'interpretation': 'High risk with limited upside potential',
        'example': 'Poor technicals + high volatility + negative momentum'
    }
}

# Risk level definitions based on the analysis service thresholds
RISK_LEVEL_DEFINITIONS = {
    'Low': {
        'explanation': 'Key risk metrics within normal ranges',
        'interpretation': 'Suitable for most investment strategies',
        'example': 'Volatility < 15%, healthy Sharpe ratio, moderate drawdowns'
    },
    'Medium': {
        'explanation': 'Some risk metrics showing elevated levels',
        'interpretation': 'Consider position sizing and monitoring',
        'example': 'Volatility 15-25% or larger drawdowns'
    },
    'High': {
        'explanation': 'Multiple risk metrics at concerning levels',
        'interpretation': 'Requires active risk management',
        'example': 'High volatility, poor Sharpe ratio, large drawdowns'
    }
}
