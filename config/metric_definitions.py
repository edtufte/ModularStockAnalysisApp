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
        'explanation': 'Measures how much extra return you get for taking on more risk',
        'interpretation': '<0.5: Poor, 0.5-1: Fair, 1-2: Good, >2: Excellent',
        'example': '1.5 means you\'re well-compensated for the risk taken'
    },
    
    'sortino_ratio': {
        'title': 'Sortino Ratio',
        'explanation': 'Similar to Sharpe, but focuses only on harmful volatility',
        'interpretation': '<1: High risk of losses, 1-2: Good, >2: Excellent',
        'example': '2.0 means strong returns without major downside risks'
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
        'explanation': 'Your likely worst-case daily loss scenario',
        'interpretation': 'Larger negative numbers mean higher risk',
        'example': '-2% means you probably won\'t lose more than 2% in a day'
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
    },

    'calmar_ratio': {
        'title': 'Calmar Ratio',
        'explanation': 'Compares returns to your worst possible loss',
        'interpretation': '<1: Poor, 1-3: Good, >3: Excellent',
        'example': '2.5 means your returns are 2.5x larger than your worst loss'
    },

    'treynor_ratio': {
        'title': 'Treynor Ratio',
        'explanation': 'Shows return compared to market risk taken',
        'interpretation': 'Higher numbers mean better market risk management',
        'example': '0.15 means 15% extra return for each unit of market risk'
    },

    'capture_ratio_up': {
        'title': 'Upside Capture Ratio',
        'explanation': 'Measures performance relative to benchmark in up markets',
        'interpretation': '>100%: Outperforms in up markets, <100%: Underperforms',
        'example': '120% means 20% better returns than benchmark in up markets'
    },

    'capture_ratio_down': {
        'title': 'Downside Capture Ratio',
        'explanation': 'Measures performance relative to benchmark in down markets',
        'interpretation': '<100%: Better downside protection, >100%: More downside risk',
        'example': '80% means 20% less loss than benchmark in down markets'
    },

    'omega_ratio': {
        'title': 'Omega Ratio',
        'explanation': 'Compares the size and frequency of gains versus losses',
        'interpretation': '>1: More gains than losses, <1: More losses than gains',
        'example': '1.5 means your gains are 50% bigger than your losses'
    },

    'tail_ratio': {
        'title': 'Tail Ratio',
        'explanation': 'Compares your best days to your worst days',
        'interpretation': '>1: Better upside potential, <1: More downside risk',
        'example': '1.2 means your best days are 20% better than your worst days'
    }
}

SIGNAL_DEFINITIONS = {
    'Strong Buy': {
        'explanation': 'Multiple indicators suggest this is a good opportunity',
        'interpretation': 'Very attractive risk vs. reward balance',
        'example': 'Strong performance + low risk + positive momentum'
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
        'explanation': 'All risk measurements look healthy',
        'interpretation': 'Generally safe for most investors',
        'example': 'Stable performance with minimal surprises'
    },
    'Medium': {
        'explanation': 'Some risk indicators are elevated',
        'interpretation': 'Watch position size and monitor closely',
        'example': 'More price swings or larger drops than normal'
    },
    'High': {
        'explanation': 'Multiple warning signs in risk metrics',
        'interpretation': 'Needs careful risk management',
        'example': 'Large price swings and significant drops possible'
    }
}
