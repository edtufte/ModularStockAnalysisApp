# services/technical_indicators.py
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from scipy import stats

class TechnicalIndicators:
    """Class for calculating technical indicators and risk metrics"""
    
    RISK_FREE_RATE = 0.02  # Treasury bill rate - should be updated periodically
    TRADING_DAYS = 252     # Standard trading days in a year
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        df['SMA_20'] = df['Adj Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
        
        # Add MACD
        exp1 = df['Adj Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Adj Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['BB_middle'] = df['Adj Close'].rolling(window=20).mean()
        std = df['Adj Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * std
        df['BB_lower'] = df['BB_middle'] - 2 * std
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Adj Close'].shift())
        low_close = np.abs(df['Low'] - df['Adj Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def calculate_on_balance_volume(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On Balance Volume"""
        df['OBV'] = (np.sign(df['Adj Close'].diff()) * df['Volume']).fillna(0).cumsum()
        return df

    @staticmethod
    def calculate_risk_metrics(df: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate comprehensive risk metrics
        
        Args:
            df: DataFrame with price data
            benchmark_returns: Optional benchmark returns (e.g., S&P 500)
        """
        # Calculate daily returns
        df['Daily_Return'] = df['Adj Close'].pct_change()
        
        # Annualized Volatility
        df['Volatility'] = df['Daily_Return'].rolling(window=21).std() * np.sqrt(TechnicalIndicators.TRADING_DAYS)
        
        # Calculate rolling Sharpe Ratio
        excess_returns = df['Daily_Return'] - TechnicalIndicators.RISK_FREE_RATE/TechnicalIndicators.TRADING_DAYS
        df['Sharpe_Ratio'] = (excess_returns.rolling(window=252).mean() * TechnicalIndicators.TRADING_DAYS) / \
                            (df['Daily_Return'].rolling(window=252).std() * np.sqrt(TechnicalIndicators.TRADING_DAYS))
        
        # Calculate rolling Sortino Ratio
        negative_returns = df['Daily_Return'].copy()
        negative_returns[negative_returns > 0] = 0
        downside_std = negative_returns.rolling(window=252).std() * np.sqrt(TechnicalIndicators.TRADING_DAYS)
        df['Sortino_Ratio'] = np.where(
            downside_std != 0,
            (excess_returns.rolling(window=252).mean() * TechnicalIndicators.TRADING_DAYS) / downside_std,
            np.nan
        )
        
        # Maximum Drawdown
        rolling_max = df['Adj Close'].rolling(window=252, min_periods=1).max()
        daily_drawdown = df['Adj Close']/rolling_max - 1
        df['Max_Drawdown'] = daily_drawdown.rolling(window=252, min_periods=1).min()
        
        # Value at Risk (95% confidence)
        df['VaR_95'] = df['Daily_Return'].rolling(window=252).quantile(0.05)
        
        # Calculate Beta and Alpha if benchmark provided
        if benchmark_returns is not None:
            # Ensure benchmark_returns index matches df
            benchmark_returns = benchmark_returns.reindex(df.index)
            
            # Rolling beta calculation
            covariance = df['Daily_Return'].rolling(window=252).cov(benchmark_returns)
            variance = benchmark_returns.rolling(window=252).var()
            df['Beta'] = covariance / variance
            
            # Rolling alpha calculation
            df['Alpha'] = df['Daily_Return'].rolling(window=252).mean() - \
                         (TechnicalIndicators.RISK_FREE_RATE/TechnicalIndicators.TRADING_DAYS + \
                          df['Beta'] * (benchmark_returns.rolling(window=252).mean() - \
                          TechnicalIndicators.RISK_FREE_RATE/TechnicalIndicators.TRADING_DAYS))
            
            # Information Ratio
            active_return = df['Daily_Return'] - benchmark_returns
            df['Information_Ratio'] = (active_return.rolling(window=252).mean() * TechnicalIndicators.TRADING_DAYS) / \
                                    (active_return.rolling(window=252).std() * np.sqrt(TechnicalIndicators.TRADING_DAYS))
        
        return df

    @staticmethod
    def calculate_statistical_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical properties of returns"""
        returns = df['Daily_Return']
        
        # Rolling Skewness
        df['Return_Skew'] = returns.rolling(window=252).skew()
        
        # Rolling Kurtosis
        df['Return_Kurtosis'] = returns.rolling(window=252).kurt()
        
        # Rolling Z-Score
        df['Return_ZScore'] = (returns - returns.rolling(window=20).mean()) / returns.rolling(window=20).std()
        
        return df

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate all technical indicators and risk metrics"""
        df = TechnicalIndicators.calculate_moving_averages(df)
        df = TechnicalIndicators.calculate_bollinger_bands(df)
        df = TechnicalIndicators.calculate_rsi(df)
        df = TechnicalIndicators.calculate_atr(df)
        df = TechnicalIndicators.calculate_on_balance_volume(df)
        df = TechnicalIndicators.calculate_risk_metrics(df, benchmark_returns)
        df = TechnicalIndicators.calculate_statistical_metrics(df)
        return df

    @staticmethod
    def calculate_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various performance metrics"""
        df_yearly = df.resample('Y').agg({
            'Open': 'first',
            'Adj Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum',
            'Volatility': 'mean',
            'Sharpe_Ratio': 'mean',
            'Sortino_Ratio': 'mean',
            'Max_Drawdown': 'min',
            'VaR_95': 'mean'
        })
        
        df_yearly['Year'] = df_yearly.index.year
        df_yearly['Return (%)'] = ((df_yearly['Adj Close'] - df_yearly['Open']) / df_yearly['Open'] * 100).round(2)
        df_yearly['Max Drawdown (%)'] = (df_yearly['Max_Drawdown'] * 100).round(2)
        df_yearly['Volatility (%)'] = (df_yearly['Volatility'] * 100).round(2)
        
        # Add Beta and Information Ratio if available
        if 'Beta' in df.columns:
            df_yearly['Beta'] = df.groupby(pd.Grouper(freq='Y'))['Beta'].mean()
        if 'Information_Ratio' in df.columns:
            df_yearly['Information_Ratio'] = df.groupby(pd.Grouper(freq='Y'))['Information_Ratio'].mean()
        
        return df_yearly

    @staticmethod
    def get_technical_summary(df: pd.DataFrame) -> Dict[str, str]:
        """Generate technical analysis summary and signals"""
        latest = df.iloc[-1]
        signals = []
        
        # MACD Signal
        if latest['MACD'] > latest['Signal_Line']:
            signals.append(('MACD', 'Bullish', 'MACD above signal line'))
        else:
            signals.append(('MACD', 'Bearish', 'MACD below signal line'))
        
        # RSI Signal
        if latest['RSI'] > 70:
            signals.append(('RSI', 'Overbought', f'RSI at {latest["RSI"]:.1f}'))
        elif latest['RSI'] < 30:
            signals.append(('RSI', 'Oversold', f'RSI at {latest["RSI"]:.1f}'))
        
        # Bollinger Bands Signal
        if latest['Adj Close'] > latest['BB_upper']:
            signals.append(('BB', 'Overbought', 'Price above upper band'))
        elif latest['Adj Close'] < latest['BB_lower']:
            signals.append(('BB', 'Oversold', 'Price below lower band'))
        
        # Trend Analysis
        if latest['SMA_20'] > latest['SMA_50']:
            signals.append(('Trend', 'Bullish', 'Short-term trend above medium-term'))
        else:
            signals.append(('Trend', 'Bearish', 'Short-term trend below medium-term'))
        
        return {
            'signals': signals,
            'strength': len([s for s in signals if s[1] in ['Bullish', 'Oversold']]) / len(signals)
        }