import pandas as pd
import numpy as np
from typing import Optional

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    @staticmethod
    def calculate_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages"""
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        return df

    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * std
        df['BB_lower'] = df['BB_middle'] - 2 * std
        return df

    @staticmethod
    def calculate_rsi(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility metrics"""
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=21).std() * np.sqrt(252)
        return df

    @staticmethod
    def calculate_performance_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various performance metrics"""
        df_yearly = df.resample('Y').agg({
            'Open': 'first',
            'Close': 'last',
            'High': 'max',
            'Low': 'min',
            'Volume': 'sum'
        })
        df_yearly['Year'] = df_yearly.index.year
        df_yearly['Return (%)'] = ((df_yearly['Close'] - df_yearly['Open']) / df_yearly['Open'] * 100).round(2)
        df_yearly['Max Drawdown (%)'] = ((df_yearly['Low'] - df_yearly['High']) / df_yearly['High'] * 100).round(2)
        return df_yearly

    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        df = TechnicalIndicators.calculate_moving_averages(df)
        df = TechnicalIndicators.calculate_bollinger_bands(df)
        df = TechnicalIndicators.calculate_rsi(df)
        df = TechnicalIndicators.calculate_volatility(df)
        return df