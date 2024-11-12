# services/stock_data_service.py
import pandas as pd
import numpy as np
import requests
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from config import secrets

class StockDataService:
    """Service class for fetching and validating stock data using Alpha Vantage API"""
    
    TIME_SERIES_MAPPING = {
        '6mo': 'TIME_SERIES_DAILY',
        '1y': 'TIME_SERIES_DAILY',
        '3y': 'TIME_SERIES_WEEKLY',
        '5y': 'TIME_SERIES_WEEKLY',
        'max': 'TIME_SERIES_MONTHLY'
    }
    
    @staticmethod
    def _make_api_request(function: str, symbol: str, **kwargs) -> Optional[Dict]:
        """Make request to Alpha Vantage API"""
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': secrets.alpha_vantage_key,
            **kwargs
        }
        
        try:
            response = requests.get(secrets.api_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                logging.error(f"API Error: {data['Error Message']}")
                return None
                
            return data
        except Exception as e:
            logging.error(f"API request failed: {str(e)}")
            return None
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, Optional[str]]:
        """Validate if the ticker exists"""
        try:
            data = StockDataService._make_api_request(
                'SYMBOL_SEARCH',
                ticker
            )
            
            if not data or 'bestMatches' not in data:
                return False, "Unable to validate ticker"
                
            matches = data['bestMatches']
            exact_match = next(
                (m for m in matches if m['1. symbol'].upper() == ticker.upper()),
                None
            )
            
            if exact_match:
                return True, None
            return False, "Ticker not found"
            
        except Exception as e:
            error_str = str(e).lower()
            if 'timeout' in error_str:
                return False, "Request timed out - please try again"
            elif 'connection' in error_str:
                return False, "Connection error - please check your internet connection"
            else:
                return False, "Unable to validate ticker"
    
    @staticmethod
    def fetch_stock_data(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch stock data and perform initial processing"""
        try:
            # Get appropriate time series function
            function = StockDataService.TIME_SERIES_MAPPING.get(timeframe, 'TIME_SERIES_DAILY')
            
            # Make API request
            data = StockDataService._make_api_request(
                function,
                ticker,
                outputsize='full'
            )
            
            if not data:
                return None
            
            # Get the time series data
            time_series_key = next(
                (k for k in data.keys() if 'Time Series' in k),
                None
            )
            
            if not time_series_key:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(
                data[time_series_key],
                orient='index'
            )
            
            # Rename columns
            df.columns = [col.split('. ')[1] for col in df.columns]
            
            # Convert values to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter based on timeframe
            if timeframe != 'max':
                months = int(timeframe.replace('mo', '').replace('y', '') * 12)
                start_date = datetime.now() - timedelta(days=months * 30)
                df = df[df.index >= start_date.strftime('%Y-%m-%d')]
            
            # Sort index
            df.sort_index(inplace=True)
            
            # Add adjusted close (Alpha Vantage daily adjusted could be used instead)
            df['Adj Close'] = df['close']
            
            # Rename columns to match existing code
            df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching stock data: {str(e)}")
            return None
    
    @staticmethod
    def get_fundamental_data(ticker: str) -> Dict[str, Any]:
        """Fetch fundamental data for a stock"""
        try:
            # Get company overview
            data = StockDataService._make_api_request(
                'OVERVIEW',
                ticker
            )
            
            if not data:
                return {}
            
            return {
                'pe_ratio': float(data.get('PERatio', 0)) or None,
                'industry_pe': float(data.get('PEGRatio', 20)) or 20,
                'market_cap': float(data.get('MarketCapitalization', 0)) or None,
                'volume': float(data.get('Volume', 0)) or None,
                'dividend_yield': float(data.get('DividendYield', 0)) or None
            }
            
        except Exception as e:
            logging.error(f"Error fetching fundamental data: {str(e)}")
            return {}