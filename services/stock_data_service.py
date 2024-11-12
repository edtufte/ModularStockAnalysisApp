import yfinance as yf
import logging
import pandas as pd
from typing import Optional, Dict, Any, Tuple

class StockDataService:
    """Service class for fetching and validating stock data"""
    
    @staticmethod
    def setup_logging():
        yf_logger = logging.getLogger('yfinance')
        yf_logger.setLevel(logging.CRITICAL)

    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if the ticker exists
        Returns (is_valid, error_message)
        """
        try:
            StockDataService.setup_logging()
            data = yf.download(ticker, period="1d", progress=False, timeout=5)
            
            if not data.empty:
                return True, None
                
            # Try getting ticker info as fallback
            stock = yf.Ticker(ticker)
            try:
                price = stock.fast_info['lastPrice']
                if price is not None and price > 0:
                    return True, None
            except:
                pass
            
            return False, "Unable to fetch data for this ticker"
            
        except Exception as e:
            error_str = str(e).lower()
            if '404' in error_str:
                return False, "Ticker not found"
            elif 'timeout' in error_str:
                return False, "Request timed out - please try again"
            elif 'connection' in error_str:
                return False, "Connection error - please check your internet connection"
            else:
                return False, "Unable to validate ticker"

    @staticmethod
    def fetch_stock_data(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch stock data and perform initial processing"""
        try:
            StockDataService.setup_logging()
            df = yf.download(ticker, period=timeframe, progress=False)
            
            if df.empty:
                return None
                
            # Convert to DataFrame if needed
            df = pd.DataFrame(df)
            
            # Calculate adjusted prices
            df['Open'] = df['Open'] * (df['Adj Close'] / df['Close'])
            df['High'] = df['High'] * (df['Adj Close'] / df['Close'])
            df['Low'] = df['Low'] * (df['Adj Close'] / df['Close'])
            df['Close'] = df['Adj Close']
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching stock data: {str(e)}")
            return None

    @staticmethod
    def get_fundamental_data(ticker: str) -> Dict[str, Any]:
        """Fetch fundamental data for a stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'pe_ratio': info.get('forwardPE', info.get('trailingPE', None)),
                'industry_pe': info.get('industryPE', 20),
                'market_cap': info.get('marketCap', None),
                'volume': info.get('volume', None),
                'dividend_yield': info.get('dividendYield', None)
            }
        except Exception as e:
            logging.error(f"Error fetching fundamental data: {str(e)}")
            return {}