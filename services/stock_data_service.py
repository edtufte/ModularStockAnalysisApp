# services/stock_data_service.py
import pandas as pd
import numpy as np
import yfinance as yf
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import requests
import time
from requests.exceptions import RequestException
import pandas_datareader.data as web

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataService:
    """Service class for fetching and validating stock data using multiple sources"""
    
    PERIODS = {
        '6mo': '6mo',
        '1y': '1y',
        '3y': '3y',
        '5y': '5y',
        'max': 'max'
    }
    
    @staticmethod
    def _retry_with_backoff(func, *args, max_attempts=3, initial_delay=1, **kwargs):
        """Helper method to retry operations with exponential backoff"""
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                delay = initial_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)
    
    @staticmethod
    def _fetch_from_yfinance(ticker: str, period: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                raise ValueError("Empty dataset returned")
            return df
        except Exception as e:
            logger.error(f"YFinance error: {str(e)}")
            return None
    
    @staticmethod
    def _fetch_from_alternative(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch data from alternative source (pandas_datareader)"""
        try:
            df = web.DataReader(
                ticker,
                'stooq',  # Using Stooq as alternative source
                start_date,
                end_date
            )
            if df.empty:
                raise ValueError("Empty dataset returned")
            return df
        except Exception as e:
            logger.error(f"Alternative source error: {str(e)}")
            return None
    
    @staticmethod
    def _format_market_cap(market_cap: float) -> str:
        """Format market cap with appropriate suffix and precision"""
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        elif market_cap >= 1e6:
            return f"${market_cap/1e6:.2f}M"
        else:
            return f"${market_cap:.2f}"
    
    @staticmethod
    def _calculate_market_cap(price: float, volume: float, shares_outstanding: Optional[float] = None) -> Optional[float]:
        """
        Calculate market cap using the most accurate available data
        
        Args:
            price (float): Current stock price
            volume (float): Trading volume
            shares_outstanding (float, optional): Number of shares outstanding
        
        Returns:
            float: Calculated market cap if data is available; None if unavailable.
        """
        # Primary Method: Use price * shares_outstanding if shares outstanding data is available
        if shares_outstanding is not None:
            return price * shares_outstanding

        # Fallback Method: Use price * (average monthly volume) as a proxy
        # This assumes volume is daily; adjust if volume represents another period.
        if volume is not None:
            monthly_volume = volume * 30  # Using monthly volume as a fallback proxy
            return price * monthly_volume

        # If no reliable data is available, return None
        return None
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """Validate stock ticker symbol with multiple attempts and sources"""
        try:
            # Basic cleanups
            ticker = ticker.strip().upper()
            
            # Length check
            if len(ticker) == 0:
                return False, "Ticker symbol cannot be empty"
            if len(ticker) > 6:
                return False, "Ticker symbol too long"
                
            # Character validation
            valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ.-")
            if not all(char in valid_chars for char in ticker):
                return False, "Ticker contains invalid characters"
                
            # Must start with a letter
            if not ticker[0].isalpha():
                return False, "Ticker must start with a letter"
            
            # Try multiple data sources
            try:
                # Try yfinance first
                stock = yf.Ticker(ticker)
                info = StockDataService._retry_with_backoff(
                    lambda: stock.info,
                    max_attempts=2
                )
                if info and isinstance(info, dict):
                    return True, ""
            except Exception as e:
                logger.warning(f"YFinance validation failed: {str(e)}")
                
            # Try alternative source
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)
                df = web.DataReader(ticker, 'stooq', start_date, end_date)
                if not df.empty:
                    return True, ""
            except Exception as e:
                logger.warning(f"Alternative source validation failed: {str(e)}")
            
            return False, "Unable to verify ticker existence"
            
        except Exception as e:
            return False, f"Error validating ticker: {str(e)}"
    
    @staticmethod
    def fetch_stock_data(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with fallback sources"""
        try:
            logger.info(f"Fetching {timeframe} data for {ticker}")
            
            # Get appropriate time period
            period = StockDataService.PERIODS.get(timeframe)
            if not period:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
                
            # Try YFinance first
            df = StockDataService._retry_with_backoff(
                StockDataService._fetch_from_yfinance,
                ticker,
                period,
                max_attempts=2
            )
            
            # If YFinance fails, try alternative source
            if df is None:
                end_date = datetime.now()
                if period == 'max':
                    start_date = end_date - timedelta(days=3650)  # 10 years
                else:
                    days = {'6mo': 180, '1y': 365, '3y': 1095, '5y': 1825}
                    start_date = end_date - timedelta(days=days[period])
                
                df = StockDataService._retry_with_backoff(
                    StockDataService._fetch_from_alternative,
                    ticker,
                    start_date,
                    end_date,
                    max_attempts=2
                )
            
            if df is None:
                logger.error(f"No data returned for {ticker}")
                return None
            
            # Process the dataframe
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Standardize column names
            column_map = {
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume',
                'Adj Close': 'Adj Close',
                'Adj. Close': 'Adj Close',  # For alternative source
                'Adj. Volume': 'Volume'      # For alternative source
            }
            df.rename(columns=column_map, inplace=True)
            
            # Ensure all required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns in data for {ticker}")
                return None
            
            logger.info(f"Successfully processed data. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return None
    
    @staticmethod
    def get_company_overview(ticker: str) -> Dict[str, Any]:
        """Get company overview with multiple data source attempts"""
        try:
            stock = yf.Ticker(ticker)
            
            # Try to get company info from primary source (e.g., Yahoo Finance)
            info = StockDataService._retry_with_backoff(
                lambda: stock.info,
                max_attempts=2
            )
            
            # Fallback in case of missing data
            if not info:
                logger.warning("Failed to get detailed info, trying alternative source")
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=30)
                    df = web.DataReader(ticker, 'stooq', start_date, end_date)
                    
                    if df is not None and not df.empty:
                        latest_price = df['Close'].iloc[-1]
                        avg_volume = df['Volume'].mean() if 'Volume' in df.columns else None
                        market_cap = None
                        shares_outstanding = info.get('sharesOutstanding') if info else None
                        if shares_outstanding:
                            market_cap = latest_price * shares_outstanding
                        elif avg_volume:
                            market_cap = StockDataService._calculate_market_cap(latest_price, avg_volume)
                        
                        info = {
                            'longName': ticker.upper(),
                            'marketCap': market_cap,
                            'fiftyTwoWeekHigh': df['High'].max(),
                            'fiftyTwoWeekLow': df['Low'].min()
                        }
                except Exception as e:
                    logger.error(f"Alternative source error: {str(e)}")
                    info = {}
            
            # Use `.get()` to safely access all attributes
            overview = {
                'Name': info.get('longName', ticker.upper()),
                'Description': info.get('longBusinessSummary', 'Company description temporarily unavailable'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'MarketCap': info.get('marketCap', 'N/A'),
                'PERatio': info.get('trailingPE', 'N/A'),
                'DividendYield': info.get('dividendYield', 'N/A'),
                '52WeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
                '52WeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
                'Exchange': info.get('exchange', 'N/A'),  # Safe access
                'Address': ', '.join(filter(None, [
                    info.get('city', ''),
                    info.get('state', ''),
                    info.get('country', '')
                ])) or 'N/A',
                'FullTimeEmployees': info.get('fullTimeEmployees', 'N/A')
            }
            
            # Format market cap and dividend yield
            if overview['MarketCap'] != 'N/A':
                overview['MarketCap'] = StockDataService._format_market_cap(float(overview['MarketCap']))
            if overview['DividendYield'] != 'N/A' and overview['DividendYield'] is not None:
                overview['DividendYield'] = f"{float(overview['DividendYield'])*100:.2f}%"
            
            return overview
            
        except Exception as e:
            logger.error(f"Error fetching company overview: {str(e)}")
            return {}
    
    @staticmethod
    def get_fundamental_data(ticker: str) -> Dict[str, Any]:
        """Get fundamental data with fallback options"""
        try:
            stock = yf.Ticker(ticker)
            
            # Try to get info with retries
            info = StockDataService._retry_with_backoff(
                lambda: stock.info,
                max_attempts=2
            )
            
            if not info:
                # Try to calculate some basic metrics from price data
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)
                    df = web.DataReader(ticker, 'stooq', start_date, end_date)
                    if not df.empty:
                        returns = df['Close'].pct_change()
                        beta = returns.std() * np.sqrt(252)  # Simple volatility as beta proxy
                        avg_volume = df['Volume'].mean()
                        latest_price = df['Close'].iloc[-1]
                        market_cap = StockDataService._calculate_market_cap(
                            latest_price, avg_volume
                        )
                        return {
                            'pe_ratio': None,
                            'industry_pe': None,
                            'market_cap': market_cap,
                            'dividend_yield': None,
                            'beta': beta
                        }
                except Exception:
                    pass
            
            return {
                'pe_ratio': info.get('trailingPE'),
                'industry_pe': info.get('industryPE'),
                'market_cap': info.get('marketCap'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta')
            }
            
        except Exception as e:
            logger.error(f"Error getting fundamental data: {str(e)}")
            return {
                'pe_ratio': None,
                'industry_pe': None,
                'market_cap': None,
                'dividend_yield': None,
                'beta': None
            }