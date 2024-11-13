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
import sqlite3
import json
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataCache:
    """Handle caching of stock data in SQLite database"""
    
    def __init__(self, db_path: str = "stock_cache.db"):
        self.db_path = db_path
        self.lock = Lock()  # Thread safety for cache operations
        self._create_tables()

    def _create_tables(self):
        """Create necessary tables for caching"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table for stock price data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        ticker TEXT,
                        date DATE,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        adj_close REAL,
                        last_updated TIMESTAMP,
                        PRIMARY KEY (ticker, date)
                    )
                """)
                
                # Table for company information
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS company_info (
                        ticker TEXT PRIMARY KEY,
                        info JSON,
                        last_updated TIMESTAMP
                    )
                """)
                
                # Table for fundamental data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fundamental_data (
                        ticker TEXT PRIMARY KEY,
                        data JSON,
                        last_updated TIMESTAMP
                    )
                """)
                
                # Table for data ranges to track what we have cached
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_ranges (
                        ticker TEXT PRIMARY KEY,
                        start_date DATE,
                        end_date DATE,
                        last_updated TIMESTAMP
                    )
                """)
                
                # Index for faster date range queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stock_data_date 
                    ON stock_data(ticker, date)
                """)
                
                conn.commit()

    def get_cached_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Retrieve cached stock data for the given period"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    query = """
                        SELECT date, open, high, low, close, volume, adj_close
                        FROM stock_data
                        WHERE ticker = ? AND date BETWEEN ? AND ?
                        ORDER BY date
                    """
                    df = pd.read_sql_query(
                        query,
                        conn,
                        params=(ticker, start_date.date(), end_date.date()),
                        parse_dates=['date']
                    )
                    if not df.empty:
                        df.set_index('date', inplace=True)
                        df.rename(columns={'adj_close': 'Adj Close'}, inplace=True)
                        return df
                    return None
        except Exception as e:
            logger.error(f"Error retrieving cached data: {str(e)}")
            return None

    def get_data_range(self, ticker: str) -> Optional[Tuple[datetime, datetime]]:
        """Get the date range of cached data for a ticker"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT start_date, end_date
                        FROM data_ranges
                        WHERE ticker = ?
                    """, (ticker,))
                    result = cursor.fetchone()
                    if result:
                        return (
                            datetime.strptime(result[0], '%Y-%m-%d'),
                            datetime.strptime(result[1], '%Y-%m-%d')
                        )
                    return None
        except Exception as e:
            logger.error(f"Error getting data range: {str(e)}")
            return None
        
    def update_stock_data(self, ticker: str, df: pd.DataFrame):
        """Update or insert new stock data"""
        if df is None or df.empty:
            return
            
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Convert DataFrame to records
                    records = []
                    for date, row in df.iterrows():
                        records.append((
                            ticker,
                            date.date(),
                            row['Open'],
                            row['High'],
                            row['Low'],
                            row['Close'],
                            row['Volume'],
                            row.get('Adj Close', row['Close']),  # Use Close if Adj Close not available
                            datetime.now()
                        ))
                    
                    # Use REPLACE to update existing records or insert new ones
                    conn.executemany("""
                        REPLACE INTO stock_data 
                        (ticker, date, open, high, low, close, volume, adj_close, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, records)
                    
                    # Update data range
                    if records:
                        start_date = min(r[1] for r in records)
                        end_date = max(r[1] for r in records)
                        
                        conn.execute("""
                            REPLACE INTO data_ranges 
                            (ticker, start_date, end_date, last_updated)
                            VALUES (?, ?, ?, ?)
                        """, (ticker, start_date, end_date, datetime.now()))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating stock data cache: {str(e)}")

    def update_partial_data(self, ticker: str, df: pd.DataFrame, start_date: datetime, end_date: datetime):
        """Update a specific date range of data"""
        if df is None or df.empty:
            return
            
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Delete existing data in the date range
                    conn.execute("""
                        DELETE FROM stock_data 
                        WHERE ticker = ? AND date BETWEEN ? AND ?
                    """, (ticker, start_date.date(), end_date.date()))
                    
                    # Insert new data
                    records = []
                    for date, row in df.iterrows():
                        if start_date.date() <= date.date() <= end_date.date():
                            records.append((
                                ticker,
                                date.date(),
                                row['Open'],
                                row['High'],
                                row['Low'],
                                row['Close'],
                                row['Volume'],
                                row.get('Adj Close', row['Close']),
                                datetime.now()
                            ))
                    
                    if records:
                        conn.executemany("""
                            INSERT INTO stock_data 
                            (ticker, date, open, high, low, close, volume, adj_close, last_updated)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, records)
                        
                        # Update data range if necessary
                        current_range = self.get_data_range(ticker)
                        if current_range:
                            new_start = min(current_range[0], start_date)
                            new_end = max(current_range[1], end_date)
                        else:
                            new_start = start_date
                            new_end = end_date
                            
                        conn.execute("""
                            REPLACE INTO data_ranges 
                            (ticker, start_date, end_date, last_updated)
                            VALUES (?, ?, ?, ?)
                        """, (ticker, new_start.date(), new_end.date(), datetime.now()))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating partial data: {str(e)}")

    def get_cached_company_info(self, ticker: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Retrieve cached company info if not expired"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT info, last_updated
                        FROM company_info
                        WHERE ticker = ?
                    """, (ticker,))
                    result = cursor.fetchone()
                    
                    if result:
                        info, last_updated = result
                        last_updated = datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S.%f')
                        
                        if datetime.now() - last_updated < timedelta(hours=max_age_hours):
                            return json.loads(info)
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving cached company info: {str(e)}")
            return None
    def update_company_info(self, ticker: str, info: Dict):
        """Update or insert company information"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        REPLACE INTO company_info (ticker, info, last_updated)
                        VALUES (?, ?, ?)
                    """, (ticker, json.dumps(info), datetime.now()))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating company info cache: {str(e)}")

    def get_cached_fundamental_data(self, ticker: str, max_age_hours: int = 24) -> Optional[Dict]:
        """Retrieve cached fundamental data if not expired"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT data, last_updated
                        FROM fundamental_data
                        WHERE ticker = ?
                    """, (ticker,))
                    result = cursor.fetchone()
                    
                    if result:
                        data, last_updated = result
                        last_updated = datetime.strptime(last_updated, '%Y-%m-%d %H:%M:%S.%f')
                        
                        if datetime.now() - last_updated < timedelta(hours=max_age_hours):
                            return json.loads(data)
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving cached fundamental data: {str(e)}")
            return None

    def update_fundamental_data(self, ticker: str, data: Dict):
        """Update or insert fundamental data"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        REPLACE INTO fundamental_data (ticker, data, last_updated)
                        VALUES (?, ?, ?)
                    """, (ticker, json.dumps(data), datetime.now()))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating fundamental data cache: {str(e)}")

    def clear_old_cache(self, max_age_days: int = 30):
        """Clear cache entries older than specified days"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cutoff_date = datetime.now() - timedelta(days=max_age_days)
                    
                    conn.execute("""
                        DELETE FROM company_info 
                        WHERE last_updated < ?
                    """, (cutoff_date,))
                    
                    conn.execute("""
                        DELETE FROM fundamental_data 
                        WHERE last_updated < ?
                    """, (cutoff_date,))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error clearing old cache: {str(e)}")

class StockDataService:
    """Service class for fetching and managing stock data with caching"""
    
    PERIODS = {
        '6mo': '6mo',
        '1y': '1y',
        '3y': '3y',
        '5y': '5y',
        'max': 'max'
    }
    
    _cache = StockDataCache()
    
    @staticmethod
    def get_date_range(timeframe: str) -> Tuple[datetime, datetime]:
        """Convert timeframe string to start and end dates with validation
        
        Args:
            timeframe (str): Time period ('6mo', '1y', '3y', '5y', 'max')
            
        Returns:
            Tuple[datetime, datetime]: (start_date, end_date)
            
        Raises:
            ValueError: If invalid timeframe provided
        """
        timeframe_days = {
            '6mo': 180,
            '1y': 365,
            '3y': 1095,
            '5y': 1825,
            'max': 3650  # 10 years
        }
        
        if timeframe not in timeframe_days:
            logger.warning(f"Invalid timeframe '{timeframe}', defaulting to '1y'")
            timeframe = '1y'
        
        end_date = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999)
        days = timeframe_days[timeframe]
        start_date = end_date - timedelta(days=days)
        
        # Adjust for weekends and common holidays
        while start_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
            start_date -= timedelta(days=1)
        
        logger.debug(f"Calculated date range for {timeframe}: {start_date} to {end_date}")
        return start_date, end_date

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
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """Validate stock ticker symbol with multiple attempts and sources"""
        try:
            ticker = ticker.strip().upper()
            
            if len(ticker) == 0:
                return False, "Ticker symbol cannot be empty"
            if len(ticker) > 6:
                return False, "Ticker symbol too long"
                
            valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ.-")
            if not all(char in valid_chars for char in ticker):
                return False, "Ticker contains invalid characters"
                
            if not ticker[0].isalpha():
                return False, "Ticker must start with a letter"
            
            try:
                stock = yf.Ticker(ticker)
                info = StockDataService._retry_with_backoff(
                    lambda: stock.info,
                    max_attempts=2
                )
                if info and isinstance(info, dict):
                    return True, ""
            except Exception as e:
                logger.warning(f"YFinance validation failed: {str(e)}")
                
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
        """Fetch stock data with smart caching and error handling"""
        try:
            logger.info(f"Fetching {timeframe} data for {ticker}")
            
            start_date, end_date = StockDataService.get_date_range(timeframe)
            
            # Check cache first
            cached_data = StockDataService._cache.get_cached_data(ticker, start_date, end_date)
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Using cached data for {ticker}")
                return cached_data
            
            # Check if we have partial data that needs updating
            cached_range = StockDataService._cache.get_data_range(ticker)
            if cached_range:
                cached_start, cached_end = cached_range
                
                # If we need older data
                if start_date < cached_start:
                    old_data = StockDataService._fetch_from_yfinance(
                        ticker,
                        start=start_date,
                        end=cached_start - timedelta(days=1)
                    )
                    if old_data is not None:
                        StockDataService._cache.update_partial_data(
                            ticker, old_data, start_date, cached_start - timedelta(days=1)
                        )
                
                # If we need newer data
                if end_date > cached_end:
                    new_data = StockDataService._fetch_from_yfinance(
                        ticker,
                        start=cached_end + timedelta(days=1),
                        end=end_date
                    )
                    if new_data is not None:
                        StockDataService._cache.update_partial_data(
                            ticker, new_data, cached_end + timedelta(days=1), end_date
                        )
                
                # Get complete dataset from cache
                return StockDataService._cache.get_cached_data(ticker, start_date, end_date)
            
            # Fetch complete new dataset
            df = StockDataService._fetch_from_yfinance(ticker, timeframe)
            if df is None:
                df = StockDataService._fetch_from_alternative(ticker, start_date, end_date)
            
            if df is not None and not df.empty:
                df = StockDataService._process_dataframe(df)
                StockDataService._cache.update_stock_data(ticker, df)
                return df
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return None

    @staticmethod
    def get_company_overview(ticker: str) -> Dict[str, Any]:
        """Get company overview with caching"""
        try:
            cached_info = StockDataService._cache.get_cached_company_info(ticker)
            if cached_info:
                return cached_info
                
            stock = yf.Ticker(ticker)
            info = StockDataService._retry_with_backoff(
                lambda: stock.info,
                max_attempts=2
            )
            
            if not info:
                logger.warning("Failed to get detailed info, using alternative source")
                try:
                    df = StockDataService._fetch_from_alternative(
                        ticker,
                        datetime.now() - timedelta(days=30),
                        datetime.now()
                    )
                    if df is not None and not df.empty:
                        info = {
                            'longName': ticker.upper(),
                            'marketCap': StockDataService._calculate_market_cap(
                                df['Close'].iloc[-1],
                                df['Volume'].mean()
                            ),
                            'fiftyTwoWeekHigh': df['High'].max(),
                            'fiftyTwoWeekLow': df['Low'].min()
                        }
                except Exception as e:
                    logger.error(f"Alternative source error: {str(e)}")
                    info = {}
            
            overview = StockDataService._format_company_overview(info)
            StockDataService._cache.update_company_info(ticker, overview)
            return overview
            
        except Exception as e:
            logger.error(f"Error fetching company overview: {str(e)}")
            return {}
    
    @staticmethod
    def _format_company_overview(info: Dict) -> Dict:
        """Format raw company info into standardized overview"""
        return {
            'Name': info.get('longName', 'N/A'),
            'Description': info.get('longBusinessSummary', 'Company description unavailable'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'MarketCap': StockDataService._format_market_cap(info.get('marketCap')),
            'PERatio': info.get('trailingPE', 'N/A'),
            'DividendYield': f"{float(info.get('dividendYield', 0))*100:.2f}%" if info.get('dividendYield') else 'N/A',
            '52WeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52WeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
            'Exchange': info.get('exchange', 'N/A'),
            'Address': ', '.join(filter(None, [
                info.get('city', ''),
                info.get('state', ''),
                info.get('country', '')
            ])) or 'N/A',
            'FullTimeEmployees': info.get('fullTimeEmployees', 'N/A')
        }

    @staticmethod
    def _format_market_cap(market_cap: Optional[float]) -> str:
        """Format market cap with appropriate suffix"""
        if not market_cap:
            return 'N/A'
        if market_cap >= 1e12:
            return f"${market_cap/1e12:.2f}T"
        if market_cap >= 1e9:
            return f"${market_cap/1e9:.2f}B"
        if market_cap >= 1e6:
            return f"${market_cap/1e6:.2f}M"
        return f"${market_cap:.2f}"

    @staticmethod
    def _calculate_market_cap(price: float, volume: float) -> float:
        """Estimate market cap from price and volume"""
        return price * (volume * 30)  # Rough estimation using monthly volume

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame format"""
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        
        column_map = {
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Adj Close': 'Adj Close',
            'Adj. Close': 'Adj Close',
            'Adj. Volume': 'Volume'
        }
        df.rename(columns=column_map, inplace=True)
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns")
        
        return df