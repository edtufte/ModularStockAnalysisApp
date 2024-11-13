from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import time
import yfinance as yf
import pandas as pd
import requests
from pandas_datareader import data as pdr
import pandas_datareader.data as web
import logging
import json
import sqlite3

logger = logging.getLogger(__name__)
yf.pdr_override()

class Cache:
    def __init__(self, db_path="cache.db"):
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    ticker TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS company_info (
                    ticker TEXT PRIMARY KEY,
                    info TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_cached_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Retrieve cached stock data with proper column names and date index."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT date, open, high, low, close, adj_close as 'Adj Close', volume 
                FROM stock_data
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            try:
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(ticker, start_date, end_date),
                    parse_dates=['date']
                )
                
                if not df.empty:
                    # Rename columns to match yfinance format
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    df.set_index('Date', inplace=True)
                    return df
                    
            except Exception as e:
                logging.error(f"Error retrieving cached data: {str(e)}")
            return None

    def update_stock_data(self, ticker: str, df: pd.DataFrame):
        """Update cache with properly formatted stock data."""
        if df is None or df.empty:
            return
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"Missing required columns. Found: {df.columns}")
            return

        with sqlite3.connect(self.db_path) as conn:
            try:
                # Reset index to get date as column
                df_to_cache = df.reset_index()
                
                # Rename columns to match database schema
                df_to_cache.columns = [
                    'date' if col == 'Date' or col == df.index.name
                    else col.lower().replace(' ', '_')
                    for col in df_to_cache.columns
                ]
                
                # Ensure date is in string format
                df_to_cache['date'] = df_to_cache['date'].dt.strftime('%Y-%m-%d')
                
                # Add ticker column
                df_to_cache['ticker'] = ticker
                
                # Delete existing data for this ticker and date range
                conn.execute(
                    "DELETE FROM stock_data WHERE ticker = ? AND date BETWEEN ? AND ?",
                    (ticker, df_to_cache['date'].min(), df_to_cache['date'].max())
                )
                
                # Insert new data
                for _, row in df_to_cache.iterrows():
                    conn.execute("""
                        INSERT INTO stock_data 
                        (ticker, date, open, high, low, close, adj_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['ticker'], row['date'], row['open'], row['high'],
                        row['low'], row['close'], row['adj_close'], row['volume']
                    ))
                    
                conn.commit()
                
            except Exception as e:
                logging.error(f"Error caching data: {str(e)}")
                conn.rollback()

    def get_cached_company_info(self, ticker: str) -> Optional[dict]:
        """Retrieve cached company info."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Get cached data that's less than 24 hours old
            cursor.execute("""
                SELECT info FROM company_info 
                WHERE ticker = ? AND 
                datetime(last_updated) > datetime('now', '-24 hours')
            """, (ticker,))
            row = cursor.fetchone()
            
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    logging.error(f"Error decoding cached company info for {ticker}")
            return None

    def update_company_info(self, ticker: str, info: dict):
        """Update cache with company info."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                json_str = json.dumps(info)
                conn.execute("""
                    INSERT OR REPLACE INTO company_info (ticker, info, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (ticker, json_str))
                conn.commit()
            except Exception as e:
                logging.error(f"Error updating company info cache: {str(e)}")
                conn.rollback()

    def clear_cache(self, ticker: Optional[str] = None):
        """Clear cache for a specific ticker or all tickers."""
        with sqlite3.connect(self.db_path) as conn:
            try:
                if ticker:
                    conn.execute("DELETE FROM stock_data WHERE ticker = ?", (ticker,))
                    conn.execute("DELETE FROM company_info WHERE ticker = ?", (ticker,))
                else:
                    conn.execute("DELETE FROM stock_data")
                    conn.execute("DELETE FROM company_info")
                conn.commit()
            except Exception as e:
                logging.error(f"Error clearing cache: {str(e)}")
                conn.rollback()

class StockDataService:
    """Service class for fetching and managing stock data with caching"""
    
    _cache = Cache()
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    
    PERIODS = {
        '6mo': '6mo',
        '1y': '1y',
        '3y': '3y',
        '5y': '5y',
        'max': 'max'
    }

    @staticmethod
    def get_date_range(timeframe: str) -> Tuple[datetime, datetime]:
        """Calculate date range based on timeframe"""
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
        start_date = end_date - timedelta(days=timeframe_days[timeframe])
        while start_date.weekday() > 4:
            start_date -= timedelta(days=1)

        logger.debug(f"Calculated date range for {timeframe}: {start_date} to {end_date}")
        return start_date, end_date

    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """Validate ticker symbol with improved error handling"""
        ticker = ticker.strip().upper()
        if len(ticker) == 0 or len(ticker) > 6:
            return False, "Ticker symbol is invalid"
        
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ.-")
        if not all(char in valid_chars for char in ticker) or not ticker[0].isalpha():
            return False, "Ticker contains invalid characters"
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if info and isinstance(info, dict):
                return True, ""
        except Exception as e:
            logger.warning(f"YFinance validation failed: {str(e)}")
        
        try:
            start_date = datetime.now() - timedelta(days=5)
            df = web.DataReader(ticker, 'stooq', start_date, datetime.now())
            if not df.empty:
                return True, ""
        except Exception as e:
            logger.warning(f"Alternative validation failed: {str(e)}")
        
        return False, "Unable to verify ticker existence"

    @staticmethod
    def fetch_from_yfinance(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetches data from Yahoo Finance with improved error handling."""
        try:
            start_date, end_date = StockDataService.get_date_range(timeframe)
            
            # First try using pandas_datareader
            try:
                df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
                if not df.empty and all(col in df.columns for col in StockDataService.REQUIRED_COLUMNS):
                    return df
            except Exception as e:
                logger.warning(f"pandas_datareader fetch failed for {ticker}: {str(e)}")
            
            # If that fails, try direct yfinance
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period=timeframe, interval='1d')
                
                # Ensure column names match required format
                df = df.rename(columns={
                    'Open': 'Open',
                    'High': 'High',
                    'Low': 'Low',
                    'Close': 'Close',
                    'Adj Close': 'Adj Close',
                    'Volume': 'Volume'
                })
                
                if not df.empty and all(col in df.columns for col in StockDataService.REQUIRED_COLUMNS):
                    return df
            except Exception as e:
                logger.warning(f"yfinance direct fetch failed for {ticker}: {str(e)}")
            
            # If both methods fail, try one more time with backup method
            try:
                df = web.DataReader(ticker, 'yahoo', start_date, end_date)
                if not df.empty and all(col in df.columns for col in StockDataService.REQUIRED_COLUMNS):
                    return df
            except Exception as e:
                logger.warning(f"web.DataReader fetch failed for {ticker}: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.error(f"All fetch attempts failed for {ticker}: {str(e)}")
            return None

    @staticmethod
    def fetch_stock_data(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with improved error handling and caching"""
        try:
            logger.info(f"Fetching {timeframe} data for {ticker}")
            
            if not ticker or not timeframe:
                raise ValueError("Ticker and timeframe must be provided")
                
            start_date, end_date = StockDataService.get_date_range(timeframe)
            
            # Check cache first
            cached_data = StockDataService._cache.get_cached_data(
                ticker, 
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Using cached data for {ticker}")
                return cached_data
                
            # Try fetching fresh data
            df = StockDataService.fetch_from_yfinance(ticker, timeframe)
            
            # Validate and process the data
            if df is not None and not df.empty:
                # Check required columns
                missing_columns = [col for col in StockDataService.REQUIRED_COLUMNS if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns for {ticker}: {missing_columns}")
                    # Try to fix common column name mismatches
                    df = StockDataService._fix_column_names(df)
                    
                    # Check again after fixing
                    missing_columns = [col for col in StockDataService.REQUIRED_COLUMNS if col not in df.columns]
                    if missing_columns:
                        return None
                
                # Ensure data types
                df = StockDataService._ensure_data_types(df)
                
                # Cache the valid data
                StockDataService._cache.update_stock_data(ticker, df)
                return df
            
            logger.error(f"No valid data available for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            return None

    @staticmethod
    def _fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Fix common column name mismatches."""
        # Common variations of column names
        name_map = {
            'Adj Close': ['Adj_Close', 'AdjClose', 'Adjusted_Close', 'Adjusted Close'],
            'Adj. Close': ['Adj Close', 'AdjClose', 'Adjusted_Close', 'Adjusted Close'],
            'adj_close': ['Adj Close', 'AdjClose', 'Adjusted_Close', 'Adjusted Close'],
            'adj close': ['Adj Close', 'AdjClose', 'Adjusted_Close', 'Adjusted Close']
        }
        
        for standard_name, variants in name_map.items():
            for variant in variants:
                if variant in df.columns and standard_name not in df.columns:
                    df = df.rename(columns={variant: standard_name})
                    
        return df

    @staticmethod
    def _ensure_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure correct data types for each column."""
        try:
            # Ensure index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Ensure numeric columns are float
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Ensure Volume is int
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
            
            return df
        except Exception as e:
            logger.error(f"Error ensuring data types: {str(e)}")
            return df

    @staticmethod
    def get_company_overview(ticker: str) -> Dict[str, Any]:
        """Fetches company overview with improved caching."""
        try:
            # Check cache first
            cached_info = StockDataService._cache.get_cached_company_info(ticker)
            if cached_info:
                return cached_info
            
            # Fetch from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info and isinstance(info, dict):
                # Cache the data
                StockDataService._cache.update_company_info(ticker, info)
                return info
            
            logger.warning(f"No company info available for {ticker}")
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching company overview: {str(e)}")
            return {}