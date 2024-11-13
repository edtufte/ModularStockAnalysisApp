from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, List
import time
import yfinance as yf
import pandas as pd
import requests
from pandas_datareader import data as pdr
import pandas_datareader.data as web
import logging
import json
import re
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

    def get_cached_data(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
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
                    params=(
                        ticker, 
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    ),
                    parse_dates=['date']
                )
                
                if not df.empty:
                    # Rename columns to match yfinance format and set index
                    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                    df.set_index('Date', inplace=True)
                    return df
                    
            except Exception as e:
                logging.error(f"Error retrieving cached data: {str(e)}")
            return None

    def update_stock_data(self, ticker: str, df: pd.DataFrame):
        """Update cache with new stock data, avoiding duplicates."""
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
                
                # Use INSERT OR REPLACE to handle duplicates
                conn.executemany("""
                    INSERT OR REPLACE INTO stock_data 
                    (ticker, date, open, high, low, close, adj_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        row['ticker'],
                        row['date'],
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row['adj_close'],
                        row['volume']
                    ) for _, row in df_to_cache.iterrows()
                ])
                
                conn.commit()
                
            except Exception as e:
                logging.error(f"Error caching data: {str(e)}")
                conn.rollback()

    def get_cached_company_info(self, ticker: str) -> Optional[dict]:
        """Retrieve cached company info that's less than 24 hours old."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
    
    @staticmethod
    def normalize_stock_data(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize stock data to have consistent columns and timezone-naive dates"""
        if df is None or df.empty:
            return df
            
        # Convert index to timezone-naive datetime
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Normalize column names
        df.columns = [col.replace(' ', '_') for col in df.columns]
        
        # Handle missing Adj Close
        if 'Adj_Close' not in df.columns and 'Close' in df.columns:
            df['Adj_Close'] = df['Close']
        
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, using Close price")
                df[col] = df['Close'] if 'Close' in df.columns else None
        
        # Remove unnecessary columns
        df = df[required_columns]
        
        # Rename columns to standard format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        return df

    @staticmethod
    def should_refresh_data(ticker: str, cached_data: Optional[pd.DataFrame] = None) -> bool:
        """Determine if most recent data should be refreshed"""
        if cached_data is None:
            return True
            
        now = pd.Timestamp.now().normalize()
        last_data_time = cached_data.index[-1]
        data_age = now - last_data_time
        
        # Refresh if last data point is more than a day old
        return data_age > pd.Timedelta(days=1)
    
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

        if timeframe in timeframe_days:
            days = timeframe_days[timeframe]
        else:
            match = re.match(r"(\d+)(mo|y)", timeframe)
            if match:
                amount, unit = match.groups()
                if unit == "mo":
                    days = int(amount) * 30
                elif unit == "y":
                    days = int(amount) * 365
            else:
                logger.warning(f"Invalid timeframe '{timeframe}', defaulting to '1y'")
                days = timeframe_days['1y']

        # Set end_date to today at start of day (00:00:00)
        end_date = pd.Timestamp.now().normalize()
        start_date = (end_date - pd.Timedelta(days=days)).normalize()
        
        # Ensure timezone-naive
        if end_date.tz is not None:
            end_date = end_date.tz_localize(None)
        if start_date.tz is not None:
            start_date = start_date.tz_localize(None)
        
        logger.debug(f"Calculated date range for {timeframe}: {start_date} to {end_date}")
        return start_date, end_date

    @staticmethod
    def get_required_date_ranges(
        target_start: datetime,
        target_end: datetime,
        cached_data: Optional[pd.DataFrame]
    ) -> List[Tuple[datetime, datetime]]:
        """Calculate what date ranges need to be fetched based on cache"""
        if cached_data is None or cached_data.empty:
            return [(target_start, target_end)]
            
        # Convert to pandas timestamps and ensure timezone-naive
        target_start = pd.Timestamp(target_start).tz_localize(None)
        target_end = pd.Timestamp(target_end).tz_localize(None)
        
        # Get actual data range from cache
        cached_start = cached_data.index.min()
        cached_end = cached_data.index.max()
        
        missing_ranges = []
        
        # If we need earlier data
        if target_start < cached_start:
            missing_ranges.append((target_start, cached_start))
            
        # If we need later data
        if target_end > cached_end:
            missing_ranges.append((cached_end, target_end))
            
        return missing_ranges

    @staticmethod
    def fetch_from_yfinance(ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetches data from Yahoo Finance for a specific date range"""
        try:
            stock = yf.Ticker(ticker)
            
            # Ensure dates are timezone-naive
            start_date = pd.Timestamp(start_date).tz_localize(None)
            end_date = pd.Timestamp(end_date).tz_localize(None)
            
            df = stock.history(start=start_date, end=end_date)
            if not df.empty:
                return StockDataService.normalize_stock_data(df)
            return None
                
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
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
    def fetch_stock_data(ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with improved error handling and caching"""
        try:
            logger.info(f"Fetching {timeframe} data for {ticker}")
            
            if not ticker or not timeframe:
                raise ValueError("Ticker and timeframe must be provided")
                
            target_start, target_end = StockDataService.get_date_range(timeframe)
            logger.info(f"Target date range for {ticker}: {target_start} to {target_end}")
            
            # Get any existing cached data within our target range
            cached_data = StockDataService._cache.get_cached_data(
                ticker, 
                target_start,
                target_end
            )
            
            df_list = []
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Found cached data for {ticker} from {cached_data.index.min()} to {cached_data.index.max()}")
                df_list.append(cached_data)
                
                # Calculate what ranges we're missing
                missing_ranges = StockDataService.get_required_date_ranges(
                    target_start, target_end, cached_data
                )
            else:
                logger.info(f"No cached data found for {ticker} in target range")
                missing_ranges = [(target_start, target_end)]
            
            # Fetch any missing data
            for start_date, end_date in missing_ranges:
                logger.info(f"Fetching missing data for {ticker} from {start_date} to {end_date}")
                
                missing_df = StockDataService.fetch_from_yfinance(ticker, start_date, end_date)
                
                if missing_df is not None and not missing_df.empty:
                    df_list.append(missing_df)
                    StockDataService._cache.update_stock_data(ticker, missing_df)
            
            if not df_list:
                logger.warning(f"No data available for {ticker}")
                return None
                
            # Combine all data pieces
            final_df = pd.concat(df_list, axis=0)
            final_df = final_df[~final_df.index.duplicated(keep='last')]
            final_df.sort_index(inplace=True)
            
            # Filter to requested timeframe and ensure timezone-naive
            final_df = final_df[
                (final_df.index >= pd.Timestamp(target_start)) & 
                (final_df.index <= pd.Timestamp(target_end))
            ]
            
            logger.info(f"Final dataset for {ticker}: {len(final_df)} rows from {final_df.index.min()} to {final_df.index.max()}")
            return final_df
                
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            return None

    @staticmethod
    def get_company_overview(ticker: str) -> Dict[str, Any]:
        """Fetches company overview with improved caching"""
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