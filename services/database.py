# services/database.py
import sqlite3
from typing import List, Tuple
from threading import local
from contextlib import contextmanager
import logging

class Database:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self._thread_local = local()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    @property
    def conn(self):
        if not hasattr(self._thread_local, 'conn'):
            self._thread_local.conn = sqlite3.connect(self.db_name)
        return self._thread_local.conn

    @contextmanager
    def get_connection(self):
        """Context manager to handle connections per thread"""
        connection = None
        try:
            connection = sqlite3.connect(self.db_name)
            yield connection
        finally:
            if connection:
                connection.close()

    @contextmanager
    def get_cursor(self):
        """Context manager to handle cursors with automatic commit/rollback"""
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                yield cursor
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def create_tables(self):
        """Create database tables"""
        with self.get_cursor() as cursor:
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password TEXT
                )
            """)
            
            # Create portfolios table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    name TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Create portfolio_holdings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_holdings (
                    id INTEGER PRIMARY KEY,
                    portfolio_id INTEGER,
                    ticker TEXT,
                    allocation REAL,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios (id) ON DELETE CASCADE
                )
            """)
            self.logger.info("Database tables created successfully")

    def insert_portfolio(self, user_id: int, name: str) -> int:
        """Insert a new portfolio and return its ID"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO portfolios (user_id, name) VALUES (?, ?)",
                (user_id, name)
            )
            portfolio_id = cursor.lastrowid
            self.logger.info(f"Created portfolio {portfolio_id} for user {user_id}")
            return portfolio_id

    def insert_portfolio_holding(self, portfolio_id: int, ticker: str, allocation: float):
        """Insert a new holding into a portfolio"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "INSERT INTO portfolio_holdings (portfolio_id, ticker, allocation) VALUES (?, ?, ?)",
                (portfolio_id, ticker, allocation)
            )
            self.logger.info(f"Added holding {ticker} with {allocation}% to portfolio {portfolio_id}")

    def get_portfolios_by_user_id(self, user_id: int) -> List[Tuple]:
        """Get all portfolios for a user"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT id, name FROM portfolios WHERE user_id = ?",
                (user_id,)
            )
            portfolios = cursor.fetchall()
            self.logger.info(f"Retrieved {len(portfolios)} portfolios for user {user_id}")
            return portfolios

    def get_portfolio_holdings(self, portfolio_id: int) -> List[Tuple]:
        """Get all holdings for a portfolio"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "SELECT * FROM portfolio_holdings WHERE portfolio_id = ?",
                (portfolio_id,)
            )
            holdings = cursor.fetchall()
            self.logger.info(f"Retrieved {len(holdings)} holdings for portfolio {portfolio_id}")
            self.logger.debug(f"Holdings data: {holdings}")  # Debug log
            return holdings

    def update_portfolio(self, portfolio_id: int, name: str):
        """Update a portfolio's name"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE portfolios SET name = ? WHERE id = ?",
                (name, portfolio_id)
            )
            self.logger.info(f"Updated portfolio {portfolio_id} name to {name}")

    def delete_portfolio(self, portfolio_id: int):
        """Delete a portfolio and its holdings"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM portfolio_holdings WHERE portfolio_id = ?",
                (portfolio_id,)
            )
            cursor.execute(
                "DELETE FROM portfolios WHERE id = ?",
                (portfolio_id,)
            )
            self.logger.info(f"Deleted portfolio {portfolio_id} and its holdings")

    def update_holding(self, holding_id: int, allocation: float):
        """Update a holding's allocation"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "UPDATE portfolio_holdings SET allocation = ? WHERE id = ?",
                (allocation, holding_id)
            )
            self.logger.info(f"Updated holding {holding_id} allocation to {allocation}%")

    def delete_holding(self, holding_id: int):
        """Delete a holding from a portfolio"""
        with self.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM portfolio_holdings WHERE id = ?",
                (holding_id,)
            )
            self.logger.info(f"Deleted holding {holding_id}")