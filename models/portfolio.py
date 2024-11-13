# models/portfolio.py
from typing import List, Dict
from services.database import Database
import logging

class Portfolio:
    def __init__(self, db: Database):
        self.db = db
        self.logger = logging.getLogger(__name__)

    def get_user_portfolios(self, user_id: int) -> List[Dict]:
        """Get all portfolios for a user"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "SELECT id, name FROM portfolios WHERE user_id = ?",
                    (user_id,)
                )
                portfolios = cursor.fetchall()
                # Convert to list of dicts
                return [{"id": p[0], "name": p[1]} for p in portfolios]
        except Exception as e:
            self.logger.error(f"Error getting user portfolios: {str(e)}")
            raise

    def create_portfolio(self, user_id: int, name: str) -> int:
        """Create a new portfolio"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO portfolios (user_id, name) VALUES (?, ?)",
                    (user_id, name)
                )
                portfolio_id = cursor.lastrowid
                self.logger.info(f"Created portfolio {portfolio_id} for user {user_id}")
                return portfolio_id
        except Exception as e:
            self.logger.error(f"Error creating portfolio: {str(e)}")
            raise

    def add_holding(self, portfolio_id: int, ticker: str, allocation: float):
        """Add a holding to a portfolio"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO portfolio_holdings (portfolio_id, ticker, allocation) VALUES (?, ?, ?)",
                    (portfolio_id, ticker, allocation)
                )
            self.logger.info(f"Added holding {ticker} with {allocation}% to portfolio {portfolio_id}")
        except Exception as e:
            self.logger.error(f"Error adding holding: {str(e)}")
            raise

    def get_portfolio_details(self, portfolio_id: int) -> Dict:
        """Get details of a specific portfolio"""
        try:
            with self.db.get_cursor() as cursor:
                # Get portfolio info
                cursor.execute(
                    "SELECT name FROM portfolios WHERE id = ?",
                    (portfolio_id,)
                )
                portfolio_info = cursor.fetchone()
                
                # Get holdings
                cursor.execute(
                    "SELECT id, ticker, allocation FROM portfolio_holdings WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                holdings = cursor.fetchall()
                
                return {
                    "id": portfolio_id,
                    "name": portfolio_info[0] if portfolio_info else None,
                    "holdings": [
                        {"id": h[0], "ticker": h[1], "allocation": h[2]}
                        for h in holdings
                    ]
                }
        except Exception as e:
            self.logger.error(f"Error getting portfolio details: {str(e)}")
            raise

    def clear_user_portfolios(self, user_id: int):
        """Clear all portfolios for a user"""
        try:
            portfolios = self.get_user_portfolios(user_id)
            for p in portfolios:
                self.delete_portfolio(p["id"])
            self.logger.info(f"Cleared all portfolios for user {user_id}")
        except Exception as e:
            self.logger.error(f"Error clearing portfolios: {str(e)}")
            raise

    def delete_portfolio(self, portfolio_id: int):
        """Delete a portfolio and its holdings"""
        try:
            with self.db.get_cursor() as cursor:
                # Delete holdings first due to foreign key constraint
                cursor.execute(
                    "DELETE FROM portfolio_holdings WHERE portfolio_id = ?",
                    (portfolio_id,)
                )
                cursor.execute(
                    "DELETE FROM portfolios WHERE id = ?",
                    (portfolio_id,)
                )
            self.logger.info(f"Deleted portfolio {portfolio_id} and its holdings")
        except Exception as e:
            self.logger.error(f"Error deleting portfolio: {str(e)}")
            raise

    def delete_holding(self, holding_id: int):
        """Delete a holding from a portfolio"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "DELETE FROM portfolio_holdings WHERE id = ?",
                    (holding_id,)
                )
            self.logger.info(f"Deleted holding {holding_id}")
        except Exception as e:
            self.logger.error(f"Error deleting holding: {str(e)}")
            raise

    def update_holding(self, holding_id: int, allocation: float):
        """Update a holding's allocation"""
        try:
            with self.db.get_cursor() as cursor:
                cursor.execute(
                    "UPDATE portfolio_holdings SET allocation = ? WHERE id = ?",
                    (allocation, holding_id)
                )
            self.logger.info(f"Updated holding {holding_id} allocation to {allocation}%")
        except Exception as e:
            self.logger.error(f"Error updating holding: {str(e)}")
            raise