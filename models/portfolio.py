# models/portfolio.py
from typing import List, Dict
from services.database import Database
import logging

class Portfolio:
    def __init__(self, db: Database):
        self.db = db

    def create_portfolio(self, user_id: int, name: str) -> int:
        """Create a new portfolio"""
        try:
            return self.db.insert_portfolio(user_id, name)
        except Exception as e:
            logging.error(f"Error creating portfolio: {str(e)}")
            raise

    def add_holding(self, portfolio_id: int, ticker: str, allocation: float):
        """Add a holding to a portfolio"""
        try:
            self.db.insert_portfolio_holding(portfolio_id, ticker, allocation)
        except Exception as e:
            logging.error(f"Error adding holding: {str(e)}")
            raise

    def get_user_portfolios(self, user_id: int) -> List[Dict]:
        """Get all portfolios for a user"""
        try:
            portfolios = self.db.get_portfolios_by_user_id(user_id)
            return [{"id": p[0], "name": p[1]} for p in portfolios]  # Changed from p[2] to p[1]
        except Exception as e:
            logging.error(f"Error getting user portfolios: {str(e)}")
            raise

    def get_portfolio_details(self, portfolio_id: int) -> Dict:
        """Get details of a specific portfolio"""
        try:
            holdings = self.db.get_portfolio_holdings(portfolio_id)
            logging.info(f"Retrieved holdings for portfolio {portfolio_id}: {holdings}")  # Debug log
            return {
                "id": portfolio_id,
                "holdings": [{"ticker": h[2], "allocation": h[3]} for h in holdings]
            }
        except Exception as e:
            logging.error(f"Error getting portfolio details: {str(e)}")
            raise

    def update_portfolio(self, portfolio_id: int, name: str):
        """Update a portfolio's name"""
        try:
            self.db.update_portfolio(portfolio_id, name)
        except Exception as e:
            logging.error(f"Error updating portfolio: {str(e)}")
            raise

    def delete_portfolio(self, portfolio_id: int):
        """Delete a portfolio"""
        try:
            self.db.delete_portfolio(portfolio_id)
        except Exception as e:
            logging.error(f"Error deleting portfolio: {str(e)}")
            raise

    def update_holding(self, holding_id: int, allocation: float):
        """Update a holding's allocation"""
        try:
            self.db.update_holding(holding_id, allocation)
        except Exception as e:
            logging.error(f"Error updating holding: {str(e)}")
            raise

    def delete_holding(self, holding_id: int):
        """Delete a holding"""
        try:
            self.db.delete_holding(holding_id)
        except Exception as e:
            logging.error(f"Error deleting holding: {str(e)}")
            raise

    def get_total_allocation(self, portfolio_id: int) -> float:
        """Get total allocation for a portfolio"""
        try:
            holdings = self.db.get_portfolio_holdings(portfolio_id)
            return sum(h[3] for h in holdings)
        except Exception as e:
            logging.error(f"Error calculating total allocation: {str(e)}")
            raise