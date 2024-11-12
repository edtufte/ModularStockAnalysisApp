import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

class AnalysisService:
    """Service for analyzing stock data and generating recommendations"""
    
    @staticmethod
    def calculate_price_targets(stock_ticker: str, df: pd.DataFrame, fundamental_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate price targets using multiple methods"""
        if df is None or df.empty:
            logging.error(f"Error: No valid data for {stock_ticker}")
            return None
            
        try:
            df = df.astype(float)
            price_targets = {}
            current_price = float(df['Close'].iloc[-1])
            
            # Technical Analysis Targets
            recent_high = df['High'].tail(20).max()
            recent_low = df['Low'].tail(20).min()
            
            # Moving Average Based Targets
            ma_20 = df['Close'].rolling(window=20).mean().iloc[-1]
            ma_50 = df['Close'].rolling(window=50).mean().iloc[-1]
            ma_200 = df['Close'].rolling(window=200).mean().iloc[-1]
            
            # Volatility Based Targets
            daily_returns = df['Close'].pct_change()
            volatility = daily_returns.std() * np.sqrt(252)
            price_range = current_price * volatility
            
            # Fundamental Analysis
            target_price_fundamental = None
            pe_ratio = fundamental_data.get('pe_ratio')
            industry_pe = fundamental_data.get('industry_pe', 20)
            
            if pe_ratio and industry_pe:
                pe_adjustment = industry_pe / pe_ratio
                target_price_fundamental = current_price * pe_adjustment
            
            # Calculate Consensus Target
            technical_target = (ma_20 + ma_50 + ma_200) / 3
            
            return {
                'current_price': current_price,
                'technical_target': technical_target,
                'support_level': recent_low,
                'resistance_level': recent_high,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'fundamental_target': target_price_fundamental,
                'volatility_range': (current_price - price_range, current_price + price_range)
            }
            
        except Exception as e:
            logging.error(f"Error in calculate_price_targets: {str(e)}")
            return None

    @staticmethod
    def generate_recommendation(price_targets: Optional[Dict[str, Any]], df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading recommendation based on multiple factors"""
        if price_targets is None:
            return {
                'recommendation': 'No Data',
                'confidence': 0,
                'reasons': ['Insufficient data for analysis'],
                'color': 'gray'
            }
        
        try:
            current_price = price_targets['current_price']
            score = 0
            max_score = 0
            reasons = []
            
            # Moving Average Analysis
            max_score += 3
            if current_price > price_targets['ma_200']:
                score += 1
                reasons.append("Price above 200-day MA (long-term uptrend)")
            if current_price > price_targets['ma_50']:
                score += 1
                reasons.append("Price above 50-day MA (medium-term uptrend)")
            if price_targets['ma_20'] > price_targets['ma_50']:
                score += 1
                reasons.append("20-day MA above 50-day MA (positive momentum)")
            
            # Support/Resistance Analysis
            max_score += 2
            if current_price < price_targets['resistance_level']:
                distance_to_resistance = (price_targets['resistance_level'] - current_price) / current_price
                if distance_to_resistance > 0.05:
                    score += 1
                    reasons.append(f"Room for upside: {distance_to_resistance*100:.1f}% to resistance")
            if current_price > price_targets['support_level']:
                score += 1
                reasons.append("Price above support level")
            
            # Fundamental Analysis
            if price_targets['fundamental_target']:
                max_score += 2
                upside = (price_targets['fundamental_target'] - current_price) / current_price
                if upside > 0.1:
                    score += 2
                    reasons.append(f"Fundamentally undervalued by {upside*100:.1f}%")
                elif upside > 0:
                    score += 1
                    reasons.append("Slightly undervalued based on fundamentals")
            
            # Calculate final recommendation
            confidence = (score / max_score) * 100 if max_score > 0 else 0
            
            recommendation_map = {
                (70, float('inf')): ('Strong Buy', 'green'),
                (60, 70): ('Buy', 'lightgreen'),
                (40, 60): ('Hold', 'yellow'),
                (30, 40): ('Sell', 'pink'),
                (float('-inf'), 30): ('Strong Sell', 'red')
            }
            
            for (lower, upper), (rec, color) in recommendation_map.items():
                if lower <= confidence < upper:
                    return {
                        'recommendation': rec,
                        'confidence': confidence,
                        'reasons': reasons,
                        'color': color
                    }
            
        except Exception as e:
            logging.error(f"Error in generate_recommendation: {str(e)}")
            
        return {
            'recommendation': 'Error',
            'confidence': 0,
            'reasons': ['Error in analysis'],
            'color': 'gray'
        }