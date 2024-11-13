# services/analysis_service.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from technical_indicators import TechnicalIndicators

class AnalysisService:
    """Service for analyzing stock data and generating recommendations"""
    
    # Risk thresholds
    HIGH_VOLATILITY_THRESHOLD = 0.30  # 30% annualized volatility
    LOW_SHARPE_THRESHOLD = 0.5
    HIGH_SHARPE_THRESHOLD = 1.5
    HIGH_SORTINO_THRESHOLD = 1.0
    SEVERE_DRAWDOWN_THRESHOLD = -0.20  # 20% drawdown
    HIGH_VAR_THRESHOLD = -0.03  # 3% daily VaR
    
    # Technical thresholds
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    @staticmethod
    def calculate_price_targets(
        stock_ticker: str, 
        df: pd.DataFrame, 
        fundamental_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Calculate price targets using multiple methods including risk metrics"""
        if df is None or df.empty:
            logging.error(f"Error: No valid data for {stock_ticker}")
            return None
            
        try:
            current_price = float(df['Close'].iloc[-1])
            
            # Technical Analysis Targets
            technical_targets = AnalysisService._calculate_technical_targets(df)
            
            # Risk-Adjusted Targets
            risk_adjusted_targets = AnalysisService._calculate_risk_adjusted_targets(df, current_price)
            
            # Fundamental Targets
            fundamental_targets = AnalysisService._calculate_fundamental_targets(
                current_price, fundamental_data
            )
            
            # Combine all targets
            targets = {
                'current_price': current_price,
                **technical_targets,
                **risk_adjusted_targets,
                **fundamental_targets
            }
            
            # Calculate consensus target
            valid_targets = [v for k, v in targets.items() 
                           if isinstance(v, (int, float)) and 
                           k not in ['current_price', 'volatility_range']]
            
            if valid_targets:
                targets['consensus_target'] = np.median(valid_targets)
                
            return targets
            
        except Exception as e:
            logging.error(f"Error in calculate_price_targets: {str(e)}")
            return None
    
    @staticmethod
    def _calculate_technical_targets(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical analysis based price targets"""
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()
        
        # Moving averages
        ma_20 = df['SMA_20'].iloc[-1]
        ma_50 = df['SMA_50'].iloc[-1]
        ma_200 = df['SMA_200'].iloc[-1]
        
        # MACD and signal line for trend strength
        macd = df['MACD'].iloc[-1]
        signal = df['Signal_Line'].iloc[-1]
        
        # ATR for volatility bands
        atr = df['ATR'].iloc[-1]
        current_price = df['Close'].iloc[-1]
        
        return {
            'technical_target': (ma_20 + ma_50 + ma_200) / 3,
            'support_level': recent_low,
            'resistance_level': recent_high,
            'volatility_range': (current_price - 2*atr, current_price + 2*atr)
        }
    
    @staticmethod
    def _calculate_risk_adjusted_targets(
        df: pd.DataFrame, 
        current_price: float
    ) -> Dict[str, float]:
        """Calculate risk-adjusted price targets"""
        
        # Get latest risk metrics
        latest = df.iloc[-1]
        volatility = latest.get('Volatility', 0)
        var_95 = latest.get('VaR_95', 0)
        
        # Calculate risk-adjusted range
        risk_adjustment = 1 + (2 * volatility)  # Wider range for higher volatility
        upside_potential = current_price * risk_adjustment
        downside_risk = current_price * (1 - abs(var_95 * 2))  # Use VaR for downside
        
        return {
            'risk_adjusted_target': (upside_potential + downside_risk) / 2,
            'upside_target': upside_potential,
            'downside_target': downside_risk
        }
    
    @staticmethod
    def _calculate_fundamental_targets(
        current_price: float, 
        fundamental_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate fundamental analysis based price targets"""
        pe_ratio = fundamental_data.get('pe_ratio')
        industry_pe = fundamental_data.get('industry_pe', 20)
        beta = fundamental_data.get('beta', 1.0)
        
        fundamental_target = None
        if pe_ratio and industry_pe:
            # Adjust target based on PE ratio comparison
            pe_adjustment = industry_pe / pe_ratio
            base_target = current_price * pe_adjustment
            
            # Further adjust based on beta
            if beta:
                risk_adjustment = 1 + (1 - beta) * 0.1  # Adjust less for lower beta
                fundamental_target = base_target * risk_adjustment
        
        return {'fundamental_target': fundamental_target}

    @staticmethod
    def generate_recommendation(
        price_targets: Optional[Dict[str, Any]], 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive trading recommendation based on multiple factors"""
        if price_targets is None:
            return {
                'recommendation': 'No Data',
                'confidence': 0,
                'reasons': ['Insufficient data for analysis'],
                'color': 'gray',
                'risk_level': 'Unknown'
            }
        
        try:
            current_price = price_targets['current_price']
            latest = df.iloc[-1]
            
            # Initialize scoring system
            signals = []
            risk_signals = []
            total_score = 0
            max_score = 0
            
            # Technical Analysis Signals
            technical_signals = AnalysisService._analyze_technical_signals(
                df, current_price, price_targets
            )
            signals.extend(technical_signals['signals'])
            total_score += technical_signals['score']
            max_score += technical_signals['max_score']
            
            # Risk Metrics Signals
            risk_analysis = AnalysisService._analyze_risk_metrics(
                df, latest
            )
            signals.extend(risk_analysis['signals'])
            risk_signals.extend(risk_analysis['risk_signals'])
            total_score += risk_analysis['score']
            max_score += risk_analysis['max_score']
            
            # Price Target Analysis
            target_analysis = AnalysisService._analyze_price_targets(
                current_price, price_targets
            )
            signals.extend(target_analysis['signals'])
            total_score += target_analysis['score']
            max_score += target_analysis['max_score']
            
            # Calculate final recommendation
            confidence = (total_score / max_score * 100) if max_score > 0 else 0
            recommendation = AnalysisService._get_recommendation(
                confidence, risk_signals
            )
            
            return {
                'recommendation': recommendation['action'],
                'confidence': confidence,
                'reasons': signals,
                'color': recommendation['color'],
                'risk_level': recommendation['risk_level'],
                'risk_factors': risk_signals
            }
            
        except Exception as e:
            logging.error(f"Error in generate_recommendation: {str(e)}")
            return {
                'recommendation': 'Error',
                'confidence': 0,
                'reasons': ['Error in analysis'],
                'color': 'gray',
                'risk_level': 'Unknown'
            }
    
    @staticmethod
    def _analyze_technical_signals(
        df: pd.DataFrame, 
        current_price: float,
        price_targets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze technical indicators for trading signals"""
        latest = df.iloc[-1]
        signals = []
        score = 0
        max_score = 6  # Adjust based on number of signals
        
        # MACD Analysis
        if latest['MACD'] > latest['Signal_Line']:
            signals.append("MACD showing bullish momentum")
            score += 1
        
        # RSI Analysis
        if latest['RSI'] < AnalysisService.RSI_OVERSOLD:
            signals.append(f"RSI indicating oversold at {latest['RSI']:.1f}")
            score += 1
        elif latest['RSI'] > AnalysisService.RSI_OVERBOUGHT:
            signals.append(f"RSI indicating overbought at {latest['RSI']:.1f}")
            score -= 1
        
        # Moving Average Analysis
        if latest['SMA_20'] > latest['SMA_50']:
            signals.append("Short-term trend is bullish")
            score += 1
        if current_price > latest['SMA_200']:
            signals.append("Price above 200-day MA (long-term uptrend)")
            score += 1
        
        # Bollinger Bands Analysis
        if current_price < latest['BB_lower']:
            signals.append("Price below lower Bollinger Band (potential oversold)")
            score += 1
        elif current_price > latest['BB_upper']:
            signals.append("Price above upper Bollinger Band (potential overbought)")
            score -= 1
        
        # Volume Analysis
        if 'OBV' in df.columns:
            obv_trend = df['OBV'].diff().tail(5).mean()
            if obv_trend > 0:
                signals.append("Positive volume trend supporting price action")
                score += 1
        
        return {
            'signals': signals,
            'score': score,
            'max_score': max_score
        }
    
    @staticmethod
    def _analyze_risk_metrics(
        df: pd.DataFrame,
        latest: pd.Series
    ) -> Dict[str, Any]:
        """Analyze risk metrics for trading signals"""
        signals = []
        risk_signals = []
        score = 0
        max_score = 5  # Adjust based on number of signals
        
        # Volatility Analysis
        if latest['Volatility'] > AnalysisService.HIGH_VOLATILITY_THRESHOLD:
            risk_signals.append(f"High volatility: {latest['Volatility']*100:.1f}%")
            score -= 1
        
        # Sharpe Ratio Analysis
        if 'Sharpe_Ratio' in latest:
            if latest['Sharpe_Ratio'] > AnalysisService.HIGH_SHARPE_THRESHOLD:
                signals.append(f"Strong risk-adjusted returns (Sharpe: {latest['Sharpe_Ratio']:.2f})")
                score += 1
            elif latest['Sharpe_Ratio'] < AnalysisService.LOW_SHARPE_THRESHOLD:
                risk_signals.append(f"Poor risk-adjusted returns (Sharpe: {latest['Sharpe_Ratio']:.2f})")
                score -= 1
        
        # Sortino Ratio Analysis
        if 'Sortino_Ratio' in latest:
            if latest['Sortino_Ratio'] > AnalysisService.HIGH_SORTINO_THRESHOLD:
                signals.append(f"Strong downside risk-adjusted returns (Sortino: {latest['Sortino_Ratio']:.2f})")
                score += 1
        
        # Maximum Drawdown Analysis
        if latest['Max_Drawdown'] < AnalysisService.SEVERE_DRAWDOWN_THRESHOLD:
            risk_signals.append(f"Severe drawdown: {latest['Max_Drawdown']*100:.1f}%")
            score -= 1
        
        # Value at Risk Analysis
        if latest['VaR_95'] < AnalysisService.HIGH_VAR_THRESHOLD:
            risk_signals.append(f"High Value at Risk: {latest['VaR_95']*100:.1f}%")
            score -= 1
        
        return {
            'signals': signals,
            'risk_signals': risk_signals,
            'score': score,
            'max_score': max_score
        }
    
    @staticmethod
    def _analyze_price_targets(
        current_price: float,
        price_targets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze price targets for trading signals"""
        signals = []
        score = 0
        max_score = 3  # Adjust based on number of signals
        
        if 'consensus_target' in price_targets:
            target = price_targets['consensus_target']
            upside = (target - current_price) / current_price
            
            if upside > 0.20:  # More than 20% upside
                signals.append(f"Strong upside potential: {upside*100:.1f}%")
                score += 2
            elif upside > 0.10:  # More than 10% upside
                signals.append(f"Moderate upside potential: {upside*100:.1f}%")
                score += 1
            elif upside < -0.10:  # More than 10% downside
                signals.append(f"Significant downside risk: {upside*100:.1f}%")
                score -= 1
        
        return {
            'signals': signals,
            'score': score,
            'max_score': max_score
        }
    
    @staticmethod
    def _get_recommendation(
        confidence: float,
        risk_signals: List[str]
    ) -> Dict[str, str]:
        """Get final recommendation based on confidence score and risk signals"""
        risk_level = 'Low' if len(risk_signals) == 0 else \
                    'Medium' if len(risk_signals) <= 2 else 'High'
        
        if risk_level == 'High':
            confidence = max(confidence * 0.8, 0)  # Reduce confidence for high risk
        
        recommendation_map = {
            (80, float('inf')): ('Strong Buy', 'green'),
            (60, 80): ('Buy', 'lightgreen'),
            (40, 60): ('Hold', 'yellow'),
            (20, 40): ('Sell', 'pink'),
            (float('-inf'), 20): ('Strong Sell', 'red')
        }
        
        for (lower, upper), (action, color) in recommendation_map.items():
            if lower <= confidence < upper:
                return {
                    'action': action,
                    'color': color,
                    'risk_level': risk_level
                }
        
        return {
            'action': 'Hold',
            'color': 'yellow',
            'risk_level': risk_level
        }