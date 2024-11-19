# services/analysis_service.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from services.technical_indicators import TechnicalIndicators
from services.stock_data_service import StockDataService

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
    def calculate_price_targets(stock_ticker: str, df: pd.DataFrame, fundamental_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate price targets using multiple methods including risk metrics"""
        if df is None or df.empty:
            logging.error(f"Error: No valid data for {stock_ticker}")
            return None
            
        try:
            current_price = float(df['Adj Close'].iloc[-1])
                
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
            recommendation = AnalysisService.get_recommendation(
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
    def get_recommendation(
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
    
    @staticmethod
    def _calculate_period(start_date: pd.Timestamp, end_date: pd.Timestamp) -> str:
        """Calculate the appropriate period string for data fetching"""
        days_diff = (end_date - start_date).days
        
        if days_diff <= 180:
            return '6mo'
        elif days_diff <= 365:
            return '1y'
        elif days_diff <= 365 * 3:
            return '3y'
        elif days_diff <= 365 * 5:
            return '5y'
        else:
            return 'max'
    
    @staticmethod
    def run_portfolio_backtesting(holdings: List[Dict], start_date: str, end_date: str, benchmark_ticker: str = 'SPY') -> Dict:
        """Run backtesting analysis on a portfolio
        
        Args:
            holdings: List of dictionaries containing 'ticker' and 'allocation'
            start_date: Start date for backtesting period
            end_date: End date for backtesting period
            benchmark_ticker: Ticker symbol for benchmark (default: 'SPY')
            
        Returns:
            Dict containing backtest results including returns, risk metrics, etc.
        """
        try:
            logger = logging.getLogger(__name__)
            logger.info(f"Starting backtesting from {start_date} to {end_date}")
            
            # Convert dates to datetime
            start_date_obj = pd.to_datetime(start_date)
            end_date_obj = pd.to_datetime(end_date)
            
            # Calculate the period for fetching data
            period = AnalysisService._calculate_period(start_date_obj, end_date_obj)
            
            # Initialize DataFrames dictionary to store price data
            price_data = {}
            weights = {}
            
            # Fetch data for each holding
            for holding in holdings:
                ticker = holding["ticker"]
                allocation = holding["allocation"] / 100.0  # Convert percentage to decimal
                
                logger.info(f"Fetching data for {ticker} with {allocation*100}% allocation")
                
                # Fetch the data
                df = StockDataService().fetch_stock_data(ticker, period)
                
                if df is not None and not df.empty:
                    # Store only the Adj Close prices and allocation
                    price_data[ticker] = df['Adj Close']
                    weights[ticker] = allocation
                else:
                    logger.warning(f"No data available for {ticker}")
            
            if not price_data:
                raise ValueError("No valid data available for any holdings")
            
            # Create a DataFrame with all close prices
            prices_df = pd.DataFrame(price_data)
            
            # Filter to the specified date range
            prices_df = prices_df[start_date_obj:end_date_obj]
            
            if prices_df.empty:
                raise ValueError("No data available for the specified date range")
            
            # Calculate daily returns
            returns_df = prices_df.pct_change()
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0.0, index=returns_df.index)
            for ticker, weight in weights.items():
                portfolio_returns += returns_df[ticker] * weight
            
            # Calculate cumulative returns
            portfolio_cumulative = (1 + portfolio_returns).cumprod()
            
            # Get benchmark data
            benchmark_df = StockDataService().fetch_stock_data(benchmark_ticker, period)
            benchmark_returns = benchmark_df['Adj Close'].pct_change()
            benchmark_returns = benchmark_returns[start_date_obj:end_date_obj]
            benchmark_cumulative = (1 + benchmark_returns).cumprod()

            # Store the values for plotting
            dates = benchmark_returns.index
            portfolio_values = portfolio_cumulative
            benchmark_values = benchmark_cumulative
            
            # Calculate metrics
            total_return = portfolio_cumulative.iloc[-1] - 1
            benchmark_return = benchmark_cumulative.iloc[-1] - 1
            
            # Calculate alpha (excess return over benchmark)
            alpha = total_return - benchmark_return
            
            # Calculate Sharpe Ratio
            risk_free_rate = 0.02  # Assume 2% risk-free rate
            excess_returns = portfolio_returns - risk_free_rate/252
            sharpe_ratio = np.sqrt(252) * (excess_returns.mean() / portfolio_returns.std())
            
            # Calculate volatility
            volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Calculate maximum drawdown
            rolling_max = portfolio_cumulative.expanding().max()
            drawdowns = portfolio_cumulative / rolling_max - 1
            max_drawdown = drawdowns.min()
            
            # Calculate beta
            covariance = portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
            
            # Calculate additional risk metrics
            daily_returns = portfolio_returns.dropna()
            benchmark_returns = benchmark_returns.dropna()

            # Calculate Calmar Ratio
            calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else np.nan

            # Calculate up/down capture ratios
            up_market_returns = daily_returns[benchmark_returns > 0]
            down_market_returns = daily_returns[benchmark_returns < 0]
            up_market_benchmark = benchmark_returns[benchmark_returns > 0]
            down_market_benchmark = benchmark_returns[benchmark_returns < 0]

            capture_ratio_up = (up_market_returns.mean() / up_market_benchmark.mean() * 100) if not up_market_benchmark.empty else np.nan
            capture_ratio_down = (down_market_returns.mean() / down_market_benchmark.mean() * 100) if not down_market_benchmark.empty else np.nan

            # Calculate Treynor Ratio
            treynor_ratio = (total_return - risk_free_rate) / beta if beta != 0 else np.nan

            # Calculate Omega Ratio (using 0 as threshold)
            threshold = 0
            omega_ratio = np.sum(daily_returns[daily_returns > threshold]) / abs(np.sum(daily_returns[daily_returns < threshold])) if np.sum(daily_returns[daily_returns < threshold]) != 0 else np.nan

            # Calculate Tail Ratio
            tail_ratio = abs(np.percentile(daily_returns, 95)) / abs(np.percentile(daily_returns, 5)) if np.percentile(daily_returns, 5) != 0 else np.nan

            results = {
                "portfolio_return": total_return,
                "benchmark_return": benchmark_return,
                "dates": dates,
                "portfolio_values": portfolio_values,
                "benchmark_values": benchmark_values,
                "alpha": alpha,
                "beta": beta,
                "sharpe_ratio": sharpe_ratio,
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                
                # Add period-specific metrics
                "start_date": start_date_obj.strftime('%Y-%m-%d'),
                "end_date": end_date_obj.strftime('%Y-%m-%d'),
                "trading_days": len(portfolio_returns),
                
                # Add additional analytics
                "annualized_return": (1 + total_return) ** (252/len(portfolio_returns)) - 1,
                "annualized_benchmark_return": (1 + benchmark_return) ** (252/len(benchmark_returns)) - 1,
                "tracking_error": (portfolio_returns - benchmark_returns).std() * np.sqrt(252),
                "information_ratio": alpha / ((portfolio_returns - benchmark_returns).std() * np.sqrt(252)),
                'calmar_ratio': calmar_ratio,
                'treynor_ratio': treynor_ratio,
                'capture_ratio_up': capture_ratio_up,
                'capture_ratio_down': capture_ratio_down,
                'omega_ratio': omega_ratio,
                'tail_ratio': tail_ratio
            }
            
            logger.info(f"Backtesting results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error in portfolio backtesting: {str(e)}")
            raise

    @staticmethod
    def _calculate_portfolio_performance(
        portfolio_data: Dict,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> Dict:
        """Calculate portfolio performance metrics
        
        Args:
            portfolio_data: Dictionary of ticker data and allocations
            start_date: Start date for analysis
            end_date: End date for analysis
        """
        # Initialize portfolio returns series
        portfolio_returns = pd.Series(0.0, index=next(iter(portfolio_data.values()))["data"].index)
        
        # Calculate weighted returns
        for ticker, data in portfolio_data.items():
            df = data["data"]
            allocation = data["allocation"]
            
            # Calculate daily returns
            returns = df["Close"].pct_change()
            portfolio_returns += returns * allocation
        
        # Calculate performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        
        # Get S&P 500 as benchmark
        benchmark_df = StockDataService().fetch_stock_data("SPY", "1y")
        benchmark_returns = benchmark_df["Close"].pct_change()
        benchmark_total_return = (1 + benchmark_returns).prod() - 1
        
        # Calculate alpha (simplified)
        alpha = total_return - benchmark_total_return
        
        # Calculate Sharpe Ratio (annualized)
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = portfolio_returns - risk_free_rate/252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / portfolio_returns.std()
        
        return {
            "portfolio_return": total_return,
            "benchmark_return": benchmark_total_return,
            "alpha": alpha,
            "sharpe_ratio": sharpe_ratio,
            "volatility": portfolio_returns.std() * np.sqrt(252),  # Annualized volatility
            "max_drawdown": (1 + portfolio_returns).cumprod().div(
                (1 + portfolio_returns).cumprod().cummax()
            ).min() - 1
        }