# callbacks/dashboard_callbacks.py
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash import html, callback_context
import time
import logging
from typing import Dict, Any, List, Union, Tuple, Optional
import pandas as pd
from functools import lru_cache
from datetime import datetime, timedelta

from services.stock_data_service import StockDataService
from services.analysis_service import AnalysisService
from components.dashboard_components import DashboardComponents
from components.charts import ChartComponents
from technical_indicators import TechnicalIndicators

class DashboardCallbacks:
    """Class for managing dashboard callbacks with optimized data handling"""
    
    # Class-level cache for benchmark data
    _benchmark_cache = {}
    
    @staticmethod
    @lru_cache(maxsize=10)
    def _get_cached_benchmark(benchmark_ticker: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Get cached benchmark data with LRU caching"""
        cache_key = f"{benchmark_ticker}_{timeframe}"
        if cache_key in DashboardCallbacks._benchmark_cache:
            return DashboardCallbacks._benchmark_cache[cache_key]
        return None

    @staticmethod
    def _cache_benchmark(benchmark_ticker: str, timeframe: str, df: pd.DataFrame):
        """Cache benchmark data"""
        cache_key = f"{benchmark_ticker}_{timeframe}"
        DashboardCallbacks._benchmark_cache[cache_key] = df

    @staticmethod
    def _should_update_data(current_data: Optional[pd.DataFrame], 
                          new_timeframe: str, 
                          old_timeframe: str) -> bool:
        """Determine if data needs to be updated based on timeframe change"""
        if current_data is None or current_data.empty:
            return True
            
        time_periods = {
            '6mo': 180,
            '1y': 365,
            '3y': 1095,
            '5y': 1825,
            'max': 3650
        }
        
        new_days = time_periods[new_timeframe]
        old_days = time_periods[old_timeframe] if old_timeframe else 0
        
        if new_days <= old_days:
            return False
            
        oldest_date = current_data.index.min()
        required_start = datetime.now() - timedelta(days=new_days)
        
        return oldest_date > required_start

    @staticmethod
    def register_callbacks(app):
        """Register all callbacks for the dashboard"""
        
        @app.callback(
            [Output('ticker-error', 'children'),
             Output('main-content', 'style'),
             Output('benchmark-error', 'children')],
            [Input('analyze-button', 'n_clicks'),
             Input('stock-input', 'n_submit'),
             Input('benchmark-dropdown', 'value')],
            [State('stock-input', 'value'),
             State('timeframe-dropdown', 'value')]
        )
        def validate_input(n_clicks, n_submit, benchmark_ticker, ticker_value, timeframe):
            if not n_clicks and not n_submit:
                raise PreventUpdate
            
            benchmark_error = ""
            
            # Basic input validation
            if not ticker_value:
                return "Please enter a ticker symbol", {'display': 'none'}, benchmark_error
            
            ticker_value = ticker_value.strip().upper()
            if not all(c.isalpha() or c in '.-' for c in ticker_value):
                return "Invalid ticker format", {'display': 'none'}, benchmark_error
            
            # Validate benchmark if provided
            if benchmark_ticker:
                try:
                    # Check cache first
                    cached_benchmark = DashboardCallbacks._get_cached_benchmark(benchmark_ticker, timeframe)
                    if cached_benchmark is None:
                        is_valid, error_msg = StockDataService.validate_ticker(benchmark_ticker)
                        if not is_valid:
                            benchmark_error = f"Benchmark error: {error_msg}"
                except Exception as e:
                    benchmark_error = f"Benchmark validation error: {str(e)}"
            
            # Validate stock ticker
            try:
                is_valid, error_message = StockDataService.validate_ticker(ticker_value)
                if is_valid:
                    return "", {'display': 'block'}, benchmark_error
            except Exception as e:
                return f"Error validating ticker: {str(e)}", {'display': 'none'}, benchmark_error
            
            return error_message or "Unable to validate ticker", {'display': 'none'}, benchmark_error

        @app.callback(
            [Output({'type': 'stock-table', 'user_id': MATCH}, 'children'),
             Output({'type': 'stock-chart', 'user_id': MATCH}, 'figure'),
             Output({'type': 'volume-chart', 'user_id': MATCH}, 'figure'),
             Output({'type': 'technical-chart', 'user_id': MATCH}, 'figure'),
             Output({'type': 'current-price-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'change-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'volatility-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'sharpe-ratio-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'recommendation-container', 'user_id': MATCH}, 'children'),
             Output({'type': 'company-overview', 'user_id': MATCH}, 'children'),
             Output({'type': 'benchmark-status', 'user_id': MATCH}, 'children')],
            [Input('analyze-button', 'n_clicks'),
             Input('stock-input', 'n_submit'),
             Input('timeframe-dropdown', 'value'),
             Input('benchmark-dropdown', 'value')],
            [State('stock-input', 'value'),
             State('user-id', 'data'),
             State('timeframe-dropdown', 'value')]
        )
        def update_dashboard(n_clicks, n_submit, timeframe, benchmark_ticker, ticker, user_id, prev_timeframe):
            user_id = user_id or 'default'
            if not (n_clicks or n_submit) or not ticker:
                raise PreventUpdate
                
            # Initialize error components
            empty_fig, error_card, error_message = DashboardComponents.create_error_components()
            empty_response = [
                error_message,     # stock-table
                empty_fig,         # stock-chart
                empty_fig,         # volume-chart
                empty_fig,         # technical-chart
                error_card,        # current-price-card
                error_card,        # change-card
                error_card,        # volatility-card
                error_card,        # sharpe-ratio-card
                error_message,     # recommendation-container
                error_message,     # company-overview
                None              # benchmark-status
            ]
            
            try:
                ticker = ticker.strip().upper()
                logger = logging.getLogger(__name__)
                logger.info(f"Fetching data for {ticker} with timeframe {timeframe}")
                
                # Check if we need to update the data based on timeframe change
                should_update = DashboardCallbacks._should_update_data(
                    StockDataService._cache.get_cached_data(ticker, *StockDataService._get_date_range(timeframe)),
                    timeframe,
                    prev_timeframe
                )
                
                # Fetch stock data
                df = StockDataService.fetch_stock_data(ticker, timeframe if should_update else prev_timeframe)
                if df is None or df.empty:
                    logger.error(f"No data available for {ticker}")
                    return empty_response
                
                # Handle benchmark data
                benchmark_df = None
                benchmark_name = None
                benchmark_status = None
                
                if benchmark_ticker:
                    try:
                        # Try to get cached benchmark data first
                        benchmark_df = DashboardCallbacks._get_cached_benchmark(benchmark_ticker, timeframe)
                        
                        if benchmark_df is None:
                            benchmark_df = StockDataService.fetch_stock_data(benchmark_ticker, timeframe)
                            if benchmark_df is not None and not benchmark_df.empty:
                                DashboardCallbacks._cache_benchmark(benchmark_ticker, timeframe, benchmark_df)
                        
                        if benchmark_df is not None and not benchmark_df.empty:
                            benchmark_info = StockDataService.get_company_overview(benchmark_ticker)
                            benchmark_name = benchmark_info.get('Name', benchmark_ticker)
                            
                            benchmark_status = html.Div([
                                html.I(className="fas fa-check-circle mr-2", style={"color": "green"}),
                                html.Span(f"Benchmark loaded: {benchmark_name}")
                            ], className="benchmark-status-success")
                        else:
                            raise ValueError("No benchmark data available")
                            
                    except Exception as e:
                        logger.warning(f"Error fetching benchmark data: {str(e)}")
                        benchmark_status = html.Div([
                            html.I(className="fas fa-exclamation-circle mr-2", style={"color": "red"}),
                            html.Span(f"Benchmark error: {str(e)}")
                        ], className="benchmark-status-error")
                        benchmark_df = None
                else:
                    benchmark_status = html.Div([
                        html.I(className="fas fa-info-circle mr-2", style={"color": "gray"}),
                        html.Span("No benchmark selected")
                    ], className="benchmark-status-info")
                
                # Calculate indicators and generate visualizations
                df = TechnicalIndicators.calculate_all_indicators(
                    df,
                    benchmark_returns=benchmark_df['Close'].pct_change() if benchmark_df is not None else None
                )
                
                # Generate all components
                components = DashboardCallbacks._generate_dashboard_components(
                    df, ticker, benchmark_df, benchmark_name
                )
                
                return components
                
            except Exception as e:
                logger.error(f"Fatal error updating dashboard for {ticker}: {str(e)}")
                return empty_response

    @staticmethod
    def _generate_dashboard_components(
        df: pd.DataFrame,
        ticker: str,
        benchmark_df: Optional[pd.DataFrame],
        benchmark_name: Optional[str]
    ) -> List[Any]:
        """Generate all dashboard components"""
        try:
            # Calculate performance metrics
            df_yearly = TechnicalIndicators.calculate_performance_metrics(df)
            
            # Get fundamental data and company overview
            fundamental_data = StockDataService.get_fundamental_data(ticker)
            company_overview = StockDataService.get_company_overview(ticker)
            company_section = DashboardComponents.create_company_overview(company_overview)
            
            # Calculate analysis and recommendations
            price_targets = AnalysisService.calculate_price_targets(ticker, df, fundamental_data)
            recommendation = AnalysisService.generate_recommendation(price_targets, df)
            
            # Create visualizations
            price_chart = ChartComponents.create_price_chart(df, ticker, benchmark_df, benchmark_name)
            volume_chart = ChartComponents.create_volume_chart(df, ticker)
            technical_chart = ChartComponents.create_technical_chart(df, ticker)
            
            # Calculate metrics for cards
            metrics = DashboardCallbacks._calculate_metrics(df, benchmark_df)
            
            return [
                DashboardComponents.create_performance_table(df_yearly),  # stock-table
                price_chart,        # stock-chart
                volume_chart,       # volume-chart
                technical_chart,    # technical-chart
                metrics['price_card'],         # current-price-card
                metrics['change_card'],        # change-card
                metrics['volatility_card'],    # volatility-card
                metrics['sharpe_card'],        # sharpe-ratio-card
                DashboardComponents.create_recommendation_layout(
                    recommendation, price_targets
                ),  # recommendation-container
                company_section,    # company-overview
                metrics['benchmark_status']    # benchmark-status
            ]
            
        except Exception as e:
            logger.error(f"Error generating dashboard components: {str(e)}")
            raise

    @staticmethod
    def _calculate_metrics(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Calculate all metrics for dashboard cards"""
        current_price = df['Adj Close'].iloc[-1]
        price_change = ((df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) - 1) * 100
        volatility = df['Volatility'].iloc[-1] * 100
        
        daily_returns = df['Adj Close'].pct_change()
        sharpe_ratio = ((daily_returns.mean() * 252) - 0.02) / (daily_returns.std() * (252 ** 0.5))
        
        # Add benchmark comparison to change card if available
        change_text = f"{price_change:.1f}%"
        if benchmark_df is not None:
            benchmark_change = ((benchmark_df['Adj Close'].iloc[-1] / benchmark_df['Adj Close'].iloc[0]) - 1) * 100
            relative_performance = price_change - benchmark_change
            change_text += (f" ({'+' if relative_performance > 0 else ''}{relative_performance:.1f}% "
                        f"vs {benchmark_df.name})")
        
        return {
            'price_card': DashboardComponents.create_metric_card(
                "Current Price", f"${current_price:.2f}"
            ),
            'change_card': DashboardComponents.create_metric_card(
                "Period Return", change_text
            ),
            'volatility_card': DashboardComponents.create_metric_card(
                "Annualized Volatility", f"{volatility:.1f}%"
            ),
            'sharpe_card': DashboardComponents.create_metric_card(
                "Sharpe Ratio", f"{sharpe_ratio:.2f}"
            )
        }