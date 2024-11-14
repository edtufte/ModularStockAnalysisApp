# callbacks/dashboard_callbacks.py
import dash
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
    """Class for managing dashboard callbacks with improved refresh handling"""
    
    def __init__(self):
        self._data_cache = {}
        self._last_update = {}
        self.logger = logging.getLogger(__name__)
    
    def should_refresh_data(self, ticker: str, timeframe: str, cached_data: Optional[pd.DataFrame] = None) -> bool:
        """Determine if data should be refreshed based on cache and market hours"""
        if cached_data is None:
            return True
            
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Always refresh during market hours
        if market_open <= now <= market_close and now.weekday() < 5:
            return True
            
        # Check last data point
        last_data_time = cached_data.index[-1]
        data_age = now - last_data_time
        
        # Refresh if data is too old based on timeframe
        refresh_thresholds = {
            '6mo': timedelta(hours=1),
            '1y': timedelta(hours=4),
            '3y': timedelta(days=1),
            '5y': timedelta(days=1),
            'max': timedelta(days=7)
        }
        
        return data_age > refresh_thresholds.get(timeframe, timedelta(hours=1))

    def register_callbacks(self, app):
        """Register all callbacks for the dashboard with improved refresh handling"""
        
        @app.callback(
            [Output('benchmark-dropdown', 'value'),
             Output('benchmark-dropdown-backtesting', 'value')],
            [Input('analyze-button', 'n_clicks'),
             Input('stock-input', 'n_submit'),
             Input('timeframe-dropdown', 'value')],
            [State('benchmark-dropdown', 'value'),
             State('user-id', 'data')]
        )
        def refresh_benchmark_data(n_clicks, n_submit, timeframe, current_benchmark, user_id):
            """Refresh benchmark data when analyze is clicked"""
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id in ['analyze-button', 'stock-input']:
                # Update benchmark data
                if current_benchmark:
                    try:
                        self.logger.info(f"Refreshing benchmark data for {current_benchmark}")
                        StockDataService._cache.clear_cache(current_benchmark)  # Clear cache for benchmark
                        
                        # Fetch fresh benchmark data
                        df = StockDataService.fetch_stock_data(current_benchmark, timeframe)
                        if df is not None and not df.empty:
                            self.logger.info(f"Successfully refreshed benchmark data for {current_benchmark}")
                            # Keep the same benchmark selected
                            return current_benchmark, current_benchmark
                            
                    except Exception as e:
                        self.logger.error(f"Error refreshing benchmark data: {str(e)}")
                        
            raise PreventUpdate

        @app.callback(
            [Output('main-content', 'style'),
             Output('ticker-error', 'children'),
             Output('benchmark-error', 'children')],
            [Input('analyze-button', 'n_clicks'),
             Input('stock-input', 'n_submit'),
             Input('timeframe-dropdown', 'value'),
             Input('benchmark-dropdown', 'value')],
            [State('stock-input', 'value')]
        )
        def validate_and_trigger_update(n_clicks, n_submit, timeframe, benchmark_ticker, ticker_value):
            """Validate input and trigger data refresh when needed"""
            ctx = callback_context
            if not ctx.triggered:
                raise PreventUpdate
                
            # Get the ID of the component that triggered the callback
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if not ticker_value and trigger_id in ['analyze-button', 'stock-input']:
                return {'display': 'none'}, "Please enter a ticker symbol", ""
                
            if ticker_value:
                ticker_value = ticker_value.strip().upper()
                
                # Validate ticker format
                if not all(c.isalpha() or c in '.-' for c in ticker_value):
                    return {'display': 'none'}, "Invalid ticker format", ""
                
                try:
                    # Validate stock ticker
                    is_valid, error_message = StockDataService.validate_ticker(ticker_value)
                    if not is_valid:
                        return {'display': 'none'}, error_message, ""
                        
                    # Validate benchmark if provided
                    benchmark_error = ""
                    if benchmark_ticker:
                        is_valid, error_message = StockDataService.validate_ticker(benchmark_ticker)
                        if not is_valid:
                            benchmark_error = f"Benchmark error: {error_message}"
                    
                    return {'display': 'block'}, "", benchmark_error
                    
                except Exception as e:
                    return {'display': 'none'}, f"Error validating ticker: {str(e)}", ""
            
            return {'display': 'none'}, "", ""

        @app.callback(
            [Output({'type': 'stock-chart', 'user_id': MATCH}, 'figure'),
             Output({'type': 'volume-chart', 'user_id': MATCH}, 'figure'),
             Output({'type': 'technical-chart', 'user_id': MATCH}, 'figure'),
             Output({'type': 'current-price-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'change-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'volatility-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'sharpe-ratio-card', 'user_id': MATCH}, 'children'),
             Output({'type': 'stock-table', 'user_id': MATCH}, 'children'),
             Output({'type': 'recommendation-container', 'user_id': MATCH}, 'children'),
             Output({'type': 'company-overview', 'user_id': MATCH}, 'children'),
             Output({'type': 'benchmark-status', 'user_id': MATCH}, 'children')],
            [Input('analyze-button', 'n_clicks'),
             Input('stock-input', 'n_submit'),
             Input('timeframe-dropdown', 'value'),
             Input('benchmark-dropdown', 'value')],
            [State('stock-input', 'value'),
             State('user-id', 'data')]
        )
        def update_dashboard(n_clicks, n_submit, timeframe, benchmark_ticker, ticker, user_id):
            """Update dashboard with improved refresh handling"""
            ctx = callback_context
            if not ctx.triggered or not ticker:
                raise PreventUpdate
                
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            ticker = ticker.strip().upper()
            
            try:
                # Clear cache if analyze button clicked
                if trigger_id == 'analyze-button':
                    self._data_cache.clear()
                
                # Get cached data
                cache_key = f"{ticker}_{timeframe}"
                cached_data = self._data_cache.get(cache_key)
                
                # Check if refresh needed
                if self.should_refresh_data(ticker, timeframe, cached_data):
                    self.logger.info(f"Refreshing data for {ticker}")
                    df = StockDataService.fetch_stock_data(ticker, timeframe)
                    if df is not None:
                        self._data_cache[cache_key] = df
                else:
                    self.logger.info(f"Using cached data for {ticker}")
                    df = cached_data
                
                if df is None or df.empty:
                    raise ValueError(f"No data available for {ticker}")
                
                # Handle benchmark data
                benchmark_df = None
                if benchmark_ticker:
                    try:
                        self.logger.info(f"Fetching fresh benchmark data for {benchmark_ticker}")
                        # Clear cache for benchmark to ensure fresh data
                        StockDataService._cache.clear_cache(benchmark_ticker)
                        benchmark_df = StockDataService.fetch_stock_data(benchmark_ticker, timeframe)
                        
                        if benchmark_df is not None and not benchmark_df.empty:
                            # Align benchmark data with stock data
                            common_dates = df.index.intersection(benchmark_df.index)
                            if len(common_dates) > 0:
                                df = df.loc[common_dates]
                                benchmark_df = benchmark_df.loc[common_dates]
                                self.logger.info(f"Successfully aligned benchmark data with {len(common_dates)} data points")
                            else:
                                self.logger.warning("No overlapping dates between stock and benchmark data")
                                benchmark_df = None
                    except Exception as e:
                        self.logger.error(f"Error fetching benchmark data: {str(e)}")
                        benchmark_df = None
                
                # Calculate indicators and metrics
                df = TechnicalIndicators.calculate_all_indicators(
                    df,
                    benchmark_returns=benchmark_df['Close'].pct_change() if benchmark_df is not None else None
                )
                
                # Calculate performance metrics
                df_yearly = TechnicalIndicators.calculate_performance_metrics(df)
                
                # Get fundamental data and create components
                fundamental_data = StockDataService.get_company_overview(ticker)
                company_overview = StockDataService.get_company_overview(ticker)
                
                # Calculate price targets and recommendations
                price_targets = AnalysisService.calculate_price_targets(ticker, df, fundamental_data)
                recommendation = AnalysisService.generate_recommendation(price_targets, df)
                
                # Create charts
                price_chart = ChartComponents.create_price_chart(df, ticker, benchmark_df, benchmark_ticker)
                volume_chart = ChartComponents.create_volume_chart(df, ticker)
                technical_chart = ChartComponents.create_technical_chart(df, ticker)
                
                # Calculate metrics
                metrics = DashboardComponents._calculate_metrics(df, benchmark_df, benchmark_ticker)
                
                return [
                    price_chart,      # stock-chart
                    volume_chart,     # volume-chart
                    technical_chart,  # technical-chart
                    metrics['price_card'],      # current-price-card
                    metrics['change_card'],     # change-card
                    metrics['volatility_card'], # volatility-card
                    metrics['sharpe_card'],     # sharpe-ratio-card
                    DashboardComponents.create_performance_table(df_yearly),  # stock-table
                    DashboardComponents.create_recommendation_layout(  # recommendation-container
                        recommendation, price_targets
                    ),
                    DashboardComponents.create_company_overview(company_overview),  # company-overview
                    metrics.get('benchmark_status', html.Div())  # benchmark-status
                ]
                
            except Exception as e:
                self.logger.error(f"Error updating dashboard: {str(e)}")
                empty_fig, error_card, error_message = DashboardComponents.create_error_components()
                return [
                    empty_fig,        # stock-chart
                    empty_fig,        # volume-chart
                    empty_fig,        # technical-chart
                    error_card,       # current-price-card
                    error_card,       # change-card
                    error_card,       # volatility-card
                    error_card,       # sharpe-ratio-card
                    error_message,    # stock-table
                    error_message,    # recommendation-container
                    error_message,    # company-overview
                    html.Div(        # benchmark-status
                        [html.I(className="fas fa-exclamation-circle mr-2", 
                               style={"color": "red"}),
                         html.Span("Error loading data")],
                        className="benchmark-status-error"
                    )
                ]

        # Additional callback for interval-based refresh during market hours
        @app.callback(
            Output('refresh-trigger', 'children'),
            [Input('refresh-interval', 'n_intervals')],
            [State('stock-input', 'value'),
             State('timeframe-dropdown', 'value')]
        )
        def handle_auto_refresh(n_intervals, ticker, timeframe):
            if not ticker:
                raise PreventUpdate
                
            ticker = ticker.strip().upper()
            
            # Clear cache for current ticker during market hours
            if self.should_refresh_data(ticker, timeframe):
                cache_key = f"{ticker}_{timeframe}"
                if cache_key in self._data_cache:
                    del self._data_cache[cache_key]
            
            return ""