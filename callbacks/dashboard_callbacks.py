# callbacks/dashboard_callbacks.py
from dash.dependencies import Input, Output, State, MATCH, ALL
from dash.exceptions import PreventUpdate
from dash import html, callback_context
import time
import logging
from typing import Dict, Any, List, Union, Tuple, Optional
import pandas as pd

from services.stock_data_service import StockDataService
from services.analysis_service import AnalysisService
from components.dashboard_components import DashboardComponents
from components.charts import ChartComponents
from technical_indicators import TechnicalIndicators

class DashboardCallbacks:
    """Class for managing dashboard callbacks"""
    
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
            [State('stock-input', 'value')]
        )
        def validate_input(n_clicks, n_submit, benchmark_ticker, ticker_value):
            if not n_clicks and not n_submit:
                raise PreventUpdate
            
            benchmark_error = ""
            
            # Check for empty input
            if not ticker_value:
                return "Please enter a ticker symbol", {'display': 'none'}, benchmark_error
            
            # Convert to uppercase and remove whitespace
            ticker_value = ticker_value.strip().upper()
            
            # Basic format validation
            if not all(c.isalpha() or c in '.-' for c in ticker_value):
                return "Invalid ticker format", {'display': 'none'}, benchmark_error
            
            # Validate benchmark ticker if provided
            if benchmark_ticker:
                try:
                    is_valid, error_msg = StockDataService.validate_ticker(benchmark_ticker)
                    if not is_valid:
                        benchmark_error = f"Benchmark error: {error_msg}"
                except Exception as e:
                    benchmark_error = f"Benchmark validation error: {str(e)}"
            
            # Add small delay to prevent too rapid requests
            time.sleep(0.1)
            
            # Try validation up to 2 times
            for attempt in range(2):
                try:
                    is_valid, error_message = StockDataService.validate_ticker(ticker_value)
                    if is_valid:
                        return "", {'display': 'block'}, benchmark_error
                    if attempt < 1:
                        time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Validation error on attempt {attempt + 1}: {str(e)}")
                    if attempt == 1:  # Only return error on final attempt
                        return f"Error validating ticker: {str(e)}", {'display': 'none'}, benchmark_error
            
            return error_message or "Unable to validate ticker", {'display': 'none'}, benchmark_error
            
            pass

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
             State('user-id', 'data')]
        )

        def update_dashboard(n_clicks, n_submit, timeframe, benchmark_ticker, ticker, user_id):
            # If no user_id is provided, use default
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
                                
                # Configure logging
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)
                logger.info(f"Fetching data for {ticker} with timeframe {timeframe}")
                
                # Fetch stock data
                df = StockDataService.fetch_stock_data(ticker, timeframe)
                if df is None or df.empty:
                    logger.error(f"No data available for {ticker}")
                    return empty_response
                
                # Fetch benchmark data if provided
                benchmark_df = None
                benchmark_name = None
                benchmark_status = None
                
                if benchmark_ticker:
                    try:
                        benchmark_df = StockDataService.fetch_stock_data(benchmark_ticker, timeframe)
                        if benchmark_df is None or benchmark_df.empty:
                            raise ValueError("No benchmark data available")
                            
                        benchmark_info = StockDataService.get_company_overview(benchmark_ticker)
                        benchmark_name = benchmark_info.get('Name', benchmark_ticker)
                        
                        # Create success status indicator
                        benchmark_status = html.Div([
                            html.I(className="fas fa-check-circle mr-2", 
                                  style={"color": "green"}),
                            html.Span(f"Benchmark loaded: {benchmark_name}")
                        ], className="benchmark-status-success")
                        
                    except Exception as e:
                        logger.warning(f"Error fetching benchmark data: {str(e)}")
                        benchmark_status = html.Div([
                            html.I(className="fas fa-exclamation-circle mr-2", 
                                  style={"color": "red"}),
                            html.Span(f"Benchmark error: {str(e)}")
                        ], className="benchmark-status-error")
                        benchmark_df = None
                else:
                    benchmark_status = html.Div([
                        html.I(className="fas fa-info-circle mr-2", 
                              style={"color": "gray"}),
                        html.Span("No benchmark selected")
                    ], className="benchmark-status-info")
                
                # Fetch company overview
                company_overview = StockDataService.get_company_overview(ticker)
                company_section = DashboardComponents.create_company_overview(company_overview)
                
                logger.info(f"Successfully fetched data for {ticker}. Calculating indicators...")
                
                # Calculate technical indicators
                df = TechnicalIndicators.calculate_all_indicators(df, 
                                                               benchmark_returns=benchmark_df['Close'].pct_change() if benchmark_df is not None else None)
                df_yearly = TechnicalIndicators.calculate_performance_metrics(df)
                
                # Get fundamental data
                fundamental_data = StockDataService.get_fundamental_data(ticker)
                logger.info("Fetched fundamental data")
                
                # Calculate analysis and recommendations
                try:
                    price_targets = AnalysisService.calculate_price_targets(
                        ticker, df, fundamental_data
                    )
                    recommendation = AnalysisService.generate_recommendation(
                        price_targets, df
                    )
                    logger.info("Generated analysis and recommendations")
                except Exception as e:
                    logger.error(f"Error in analysis: {str(e)}")
                    price_targets = None
                    recommendation = {
                        'recommendation': 'Analysis Error',
                        'confidence': 0,
                        'reasons': ['Error in analysis calculations'],
                        'color': 'gray'
                    }
                
                # Create visualizations
                try:
                    price_chart = ChartComponents.create_price_chart(
                        df, ticker, benchmark_df, benchmark_name
                    )
                    volume_chart = ChartComponents.create_volume_chart(df, ticker)
                    technical_chart = ChartComponents.create_technical_chart(df, ticker)
                    logger.info("Created charts")
                except Exception as e:
                    logger.error(f"Error creating charts: {str(e)}")
                    price_chart = empty_fig
                    volume_chart = empty_fig
                    technical_chart = empty_fig
                
                # Calculate metrics for cards
                try:
                    current_price = df['Close'].iloc[-1]
                    price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                    volatility = df['Volatility'].iloc[-1] * 100 if 'Volatility' in df else 0
                    daily_returns = df['Close'].pct_change()
                    sharpe_ratio = ((daily_returns.mean() * 252) - 0.02) / (daily_returns.std() * (252 ** 0.5))
                    
                    # Add benchmark comparison to change card if available
                    change_text = f"{price_change:.1f}%"
                    if benchmark_df is not None:
                        benchmark_change = ((benchmark_df['Close'].iloc[-1] / benchmark_df['Close'].iloc[0]) - 1) * 100
                        relative_performance = price_change - benchmark_change
                        change_text += (f" ({'+' if relative_performance > 0 else ''}{relative_performance:.1f}% "
                                      f"vs {benchmark_ticker})")
                    
                    # Create metric cards
                    price_card = DashboardComponents.create_metric_card(
                        "Current Price", f"${current_price:.2f}"
                    )
                    change_card = DashboardComponents.create_metric_card(
                        "Period Return", change_text
                    )
                    volatility_card = DashboardComponents.create_metric_card(
                        "Annualized Volatility", f"{volatility:.1f}%"
                    )
                    sharpe_card = DashboardComponents.create_metric_card(
                        "Sharpe Ratio", f"{sharpe_ratio:.2f}"
                    )
                    logger.info("Calculated metrics")
                except Exception as e:
                    logger.error(f"Error calculating metrics: {str(e)}")
                    price_card = error_card
                    change_card = error_card
                    volatility_card = error_card
                    sharpe_card = error_card
                
                # Create performance table and analysis
                try:
                    table = DashboardComponents.create_performance_table(df_yearly)
                    analysis = DashboardComponents.create_recommendation_layout(
                        recommendation, price_targets
                    )
                    logger.info("Created table and analysis")
                except Exception as e:
                    logger.error(f"Error creating table/analysis: {str(e)}")
                    table = error_message
                    analysis = error_message
                
                return [
                    table,              # stock-table
                    price_chart,        # stock-chart
                    volume_chart,       # volume-chart
                    technical_chart,    # technical-chart
                    price_card,         # current-price-card
                    change_card,        # change-card
                    volatility_card,    # volatility-card
                    sharpe_card,        # sharpe-ratio-card
                    analysis,           # recommendation-container
                    company_section,    # company-overview
                    benchmark_status    # benchmark-status
                ]
                
            except Exception as e:
                logger.error(f"Fatal error updating dashboard for {ticker}: {str(e)}")
                return empty_response

    @staticmethod
    def _get_component_id(base_id: str, user_id: str) -> Dict:
        """Helper method to create component IDs with user_id"""
        return {'type': base_id, 'user_id': user_id}