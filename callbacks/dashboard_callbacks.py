# callbacks/dashboard_callbacks.py
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html
import time
import logging
from typing import Dict, Any, List, Union
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
             Output('main-content', 'style')],
            [Input('analyze-button', 'n_clicks'),
             Input('stock-input', 'n_submit')],
            [State('stock-input', 'value')]
        )
        def validate_input(n_clicks, n_submit, ticker_value):
            """Callback to validate stock ticker input"""
            # If no trigger, prevent update
            if not n_clicks and not n_submit:
                raise PreventUpdate
            
            # Check for empty input
            if not ticker_value:
                return "Please enter a ticker symbol", {'display': 'none'}
            
            # Convert to uppercase and remove whitespace
            ticker_value = ticker_value.strip().upper()
            
            # Basic format validation
            if not all(c.isalpha() or c in '.-' for c in ticker_value):
                return "Invalid ticker format", {'display': 'none'}
            
            # Add small delay to prevent too rapid requests
            time.sleep(0.1)
            
            # Try validation up to 2 times
            for attempt in range(2):
                try:
                    is_valid, error_message = StockDataService.validate_ticker(ticker_value)
                    if is_valid:
                        return "", {'display': 'block'}
                    if attempt < 1:
                        time.sleep(0.5)
                except Exception as e:
                    logging.error(f"Validation error on attempt {attempt + 1}: {str(e)}")
                    if attempt == 1:  # Only return error on final attempt
                        return f"Error validating ticker: {str(e)}", {'display': 'none'}
            
            return error_message or "Unable to validate ticker", {'display': 'none'}

        @app.callback(
            [Output('stock-table', 'children'),
            Output('stock-chart', 'figure'),
            Output('volume-chart', 'figure'),
            Output('technical-chart', 'figure'),
            Output('current-price-card', 'children'),
            Output('change-card', 'children'),
            Output('volatility-card', 'children'),
            Output('sharpe-ratio-card', 'children'),
            Output('recommendation-container', 'children'),
            Output('company-overview', 'children')],  # Added this output
            [Input('analyze-button', 'n_clicks'),
            Input('stock-input', 'n_submit'),
            Input('timeframe-dropdown', 'value')],
            [State('stock-input', 'value')]
        )
        def update_dashboard(n_clicks, n_submit, timeframe, ticker):
            """Main callback to update all dashboard components"""
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
                error_message      # company-overview
            ]
            
            try:
                ticker = ticker.strip().upper()
                
                # Fetch company overview
                company_overview = StockDataService.get_company_overview(ticker)
                company_section = DashboardComponents.create_company_overview(company_overview)
                
                # Configure logging
                logging.basicConfig(level=logging.INFO)
                logger = logging.getLogger(__name__)
                logger.info(f"Fetching data for {ticker} with timeframe {timeframe}")
                
                # Fetch stock data
                df = StockDataService.fetch_stock_data(ticker, timeframe)
                if df is None or df.empty:
                    logger.error(f"No data available for {ticker}")
                    return empty_response
                
                logger.info(f"Successfully fetched data for {ticker}. Calculating indicators...")
                
                # Calculate technical indicators
                df = TechnicalIndicators.calculate_all_indicators(df)
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
                    price_chart = ChartComponents.create_price_chart(df, ticker)
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
                    
                    # Create metric cards
                    price_card = DashboardComponents.create_metric_card(
                        "Current Price", f"${current_price:.2f}"
                    )
                    change_card = DashboardComponents.create_metric_card(
                        "Period Return", f"{price_change:.1f}%"
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
                    company_section     # company-overview
                ]
                
            except Exception as e:
                logger.error(f"Fatal error updating dashboard for {ticker}: {str(e)}")
                return empty_response