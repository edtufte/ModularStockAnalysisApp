from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html
import time

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
            if not n_clicks and not n_submit:
                raise PreventUpdate
            
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
                is_valid, error_message = StockDataService.validate_ticker(ticker_value)
                if is_valid:
                    return "", {'display': 'block'}
                if attempt < 1:
                    time.sleep(0.5)
            
            return error_message, {'display': 'none'}

        @app.callback(
            [Output('stock-table', 'children'),
             Output('stock-chart', 'figure'),
             Output('volume-chart', 'figure'),
             Output('technical-chart', 'figure'),
             Output('current-price-card', 'children'),
             Output('change-card', 'children'),
             Output('volatility-card', 'children'),
             Output('sharpe-ratio-card', 'children'),
             Output('recommendation-container', 'children')],
            [Input('analyze-button', 'n_clicks'),
             Input('stock-input', 'n_submit'),
             Input('timeframe-dropdown', 'value')],
            [State('stock-input', 'value')]
        )
        def update_dashboard(n_clicks, n_submit, timeframe, ticker):
            """Main callback to update all dashboard components"""
            if not (n_clicks or n_submit) or not ticker:
                raise PreventUpdate
            
            ticker = ticker.strip().upper()
            
            try:
                # Fetch stock data
                df = StockDataService.fetch_stock_data(ticker, timeframe)
                if df is None:
                    raise ValueError(f"No data available for {ticker}")
                
                # Calculate technical indicators
                df = TechnicalIndicators.calculate_all_indicators(df)
                df_yearly = TechnicalIndicators.calculate_performance_metrics(df)
                
                # Get fundamental data
                fundamental_data = StockDataService.get_fundamental_data(ticker)
                
                # Calculate analysis
                price_targets = AnalysisService.calculate_price_targets(
                    ticker, df, fundamental_data
                )
                recommendation = AnalysisService.generate_recommendation(
                    price_targets, df
                )
                
                # Create charts
                price_chart = ChartComponents.create_price_chart(df, ticker)
                volume_chart = ChartComponents.create_volume_chart(df, ticker)
                technical_chart = ChartComponents.create_technical_chart(df, ticker)
                
                # Calculate metrics for cards
                current_price = df['Close'].iloc[-1]
                price_change = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
                volatility = df['Volatility'].iloc[-1] * 100
                sharpe_ratio = ((df['Daily_Return'].mean() * 252) - 0.02) / (df['Daily_Return'].std() * (252 ** 0.5))
                
                # Create components
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
                
                # Create performance table
                table = DashboardComponents.create_performance_table(df_yearly)
                
                # Create recommendation layout
                analysis = DashboardComponents.create_recommendation_layout(
                    recommendation, price_targets
                )
                
                return (
                    table,              # stock-table
                    price_chart,        # stock-chart
                    volume_chart,       # volume-chart
                    technical_chart,    # technical-chart
                    price_card,         # current-price-card
                    change_card,        # change-card
                    volatility_card,    # volatility-card
                    sharpe_card,        # sharpe-ratio-card
                    analysis           # recommendation-container
                )
                
            except Exception as e:
                # Get error components
                empty_fig, error_card, error_message = DashboardComponents.create_error_components()
                
                # Return error state for all outputs
                return (
                    error_message,     # stock-table
                    empty_fig,         # stock-chart
                    empty_fig,         # volume-chart
                    empty_fig,         # technical-chart
                    error_card,        # current-price-card
                    error_card,        # change-card
                    error_card,        # volatility-card
                    error_card,        # sharpe-ratio-card
                    error_message      # recommendation-container
                )