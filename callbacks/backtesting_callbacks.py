# callbacks/backtesting_callbacks.py
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html
import logging
from datetime import datetime
import pandas as pd
from models.portfolio import Portfolio
from services.analysis_service import AnalysisService
from services.stock_data_service import StockDataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_callbacks(app, db):
    portfolio = Portfolio(db)
    
    @app.callback(
        [Output("backtesting-date-range", "min_date_allowed"),
         Output("backtesting-date-range", "max_date_allowed"),
         Output("backtesting-date-range", "start_date"),
         Output("backtesting-date-range", "end_date")],
        [Input("portfolio-dropdown-backtesting", "value")]
    )
    def update_date_range(portfolio_id):
        """Update the allowed date range based on available data"""
        if not portfolio_id:
            today = datetime.now()
            one_year_ago = today.replace(year=today.year - 1)
            return (
                "2010-01-01",
                today.strftime("%Y-%m-%d"),
                one_year_ago.strftime("%Y-%m-%d"),
                today.strftime("%Y-%m-%d")
            )
            
        try:
            details = portfolio.get_portfolio_details(portfolio_id)
            if not details["holdings"]:
                raise PreventUpdate
                
            # Find the earliest available data across all holdings
            min_dates = []
            max_dates = []
            
            for holding in details["holdings"]:
                ticker = holding["ticker"]
                # Fetch max available data
                df = StockDataService.fetch_stock_data(ticker, 'max')
                if df is not None and not df.empty:
                    min_dates.append(df.index.min())
                    max_dates.append(df.index.max())
            
            if not min_dates or not max_dates:
                raise PreventUpdate
                
            earliest_date = min(min_dates)
            latest_date = max(max_dates)
            
            # Set default range to last year if available
            default_end = latest_date
            default_start = (latest_date - pd.Timedelta(days=365))
            if default_start < earliest_date:
                default_start = earliest_date
            
            return (
                earliest_date.strftime("%Y-%m-%d"),
                latest_date.strftime("%Y-%m-%d"),
                default_start.strftime("%Y-%m-%d"),
                default_end.strftime("%Y-%m-%d")
            )
            
        except Exception as e:
            logger.error(f"Error updating date range: {str(e)}")
            raise PreventUpdate

    @app.callback(
        Output("portfolio-details-backtesting", "children"),
        [Input("portfolio-dropdown-backtesting", "value")]
    )
    def display_portfolio_details_backtesting(portfolio_id):
        if not portfolio_id:
            return "Select a portfolio to view its details"
            
        try:
            logger.info(f"Fetching details for portfolio {portfolio_id}")
            details = portfolio.get_portfolio_details(portfolio_id)
            
            if not details["holdings"]:
                return html.Div(
                    "No holdings in this portfolio yet.",
                    className="info-message"
                )
                
            return html.Div([
                html.H4("Portfolio Composition", className="subsection-title"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Ticker"),
                            html.Th("Allocation (%)")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(h["ticker"]),
                            html.Td(f"{h['allocation']}%")
                        ]) for h in details["holdings"]
                    ])
                ], className="holdings-table")
            ])
        except Exception as e:
            logger.error(f"Error displaying portfolio details: {str(e)}")
            return html.Div(
                f"Error loading portfolio details: {str(e)}",
                className="error-message"
            )

    @app.callback(
        [Output("backtesting-results", "children"),
         Output("backtesting-chart", "figure")],
        [Input("run-backtest-button", "n_clicks"),
         Input("backtesting-date-range", "start_date"),
         Input("backtesting-date-range", "end_date")],
        [State("portfolio-dropdown-backtesting", "value"),
         State("benchmark-dropdown-backtesting", "value")]
    )
    def run_backtesting(n_clicks, start_date, end_date, portfolio_id, benchmark_ticker):
        if not n_clicks or not all([portfolio_id, start_date, end_date]):
            raise PreventUpdate
            
        logger.info(f"Running backtest for portfolio {portfolio_id} from {start_date} to {end_date}")
            
        try:
            # Clear any cached data for the benchmark
            if benchmark_ticker:
                try:
                    logger.info(f"Clearing cache for benchmark {benchmark_ticker}")
                    StockDataService._cache.clear_cache(benchmark_ticker)
                except Exception as e:
                    logger.warning(f"Error clearing benchmark cache: {str(e)}")
            
            details = portfolio.get_portfolio_details(portfolio_id)
            if not details["holdings"]:
                empty_fig = {
                    'data': [],
                    'layout': {
                        'title': 'No data available',
                        'showlegend': False
                    }
                }
                return html.Div(
                    "Cannot run backtest: portfolio has no holdings",
                    className="warning-message"
                ), empty_fig
            
            # Clear cache for all holdings
            for holding in details["holdings"]:
                try:
                    logger.info(f"Clearing cache for holding {holding['ticker']}")
                    StockDataService._cache.clear_cache(holding['ticker'])
                except Exception as e:
                    logger.warning(f"Error clearing cache for {holding['ticker']}: {str(e)}")
            
            results = AnalysisService.run_portfolio_backtesting(
                details["holdings"],
                start_date,
                end_date,
                benchmark_ticker
            )
            
            # Create performance chart
            fig = {
                'data': [
                    {
                        'x': results['dates'],
                        'y': results['portfolio_values'],
                        'name': 'Portfolio',
                        'type': 'scatter',
                        'line': {'color': '#3366CC'}
                    }
                ],
                'layout': {
                    'title': 'Portfolio Performance',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Value ($)'},
                    'hovermode': 'x unified',
                    'template': 'plotly_white'
                }
            }
            
            # Add benchmark to chart if available
            if 'benchmark_values' in results and len(results['benchmark_values']) > 0:
                fig['data'].append({
                    'x': results['dates'],
                    'y': results['benchmark_values'],
                    'name': f'Benchmark ({benchmark_ticker})',
                    'type': 'scatter',
                    'line': {'color': '#DC3912'}
                })
                fig['layout']['title'] = 'Portfolio Performance vs Benchmark'
            
            # Create metrics table
            metrics_table = html.Div([
                html.H4("Backtesting Results", className="subsection-title"),
                html.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("Total Return"),
                            html.Td(f"{results['portfolio_return']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td("Annualized Return"),
                            html.Td(f"{results['annualized_return']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td("Benchmark Return"),
                            html.Td(f"{results['benchmark_return']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td("Alpha"),
                            html.Td(f"{results['alpha']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td("Beta"),
                            html.Td(f"{results['beta']:.2f}")
                        ]),
                        html.Tr([
                            html.Td("Sharpe Ratio"),
                            html.Td(f"{results['sharpe_ratio']:.2f}")
                        ]),
                        html.Tr([
                            html.Td("Volatility"),
                            html.Td(f"{results['volatility']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td("Maximum Drawdown"),
                            html.Td(f"{results['max_drawdown']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td("Information Ratio"),
                            html.Td(f"{results['information_ratio']:.2f}")
                        ])
                    ])
                ], className="results-table")
            ])
            
            return metrics_table, fig
            
        except Exception as e:
            logger.error(f"Error running backtesting: {str(e)}")
            empty_fig = {
                'data': [],
                'layout': {
                    'title': 'Error running backtest',
                    'showlegend': False
                }
            }
            return html.Div(
                f"Error running backtesting: {str(e)}",
                className="error-message"
            ), empty_fig