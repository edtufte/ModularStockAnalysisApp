# callbacks/backtesting_callbacks.py
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html
import logging
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
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
        [Input("portfolio-dropdown-backtesting", "value")],
        [State("backtesting-date-range", "start_date"),
        State("backtesting-date-range", "end_date")]
    )
    def update_date_range(portfolio_id, current_start_date, current_end_date):
        """Update the allowed date range based on available data"""
        today = datetime.now()
        
        try:
            if not portfolio_id:
                # Initial state with no portfolio selected
                return (
                    "2010-01-01",
                    today.strftime("%Y-%m-%d"),
                    current_start_date or (today - timedelta(days=365)).strftime("%Y-%m-%d"),
                    current_end_date or today.strftime("%Y-%m-%d")
                )
                
            # Get portfolio details and find available date range
            details = portfolio.get_portfolio_details(portfolio_id)
            if not details["holdings"]:
                raise PreventUpdate
                
            # Find the earliest and latest available data across all holdings
            min_dates = []
            max_dates = []
            
            for holding in details["holdings"]:
                ticker = holding["ticker"]
                df = StockDataService.fetch_stock_data(ticker, 'max')
                if df is not None and not df.empty:
                    min_dates.append(df.index.min())
                    max_dates.append(df.index.max())
            
            if not min_dates or not max_dates:
                raise PreventUpdate
                
            earliest_date = min(min_dates)
            latest_date = max(max_dates)
            
            # If we have current dates selected, validate and preserve them
            if current_start_date and current_end_date:
                try:
                    # Convert string dates to timestamps for comparison
                    start_ts = pd.Timestamp(current_start_date)
                    end_ts = pd.Timestamp(current_end_date)
                    
                    # Ensure dates are within valid range
                    valid_start = max(earliest_date, start_ts)
                    valid_end = min(latest_date, end_ts)
                    
                    logger.info(f"Preserving date range: {valid_start} to {valid_end}")
                    
                    return (
                        earliest_date.strftime("%Y-%m-%d"),
                        latest_date.strftime("%Y-%m-%d"),
                        valid_start.strftime("%Y-%m-%d"),
                        valid_end.strftime("%Y-%m-%d")
                    )
                except Exception as e:
                    logger.error(f"Error validating dates: {str(e)}")
            
            # Only set default dates if no dates are currently selected
            default_end = latest_date
            default_start = latest_date - pd.Timedelta(days=365)
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
            # Return current dates if there's an error
            return (
                "2010-01-01",
                today.strftime("%Y-%m-%d"),
                current_start_date or (today - timedelta(days=365)).strftime("%Y-%m-%d"),
                current_end_date or today.strftime("%Y-%m-%d")
            )

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
            
            # Create metrics table with tooltips
            metrics_table = html.Div([
                html.H4("Backtesting Results", className="subsection-title"),
                html.Table([
                    html.Tbody([
                        # Portfolio Returns Section
                        html.Tr([
                            html.Td([
                                "Portfolio Total Return",
                                html.I(className="fas fa-info-circle ml-1", 
                                      id="total-return-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['portfolio_return']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td([
                                "Portfolio Annualized Return",
                                html.I(className="fas fa-info-circle ml-1", 
                                      id="annualized-return-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['annualized_return']*100:.2f}%")
                        ]),
                        
                        # Benchmark Returns Section
                        html.Tr([html.Td("", colSpan=2, style={'borderTop': '1px solid #e5e7eb'})]),  # Divider
                        html.Tr([
                            html.Td("Benchmark Total Return", style={'fontWeight': 'bold'}),
                            html.Td(f"{results['benchmark_return']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td("Benchmark Annualized Return", style={'fontWeight': 'bold'}),
                            html.Td(f"{results['annualized_benchmark_return']*100:.2f}%")
                        ]),
                        
                        # Risk Metrics Section
                        html.Tr([html.Td("", colSpan=2, style={'borderTop': '1px solid #e5e7eb'})]),  # Divider
                        html.Tr([
                            html.Td([
                                "Alpha",
                                html.I(className="fas fa-info-circle ml-1", id="alpha-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['alpha']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td([
                                "Beta",
                                html.I(className="fas fa-info-circle ml-1", id="beta-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['beta']:.2f}")
                        ]),
                        html.Tr([
                            html.Td([
                                "Sharpe Ratio",
                                html.I(className="fas fa-info-circle ml-1", id="sharpe-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['sharpe_ratio']:.2f}")
                        ]),
                        html.Tr([
                            html.Td([
                                "Volatility",
                                html.I(className="fas fa-info-circle ml-1", id="volatility-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['volatility']*100:.2f}%")
                        ]),
                        html.Tr([
                            html.Td([
                                "Maximum Drawdown",
                                html.I(className="fas fa-info-circle ml-1", id="drawdown-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['max_drawdown']*100:.2f}%")
                        ]),
                        # Advanced metrics section
                        html.Tr([
                            html.Td([
                                "Calmar Ratio",
                                html.I(className="fas fa-info-circle ml-1", id="calmar-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['calmar_ratio']:.2f}")
                        ]),
                        html.Tr([
                            html.Td([
                                "Treynor Ratio",
                                html.I(className="fas fa-info-circle ml-1", id="treynor-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['treynor_ratio']:.2f}")
                        ]),
                        html.Tr([
                            html.Td([
                                "Upside Capture",
                                html.I(className="fas fa-info-circle ml-1", id="upside-capture-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['capture_ratio_up']:.1f}%")
                        ]),
                        html.Tr([
                            html.Td([
                                "Downside Capture",
                                html.I(className="fas fa-info-circle ml-1", id="downside-capture-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['capture_ratio_down']:.1f}%")
                        ]),
                        html.Tr([
                            html.Td([
                                "Omega Ratio",
                                html.I(className="fas fa-info-circle ml-1", id="omega-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['omega_ratio']:.2f}")
                        ]),
                        html.Tr([
                            html.Td([
                                "Tail Ratio",
                                html.I(className="fas fa-info-circle ml-1", id="tail-tooltip")
                            ], style={'fontWeight': 'bold'}),
                            html.Td(f"{results['tail_ratio']:.2f}")
                        ])
                    ])
                ], className="results-table"),
                
                # Add tooltip components with custom styling
                dbc.Tooltip(
                    "Total return of the portfolio over the selected period",
                    target="total-return-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Annualized return of the portfolio over the selected period",
                    target="annualized-return-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Measures portfolio's risk-adjusted excess return relative to benchmark",
                    target="info-ratio-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                # Risk metrics tooltips
                dbc.Tooltip(
                    "Excess return of portfolio over the risk-free rate relative to benchmark",
                    target="alpha-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Measure of portfolio's volatility relative to the benchmark",
                    target="beta-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Risk-adjusted return relative to the risk-free rate",
                    target="sharpe-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Standard deviation of portfolio returns",
                    target="volatility-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Largest peak-to-trough decline in portfolio value",
                    target="drawdown-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                # Advanced metrics tooltips
                dbc.Tooltip(
                    "Annual return divided by maximum drawdown",
                    target="calmar-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Excess return per unit of systematic risk",
                    target="treynor-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Percentage of benchmark gains captured by portfolio",
                    target="upside-capture-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Percentage of benchmark losses captured by portfolio",
                    target="downside-capture-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Probability weighted ratio of gains vs losses",
                    target="omega-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                ),
                dbc.Tooltip(
                    "Ratio of right-tail returns to left-tail returns",
                    target="tail-tooltip",
                    placement="right",
                    style={
                        "backgroundColor": "rgba(0, 0, 0, 0.8)",
                        "color": "white",
                        "fontSize": "0.8rem",
                        "maxWidth": "200px"
                    }
                )
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