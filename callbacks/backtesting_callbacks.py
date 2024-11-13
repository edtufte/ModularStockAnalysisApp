# callbacks/backtesting_callbacks.py
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html
import logging
from models.portfolio import Portfolio
from services.analysis_service import AnalysisService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_callbacks(app, db):
    portfolio = Portfolio(db)

    @app.callback(
        Output("portfolio-details-backtesting", "children"),
        Input("portfolio-dropdown-backtesting", "value")
    )
    def display_portfolio_details_backtesting(portfolio_id):
        if not portfolio_id:
            return "Select a portfolio to view its details"
            
        try:
            logger.info(f"Fetching details for portfolio {portfolio_id} in backtesting view")
            details = portfolio.get_portfolio_details(portfolio_id)
            logger.info(f"Retrieved details: {details}")
            
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
            logger.error(f"Error displaying portfolio details in backtesting: {str(e)}")
            return html.Div(
                f"Error loading portfolio details: {str(e)}",
                className="error-message"
            )

    @app.callback(
        Output("backtesting-results", "children"),
        Input("run-backtest-button", "n_clicks"),
        [State("portfolio-dropdown-backtesting", "value"),
        State("backtesting-date-range", "start_date"),
        State("backtesting-date-range", "end_date")]
    )
    def run_backtesting(n_clicks, portfolio_id, start_date, end_date):
        if n_clicks == 0:
            raise PreventUpdate
            
        if not all([portfolio_id, start_date, end_date]):
            return html.Div(
                "Please select a portfolio and specify the date range",
                className="warning-message"
            )
            
        try:
            logger.info(f"Running backtest for portfolio {portfolio_id}")
            details = portfolio.get_portfolio_details(portfolio_id)
            logger.info(f"Portfolio details for backtest: {details}")
            
            if not details["holdings"]:
                return html.Div(
                    "Cannot run backtest: portfolio has no holdings",
                    className="warning-message"
                )
            
            results = AnalysisService.run_portfolio_backtesting(
                details["holdings"],
                start_date,
                end_date
            )
            
            return html.Div([
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
        except Exception as e:
            logger.error(f"Error running backtesting: {str(e)}")
            return html.Div(
                f"Error running backtesting: {str(e)}",
                className="error-message"
            )