# layouts/backtesting_layout.py
from dash import dcc, html

class BacktestingLayout:
    @staticmethod
    def create_layout():
        return html.Div([
            html.H2("Portfolio Backtesting", className="dashboard-title"),
            
            # Portfolio Selection Section
            html.Div([
                html.H3("Select Portfolio", className="section-subtitle"),
                dcc.Dropdown(
                    id="portfolio-dropdown-backtesting",
                    options=[],
                    placeholder="Select a portfolio to test",
                    className="dropdown"
                ),
                html.Div(
                    id="portfolio-details-backtesting",
                    className="portfolio-details"
                )
            ], className="section-container"),
            
            # Backtesting Parameters Section
            html.Div([
                html.H3("Backtesting Parameters", className="section-subtitle"),
                html.Div([
                    dcc.DatePickerRange(
                        id="backtesting-date-range",
                        start_date_placeholder_text="Start Date",
                        end_date_placeholder_text="End Date",
                        className="date-picker"
                    ),
                    html.Button(
                        "Run Backtest",
                        id="run-backtest-button",
                        n_clicks=0,
                        className="analyze-button"
                    ),
                ], className="input-container"),
                html.Div(
                    id="backtesting-results",
                    className="results-container"
                )
            ], className="section-container")
        ], className="dashboard-container")