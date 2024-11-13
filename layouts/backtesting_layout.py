# layouts/backtesting_layout.py
from dash import dcc, html
from datetime import datetime, timedelta
from layouts.dashboard_layout import DashboardLayout

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
                    html.Div([
                        dcc.DatePickerRange(
                            id="backtesting-date-range",
                            min_date_allowed="2010-01-01",
                            max_date_allowed=datetime.now().strftime("%Y-%m-%d"),
                            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
                            end_date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                            className="date-picker"
                        ),
                        html.Div([
                            html.Label(
                                "Benchmark",
                                className="control-label"
                            ),
                            dcc.Dropdown(
                                id="benchmark-dropdown-backtesting",
                                options=DashboardLayout.BENCHMARK_OPTIONS,  # Using the same options as main tab
                                className="dropdown"
                            )
                        ], className="benchmark-selector"),
                        html.Button(
                            "Run Backtest",
                            id="run-backtest-button",
                            n_clicks=0,
                            className="analyze-button"
                        ),
                    ], className="input-container"),
                ]),
                # Results section with side-by-side layout
                html.Div([
                    # Left side - Metrics
                    html.Div(
                        id="backtesting-results",
                        className="backtesting-metrics"
                    ),
                    # Right side - Chart
                    html.Div([
                        dcc.Graph(
                            id="backtesting-chart",
                            className="backtest-chart"
                        )
                    ], className="backtesting-chart-container")
                ], className="backtesting-results-grid")
            ], className="section-container")
        ], className="dashboard-container")