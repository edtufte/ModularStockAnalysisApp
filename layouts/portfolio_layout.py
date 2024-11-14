# layouts/portfolio_layout.py
from dash import dcc, html

class PortfolioLayout:
    @staticmethod
    def create_layout():
        return html.Div([
            # Add initialization trigger and store for portfolio updates
            dcc.Store(id='portfolios-init-trigger', data=1),  # Will trigger on page load
            dcc.Store(id='portfolio-update-trigger', data=1),  # Will trigger on updates
            
            html.H2("Portfolio Management", className="dashboard-title"),
            
            # Create Portfolio Section
            html.Div([
                html.H3("Create New Portfolio", className="section-subtitle"),
                html.Div([
                    dcc.Input(
                        id="portfolio-name-input",
                        type="text",
                        placeholder="Enter portfolio name",
                        className="stock-input",
                        n_submit=0
                    ),
                    html.Button(
                        "Create Portfolio",
                        id="create-portfolio-button",
                        n_clicks=0,
                        className="analyze-button"
                    ),
                    html.Div(id="create-portfolio-message", className="message-container")
                ], className="input-container")
            ], className="section-container"),
            
            # Portfolio Management Section
            html.Div([
                html.H3("Manage Portfolios", className="section-subtitle"),
                html.Div([
                    # Portfolio Selection
                    html.Div([
                        dcc.Dropdown(
                            id="portfolio-dropdown",
                            options=[],
                            placeholder="Select a portfolio",
                            className="dropdown"
                        ),
                        html.Div(
                            style={
                                'display': 'flex',
                                'gap': '10px',
                                'marginTop': '10px'
                            },
                            children=[
                                html.Button(
                                    "Delete Selected Portfolio",
                                    id="delete-portfolio-button",
                                    n_clicks=0,
                                    className="delete-button",
                                    style={
                                        'backgroundColor': '#DC2626',
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '8px 16px',
                                        'borderRadius': '4px',
                                        'cursor': 'pointer'
                                    }
                                ),
                                html.Button(
                                    "Clear All Portfolios",
                                    id="clear-portfolios-button",
                                    n_clicks=0,
                                    className="clear-button",
                                    style={
                                        'backgroundColor': '#4B5563',
                                        'color': 'white',
                                        'border': 'none',
                                        'padding': '8px 16px',
                                        'borderRadius': '4px',
                                        'cursor': 'pointer'
                                    }
                                )
                            ]
                        ),
                        html.Div(id="delete-portfolio-message", className="message-container")
                    ], className="portfolio-selection-container"),
                    
                    # Portfolio Details
                    html.Div(id="portfolio-details", className="portfolio-details")
                ], className="portfolio-management-container")
            ], className="section-container"),
            
            # Add Holding Section
            html.Div([
                html.H3("Add Holding", className="section-subtitle"),
                html.Div([
                    dcc.Input(
                        id="holding-ticker-input",
                        type="text",
                        placeholder="Enter ticker symbol",
                        className="stock-input"
                    ),
                    dcc.Input(
                        id="holding-allocation-input",
                        type="number",
                        placeholder="Enter allocation (%)",
                        min=0,
                        max=100,
                        step=0.1,
                        className="stock-input"
                    ),
                    html.Button(
                        "Add Holding",
                        id="add-holding-button",
                        n_clicks=0,
                        className="analyze-button"
                    ),
                    html.Div(id="add-holding-message", className="message-container")
                ], className="input-container")
            ], className="section-container")
        ], className="dashboard-container")