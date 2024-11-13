# layouts/portfolio_layout.py
from dash import dcc, html

class PortfolioLayout:
    @staticmethod
    def create_layout():
        return html.Div([
            html.H2("Portfolio Management", className="dashboard-title"),
            
            # Create Portfolio Section
            html.Div([
                html.H3("Create New Portfolio", className="section-subtitle"),
                html.Div([
                    dcc.Input(
                        id="portfolio-name-input",
                        type="text",
                        placeholder="Enter portfolio name",
                        className="stock-input"
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
            
            # Existing Portfolios Section
            html.Div([
                html.H3("Existing Portfolios", className="section-subtitle"),
                dcc.Dropdown(
                    id="portfolio-dropdown",
                    options=[],
                    placeholder="Select a portfolio",
                    className="dropdown"
                ),
                html.Div(id="portfolio-details", className="portfolio-details")
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