# app.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from layouts.dashboard_layout import DashboardLayout
from layouts.portfolio_layout import PortfolioLayout
from layouts.backtesting_layout import BacktestingLayout
from callbacks.dashboard_callbacks import DashboardCallbacks
from callbacks import portfolio_callbacks, backtesting_callbacks
from services.database import Database
from config.database import DB_NAME

def create_app():
    """Create and configure the Dash application"""
    
    # Initialize the Dash app
    app = dash.Dash(
        __name__,
        meta_tags=[{
            "name": "viewport",
            "content": "width=device-width, initial-scale=1"
        }],
        suppress_callback_exceptions=True
    )
    
    # Set up the server
    server = app.server

    # Initialize database
    db = Database(DB_NAME)
    
    # Create tables (this is safe to call multiple times)
    db.create_tables()
    
    app.layout = html.Div([
        dcc.Store(id="user-id", storage_type="session", data=1),  # Default user ID
        dcc.Tabs([
            dcc.Tab(label="Research", children=[DashboardLayout.create_layout()]),
            dcc.Tab(label="Portfolio", children=[PortfolioLayout.create_layout()]),
            dcc.Tab(label="Backtesting", children=[BacktestingLayout.create_layout()])
        ])
    ])

    # Register callbacks
    DashboardCallbacks.register_callbacks(app)
    portfolio_callbacks.register_callbacks(app, db)
    backtesting_callbacks.register_callbacks(app, db)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)