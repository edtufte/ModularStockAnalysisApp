# app.py
import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from callbacks import dashboard_callbacks
from layouts.dashboard_layout import DashboardLayout
from layouts.portfolio_layout import PortfolioLayout
from layouts.backtesting_layout import BacktestingLayout
from callbacks.dashboard_callbacks import DashboardCallbacks
from callbacks import portfolio_callbacks, backtesting_callbacks
from services.database import Database
from config.database import DB_NAME

_db_instance = None

def get_db():
    """Singleton pattern for database access"""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(DB_NAME)
        _db_instance.create_tables()
    return _db_instance

# In app.py
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

    # Create database
    db = get_db()
    
    app.layout = html.Div([
        dcc.Store(id="user-id", storage_type="session", data=1),  # Default user ID
        dcc.Tabs([
            dcc.Tab(label="Research", children=[DashboardLayout.create_layout()]),
            dcc.Tab(label="Portfolio", children=[PortfolioLayout.create_layout()]),
            dcc.Tab(label="Backtesting", children=[BacktestingLayout.create_layout()])
        ])
    ])

    # Register callbacks
    dashboard_callbacks.register_callbacks(app)
    portfolio_callbacks.register_callbacks(app, db)
    backtesting_callbacks.register_callbacks(app, db)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)