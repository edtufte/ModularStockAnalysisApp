import dash
from layouts.dashboard_layout import DashboardLayout
from callbacks.dashboard_callbacks import DashboardCallbacks

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
    
    # Set the layout
    app.layout = DashboardLayout.create_layout()
    
    # Register callbacks
    DashboardCallbacks.register_callbacks(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)