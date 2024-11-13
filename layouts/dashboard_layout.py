# layouts/dashboard_layout.py
from dash import dcc, html

class DashboardLayout:
    """Class for defining the main dashboard layout"""
    
    # Define common benchmarks using ETF proxies
    BENCHMARK_OPTIONS = [
        {'label': 'Total Stock Market (VTI)', 'value': 'VTI'},
        {'label': 'S&P 500 (SPY)', 'value': 'SPY'},
        {'label': 'Nasdaq 100 (QQQ)', 'value': 'QQQ'},
        {'label': 'Dow Jones (DIA)', 'value': 'DIA'},
        {'label': 'Russell 2000 (IWM)', 'value': 'IWM'},
        {'label': 'Developed Markets (EFA)', 'value': 'EFA'},
        {'label': 'Emerging Markets (EEM)', 'value': 'EEM'},
        {'label': 'Bond Aggregate (AGG)', 'value': 'AGG'},
    ]
    
    @staticmethod
    def create_layout():
        """Create the main dashboard layout"""
        return html.Div([
            # Header section
            DashboardLayout._create_header(),
            
            # Control panel
            DashboardLayout._create_control_panel(),
            
            # Loading wrapper for the main content
            dcc.Loading(
                id="loading-1",
                type="circle",
                children=[DashboardLayout._create_main_content()]
            )
        ], className='dashboard-container')

    @staticmethod
    def _create_header():
        """Create the dashboard header"""
        return html.Div([
            html.H1("Investment Analysis Dashboard", className='dashboard-title'),
            html.P("Track and analyze stock performance with advanced metrics", 
                  className='dashboard-subtitle')
        ], className='header-container')

    @staticmethod
    def _create_control_panel():
        """Create the control panel section"""
        return html.Div([
            # Stock Input Section
            html.Div([
                html.Label("Enter Stock Ticker", 
                        style={'display': 'block', 'marginBottom': '8px', 'color': '#2c3e50', 'fontWeight': '500'}),
                html.Div([
                    dcc.Input(
                        id='stock-input',
                        type='text',
                        placeholder='Enter ticker (e.g., AAPL)',
                        className='stock-input',
                        n_submit=0
                    ),
                    html.Button(
                        'Analyze', 
                        id='analyze-button', 
                        className='analyze-button'
                    )
                ], className='stock-form'),
                html.Div(id='ticker-error', className='error-message'),
                html.P(
                    "Common tickers: AAPL, GOOGL, AMZN, MSFT, TSLA", 
                    className='ticker-examples'
                )
            ], className='input-container'),
            
            # Analysis Controls Container
            html.Div([
                # Timeframe Dropdown
                html.Div([
                    html.Label(
                        "Select Timeframe",
                        className='control-label'
                    ),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '3 Years', 'value': '3y'},
                            {'label': '5 Years', 'value': '5y'},
                            {'label': 'Max', 'value': 'max'}
                        ],
                        value='1y',
                        className='dropdown'
                    )
                ], className='dropdown-container'),
                
                # Benchmark Dropdown and Status
                html.Div([
                    html.Label(
                        "Compare with Benchmark",
                        className='control-label'
                    ),
                    dcc.Dropdown(
                        id='benchmark-dropdown',
                        options=DashboardLayout.BENCHMARK_OPTIONS,
                        value='VTI',  # Default to Total Stock Market ETF
                        className='dropdown'
                    ),
                    html.Div(id='benchmark-error', className='error-message'),
                    html.Div(id='benchmark-status', className='benchmark-status')
                ], className='dropdown-container')
            ], className='analysis-controls')
        ], className='control-panel')

    @staticmethod
    def _create_main_content():
        """Create the main content area"""
        return html.Div([
            # Company Overview Section
            html.Div(id='company-overview', className='company-overview-section'),
            
            # Key metrics cards
            html.Div([
                html.Div(id='current-price-card', className='metric-card'),
                html.Div(id='change-card', className='metric-card'),
                html.Div(id='volatility-card', className='metric-card'),
                html.Div(id='sharpe-ratio-card', className='metric-card')
            ], className='metrics-container'),
            
            # Price Analysis & Recommendation
            html.Div([
                html.H3("Price Analysis & Recommendation", className='section-title'),
                html.Div(id='recommendation-container')
            ], className='analysis-container'),
            
            # Charts section
            html.Div([
                dcc.Graph(id='stock-chart', className='chart'),
                dcc.Graph(id='volume-chart', className='chart')
            ], className='charts-container'),
            
            # Technical analysis section
            html.Div([
                html.H3("Technical Analysis"),
                dcc.Graph(id='technical-chart', className='chart')
            ], className='technical-container'),
            
            # Performance table
            html.Div([
                html.H3("Historical Performance"),
                html.Div(id='stock-table', className='table-container')
            ], className='performance-container')
        ], className='main-content', id='main-content')