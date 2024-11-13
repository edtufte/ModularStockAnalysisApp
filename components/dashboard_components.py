# components/dashboard_components.py

from dash import html
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import plotly.graph_objects as go

class DashboardComponents:
    """Class for creating dashboard UI components"""
    
    @staticmethod
    def create_error_components() -> Tuple[Dict, html.Div, html.Div]:
        """Create components for error state
        
        Returns:
            Tuple containing:
            - Empty figure dict for charts
            - Error card component
            - Error message component
        """
        empty_fig = {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': 'No data available',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                'template': 'plotly_white'
            }
        }
        
        error_card = html.Div([
            html.H4("Error", className='error-title'),
            html.P("Unable to load data", className='error-text')
        ], className='metric-card error-card')
        
        error_message = html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-circle mr-2"),
                html.Span("Error loading data. Please try again later.")
            ], className='error-content')
        ], className='error-container')
        
        return empty_fig, error_card, error_message
    
    # Add this method to the DashboardComponents class in dashboard_components.py

    @staticmethod
    def create_company_overview(overview: Dict[str, Any]) -> html.Div:
        """Create company overview section
        
        Args:
            overview (Dict[str, Any]): Company overview data
            
        Returns:
            html.Div: Company overview component
        """
        try:
            if not overview:
                return html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle mr-2"),
                        html.Span("Company information unavailable")
                    ], className='error-content')
                ], className='error-container')

            return html.Div([
                # Company Header
                html.Div([
                    html.H2(overview['Name'], className='company-name'),
                    html.Div([
                        html.Span(overview['Exchange'], className='exchange-tag'),
                        html.Span(overview['Sector'], className='sector-tag'),
                    ], className='company-tags')
                ], className='company-header'),
                
                # Company Description
                html.Div([
                    html.H4("About", className='section-subtitle'),
                    html.P(overview['Description'], className='company-description')
                ], className='description-section'),
                
                # Key Details Grid
                html.Div([
                    html.H4("Company Details", className='section-subtitle'),
                    html.Div([
                        # Left Column
                        html.Div([
                            html.Div([
                                html.Strong("Industry: "),
                                html.Span(overview['Industry'])
                            ], className='detail-row'),
                            html.Div([
                                html.Strong("Market Cap: "),
                                html.Span(overview['MarketCap'])
                            ], className='detail-row'),
                            html.Div([
                                html.Strong("P/E Ratio: "),
                                html.Span(overview['PERatio'])
                            ], className='detail-row'),
                        ], className='details-column'),
                        
                        # Right Column
                        html.Div([
                            html.Div([
                                html.Strong("Dividend Yield: "),
                                html.Span(overview['DividendYield'])
                            ], className='detail-row'),
                            html.Div([
                                html.Strong("52-Week Range: "),
                                html.Span(f"${overview['52WeekLow']} - ${overview['52WeekHigh']}")
                            ], className='detail-row'),
                            html.Div([
                                html.Strong("Employees: "),
                                html.Span(overview['FullTimeEmployees'])
                            ], className='detail-row'),
                        ], className='details-column'),
                    ], className='details-grid'),
                ], className='details-section'),
                
                # Address Section
                html.Div([
                    html.H4("Headquarters", className='section-subtitle'),
                    html.P(overview['Address'], className='company-address')
                ], className='address-section'),
                
            ], className='company-overview-container')
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-circle mr-2"),
                    html.Span(f"Error creating company overview: {str(e)}")
                ], className='error-content')
            ], className='error-container')

    @staticmethod
    def create_metric_card(title: str, value: str) -> html.Div:
        """Create a metric card component
        
        Args:
            title (str): Title of the metric
            value (str): Value to display
            
        Returns:
            html.Div: Styled metric card component
        """
        try:
            return html.Div([
                html.H4(title, className='metric-title'),
                html.P(value, className='metric-value')
            ], className='metric-card')
        except Exception as e:
            return html.Div([
                html.H4("Error", className='error-title'),
                html.P(f"Error creating metric: {str(e)}", className='error-text')
            ], className='metric-card error-card')

    @staticmethod
    def create_recommendation_layout(
        recommendation: Dict[str, Any],
        price_targets: Optional[Dict[str, Any]]
    ) -> html.Div:
        """Create the recommendation and price targets layout with improved styling
        
        Args:
            recommendation (Dict): Recommendation data
            price_targets (Dict): Price target data
            
        Returns:
            html.Div: Complete recommendation layout
        """
        try:
            if not price_targets:
                raise ValueError("No price targets data available")
                
            return html.Div([
                # Main container
                html.Div([
                    # Left column - Recommendation section
                    html.Div([
                        # Recommendation Header with badge-style display
                        html.Div([
                            html.H4(
                                "Recommendation:", 
                                style={
                                    'color': '#1f2937',  # Dark gray for better readability
                                    'marginBottom': '4px',
                                    'fontSize': '18px',
                                    'fontWeight': '500'
                                }
                            ),
                            html.Div(
                                recommendation['recommendation'],
                                style={
                                    'color': 'white',  # White text
                                    'backgroundColor': recommendation['color'],
                                    'padding': '8px 16px',
                                    'borderRadius': '6px',
                                    'fontSize': '24px',
                                    'fontWeight': '600',
                                    'display': 'inline-block',
                                    'marginBottom': '12px'
                                }
                            ),
                        ]),
                        
                        # Confidence score
                        html.Div([
                            html.Span(
                                "Confidence Score: ",
                                style={
                                    'color': '#4b5563',
                                    'fontSize': '16px',
                                    'fontWeight': '500'
                                }
                            ),
                            html.Span(
                                f"{recommendation['confidence']:.1f}%",
                                style={
                                    'color': '#1f2937',
                                    'fontSize': '16px',
                                    'fontWeight': '600'
                                }
                            )
                        ], style={'marginBottom': '20px'}),
                        
                        # Analysis Factors
                        html.Div([
                            html.H5(
                                "Key Analysis Factors:",
                                style={
                                    'color': '#1f2937',
                                    'marginBottom': '12px',
                                    'fontSize': '16px',
                                    'fontWeight': '600'
                                }
                            ),
                            html.Ul(
                                [
                                    html.Li(
                                        reason,
                                        style={
                                            'color': '#4b5563',
                                            'marginBottom': '8px',
                                            'fontSize': '14px',
                                            'lineHeight': '1.5'
                                        }
                                    ) for reason in recommendation['reasons']
                                ],
                                style={
                                    'listStyleType': 'disc',
                                    'paddingLeft': '1.25rem',
                                    'marginBottom': '20px'
                                }
                            )
                        ]),
                        
                        # Risk Level if available
                        html.Div([
                            html.H5(
                                "Risk Level:",
                                style={
                                    'color': '#1f2937',
                                    'marginBottom': '8px',
                                    'fontSize': '16px',
                                    'fontWeight': '600'
                                }
                            ),
                            html.Div(
                                recommendation.get('risk_level', 'N/A'),
                                style={
                                    'color': '#4b5563',
                                    'fontSize': '14px',
                                    'fontWeight': '500',
                                    'marginBottom': '16px'
                                }
                            )
                        ]) if recommendation.get('risk_level') else None,
                        
                    ], className='recommendation-section', style={
                        'backgroundColor': '#ffffff',
                        'padding': '24px',
                        'borderRadius': '8px',
                        'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                        'flex': '1'
                    }),

                    # Right column - Price Targets section
                    html.Div([
                        html.H4(
                            "Price Targets",
                            style={
                                'color': '#1f2937',
                                'marginBottom': '20px',
                                'fontSize': '18px',
                                'fontWeight': '600'
                            }
                        ),
                        html.Div([
                            # Current Price
                            html.Div([
                                html.Div(
                                    "Current Price",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div(
                                    f"${price_targets['current_price']:.2f}",
                                    className='value',
                                    style={
                                        'color': '#1f2937',
                                        'fontSize': '16px',
                                        'fontWeight': '600'
                                    }
                                )
                            ], className='price-target-row', style={'marginBottom': '12px'}),
                            
                            # Technical Target
                            html.Div([
                                html.Div(
                                    "Technical Target",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div(
                                    f"${price_targets['technical_target']:.2f}",
                                    className='value',
                                    style={
                                        'color': '#1f2937',
                                        'fontSize': '16px',
                                        'fontWeight': '600'
                                    }
                                )
                            ], className='price-target-row', style={'marginBottom': '12px'}),
                            
                            # Support Level
                            html.Div([
                                html.Div(
                                    "Support Level",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div(
                                    f"${price_targets['support_level']:.2f}",
                                    className='value',
                                    style={
                                        'color': '#1f2937',
                                        'fontSize': '16px',
                                        'fontWeight': '600'
                                    }
                                )
                            ], className='price-target-row', style={'marginBottom': '12px'}),
                            
                            # Resistance Level
                            html.Div([
                                html.Div(
                                    "Resistance Level",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div(
                                    f"${price_targets['resistance_level']:.2f}",
                                    className='value',
                                    style={
                                        'color': '#1f2937',
                                        'fontSize': '16px',
                                        'fontWeight': '600'
                                    }
                                )
                            ], className='price-target-row', style={'marginBottom': '12px'}),
                            
                            # Volatility Range
                            html.Div([
                                html.Div(
                                    "Volatility Range",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div(
                                    f"${price_targets['volatility_range'][0]:.2f} - ${price_targets['volatility_range'][1]:.2f}",
                                    className='value',
                                    style={
                                        'color': '#1f2937',
                                        'fontSize': '16px',
                                        'fontWeight': '600'
                                    }
                                )
                            ], className='price-target-row')
                        ], className='price-targets-grid')
                    ], className='price-targets-section', style={
                        'backgroundColor': '#ffffff',
                        'padding': '24px',
                        'borderRadius': '8px',
                        'boxShadow': '0 1px 3px rgba(0, 0, 0, 0.1)',
                        'flex': '1'
                    })
                ], style={
                    'display': 'flex',
                    'gap': '24px',
                    'flexWrap': 'wrap'
                })
            ], className='analysis-container')
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(
                        className="fas fa-exclamation-circle mr-2",
                        style={'marginRight': '8px', 'color': '#dc2626'}
                    ),
                    html.Span(
                        f"Error creating recommendation layout: {str(e)}",
                        style={'color': '#dc2626'}
                    )
                ], className='error-content')
            ], className='error-container')

    @staticmethod
    def _create_price_target_row(label: str, value: str) -> html.Div:
        """Create a price target row with consistent styling
        
        Args:
            label (str): Label for the price target
            value (str): Value to display
            
        Returns:
            html.Div: Styled price target row
        """
        return html.Div([
            html.Div(label, className='label'),
            html.Div(value, className='value')
        ], className='price-target-row')

    @staticmethod
    def create_performance_table(df_yearly: pd.DataFrame) -> html.Table:
        """Create the performance table component
        
        Args:
            df_yearly (pd.DataFrame): Yearly performance data
            
        Returns:
            html.Table: Styled performance table
        """
        try:
            if df_yearly is None or df_yearly.empty:
                raise ValueError("No performance data available")
                
            return html.Table([
                html.Thead(
                    html.Tr([
                        html.Th(col) for col in [
                            'Year', 'Close', 'Return (%)',
                            'Max Drawdown (%)', 'Volume (M)'
                        ]
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Year']),
                        html.Td(f"${row['Close']:.2f}"),
                        html.Td(
                            f"{row['Return (%)']:.1f}%",
                            style={'color': 'green' if row['Return (%)'] > 0 else 'red'}
                        ),
                        html.Td(f"{row['Max Drawdown (%)']:.1f}%"),
                        html.Td(f"{row['Volume']/1e6:.1f}M")
                    ]) for _, row in df_yearly.iterrows()
                ])
            ], className='performance-table')
            
        except Exception as e:
            return html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-circle mr-2"),
                    html.Span(f"Error creating performance table: {str(e)}")
                ], className='error-content')
            ], className='error-container')

    @staticmethod
    def create_loading_components() -> Tuple[Dict, html.Div, html.Div]:
        """Create components for loading state
        
        Returns:
            Tuple containing:
            - Loading figure dict for charts
            - Loading card component
            - Loading message component
        """
        loading_fig = {
            'data': [],
            'layout': {
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'annotations': [{
                    'text': 'Loading...',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 16}
                }],
                'template': 'plotly_white'
            }
        }
        
        loading_card = html.Div([
            html.H4("Loading...", className='loading-title'),
            html.P("Fetching data", className='loading-text')
        ], className='metric-card loading-card')
        
        loading_message = html.Div([
            html.Div([
                html.I(className="fas fa-spinner fa-spin mr-2"),
                html.Span("Loading data...")
            ], className='loading-content')
        ], className='loading-container')
        
        return loading_fig, loading_card, loading_message

    @staticmethod
    def format_large_number(number: float) -> str:
        """Format large numbers with appropriate suffixes
        
        Args:
            number (float): Number to format
            
        Returns:
            str: Formatted number string
        """
        try:
            abs_num = abs(number)
            if abs_num >= 1e9:
                return f"{number/1e9:.1f}B"
            elif abs_num >= 1e6:
                return f"{number/1e6:.1f}M"
            elif abs_num >= 1e3:
                return f"{number/1e3:.1f}K"
            else:
                return f"{number:.1f}"
        except Exception:
            return "N/A"