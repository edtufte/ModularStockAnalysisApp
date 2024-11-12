from dash import html
from typing import Dict, Any

class DashboardComponents:
    """Class for creating dashboard UI components"""
    
    @staticmethod
    def create_metric_card(title: str, value: str) -> html.Div:
        """Create a metric card component"""
        return html.Div([
            html.H4(title),
            html.P(value)
        ], className='metric-card')
    
@staticmethod
def create_recommendation_layout(recommendation: Dict[str, Any], price_targets: Dict[str, Any]) -> html.Div:
    """Create the recommendation and price targets layout"""
    return html.Div([
        # Main title
        html.H3("Price Analysis & Recommendation", className='section-title'),
        
        # Grid container for the two columns
        html.Div([
            # Left column - Recommendation section
            html.Div([
                html.H4(
                    f"Recommendation: {recommendation['recommendation']}", 
                    style={
                        'color': recommendation['color'],
                        'marginBottom': '12px',
                        'fontSize': '24px',
                        'fontWeight': '600'
                    }
                ),
                html.P(
                    f"Confidence: {recommendation['confidence']:.1f}%",
                    style={'marginBottom': '20px', 'color': '#374151', 'fontSize': '16px'}
                ),
                html.H5(
                    "Key Factors:",
                    style={
                        'fontWeight': '600',
                        'color': '#1f2937',
                        'marginBottom': '12px',
                        'fontSize': '16px'
                    }
                ),
                html.Ul(
                    [html.Li(reason) for reason in recommendation['reasons']],
                    style={'listStyleType': 'disc', 'paddingLeft': '1.25rem'}
                )
            ], className='recommendation-section'),

            # Right column - Price Targets section
            html.Div([
                html.H4(
                    "Price Targets",
                    style={
                        'marginBottom': '20px',
                        'fontSize': '20px',
                        'fontWeight': '600',
                        'color': '#1f2937'
                    }
                ),
                html.Div([
                    DashboardComponents._create_price_target_row(
                        "Technical Target:", 
                        f"${price_targets['technical_target']:.2f}"
                    ),
                    DashboardComponents._create_price_target_row(
                        "Support Level:", 
                        f"${price_targets['support_level']:.2f}"
                    ),
                    DashboardComponents._create_price_target_row(
                        "Resistance Level:", 
                        f"${price_targets['resistance_level']:.2f}"
                    ),
                    DashboardComponents._create_price_target_row(
                        "Volatility Range:", 
                        f"${price_targets['volatility_range'][0]:.2f} - ${price_targets['volatility_range'][1]:.2f}"
                    )
                ], className='price-targets-grid')
            ], className='price-targets-section')
        ], className='analysis-grid')
    ], className='analysis-container')

    @staticmethod
    def _create_price_target_row(label: str, value: str) -> html.Div:
        """Create a price target row with consistent styling"""
        return html.Div([
            html.Div(label, className='label'),
            html.Div(value, className='value')
        ], className='price-target-row')

    @staticmethod
    def create_performance_table(df_yearly) -> html.Table:
        """Create the performance table component"""
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
                    html.Td(f"{row['Return (%)']:.1f}%"),
                    html.Td(f"{row['Max Drawdown (%)']:.1f}%"),
                    html.Td(f"{row['Volume']/1e6:.1f}M")
                ]) for _, row in df_yearly.iterrows()
            ])
        ], className='performance-table')

    @staticmethod
    def create_error_components():
        """Create error state components"""
        empty_fig = {
            'data': [],
            'layout': {
                'title': 'No data available',
                'annotations': [{
                    'text': 'Error loading data',
                    'xref': 'paper',
                    'yref': 'paper',
                    'showarrow': False,
                    'font': {'size': 14}
                }]
            }
        }
        
        error_card = html.Div([
            html.H4("Error"),
            html.P("Unable to load data")
        ], className='metric-card')
        
        error_message = html.Div(
            "Unable to load data",
            className='error-message'
        )
        
        return empty_fig, error_card, error_message