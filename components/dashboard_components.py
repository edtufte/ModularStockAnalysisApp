# components/dashboard_components.py

import logging
from dash import html
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
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
    
    @staticmethod
    def create_company_overview(overview: Dict[str, Any]) -> html.Div:
        """Create company overview section with improved error handling and fallbacks"""
        try:
            # Ensure we have a valid dictionary
            if not isinstance(overview, dict):
                overview = {}
                
            # Get values with defaults to prevent KeyError
            name = overview.get('Name', 'Company Overview')
            exchange = overview.get('Exchange', 'N/A')
            sector = overview.get('Sector', 'N/A')
            industry = overview.get('Industry', 'N/A')
            description = overview.get('Description', 'N/A')
            market_cap = overview.get('MarketCap', 'N/A')
            pe_ratio = overview.get('PERatio', 'N/A')
            dividend_yield = overview.get('DividendYield', 'N/A')
            week_high = overview.get('52WeekHigh', 'N/A')
            week_low = overview.get('52WeekLow', 'N/A')
            employees = overview.get('FullTimeEmployees', 'N/A')
            address = overview.get('Address', 'N/A')

            return html.Div([
                # Company Header
                html.Div([
                    html.H2(name, className='company-name'),
                    html.Div([
                        html.Span(exchange, className='exchange-tag') if exchange != 'N/A' else None,
                        html.Span(sector, className='sector-tag') if sector != 'N/A' else None,
                    ], className='company-tags')
                ], className='company-header'),
                
                # Company Description (only if available)
                html.Div([
                    html.H4("About", className='section-subtitle'),
                    html.P(
                        description, 
                        className='company-description',
                        style={'whiteSpace': 'pre-wrap'}
                    )
                ], className='description-section') if description != 'N/A' else None,
                
                # Key Details Grid (only if we have some valid data)
                html.Div([
                    html.H4("Company Details", className='section-subtitle'),
                    html.Div([
                        # Left Column
                        html.Div([
                            html.Div([
                                html.Strong("Industry: "),
                                html.Span(industry)
                            ], className='detail-row') if industry != 'N/A' else None,
                            html.Div([
                                html.Strong("Market Cap: "),
                                html.Span(market_cap)
                            ], className='detail-row') if market_cap != 'N/A' else None,
                            html.Div([
                                html.Strong("P/E Ratio: "),
                                html.Span(pe_ratio)
                            ], className='detail-row') if pe_ratio != 'N/A' else None,
                        ], className='details-column'),
                        
                        # Right Column
                        html.Div([
                            html.Div([
                                html.Strong("Dividend Yield: "),
                                html.Span(dividend_yield)
                            ], className='detail-row') if dividend_yield != 'N/A' else None,
                            html.Div([
                                html.Strong("52-Week Range: "),
                                html.Span(f"{week_low} - {week_high}")
                            ], className='detail-row') if week_low != 'N/A' and week_high != 'N/A' else None,
                            html.Div([
                                html.Strong("Employees: "),
                                html.Span(employees)
                            ], className='detail-row') if employees != 'N/A' else None,
                        ], className='details-column'),
                    ], className='details-grid'),
                ], className='details-section') if any(x != 'N/A' for x in [industry, market_cap, pe_ratio, dividend_yield, employees]) else None,
                
                # Address Section (only if available)
                html.Div([
                    html.H4("Headquarters", className='section-subtitle'),
                    html.P(address, className='company-address')
                ], className='address-section') if address != 'N/A' else None,
                
            ], className='company-overview-container')
            
        except Exception as e:
            logging.error(f"Error creating company overview: {str(e)}")
            return html.Div([
                html.Div([
                    html.H2("Company Overview", className='company-name'),
                    html.P("Error loading company information. Please try again later.",
                        className='error-message')
                ], className='company-overview-container')
            ])

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
                            # Current Price with Position Indicator
                            html.Div([
                                html.Div(
                                    "Current Price",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div([
                                    html.Span(
                                        f"${price_targets['current_price']:.2f}",
                                        style={
                                            'color': '#1f2937',
                                            'fontSize': '16px',
                                            'fontWeight': '600',
                                            'marginRight': '8px'
                                        }
                                    ),
                                    # Position indicator
                                    html.Span(
                                        "●",  # Dot indicator
                                        style={
                                            'color': '#10b981' if price_targets['current_price'] > price_targets['technical_target']
                                                    else '#ef4444' if price_targets['current_price'] < price_targets['technical_target']
                                                    else '#f59e0b',
                                            'fontSize': '16px',
                                            'marginRight': '4px'
                                        }
                                    )
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], className='price-target-row', style={'marginBottom': '16px'}),
                            
                            # Technical Target with Trend Arrow
                            html.Div([
                                html.Div(
                                    "Technical Target",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div([
                                    html.Span(
                                        f"${price_targets['technical_target']:.2f}",
                                        style={
                                            'color': '#1f2937',
                                            'fontSize': '16px',
                                            'fontWeight': '600',
                                            'marginRight': '8px'
                                        }
                                    ),
                                    # Percentage difference
                                    html.Span(
                                        f"({((price_targets['technical_target'] - price_targets['current_price']) / price_targets['current_price'] * 100):.1f}%)",
                                        style={
                                            'color': '#10b981' if price_targets['technical_target'] > price_targets['current_price']
                                                    else '#ef4444',
                                            'fontSize': '14px'
                                        }
                                    ),
                                    # Trend arrow
                                    html.I(
                                        className=f"fas fa-{'arrow-up' if price_targets['technical_target'] > price_targets['current_price'] else 'arrow-down'}",
                                        style={
                                            'color': '#10b981' if price_targets['technical_target'] > price_targets['current_price']
                                                    else '#ef4444',
                                            'marginLeft': '8px'
                                        }
                                    )
                                ], style={'display': 'flex', 'alignItems': 'center'})
                            ], className='price-target-row', style={'marginBottom': '16px'}),
                            
                            # Support and Resistance Progress Bar
                            html.Div([
                                html.Div([
                                    html.Div(
                                        "Price Range",
                                        className='label',
                                        style={
                                            'color': '#6b7280',
                                            'fontSize': '14px',
                                            'marginRight': '12px',
                                            'minWidth': '80px'
                                        }
                                    ),
                                    # Progress bar container with right alignment
                                    html.Div([
                                        # Support level
                                        html.Div(
                                            f"${price_targets['support_level']:.2f}",
                                            style={
                                                'position': 'absolute',
                                                'left': '0',
                                                'bottom': '-20px',
                                                'fontSize': '12px',
                                                'color': '#1f2937'
                                            }
                                        ),
                                        # Current price marker
                                        html.Div(
                                            style={
                                                'position': 'absolute',
                                                'left': f"{((price_targets['current_price'] - price_targets['support_level']) / (price_targets['resistance_level'] - price_targets['support_level']) * 100)}%",
                                                'top': '-12px',
                                                'transform': 'translateX(-50%)',
                                                'width': '2px',
                                                'height': '24px',
                                                'backgroundColor': '#1f2937'
                                            }
                                        ),
                                        # Progress bar
                                        html.Div(
                                            style={
                                                'height': '6px',
                                                'backgroundColor': '#e5e7eb',
                                                'borderRadius': '3px',
                                                'position': 'relative',
                                                'width': '150px'  # Fixed width for the progress bar
                                            }
                                        ),
                                        # Resistance level
                                        html.Div(
                                            f"${price_targets['resistance_level']:.2f}",
                                            style={
                                                'position': 'absolute',
                                                'right': '0',
                                                'bottom': '-20px',
                                                'fontSize': '12px',
                                                'color': '#1f2937'
                                            }
                                        )
                                    ], style={
                                        'position': 'relative',
                                        'height': '6px',
                                        'marginTop': '20px',
                                        'marginBottom': '24px'
                                    })
                                ], style={
                                    'display': 'flex',
                                    'alignItems': 'flex-start',
                                    'marginBottom': '8px',
                                    'justifyContent': 'space-between',  # This will push the progress bar to the right
                                    'width': '100%'  # Ensure the container takes full width
                                })
                            ], className='price-target-row', style={'marginBottom': '16px'}),  # Changed to price-target-row for consistent styling
                            
                            # Volatility Range with Visual Indicator
                            html.Div([
                                html.Div(
                                    "Volatility Range",
                                    className='label',
                                    style={'color': '#6b7280', 'fontSize': '14px'}
                                ),
                                html.Div([
                                    html.Span(
                                        f"${price_targets['volatility_range'][0]:.2f} - ${price_targets['volatility_range'][1]:.2f}",
                                        style={
                                            'color': '#1f2937',
                                            'fontSize': '16px',
                                            'fontWeight': '600',
                                            'marginRight': '8px'
                                        }
                                    ),
                                    # Volatility indicator
                                    html.Div(
                                        style={
                                            'width': '50px',
                                            'height': '4px',
                                            'backgroundColor': '#e5e7eb',
                                            'borderRadius': '2px',
                                            'position': 'relative',
                                            'overflow': 'hidden'
                                        },
                                        children=[
                                            html.Div(
                                                style={
                                                    'position': 'absolute',
                                                    'left': '0',
                                                    'top': '0',
                                                    'height': '100%',
                                                    'width': f"{((price_targets['volatility_range'][1] - price_targets['volatility_range'][0]) / price_targets['current_price'] * 100)}%",
                                                    'backgroundColor': '#6366f1'
                                                }
                                            )
                                        ]
                                    )
                                ], style={'display': 'flex', 'alignItems': 'center'})
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
                            'Year', 'Adj Close', 'Return (%)',
                            'Max Drawdown (%)', 'Volume (M)'
                        ]
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Year']),
                        html.Td(f"${row['Adj Close']:.2f}"),
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

    @staticmethod
    def _calculate_metrics(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame] = None, benchmark_ticker: Optional[str] = None) -> Dict[str, Any]:
        """Calculate all metrics for dashboard cards with improved error handling"""
        try:
            metrics = {}
            
            # Always calculate basic metrics
            current_price = df['Adj Close'].iloc[-1]
            price_change = ((df['Adj Close'].iloc[-1] / df['Adj Close'].iloc[0]) - 1) * 100
            volatility = df['Volatility'].iloc[-1] * 100 if 'Volatility' in df else df['Adj Close'].pct_change().std() * np.sqrt(252) * 100
            
            daily_returns = df['Adj Close'].pct_change()
            sharpe_ratio = ((daily_returns.mean() * 252) - 0.02) / (daily_returns.std() * (252 ** 0.5))
            
            # Base price text
            change_text = f"{price_change:.1f}%"
            
            # Add benchmark comparison only if benchmark data is valid
            if benchmark_df is not None and not benchmark_df.empty:
                try:
                    benchmark_change = ((benchmark_df['Adj Close'].iloc[-1] / benchmark_df['Adj Close'].iloc[0]) - 1) * 100
                    relative_performance = price_change - benchmark_change
                    
                    change_text += (f" ({'+' if relative_performance > 0 else ''}{relative_performance:.1f}% "
                                f"vs {benchmark_ticker})")
                    
                    # Add benchmark status if all benchmark calculations succeeded
                    metrics['benchmark_status'] = html.Div([
                        html.I(className="fas fa-check-circle mr-2", style={"color": "green"}),
                        html.Span(f"Using {benchmark_ticker} as benchmark")
                    ], className="benchmark-status-success")
                except Exception as e:
                    logging.warning(f"Error calculating benchmark metrics: {str(e)}")
                    metrics['benchmark_status'] = html.Div([
                        html.I(className="fas fa-exclamation-circle mr-2", style={"color": "orange"}),
                        html.Span("Benchmark data incomplete")
                    ], className="benchmark-status-warning")
            else:
                # No benchmark case
                metrics['benchmark_status'] = html.Div([
                    html.I(className="fas fa-info-circle mr-2", style={"color": "gray"}),
                    html.Span("No benchmark selected")
                ], className="benchmark-status-info")
            
            # Create metric cards
            metrics.update({
                'price_card': DashboardComponents.create_metric_card(
                    "Current Price", f"${current_price:.2f}"
                ),
                'change_card': DashboardComponents.create_metric_card(
                    "Period Return", change_text
                ),
                'volatility_card': DashboardComponents.create_metric_card(
                    "Annualized Volatility", f"{volatility:.1f}%"
                ),
                'sharpe_card': DashboardComponents.create_metric_card(
                    "Sharpe Ratio", f"{sharpe_ratio:.2f}"
                )
            })
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {str(e)}")
            # Provide fallback metrics on error
            return {
                'price_card': DashboardComponents.create_metric_card("Current Price", "N/A"),
                'change_card': DashboardComponents.create_metric_card("Period Return", "N/A"),
                'volatility_card': DashboardComponents.create_metric_card("Volatility", "N/A"),
                'sharpe_card': DashboardComponents.create_metric_card("Sharpe Ratio", "N/A"),
                'benchmark_status': html.Div([
                    html.I(className="fas fa-exclamation-circle mr-2", style={"color": "red"}),
                    html.Span("Error calculating metrics")
                ], className="benchmark-status-error")
            }