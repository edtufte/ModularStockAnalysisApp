# callbacks/portfolio_callbacks.py
import dash
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from dash import html, ctx, callback_context, no_update
import logging
from typing import Dict, Any, List
from datetime import datetime
import json

from services.stock_data_service import StockDataService
from models.portfolio import Portfolio
from components.dashboard_components import DashboardComponents
from components.charts import ChartComponents
from services.technical_indicators import TechnicalIndicators

def register_callbacks(app, db):
    portfolio = Portfolio(db)

    @app.callback(
        Output('portfolio-update-trigger', 'data'),
        [Input("create-portfolio-button", "n_clicks"),
         Input("portfolio-name-input", "n_submit"),
         Input("clear-portfolios-button", "n_clicks"),
         Input("delete-portfolio-button", "n_clicks")]
    )
    def trigger_portfolio_update(create_clicks, name_submit, clear_clicks, delete_clicks):
        """Trigger a portfolio update when any portfolio action occurs"""
        if not callback_context.triggered:
            raise PreventUpdate
        # Return the current timestamp to ensure the trigger always has a new value
        return datetime.now().timestamp()

    @app.callback(
        [Output("portfolio-dropdown", "options"),
         Output("portfolio-dropdown-backtesting", "options")],
        [Input("portfolios-init-trigger", "data"),
         Input("portfolio-update-trigger", "data")],
        [State("user-id", "data")]
    )
    def update_portfolio_dropdowns(init_trigger, update_trigger, user_id):
        """Update all portfolio dropdowns"""
        try:
            portfolios = portfolio.get_user_portfolios(user_id)
            dropdown_options = create_dropdown_options(portfolios)
            return dropdown_options, dropdown_options
        except Exception as e:
            logging.error(f"Error updating portfolio dropdowns: {str(e)}")
            return [], []

    @app.callback(
        [Output("portfolio-dropdown", "value"),
         Output("create-portfolio-message", "children"),
         Output("portfolio-name-input", "value")],
        [Input("create-portfolio-button", "n_clicks"),
         Input("portfolio-name-input", "n_submit"),
         Input("clear-portfolios-button", "n_clicks"),
         Input("delete-portfolio-button", "n_clicks")],
        [State("portfolio-name-input", "value"),
         State("portfolio-dropdown", "value"),
         State("user-id", "data")]
    )
    def manage_portfolios(create_clicks, name_submit, clear_clicks, delete_clicks, 
                         portfolio_name, current_portfolio_id, user_id):
        """Handle portfolio management and selection"""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            # Create portfolio (via button or Enter key)
            if trigger_id in ["create-portfolio-button", "portfolio-name-input"] and (create_clicks or name_submit) and portfolio_name:
                portfolio_id = portfolio.create_portfolio(user_id, portfolio_name)
                return (
                    portfolio_id,  # Select the new portfolio
                    html.Div(f"Portfolio '{portfolio_name}' created successfully! (ID: {portfolio_id})", 
                            className="success-message"),
                    ""  # Clear the input
                )
                
            # Clear all portfolios
            elif trigger_id == "clear-portfolios-button" and clear_clicks:
                portfolio.clear_user_portfolios(user_id)
                return (
                    None,  # Clear selection
                    html.Div("All portfolios cleared", className="success-message"),
                    ""  # Clear the input
                )
                
            # Delete selected portfolio
            elif trigger_id == "delete-portfolio-button" and delete_clicks:
                if not current_portfolio_id:
                    return (
                        no_update,  # Keep current selection
                        html.Div("Please select a portfolio to delete", className="warning-message"),
                        no_update  # Keep input as is
                    )
                portfolio.delete_portfolio(current_portfolio_id)
                return (
                    None,  # Clear selection
                    html.Div(f"Portfolio (ID: {current_portfolio_id}) deleted", className="success-message"),
                    no_update  # Keep input as is
                )
                
            raise PreventUpdate
            
        except Exception as e:
            logging.error(f"Error managing portfolios: {str(e)}")
            return (
                no_update,  # Keep current selection
                html.Div(f"Error managing portfolios: {str(e)}", className="error-message"),
                no_update  # Keep input as is
            )

    @app.callback(
        [Output("portfolio-details", "children"),
         Output("holding-ticker-input", "value"),
         Output("holding-allocation-input", "value"),
         Output("add-holding-message", "children")],
        [Input("portfolio-dropdown", "value"),
         Input("add-holding-button", "n_clicks"),
         Input({"type": "remove-holding-button", "index": ALL}, "n_clicks")],
        [State("holding-ticker-input", "value"),
         State("holding-allocation-input", "value"),
         State("portfolio-details", "children")]
    )
    def manage_holdings(portfolio_id, add_clicks, remove_clicks, ticker, allocation, current_details):
        """Handle portfolio holdings management"""
        ctx = callback_context
        if not ctx.triggered or not portfolio_id:
            return "Select a portfolio to view its details", "", "", ""
            
        trigger = ctx.triggered[0]
        trigger_id = trigger['prop_id'].split('.')[0]
        
        try:
            # Handle remove holding
            if '"type":"remove-holding-button"' in trigger_id:
                try:
                    button_index = json.loads(trigger_id)['index']
                    details = portfolio.get_portfolio_details(portfolio_id)
                    if details["holdings"] and button_index < len(details["holdings"]):
                        holding_to_remove = details["holdings"][button_index]
                        portfolio.delete_holding(holding_to_remove["id"])
                        return create_portfolio_view(portfolio_id, portfolio), "", "", html.Div(
                            f"Removed {holding_to_remove['ticker']}", 
                            className="success-message"
                        )
                except Exception as e:
                    logging.error(f"Error removing holding: {str(e)}")
                    return no_update, no_update, no_update, html.Div(
                        f"Error removing holding: {str(e)}", 
                        className="error-message"
                    )
            
            # Handle add holding
            elif trigger_id == "add-holding-button" and add_clicks:
                if not all([ticker, allocation]):
                    return no_update, no_update, no_update, html.Div(
                        "Please fill in all fields",
                        className="warning-message"
                    )
                try:
                    portfolio.add_holding(portfolio_id, ticker.upper(), float(allocation))
                    return create_portfolio_view(portfolio_id, portfolio), "", "", html.Div(
                        f"Added {ticker.upper()} with {allocation}% allocation",
                        className="success-message"
                    )
                except Exception as e:
                    logging.error(f"Error adding holding: {str(e)}")
                    return no_update, no_update, no_update, html.Div(
                        f"Error adding holding: {str(e)}", 
                        className="error-message"
                    )
            
            # Handle portfolio selection change
            elif trigger_id == "portfolio-dropdown":
                return create_portfolio_view(portfolio_id, portfolio), "", "", ""
            
            raise PreventUpdate
            
        except Exception as e:
            logging.error(f"Error managing holdings: {str(e)}")
            return html.Div(
                f"Error: {str(e)}",
                className="error-message"
            ), no_update, no_update, ""

def create_dropdown_options(portfolios):
    """Helper function to create consistent dropdown options"""
    return [{
        "label": html.Div([
            html.Span(p["name"], style={'marginRight': '8px'}),
            html.Span(
                f"[ID: {p['id']}]",
                style={
                    'color': '#6B7280',
                    'fontSize': '0.875rem',
                    'fontStyle': 'italic'
                }
            )
        ]),
        "value": p["id"]
    } for p in portfolios]

def create_portfolio_view(portfolio_id, portfolio_instance):
    """Helper function to create consistent portfolio view"""
    details = portfolio_instance.get_portfolio_details(portfolio_id)
    
    if not details["holdings"]:
        return html.Div(
            "No holdings in this portfolio yet.",
            className="info-message"
        )
    
    total_allocation = sum(h["allocation"] for h in details["holdings"])
    
    return html.Div([
        html.H4("Holdings", className="subsection-title"),
        html.Div([
            html.Div(
                f"Total Allocation: {total_allocation:.1f}%",
                className="total-allocation",
                style={
                    'color': '#DC2626' if total_allocation > 100 else '#059669',
                    'fontWeight': '600',
                    'marginBottom': '12px'
                }
            ) if total_allocation != 0 else None,
            html.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Ticker"),
                        html.Th("Allocation (%)"),
                        html.Th("Actions")
                    ])
                ),
                html.Tbody([
                    html.Tr([
                        html.Td(h["ticker"]),
                        html.Td(f"{h['allocation']}%"),
                        html.Td(
                            html.Button(
                                "Remove",
                                id={'type': 'remove-holding-button', 'index': i},
                                n_clicks=0,
                                className="remove-button",
                                style={
                                    'backgroundColor': '#DC2626',
                                    'color': 'white',
                                    'border': 'none',
                                    'padding': '4px 8px',
                                    'borderRadius': '4px',
                                    'fontSize': '0.875rem',
                                    'cursor': 'pointer'
                                }
                            )
                        )
                    ]) for i, h in enumerate(details["holdings"])
                ])
            ], className="holdings-table")
        ])
    ])