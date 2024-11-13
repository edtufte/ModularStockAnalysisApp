# callbacks/portfolio_callbacks.py
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash import html, ctx
import logging
from models.portfolio import Portfolio

def register_callbacks(app, db):
    portfolio = Portfolio(db)

    @app.callback(
        [Output("create-portfolio-message", "children"),
         Output("portfolio-dropdown", "options"),
         Output("portfolio-dropdown-backtesting", "options"),  # Combined with main dropdown update
         Output("portfolio-dropdown", "value"),
         Output("portfolio-name-input", "value")],
        Input("create-portfolio-button", "n_clicks"),
        [State("portfolio-name-input", "value"),
         State("user-id", "data")]
    )
    def create_portfolio(n_clicks, name, user_id):
        if n_clicks == 0 or not name:
            raise PreventUpdate
            
        try:
            # Create the portfolio
            portfolio_id = portfolio.create_portfolio(user_id, name)
            
            # Get updated portfolio list
            portfolios = portfolio.get_user_portfolios(user_id)
            dropdown_options = [{"label": p["name"], "value": p["id"]} for p in portfolios]
            
            success_message = html.Div(
                f"Portfolio '{name}' created successfully!",
                className="success-message"
            )
            
            # Return: message, updated options for both dropdowns, select new portfolio, clear input
            return success_message, dropdown_options, dropdown_options, portfolio_id, ""
            
        except Exception as e:
            logging.error(f"Error creating portfolio: {str(e)}")
            # Return: error message, keep existing options, no selection, keep input
            return html.Div(
                f"Error creating portfolio: {str(e)}",
                className="error-message"
            ), dash.no_update, dash.no_update, dash.no_update, dash.no_update

    @app.callback(
        Output("portfolio-dropdown-backtesting", "value"),
        Input("portfolio-dropdown", "value")
    )
    def sync_portfolio_selection(selected_value):
        return selected_value

    @app.callback(
        [Output("portfolio-details", "children"),
         Output("holding-ticker-input", "value"),
         Output("holding-allocation-input", "value"),
         Output("add-holding-message", "children")],
        [Input("portfolio-dropdown", "value"),
         Input("add-holding-button", "n_clicks")],
        [State("holding-ticker-input", "value"),
         State("holding-allocation-input", "value")]
    )
    def update_portfolio_view(portfolio_id, add_clicks, ticker, allocation):
        triggered_id = ctx.triggered_id if not None else None
        
        if not portfolio_id:
            return "Select a portfolio to view its details", "", "", ""
            
        try:
            # Handle adding a new holding
            if triggered_id == "add-holding-button" and add_clicks > 0:
                if not all([ticker, allocation]):
                    return dash.no_update, dash.no_update, dash.no_update, html.Div(
                        "Please fill in all fields",
                        className="warning-message"
                    )
                try:
                    portfolio.add_holding(portfolio_id, ticker.upper(), allocation)
                    success_message = html.Div(
                        f"Added {ticker.upper()} with {allocation}% allocation",
                        className="success-message"
                    )
                except Exception as e:
                    logging.error(f"Error adding holding: {str(e)}")
                    return dash.no_update, dash.no_update, dash.no_update, html.Div(
                        f"Error adding holding: {str(e)}",
                        className="error-message"
                    )
            else:
                success_message = ""
            
            # Get updated portfolio details
            details = portfolio.get_portfolio_details(portfolio_id)
            if not details["holdings"]:
                details_div = html.Div("No holdings in this portfolio yet.", className="info-message")
            else:
                details_div = html.Div([
                    html.H4("Holdings", className="subsection-title"),
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("Ticker"),
                                html.Th("Allocation (%)")
                            ])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(h["ticker"]),
                                html.Td(f"{h['allocation']}%")
                            ]) for h in details["holdings"]
                        ])
                    ], className="holdings-table")
                ])
            
            # Return: updated details, clear inputs if holding was added, message
            return (
                details_div,
                "" if triggered_id == "add-holding-button" else dash.no_update,
                "" if triggered_id == "add-holding-button" else dash.no_update,
                success_message
            )
            
        except Exception as e:
            logging.error(f"Error updating portfolio view: {str(e)}")
            return html.Div(
                f"Error loading portfolio details: {str(e)}",
                className="error-message"
            ), dash.no_update, dash.no_update, ""