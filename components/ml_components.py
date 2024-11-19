from dash import html, dcc
import plotly.graph_objs as go
from typing import Dict, Any
import pandas as pd
import dash_bootstrap_components as dbc
from config.metric_definitions import METRIC_DEFINITIONS

def create_ml_prediction_card(predictions: Dict[str, Any]) -> html.Div:
    """Create a card displaying ML predictions and insights"""
    if not predictions or 'predictions' not in predictions:
        return html.Div("No prediction data available")
    
    pred_data = predictions['predictions']
    
    # Extract prediction data
    historical_dates = pred_data['historical_dates']
    historical_values = pred_data['historical_values']
    historical_actual = pred_data['historical_actual']
    forecast_dates = pred_data['forecast_dates']
    forecast_values = pred_data['forecast_values']
    lower_bound = pred_data['forecast_lower']
    upper_bound = pred_data['forecast_upper']
    
    # Create prediction plot
    fig = go.Figure()
    
    # Convert dates to pandas Timestamps
    historical_dates = pd.to_datetime(historical_dates)
    forecast_dates = pd.to_datetime(forecast_dates)
    today = pd.Timestamp.now().floor('D')
    
    # Add historical actual prices
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_actual,
        name='Actual Price',
        line=dict(
            color='rgb(67, 67, 67)',
            width=2
        )
    ))
    
    # Add historical fitted values
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_values,
        name='Model Fit',
        line=dict(
            color='rgb(31, 119, 180)',
            width=1,
            dash='dash'
        ),
        opacity=0.7
    ))
    
    # Add future prediction
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        name='Forecast',
        line=dict(
            color='rgb(31, 119, 180)',
            width=2,
            dash='dot'
        )
    ))
    
    # Add confidence interval for future predictions
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=upper_bound,
        fill=None,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=lower_bound,
        fill='tonexty',
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(31, 119, 180, 0.1)',
        name='Confidence Interval'
    ))
    
    # Add vertical line at today's date (without annotation)
    fig.add_vline(
        x=today.timestamp() * 1000,
        line_width=2,
        line_dash="solid",
        line_color="rgba(255, 0, 0, 0.5)"
    )
    
    # Add centered "Today" annotation
    fig.add_annotation(
        x=today.timestamp() * 1000,
        y=1,
        yref="paper",
        text="Today",
        showarrow=False,
        font=dict(color="rgba(255, 0, 0, 0.7)"),
        yshift=10,
        xanchor="center",
        yanchor="bottom"
    )
    
    # Add training period annotation
    if 'training_period' in pred_data:
        training_text = f"Model trained on data from {pred_data['training_period']['start']} to {pred_data['training_period']['end']}"
        fig.add_annotation(
            x=0.5,
            y=0,
            xref="paper",
            yref="paper",
            text=training_text,
            showarrow=False,
            font=dict(
                size=10,
                color="rgba(0, 0, 0, 0.5)"
            ),
            xanchor="center",
            yanchor="bottom",
            yshift=10,
            xshift=0
        )
    
    # Find the last date in the forecast
    last_date = forecast_dates.max()
    
    fig.update_layout(
        title='Price Forecast (60 Days)',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        # Add shaded background for forecast period
        shapes=[{
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': today.timestamp() * 1000,
            'y0': 0,
            'x1': last_date.timestamp() * 1000,
            'y1': 1,
            'fillcolor': 'rgba(200, 200, 200, 0.1)',
            'layer': 'below',
            'line_width': 0,
        }]
    )
    
    prediction_chart = dcc.Graph(figure=fig)
    
    # Only show feature importance if available
    feature_chart = None
    if 'feature_importance' in pred_data:
        feature_importance = pred_data['feature_importance']
        top_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5])
        
        feature_fig = go.Figure(go.Bar(
            x=list(top_features.values()),
            y=list(top_features.keys()),
            orientation='h'
        ))
        
        feature_fig.update_layout(
            title='Top 5 Important Features',
            xaxis_title='Importance Score',
            yaxis_title='Feature',
            template='plotly_white',
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        feature_chart = dcc.Graph(figure=feature_fig)
    
    # Create metrics section if available
    metrics_section = None
    if 'metrics' in pred_data:
        metrics = pred_data['metrics']
        
        # Define metric tooltips
        metric_info = {
            'mape': {
                'name': 'In-Sample MAPE',
                'tooltip': 'In-Sample Mean Absolute Percentage Error: Measures how well the model fits historical data. Lower is better.',
                'format': lambda x: f"{x:.2f}%"
            },
            'forward_mape': {
                'name': 'Forward MAPE (30d)',
                'tooltip': 'Forward-Looking Mean Absolute Percentage Error: Measures actual 30-day prediction accuracy on historical data. This is a more realistic measure of prediction performance.',
                'format': lambda x: f"{x:.2f}%" if x is not None else "N/A"
            },
            'rmse': {
                'name': 'RMSE',
                'tooltip': 'Root Mean Square Error: Measures prediction accuracy in price units. Lower is better.',
                'format': lambda x: f"${x:.2f}"
            },
            'r2': {
                'name': 'R²',
                'tooltip': 'R-squared: Measures how well the model fits the data. Ranges from 0 to 1, higher is better.',
                'format': lambda x: f"{x:.3f}"
            },
            'forecast_accuracy': {
                'name': 'Forecast Accuracy',
                'tooltip': 'Percentage of predictions within the confidence interval. Higher is better.',
                'format': lambda x: f"{x:.1f}%"
            }
        }
        
        # Create metrics containers with tooltips
        metric_containers = []
        for metric, info in metric_info.items():
            if metric in metrics:
                metric_id = f'metric-{metric}'
                container = html.Div([
                    html.Div([
                        html.P(info['name'], className="metric-label"),
                        html.I(
                            className="fas fa-question-circle ml-1",
                            id=metric_id,
                            style={'color': 'rgb(180, 180, 180)', 'cursor': 'help'}
                        )
                    ], className="d-flex align-items-center justify-content-between"),
                    html.P(
                        info['format'](metrics.get(metric, 0)),
                        className="metric-value"
                    ),
                    dbc.Tooltip(
                        info['tooltip'],
                        target=metric_id,
                        placement="top",
                        style={
                            "backgroundColor": "rgba(50, 50, 50, 0.95)",
                            "color": "white",
                            "maxWidth": "300px",
                            "fontSize": "0.9rem",
                            "padding": "8px 12px",
                            "borderRadius": "4px",
                            "boxShadow": "0 2px 4px rgba(0,0,0,0.2)",
                            "zIndex": 1000
                        },
                        delay={"show": 200, "hide": 50}
                    )
                ], className="metric-container")
                metric_containers.append(container)
        
        metrics_section = html.Div([
            html.Div([
                html.H4("Model Performance", className="metrics-title"),
                html.I(
                    className="fas fa-info-circle ml-2",
                    id='metrics-info-icon',
                    style={'color': 'rgb(31, 119, 180)', 'cursor': 'pointer'}
                )
            ], className="d-flex align-items-center mb-3"),
            
            # Metrics grid
            html.Div(
                metric_containers,
                className="metrics-grid"
            )
        ], className="model-performance-container")
    
    # Update the main container's className for better styling
    return html.Div([
        prediction_chart,
        html.Div([
            metrics_section if metrics_section else None,
            feature_chart if feature_chart else None
        ], className="model-insights-container")
    ], className="ml-prediction-container")
