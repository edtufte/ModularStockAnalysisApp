# components/charts.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, Optional, Tuple

class ChartComponents:
    """Class for creating various chart components"""
    
    @staticmethod
    def create_price_chart(df, stock: str, benchmark_df=None, benchmark_name: str = None) -> go.Figure:
        """Create main price chart with candlesticks, moving averages, and benchmark comparison
        
        Args:
            df: DataFrame with stock data
            stock: Stock ticker symbol
            benchmark_df: Optional DataFrame with benchmark data
            benchmark_name: Name of the benchmark index
        """
        # Calculate percentage change from first point
        first_price = df['Close'].iloc[0]
        df['pct_change'] = ((df['Close'] - first_price) / first_price) * 100
        
        if benchmark_df is not None:
            first_benchmark = benchmark_df['Close'].iloc[0]
            benchmark_df['pct_change'] = ((benchmark_df['Close'] - first_benchmark) / first_benchmark) * 100
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=stock,
                yaxis='y1'
            ),
            secondary_y=False
        )
        
        # Add moving averages on price axis
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['SMA_20'],
                name='20-day SMA',
                line=dict(color='orange', width=1),
                yaxis='y1'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['SMA_50'],
                name='50-day SMA',
                line=dict(color='blue', width=1),
                yaxis='y1'
            ),
            secondary_y=False
        )
        
        # Add percentage change line for stock
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['pct_change'],
                name=f'{stock} %',
                line=dict(color='purple', width=1, dash='dot'),
                yaxis='y2'
            ),
            secondary_y=True
        )
        
        # Add benchmark if provided
        if benchmark_df is not None and not benchmark_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_df.index,
                    y=benchmark_df['pct_change'],
                    name=f'{benchmark_name} %',
                    line=dict(color='gray', width=1, dash='dot'),
                    yaxis='y2'
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=f'{stock} Price Movement vs {benchmark_name or "Market"}',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            yaxis2_title='Change (%)',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="Change (%)", secondary_y=True)
        
        return fig

    @staticmethod
    def create_volume_chart(df, stock: str) -> go.Figure:
        """Create volume chart"""
        fig = go.Figure(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.5)'
        ))
        
        fig.update_layout(
            title=f'{stock} Trading Volume',
            yaxis_title='Volume',
            template='plotly_white'
        )
        return fig

    @staticmethod
    def create_technical_chart(df, stock: str) -> go.Figure:
        """Create technical analysis chart with Bollinger Bands and RSI"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_upper'],
                name='Upper BB',
                line=dict(color='gray', dash='dash'),
                showlegend=True
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_middle'],
                name='Middle BB',
                line=dict(color='gray'),
                showlegend=True
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['BB_lower'],
                name='Lower BB',
                line=dict(color='gray', dash='dash'),
                fill='tonexty',
                showlegend=True
            ),
            secondary_y=False
        )
        
        # Add price
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['Close'],
                name='Price',
                line=dict(color='blue'),
                showlegend=True
            ),
            secondary_y=False
        )
        
        # Add RSI
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['RSI'],
                name='RSI',
                line=dict(color='red'),
                showlegend=True
            ),
            secondary_y=True
        )
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5,
                     annotation_text="Overbought (70)", secondary_y=True)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5,
                     annotation_text="Oversold (30)", secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title=f'{stock} Technical Analysis',
            template='plotly_white',
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            )
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
        fig.update_yaxes(title_text="RSI", secondary_y=True, range=[0, 100])
        
        return fig