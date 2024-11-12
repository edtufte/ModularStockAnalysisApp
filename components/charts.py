import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any

class ChartComponents:
    """Class for creating various chart components"""
    
    @staticmethod
    def create_price_chart(df, stock: str) -> go.Figure:
        """Create main price chart with candlesticks and moving averages"""
        fig = go.Figure()
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='20-day SMA',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='50-day SMA',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title=f'{stock} Price Movement',
            yaxis_title='Price (USD)',
            template='plotly_white'
        )
        
        return fig

    @staticmethod
    def create_volume_chart(df, stock: str) -> go.Figure:
        """Create volume chart"""
        fig = px.bar(
            df, x=df.index, y='Volume',
            title=f'{stock} Trading Volume'
        )
        fig.update_layout(template='plotly_white')
        return fig

    @staticmethod
    def create_technical_chart(df, stock: str) -> go.Figure:
        """Create technical analysis chart with Bollinger Bands and RSI"""
        fig = go.Figure()
        
        # Add Bollinger Bands
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_upper'],
            name='Upper BB',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_middle'],
            name='Middle BB',
            line=dict(color='gray')
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_lower'],
            name='Lower BB',
            line=dict(color='gray', dash='dash'),
            fill='tonexty'
        ))
        
        # Add price
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'],
            name='Price',
            line=dict(color='blue')
        ))
        
        # Add RSI
        fig.add_trace(go.Scatter(
            x=df.index, y=df['RSI'],
            name='RSI',
            yaxis="y2",
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'{stock} Technical Analysis',
            yaxis=dict(title='Price (USD)'),
            yaxis2=dict(
                title='RSI',
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            template='plotly_white'
        )
        
        return fig