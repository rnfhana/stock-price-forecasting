import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_interactive_chart(df, ticker):
    """Membuat chart harga, volume, dan ATR"""
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker} Price Action', 'Volume', 'Average True Range (ATR)')
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # 2. Volume
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'],
        name='Volume', marker_color='rgba(100, 149, 237, 0.5)'
    ), row=2, col=1)

    # 3. ATR
    if 'ATR' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['ATR'],
            name='ATR', line=dict(color='#ff9f43', width=2)
        ), row=3, col=1)

    fig.update_layout(
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    return fig

def plot_horizon_comparison(df_metrics):
    """Membuat grafik perbandingan error antar horizon"""
    # Transform data untuk plotting
    mape_cols = ['MAPE (%) H+1', 'MAPE (%) H+2', 'MAPE (%) H+3']
    
    fig = go.Figure()
    
    for idx, row in df_metrics.iterrows():
        fig.add_trace(go.Scatter(
            x=['H+1', 'H+2', 'H+3'],
            y=row[mape_cols],
            mode='lines+markers',
            name=row['Emiten']
        ))
        
    fig.update_layout(
        title="Degradasi Performa Model (MAPE) per Horizon",
        yaxis_title="MAPE (%)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig