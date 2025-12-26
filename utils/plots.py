import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_advanced_technical(df, emiten):
    """
    Robinhood-style Charting: Clean, Minimalist, Interactive
    """
    df_plot = df.copy()
    
    # Warna Candlestick Modern (Pluang Style)
    incr_color = '#00C853' # Vivid Green
    decr_color = '#FF3D00' # Vivid Red
    
    # Layout: Harga (70%), Volume (30%) - Indikator MACD/RSI opsional/hidden by default biar bersih
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )

    # 1. PRICE LINE (Area Chart ala Robinhood - Lebih clean drpd Candle kadang)
    # TAPI karena ini analisis teknikal, Candle tetap terbaik. Kita buat Candle yg clean.
    fig.add_trace(go.Candlestick(
        x=df_plot['date'],
        open=df_plot['X1'], high=df_plot['X2'],
        low=df_plot['X3'], close=df_plot['Yt'],
        name='OHLC',
        increasing_line_color=incr_color,
        decreasing_line_color=decr_color,
        increasing_fillcolor=incr_color, # Isi badan candle
        decreasing_fillcolor=decr_color
    ), row=1, col=1)

    # Moving Average (Pemanis wajib)
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['Yt'].rolling(window=20).mean(),
        name='MA20', line=dict(color='#2962FF', width=1.5), opacity=0.7
    ), row=1, col=1)

    # 2. VOLUME (Bar Chart Minimalis)
    vol_colors = [incr_color if c >= o else decr_color for c, o in zip(df_plot['Yt'], df_plot['X1'])]
    fig.add_trace(go.Bar(
        x=df_plot['date'], y=df_plot['X4'],
        name='Volume', marker_color=vol_colors, opacity=0.3 # Transparan biar gak ganggu
    ), row=2, col=1)

    # STYLING PARAH (Biar mahal)
    fig.update_layout(
        title=dict(text=f"<b>{emiten}</b> Price Action", font=dict(size=24, family="Inter")),
        template="plotly_white",
        height=600,
        showlegend=False,
        margin=dict(l=0, r=40, t=50, b=20),
        xaxis=dict(
            rangeslider=dict(visible=False), # Hapus slider bawah yg jelek
            showgrid=False,
            type="date"
        ),
        yaxis=dict(showgrid=True, gridcolor='#F3F4F6', side='right'), # Harga di kanan ala TradingView
        yaxis2=dict(showgrid=False, side='right', showticklabels=False), # Volume tanpa angka y-axis
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)', # Transparan
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def plot_interactive_forecast(df_hist, pred_base, pred_fuse, dates_fut, emiten):
    """
    Fan Chart untuk Halaman Prediksi
    """
    # Ambil data secukupnya untuk konteks visual (misal 3 bulan terakhir)
    last_30 = df_hist.tail(90)
    
    fig = go.Figure()
    
    # Historis
    fig.add_trace(go.Scatter(
        x=last_30['date'], y=last_30['Yt'],
        mode='lines', name='Historical',
        line=dict(color='#111827', width=2)
    ))
    
    # Baseline
    fig.add_trace(go.Scatter(
        x=dates_fut, y=pred_base,
        mode='lines+markers', name='Baseline Forecast',
        line=dict(color='#d62728', width=2, dash='dash'),
        marker=dict(symbol='circle')
    ))
    
    # Fusion
    fig.add_trace(go.Scatter(
        x=dates_fut, y=pred_fuse,
        mode='lines+markers', name='Fusion Forecast',
        line=dict(color='#0052CC', width=3),
        marker=dict(symbol='diamond', size=8)
    ))
    
    # Connector line
    fig.add_trace(go.Scatter(
        x=[last_30['date'].iloc[-1], dates_fut[0]],
        y=[last_30['Yt'].iloc[-1], pred_fuse[0]],
        mode='lines', showlegend=False,
        line=dict(color='gray', width=1, dash='dot')
    ))

    fig.update_layout(
        title=f"Forecast Scenario: {emiten} (Next 3 Days)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.05, x=1, xanchor="right")
    )
    return fig

def plot_interactive_shap(df_shap, title_text):
    """
    Plot SHAP Values secara Interaktif (Bar Chart Horizontal).
    Warna otomatis beda antara Technical vs Sentiment.
    """
    # Sort biar yang paling penting di atas
    df_sorted = df_shap.sort_values('Importance', ascending=True)
    
    # Warna: Biru (Tech), Merah (Sentiment)
    colors = ['#d62728' if cat == 'Sentiment' else '#1f77b4' for cat in df_sorted['Category']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_sorted['Feature Name'], # Pakai nama yang cantik
        x=df_sorted['Importance'],
        orientation='h',
        marker=dict(color=colors, opacity=0.9),
        text=df_sorted['Importance'].apply(lambda x: f"{x:.4f}"), # Tampilkan angka di bar
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.5f}<br>Category: %{customdata}<extra></extra>',
        customdata=df_sorted['Category']
    ))
    
    fig.update_layout(
        title=dict(text=f"<b>{title_text}</b>", font=dict(size=18)),
        xaxis_title="Mean |SHAP Value| (Impact Magnitude)",
        yaxis_title=None,
        template="plotly_white",
        height=500,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False
    )
    
    # Tambahkan Legend Manual pakai Annotation dummy (biar user tahu merah itu apa)
    fig.add_annotation(x=1, y=0, xref='paper', yref='paper', text='🟦 Technical', showarrow=False, xanchor='right', yanchor='bottom', yshift=-30, xshift=-80)
    fig.add_annotation(x=1, y=0, xref='paper', yref='paper', text='🟥 Sentiment', showarrow=False, xanchor='right', yanchor='bottom', yshift=-30)

    return fig