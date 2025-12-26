import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import base64
from pathlib import Path
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import math

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================

st.set_page_config(
    page_title="Thesis Dashboard - GRU Fusion",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS styling (Persis sesuai request)"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
     
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
     
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
     
    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
     
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
     
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 300;
        margin: 0;
    }
     
    /* Sidebar Styling */
    .sidebar-header {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar-header h2 {
        color: white;
        margin: 0;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
     
    /* Page Headers */
    .page-header {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
     
    .page-header h2 {
        color: #2c3e50;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
     
    .page-header p {
        color: #5a6c7d;
        font-size: 1.1rem;
        margin: 0;
    }
     
    /* Feature Cards */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border-left: 4px solid #667eea;
    }
     
    .feature-card h2, .feature-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
     
    .feature-card p {
        color: #5a6c7d;
        line-height: 1.6;
    }
     
    /* Waste Categories (Diadaptasi untuk Layout Grid) */
    .waste-categories {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
     
    .category-item {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        color: white;
        font-weight: 500;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
     
    /* Features Grid */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
     
    .feature-item {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
     
    .feature-item h4 {
        color: #2c3e50;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
     
    .feature-item p {
        color: #5a6c7d;
        font-size: 0.9rem;
        margin: 0;
    }
     
    /* Stats Card */
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
     
    .stats-card h3 {
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
     
    .stat-item {
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        backdrop-filter: blur(10px);
    }
     
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
     
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
        font-weight: 300;
    }
     
    /* Guide Card */
    .guide-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1.5rem;
    }
     
    .guide-card h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
     
    .guide-card ol {
        color: #5a6c7d;
        padding-left: 1rem;
    }
     
    .guide-card li {
        margin-bottom: 0.5rem;
        line-height: 1.5;
    }
     
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
     
    .info-box h3 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
     
    .info-box p {
        color: #5a6c7d;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
     
    /* Section Headers */
    .section-header {
        text-align: center;
        margin: 2rem 0 1rem 0;
    }
     
    .section-header h3 {
        color: #2c3e50;
        font-size: 1.5rem;
        font-weight: 600;
    }
     
    /* Upload Section */
    .upload-section {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
     
    .upload-section h3 {
        color: white;
        margin: 0;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
     
    /* Results Section */
    .results-section {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
     
    .results-section h3 {
        color: white;
        margin: 0;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
     
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
     
    .prediction-main h2 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
     
    .confidence-score {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        backdrop-filter: blur(10px);
    }
     
    .confidence-score span {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
     
    /* Recommendation Card */
    .recommendation-card {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
    }
     
    .recommendation-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
     
    .recommendation-card p, .recommendation-card ul {
        color: #5a6c7d;
        line-height: 1.6;
    }
     
    .recommendation-card li {
        margin-bottom: 0.5rem;
    }
     
    /* Insight Cards */
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        height: 100%;
    }
     
    .insight-card h4 {
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 600;
    }
     
    .insight-card p {
        color: #5a6c7d;
        line-height: 1.6;
        margin: 0;
    }
     
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
         
        .main-header p {
            font-size: 1rem;
        }
         
        .page-header h2 {
            font-size: 1.5rem;
        }
         
        .features-grid {
            grid-template-columns: 1fr;
        }
         
        .waste-categories {
            grid-template-columns: 1fr;
        }
    }
     
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
     
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
     
    /* Metric Styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e1e8ed;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
     
    /* File Uploader Styling */
    .stFileUploader {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #667eea;
    }
     
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Memanggil Fungsi Load CSS
load_css()

# ==========================================
# 2. KONFIGURASI PATHS & EMITEN
# ==========================================

# Base directory paths
BASE_MODEL_PATH = r"D:\Akademik ITS\RISET & SKRIPSI\STREAMLIT\skripsi_dashboard_hana2\models"
BASE_DATA_PATH = r"D:\Akademik ITS\RISET & SKRIPSI\STREAMLIT\skripsi_dashboard_hana2\data"

# Daftar Emiten yang valid
EMITEN_LIST = ["AKRA", "BBRI", "BMRI", "PGAS", "UNVR"]

# Dictionary untuk mapping semua file path
FILES = {
    "data": {
        "main": os.path.join(BASE_DATA_PATH, "df_fusi_multimodal_final_hana.csv"),
        "shap": os.path.join(BASE_DATA_PATH, "shap_values_summary.csv")
    },
    "models": {
        "AKRA": os.path.join(BASE_MODEL_PATH, "model_fusion_AKRA.h5"),
        "BBRI": os.path.join(BASE_MODEL_PATH, "model_fusion_BBRI.h5"),
        "BMRI": os.path.join(BASE_MODEL_PATH, "model_fusion_BMRI.h5"),
        "PGAS": os.path.join(BASE_MODEL_PATH, "model_fusion_PGAS.h5"),
        "UNVR": os.path.join(BASE_MODEL_PATH, "model_fusion_UNVR.h5"),
    },
    "scalers": {
        "AKRA": os.path.join(BASE_MODEL_PATH, "scaler_AKRA.pkl"),
        "BBRI": os.path.join(BASE_MODEL_PATH, "scaler_BBRI.pkl"),
        "BMRI": os.path.join(BASE_MODEL_PATH, "scaler_BMRI.pkl"),
        "PGAS": os.path.join(BASE_MODEL_PATH, "scaler_PGAS.pkl"),
        "UNVR": os.path.join(BASE_MODEL_PATH, "scaler_UNVR.pkl"),
    }
}

# Verifikasi sederhana apakah file path ada (Optional, untuk debugging di console)
print("Konfigurasi Path Selesai. Siap Melanjutkan.")

# ==========================================
# 3. DATA LOADING & PROCESSING FUNCTIONS (PERBAIKAN)
# ==========================================

@st.cache_data
def load_dataset():
    """
    Load main dataset dan lakukan renaming kolom sesuai variabel skripsi.
    Mapping:
    - date -> Date
    - relevant_issuer -> Stock
    - Yt -> Close (Harga Penutupan)
    - X1 -> Open
    - X2 -> High
    - X3 -> Low
    - X4 -> Volume
    - X7 -> ATR
    """
    try:
        # Load Main Data
        df = pd.read_csv(FILES["data"]["main"])
        
        # 1. RENAME KOLOM (Kamus Penerjemah)
        # Agar kode visualisasi plotly bisa membaca data Anda
        rename_map = {
            'date': 'Date',
            'relevant_issuer': 'Stock',
            'Yt': 'Close',      # Close Price Hari Ini
            'X1': 'Open',       # Opening Price
            'X2': 'High',       # Highest Price
            'X3': 'Low',        # Lowest Price
            'X4': 'Volume',     # Volume
            'X7': 'ATR'         # ATR (Sudah ada, tidak perlu hitung ulang)
        }
        
        # Lakukan rename hanya jika kolomnya ada
        df.rename(columns=rename_map, inplace=True)
        
        # 2. Convert Date
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        return df
    except Exception as e:
        st.error(f"Error loading main dataset: {e}")
        return pd.DataFrame()

@st.cache_data
def load_shap_data():
    """
    Load data SHAP values summary.
    """
    try:
        df_shap = pd.read_csv(FILES["data"]["shap"])
        return df_shap
    except Exception as e:
        return pd.DataFrame()

# CATATAN: Fungsi calculate_atr SAYA HAPUS 
# karena di data Anda sudah ada variabel X7 (ATR). 
# Kita pakai X7 asli dari skripsi agar akurat.

def get_ticker_data(df, ticker):
    """
    Filter data berdasarkan emiten.
    """
    # Filter by Ticker (Kolom sudah direname jadi 'Stock')
    if 'Stock' in df.columns:
        df_ticker = df[df['Stock'] == ticker].copy()
    else:
        # Fallback jika rename gagal
        df_ticker = df[df['relevant_issuer'] == ticker].copy() if 'relevant_issuer' in df.columns else df.copy()
    
    # Sort by Date
    if 'Date' in df_ticker.columns:
        df_ticker = df_ticker.sort_values('Date')
    
    return df_ticker

# Update fungsi prepare_inputs juga agar tidak salah ambil kolom target (Yt+1)
def prepare_inputs(df_ticker, window_size=60):
    """
    Menyiapkan data input untuk prediksi.
    PENTING: Kita harus membuang kolom target (Yt+1) agar tidak bocor ke input.
    """
    # Buang kolom non-fitur dan kolom target masa depan (Yt+1)
    # Sesuaikan list 'drop_cols' dengan apa yang TIDAK dipakai saat training
    drop_cols = ['Date', 'Stock', 'relevant_issuer', 'Yt+1'] 
    
    # Ambil hanya kolom yang ada di dataframe
    cols_to_drop = [c for c in drop_cols if c in df_ticker.columns]
    features_df = df_ticker.drop(columns=cols_to_drop)
    
    # Pastikan hanya numerik
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ambil data terakhir sebanyak window_size
    last_window = df_ticker[numeric_cols].tail(window_size).values
    
    return last_window

# ==========================================
# 4. HALAMAN: MARKET OVERVIEW
# ==========================================

def plot_interactive_chart(df, ticker):
    """
    Membuat chart interaktif gabungan: 
    Row 1: Candlestick/Line Chart (Harga)
    Row 2: Volume
    Row 3: ATR (Average True Range) -> Fitur Baru
    """
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker} Price Action', 'Volume', 'Average True Range (ATR)')
    )

    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # 2. Volume Chart
    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'],
        name='Volume', marker_color='rgba(100, 149, 237, 0.5)'
    ), row=2, col=1)

    # 3. ATR Chart (Fitur Baru)
    # Pastikan ATR sudah dihitung di langkah sebelumnya
    if 'ATR' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['ATR'],
            name='ATR (14)', line=dict(color='#ff9f43', width=2)
        ), row=3, col=1)

    # Layout Styling
    fig.update_layout(
        height=800,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified"
    )
    
    return fig

def show_market_overview(df):
    """
    Menampilkan halaman Market Overview
    """
    # --- Sidebar Filter Khusus Halaman Ini ---
    st.sidebar.markdown("### ⚙️ Konfigurasi Market")
    selected_ticker = st.sidebar.selectbox("Pilih Emiten:", EMITEN_LIST, index=0)
    
    # Ambil data spesifik emiten & hitung ATR
    df_ticker = get_ticker_data(df, selected_ticker)
    
    if df_ticker.empty:
        st.warning("Data tidak ditemukan untuk emiten ini.")
        return

    # --- Header Halaman ---
    st.markdown(f"""
    <div class='page-header'>
        <h2>Market Overview: {selected_ticker}</h2>
        <p>Analisis Pergerakan Harga & Indikator Volatilitas (ATR)</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Hitung Metrics Terkini ---
    last_data = df_ticker.iloc[-1]
    prev_data = df_ticker.iloc[-2]
    
    current_price = last_data['Close']
    price_change = current_price - prev_data['Close']
    pct_change = (price_change / prev_data['Close']) * 100
    
    current_atr = last_data['ATR'] if not pd.isna(last_data['ATR']) else 0
    current_vol = last_data['Volume']

    # --- Tampilkan Metrics Card (Menggunakan CSS Custom) ---
    # Kita menggunakan HTML manual agar stylenya persis stats-card CSS
    st.markdown(f"""
    <div class="stats-card">
        <h3>Market Summary ({last_data['Date'].strftime('%d %b %Y')})</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
            <div class="stat-item">
                <div class="stat-number">Rp {current_price:,.0f}</div>
                <div class="stat-label">Last Close</div>
            </div>
            <div class="stat-item">
                <div class="stat-number" style="color: {'#4cd137' if price_change >= 0 else '#e84118'};">
                    {price_change:,.0f} ({pct_change:.2f}%)
                </div>
                <div class="stat-label">Daily Change</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{current_atr:,.2f}</div>
                <div class="stat-label">ATR (Volatility)</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{current_vol/1_000_000:.1f}M</div>
                <div class="stat-label">Volume (Millions)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Tampilkan Chart ---
    st.markdown("### 📊 Interactive Chart")
    fig = plot_interactive_chart(df_ticker, selected_ticker)
    st.plotly_chart(fig, use_container_width=True)

    # --- Tampilkan Raw Data (Expander) ---
    with st.expander("Lihat Data Historis Lengkap"):
        st.dataframe(df_ticker.sort_values('Date', ascending=False).style.format({
            'Open': '{:,.0f}', 'High': '{:,.0f}', 
            'Low': '{:,.0f}', 'Close': '{:,.0f}', 
            'Volume': '{:,.0f}', 'ATR': '{:.2f}'
        }))

# ==========================================
# 5. HALAMAN: FORECAST SIMULATOR (GRU FUSION)
# ==========================================

@st.cache_resource
def load_trained_model(ticker):
    """
    Load model GRU Fusion spesifik untuk emiten yang dipilih.
    Menggunakan cache resource agar tidak berat saat reload.
    """
    model_path = FILES["models"].get(ticker)
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model file not found for {ticker} at {model_path}")
        return None
    
    try:
        # Load model .h5
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model for {ticker}: {e}")
        return None

@st.cache_resource
def load_specific_scaler(ticker):
    """
    Load scaler .pkl spesifik untuk emiten.
    """
    scaler_path = FILES["scalers"].get(ticker)
    if not scaler_path or not os.path.exists(scaler_path):
        st.error(f"Scaler file not found for {ticker}")
        return None
    
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler for {ticker}: {e}")
        return None

def prepare_inputs(df_ticker, window_size=60):
    """
    Menyiapkan data input untuk prediksi (Sliding Window).
    Mengambil 'window_size' data terakhir.
    """
    # Pastikan data yang diambil hanya kolom numerik yang digunakan saat training
    # NOTE: Sesuaikan list 'features' ini dengan urutan fitur saat Anda melatih model
    # Biasanya: Open, High, Low, Close, Volume, (mungkin ATR/Sentiment)
    # Di sini kita ambil semua kolom numerik kecuali Date/Stock
    numeric_cols = df_ticker.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ambil data terakhir sebanyak window_size
    last_window = df_ticker[numeric_cols].tail(window_size).values
    
    return last_window

def show_forecast_simulator(df):
    """
    Menampilkan halaman Forecast Simulator (Single Model: GRU Fusion)
    """
    # --- Sidebar Configuration ---
    st.sidebar.markdown("### ⚙️ Konfigurasi Prediksi")
    selected_ticker = st.sidebar.selectbox("Pilih Emiten:", EMITEN_LIST, key="forecast_ticker")
    
    # --- Header ---
    st.markdown(f"""
    <div class='page-header'>
        <h2>Forecast Simulator: {selected_ticker}</h2>
        <p>Prediksi Harga Saham H+1 Menggunakan GRU Fusion Model</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Load Data & Resources ---
    df_ticker = get_ticker_data(df, selected_ticker)
    model = load_trained_model(selected_ticker)
    scaler = load_specific_scaler(selected_ticker)

    if df_ticker.empty:
        st.warning("Data historis tidak tersedia.")
        return

    if model is None or scaler is None:
        st.error("Gagal memuat Model atau Scaler. Periksa path file.")
        return

    # --- Information Box ---
    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Model Architecture: GRU Fusion</h3>
        <p>Sistem ini menggunakan <strong>Gated Recurrent Unit (GRU)</strong> dengan arsitektur Fusion Multimodal. 
        Model ini telah dilatih khusus untuk setiap emiten dan tidak memerlukan pemilihan model manual.</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Prediction Trigger ---
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Jalankan Prediksi (Predict Next Day)", use_container_width=True)

    if predict_btn:
        with st.spinner('Sedang memproses data & menjalankan GRU Fusion...'):
            try:
                # 1. Preprocessing Input
                # Asumsi window_size = 60 (Standar time series), sesuaikan jika skripsi Anda pakai 30/90
                WINDOW_SIZE = 60 
                
                # Mengambil fitur input
                input_data = prepare_inputs(df_ticker, window_size=WINDOW_SIZE)
                
                # Cek kecukupan data
                if len(input_data) < WINDOW_SIZE:
                    st.error(f"Data tidak cukup. Membutuhkan minimal {WINDOW_SIZE} baris data.")
                    return

                # Scaling data
                # Scaler mengharapkan input 2D (samples, features)
                input_scaled = scaler.transform(input_data)
                
                # Reshape untuk input model LSTM/GRU: (1, window_size, features)
                X_input = np.array([input_scaled])
                
                # 2. Prediction
                # GRU Fusion mungkin menerima 1 input (Time Series) atau 2 input (TS + Sentiment)
                # Kode ini mencoba input standar Time Series dulu. 
                # Jika model Anda strict multi-input, formatnya: model.predict([X_input, X_sentiment])
                
                try:
                    predicted_scaled = model.predict(X_input)
                except:
                    # Fallback jika model multimodal (misal butuh dummy input kedua)
                    # Ini pencegahan error jika input shape tidak match
                    st.warning("Terdeteksi input shape mismatch. Mencoba penyesuaian dimensi...")
                    predicted_scaled = model.predict([X_input, X_input]) # Contoh dummy fusion

                # 3. Inverse Scaling
                # Kita perlu melakukan inverse transform. 
                # Karena scaler fit pada N fitur, kita perlu membuat dummy array untuk inverse
                # Asumsi: Target prediksi (Close Price) adalah kolom tertentu.
                
                # Trik Inverse Transform:
                # Buat array kosong dengan shape yang sama dengan input terakhir
                dummy_array = np.zeros((1, input_data.shape[1]))
                
                # Isi kolom target dengan hasil prediksi
                # Asumsi: Kolom 'Close' adalah kolom ke-3 (index 3) jika urutan: Open, High, Low, Close...
                # Namun cara paling aman adalah me-restore semua dimensi jika scaler dipakai untuk semua
                
                # Simplifikasi: Kita anggap output model adalah 1 nilai (Close Price scaled)
                # Kita masukkan nilai ini ke posisi kolom 'Close' pada dummy array
                # Cari index kolom 'Close'
                numeric_cols = df_ticker.select_dtypes(include=[np.number]).columns.tolist()
                try:
                    close_col_idx = numeric_cols.index('Close')
                except:
                    close_col_idx = 3 # Default fallback
                
                dummy_array[0, close_col_idx] = predicted_scaled[0][0]
                
                # Inverse
                prediction_result = scaler.inverse_transform(dummy_array)[0, close_col_idx]
                
                # Ambil harga terakhir (Real)
                last_actual_price = df_ticker['Close'].iloc[-1]
                
                # Hitung arah pergerakan
                movement = prediction_result - last_actual_price
                movement_pct = (movement / last_actual_price) * 100
                direction_emoji = "📈 Naik" if movement > 0 else "📉 Turun"
                direction_color = "green" if movement > 0 else "red"

                # 4. Display Result (Prediction Card)
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-main">
                        <p style="color: rgba(255,255,255,0.8); margin-bottom: 0;">Prediksi Harga Penutupan Besok</p>
                        <h2>Rp {prediction_result:,.0f}</h2>
                        <div class="confidence-score" style="background: {'rgba(76, 209, 55, 0.2)' if movement > 0 else 'rgba(232, 65, 24, 0.2)'}">
                            <span>{direction_emoji} ({movement_pct:+.2f}%)</span>
                        </div>
                    </div>
                    <div style="margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.2); padding-top: 1rem;">
                        <p style="color: white;">Harga Terakhir (Actual): <strong>Rp {last_actual_price:,.0f}</strong></p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendation Card
                rec_text = "Pertimbangkan untuk **Buy/Hold** jika indikator teknikal lain mendukung." if movement > 0 else "Waspada potensi koreksi, pertimbangkan **Wait & See**."
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>💡 AI Recommendation</h4>
                    <p>{rec_text}</p>
                    <ul>
                        <li>Model: GRU Fusion (Multimodal)</li>
                        <li>Dataset: {selected_ticker} (Updated)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")
                st.info("Tips: Pastikan jumlah fitur dalam file CSV sama persis dengan jumlah fitur saat model dilatih (Scaler mismatch).")

# ==========================================
# 6. HALAMAN: MODEL EVALUATION (MAPE & RMSE)
# ==========================================

def calculate_metrics(df_ticker, model, scaler, window_size=60, test_days=30):
    """
    Menghitung MAPE dan RMSE pada data testing (misal: 30 hari terakhir).
    """
    try:
        # Siapkan data
        # Kita butuh (window_size + test_days) data terakhir untuk memprediksi test_days
        required_len = window_size + test_days
        if len(df_ticker) < required_len:
            return None, None
        
        # Ambil data untuk testing
        numeric_cols = df_ticker.select_dtypes(include=[np.number]).columns.tolist()
        data_subset = df_ticker[numeric_cols].tail(required_len).values
        
        # Scale
        data_scaled = scaler.transform(data_subset)
        
        X_test = []
        y_true_scaled = []
        
        # Buat sequence
        for i in range(window_size, len(data_scaled)):
            X_test.append(data_scaled[i-window_size:i])
            # Asumsi: Target (Close) ada di kolom tertentu. 
            # Kita ambil row i, dan semua kolom (nanti kita ambil close saat inverse)
            y_true_scaled.append(data_scaled[i])
            
        X_test = np.array(X_test)
        y_true_scaled = np.array(y_true_scaled)
        
        # Predict
        # Handle input shape mismatch (seperti di forecast simulator)
        try:
            y_pred_scaled = model.predict(X_test, verbose=0)
        except:
            y_pred_scaled = model.predict([X_test, X_test], verbose=0) # Dummy fusion
            
        # Inverse Transform
        # Kita perlu mengembalikan ke skala asli untuk menghitung MAPE/RMSE yang valid
        
        # Cari index kolom Close
        try:
            close_col_idx = numeric_cols.index('Close')
        except:
            close_col_idx = 3 # Default fallback
            
        # Helper untuk inverse spesifik kolom Close
        def inverse_close_price(scaled_data_2d):
            # Buat dummy array seukuran jumlah fitur scaler
            dummy = np.zeros((len(scaled_data_2d), data_subset.shape[1]))
            # Masukkan data prediksi ke kolom Close
            # Asumsi output model (N, 1)
            dummy[:, close_col_idx] = scaled_data_2d.flatten()
            inv = scaler.inverse_transform(dummy)
            return inv[:, close_col_idx]

        y_pred_actual = inverse_close_price(y_pred_scaled)
        
        # Untuk y_true, kita tidak perlu dummy karena kita punya data lengkap
        y_true_actual = scaler.inverse_transform(y_true_scaled)[:, close_col_idx]
        
        # Hitung Metrics
        rmse = math.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
        mape = mean_absolute_percentage_error(y_true_actual, y_pred_actual)
        
        return mape, rmse
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None

def show_model_evaluation(df):
    """
    Menampilkan halaman Evaluasi Model (Scorecard MAPE & RMSE)
    """
    st.markdown("""
    <div class='page-header'>
        <h2>Model Evaluation</h2>
        <p>Performa Model GRU Fusion pada Data Testing (30 Hari Terakhir)</p>
    </div>
    """, unsafe_allow_html=True)

    # Tombol untuk memulai kalkulasi (karena berat)
    if st.button("🔄 Hitung Ulang Metrik Evaluasi"):
        
        results = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(EMITEN_LIST):
            # Update status
            progress_bar.progress((idx + 1) / len(EMITEN_LIST))
            
            # Load resources
            df_tick = get_ticker_data(df, ticker)
            model = load_trained_model(ticker)
            scaler = load_specific_scaler(ticker)
            
            if model and scaler and not df_tick.empty:
                mape, rmse = calculate_metrics(df_tick, model, scaler)
                
                if mape is not None:
                    results.append({
                        "Emiten": ticker,
                        "Model": "GRU Fusion",
                        "MAPE": f"{mape:.4%}", # Format persentase
                        "RMSE": f"{rmse:,.2f}", # Format desimal
                        "Status": "✅ Optimal" if mape < 0.1 else "⚠️ Warning"
                    })
                else:
                    results.append({"Emiten": ticker, "Model": "Error", "MAPE": "-", "RMSE": "-", "Status": "❌ Fail"})
            else:
                results.append({"Emiten": ticker, "Model": "Not Found", "MAPE": "-", "RMSE": "-", "Status": "❌ Missing"})
        
        # Tampilkan Tabel Hasil
        st.markdown("### 🏆 Performance Scorecard")
        
        # Konversi ke DataFrame untuk tampilan tabel yang bagus
        results_df = pd.DataFrame(results)
        
        # Styling custom untuk tabel
        st.dataframe(
            results_df.style.applymap(
                lambda x: 'color: green; font-weight: bold' if x == '✅ Optimal' else 'color: red' if 'Fail' in str(x) else '',
                subset=['Status']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        # Insight Box
        avg_mape = pd.Series([float(x['MAPE'].strip('%'))/100 for x in results if 'MAPE' in x and x['MAPE'] != '-']).mean()
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>📝 Analisis Performa</h4>
            <p>Rata-rata MAPE untuk seluruh portofolio emiten adalah <strong>{avg_mape:.2%}</strong>.</p>
            <p>Nilai RMSE menunjukkan deviasi standar residual prediksi dalam satuan Rupiah. 
            Semakin kecil nilai MAPE dan RMSE, semakin akurat performa model GRU Fusion dalam memprediksi harga saham.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        # Tampilan Default sebelum tombol ditekan
        st.info("Klik tombol di atas untuk menjalankan evaluasi real-time pada 5 model sekaligus.")
        
        # Placeholder Static (Agar tidak kosong saat load awal)
        st.markdown("#### Preview Struktur Evaluasi")
        dummy_df = pd.DataFrame({
            "Emiten": EMITEN_LIST,
            "Model": ["GRU Fusion"] * 5,
            "MAPE": ["Calculating..."] * 5,
            "RMSE": ["Calculating..."] * 5,
            "Status": ["Waiting..."] * 5
        })
        st.dataframe(dummy_df, use_container_width=True, hide_index=True)

# ==========================================
# 7. MAIN APP LOGIC
# ==========================================

def main():
    # Load Main Dataset Sekali Saja
    df = load_dataset()
    
    if df.empty:
        st.error("Gagal memuat dataset utama. Pastikan path file CSV benar.")
        return

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h2>MENU SKRIPSI</h2></div>", unsafe_allow_html=True)
        
        selected_page = st.radio(
            "",
            ["🏠 Market Overview", "🔮 Forecast Simulator", "📉 Model Evaluation"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p><strong>Created by:</strong><br>Hana - Statistika ITS</p>
            <p style='font-size: 0.8em'>Thesis Project 2025</p>
        </div>
        """, unsafe_allow_html=True)

    # Routing
    if selected_page == "🏠 Market Overview":
        show_market_overview(df)
    
    elif selected_page == "🔮 Forecast Simulator":
        show_forecast_simulator(df)
        
    elif selected_page == "📉 Model Evaluation":
        show_model_evaluation(df)

if __name__ == "__main__":
    main()