import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import base64
import os
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

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
    """Load custom CSS styling"""
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

load_css()

# ==========================================
# 2. KONFIGURASI PATHS & EMITEN
# ==========================================

# Path Relatif agar aman di Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_PATH = os.path.join(BASE_DIR, "models")
BASE_DATA_PATH = os.path.join(BASE_DIR, "data")

EMITEN_LIST = ["AKRA", "BBRI", "BMRI", "PGAS", "UNVR"]

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

# ==========================================
# 3. DATA LOADING & PROCESSING FUNCTIONS
# ==========================================

@st.cache_data
def load_dataset():
    """
    Load main dataset dan rename kolom sesuai data diagnosa.
    """
    try:
        df = pd.read_csv(FILES["data"]["main"])
        
        # Mapping Rename (Sesuai diagnosa user)
        rename_map = {
            'date': 'Date',
            'relevant_issuer': 'Stock',
            'Yt': 'Close',      # Close Price
            'X1': 'Open',       # Opening Price
            'X2': 'High',       # Highest Price
            'X3': 'Low',        # Lowest Price
            'X4': 'Volume',     # Volume
            'X7': 'ATR'         # ATR (Sudah ada)
        }
        
        df.rename(columns=rename_map, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        return df
    except Exception as e:
        st.error(f"Error loading main dataset: {e}")
        return pd.DataFrame()

@st.cache_data
def load_shap_data():
    try:
        df_shap = pd.read_csv(FILES["data"]["shap"])
        return df_shap
    except:
        return pd.DataFrame()

def get_ticker_data(df, ticker):
    if 'Stock' in df.columns:
        df_ticker = df[df['Stock'] == ticker].copy()
    else:
        df_ticker = df[df['relevant_issuer'] == ticker].copy() if 'relevant_issuer' in df.columns else df.copy()
    
    if 'Date' in df_ticker.columns:
        df_ticker = df_ticker.sort_values('Date')
    
    return df_ticker

def prepare_inputs(df_ticker, window_size=60):
    """
    Menyiapkan data input untuk prediksi.
    FIX: Membuang 'Yt+1' dan 'Yt-1' agar jumlah fitur pas 12.
    """
    # List kolom yang HARUS DIBUANG
    drop_cols = ['Date', 'Stock', 'relevant_issuer', 'Yt+1', 'Yt-1'] 
    
    # Buang kolom
    features_df = df_ticker.drop(columns=drop_cols, errors='ignore')
    
    # Ambil numerik sisa (Harus 12)
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Ambil window terakhir
    last_window = features_df[numeric_cols].tail(window_size).values
    
    return last_window

# ==========================================
# 4. CHART FUNCTIONS
# ==========================================

def plot_interactive_chart(df, ticker):
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker} Price Action', 'Volume', 'Average True Range (ATR)')
    )

    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['Date'], y=df['Volume'],
        name='Volume', marker_color='rgba(100, 149, 237, 0.5)'
    ), row=2, col=1)

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

def show_market_overview(df):
    st.sidebar.markdown("### ⚙️ Konfigurasi Market")
    selected_ticker = st.sidebar.selectbox("Pilih Emiten:", EMITEN_LIST, index=0)
    
    df_ticker = get_ticker_data(df, selected_ticker)
    
    if df_ticker.empty:
        st.warning("Data tidak ditemukan.")
        return

    st.markdown(f"""
    <div class='page-header'>
        <h2>Market Overview: {selected_ticker}</h2>
        <p>Analisis Pergerakan Harga & Indikator Volatilitas (ATR)</p>
    </div>
    """, unsafe_allow_html=True)

    last_data = df_ticker.iloc[-1]
    prev_data = df_ticker.iloc[-2]
    
    current_price = last_data['Close']
    price_change = current_price - prev_data['Close']
    pct_change = (price_change / prev_data['Close']) * 100
    
    current_atr = last_data['ATR'] if 'ATR' in last_data and not pd.isna(last_data['ATR']) else 0
    current_vol = last_data['Volume']

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

    st.markdown("### 📊 Interactive Chart")
    fig = plot_interactive_chart(df_ticker, selected_ticker)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat Data Historis Lengkap"):
        st.dataframe(df_ticker.sort_values('Date', ascending=False))

# ==========================================
# 5. FORECAST FUNCTIONS
# ==========================================

@st.cache_resource
def load_trained_model(ticker):
    model_path = FILES["models"].get(ticker)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        return None

@st.cache_resource
def load_specific_scaler(ticker):
    scaler_path = FILES["scalers"].get(ticker)
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        return None

def show_forecast_simulator(df):
    st.sidebar.markdown("### ⚙️ Konfigurasi Prediksi")
    selected_ticker = st.sidebar.selectbox("Pilih Emiten:", EMITEN_LIST, key="forecast_ticker")
    
    st.markdown(f"""
    <div class='page-header'>
        <h2>Forecast Simulator: {selected_ticker}</h2>
        <p>Prediksi Harga Saham 3 Hari Ke Depan (Recursive Forecasting)</p>
    </div>
    """, unsafe_allow_html=True)

    df_ticker = get_ticker_data(df, selected_ticker)
    model = load_trained_model(selected_ticker)
    scaler = load_specific_scaler(selected_ticker)

    if df_ticker.empty or model is None or scaler is None:
        st.error("Data, Model, atau Scaler tidak ditemukan.")
        return

    st.markdown("""
    <div class="info-box">
        <h3>ℹ️ Skenario Prediksi 3 Hari</h3>
        <p>Model memprediksi H+1. Untuk mendapatkan H+2 dan H+3, sistem menggunakan hasil prediksi sebelumnya sebagai input baru (metode <em>Rolling Forecast</em>).</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("🚀 Jalankan Prediksi (3 Days Horizon)", use_container_width=True)

    if predict_btn:
        with st.spinner('Menghitung prediksi berjenjang...'):
            try:
                WINDOW_SIZE = 60
                
                # --- SIAPKAN DATA AWAL ---
                # 1. Bersihkan data
                drop_cols = ['Date', 'Stock', 'relevant_issuer', 'Yt+1', 'Yt-1']
                features_df = df_ticker.drop(columns=drop_cols, errors='ignore')
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Cek index kolom Close untuk update nanti
                try:
                    close_col_idx = numeric_cols.index('Close')
                except:
                    close_col_idx = 0

                # 2. Ambil window terakhir (Data Asli belum di-scale)
                current_window = features_df[numeric_cols].tail(WINDOW_SIZE).values # Shape (60, 12)
                
                predictions = []
                last_actual = df_ticker['Close'].iloc[-1]

                # --- LOOP PREDIKSI 3 HARI ---
                for day in range(1, 4):
                    # A. Scale Data Window saat ini
                    # input_scaled shape: (60, 12)
                    input_scaled = scaler.transform(current_window)
                    
                    # B. Reshape ke (1, 60, 12)
                    X_input = np.array([input_scaled])
                    
                    # C. Split Input (Quant vs Qual)
                    X_quant = X_input[:, :, :8]
                    X_qual  = X_input[:, :, 8:]
                    
                    # D. Predict
                    try:
                        pred_scaled = model.predict([X_quant, X_qual], verbose=0)
                    except:
                        # Fallback
                        pred_scaled = model.predict(X_input, verbose=0)
                        
                    # E. Inverse Transform (Dapatkan Rupiah)
                    dummy = np.zeros((1, 12))
                    dummy[0, close_col_idx] = pred_scaled[0][0]
                    pred_rupiah = scaler.inverse_transform(dummy)[0, close_col_idx]
                    
                    # Simpan hasil
                    predictions.append(pred_rupiah)
                    
                    # F. UPDATE WINDOW UNTUK HARI BERIKUTNYA (RECURSIVE)
                    # Kita harus membuang baris paling lama (hari ke-1), dan memasukkan baris baru (hasil prediksi)
                    
                    new_row = current_window[-1].copy() # Copy baris terakhir
                    
                    # Update nilai Close dengan hasil prediksi
                    new_row[close_col_idx] = pred_rupiah
                    
                    # Asumsi: Open besok = Close hari ini (prediksi)
                    try:
                        open_idx = numeric_cols.index('Open')
                        new_row[open_idx] = pred_rupiah
                    except:
                        pass
                    
                    # Fitur lain (Sentiment/Volume) diasumsikan sama (Forward Fill) karena tidak bisa diprediksi
                    
                    # Gabungkan: Data lama (indeks 1 sampai akhir) + Baris Baru
                    current_window = np.vstack([current_window[1:], new_row])

                # --- TAMPILKAN HASIL (3 KARTU) ---
                st.markdown("### 📅 Hasil Forecast")
                cols = st.columns(3)
                
                dates = ["Besok (H+1)", "Lusa (H+2)", "H+3"]
                
                for i, (col, pred) in enumerate(zip(cols, predictions)):
                    # Hitung kenaikan dari harga sebelumnya (Chain)
                    prev_price = last_actual if i == 0 else predictions[i-1]
                    change = pred - prev_price
                    pct_change = (change / prev_price) * 100
                    emoji = "📈" if change > 0 else "📉"
                    bg_color = "rgba(76, 209, 55, 0.2)" if change > 0 else "rgba(232, 65, 24, 0.2)"
                    
                    with col:
                        st.markdown(f"""
                        <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); text-align: center; border-top: 5px solid #667eea;">
                            <h4 style="margin:0; color:#666;">{dates[i]}</h4>
                            <h2 style="margin: 10px 0; font-size: 1.8rem; color: #2c3e50;">Rp {pred:,.0f}</h2>
                            <div style="background: {bg_color}; padding: 5px 10px; border-radius: 15px; display: inline-block;">
                                <span style="font-weight: bold; font-size: 0.9rem;">{emoji} {pct_change:+.2f}%</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.success(f"Basis Data Terakhir: Rp {last_actual:,.0f}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat forecasting: {str(e)}")

# ==========================================
# 6. EVALUATION FUNCTIONS
# ==========================================

def calculate_metrics(df_ticker, model, scaler, window_size=60, test_days=30):
    """
    Hitung MAPE & RMSE dengan FIX SPLIT INPUT (Quant 8 + Qual 4)
    """
    try:
        required_len = window_size + test_days
        # Cek ketersediaan data
        if len(df_ticker) < required_len:
            # st.warning(f"Data kurang: {len(df_ticker)} < {required_len}") # Debug
            return None, None
        
        # 1. Bersihkan Data
        drop_cols = ['Date', 'Stock', 'relevant_issuer', 'Yt+1', 'Yt-1']
        features_df = df_ticker.drop(columns=drop_cols, errors='ignore')
        
        # Pastikan hanya angka
        valid_feature_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Ambil data testing
        data_subset = features_df[valid_feature_cols].tail(required_len).values
        
        # 2. Scaling
        data_scaled = scaler.transform(data_subset)
        
        X_test = []
        y_true_scaled = []
        
        # 3. Sliding Window
        for i in range(window_size, len(data_scaled)):
            X_test.append(data_scaled[i-window_size:i])
            y_true_scaled.append(data_scaled[i])
            
        X_test = np.array(X_test).astype('float32') # Pastikan float32
        y_true_scaled = np.array(y_true_scaled)
        
        # 4. SPLIT INPUT (CRUCIAL FIX)
        X_test_quant = X_test[:, :, :8] # 8 Fitur Awal
        X_test_qual  = X_test[:, :, 8:] # 4 Fitur Akhir
        
        # 5. Predict
        try:
            # Coba Dual Input
            y_pred_scaled = model.predict([X_test_quant, X_test_qual], verbose=0)
        except:
            # Fallback Single Input
            y_pred_scaled = model.predict(X_test, verbose=0)
            
        # 6. Inverse Transform
        try:
            close_col_idx = valid_feature_cols.index('Close')
        except:
            close_col_idx = 0 

        def inverse_close_price(scaled_data_2d):
            # Dummy harus selebar total fitur (12)
            dummy = np.zeros((len(scaled_data_2d), data_subset.shape[1]))
            dummy[:, close_col_idx] = scaled_data_2d.flatten()
            inv = scaler.inverse_transform(dummy)
            return inv[:, close_col_idx]

        y_pred_actual = inverse_close_price(y_pred_scaled)
        
        # Inverse Ground Truth
        y_true_inv_full = scaler.inverse_transform(y_true_scaled)
        y_true_actual = y_true_inv_full[:, close_col_idx]
        
        # 7. Hitung Error
        rmse = math.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
        mape = mean_absolute_percentage_error(y_true_actual, y_pred_actual)
        
        return mape, rmse
        
    except Exception as e:
        # Tampilkan error spesifik di console/terminal deployment untuk debug
        print(f"DEBUG ERROR {df_ticker.iloc[0]['Stock'] if not df_ticker.empty else 'Unknown'}: {str(e)}")
        return None, None

def show_model_evaluation(df):
    st.markdown("""
    <div class='page-header'>
        <h2>Model Evaluation</h2>
        <p>Performa Model GRU Fusion pada Data Testing (30 Hari Terakhir)</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Hitung Ulang Metrik Evaluasi"):
        results = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(EMITEN_LIST):
            progress_bar.progress((idx + 1) / len(EMITEN_LIST))
            df_tick = get_ticker_data(df, ticker)
            model = load_trained_model(ticker)
            scaler = load_specific_scaler(ticker)
            
            if model and scaler and not df_tick.empty:
                mape, rmse = calculate_metrics(df_tick, model, scaler)
                if mape is not None:
                    results.append({
                        "Emiten": ticker, "Model": "GRU Fusion",
                        "MAPE": f"{mape:.4%}", "RMSE": f"{rmse:,.2f}",
                        "Status": "✅ Optimal" if mape < 0.1 else "⚠️ Warning"
                    })
                else:
                    results.append({"Emiten": ticker, "Status": "❌ Fail"})
            else:
                results.append({"Emiten": ticker, "Status": "❌ Missing"})
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    else:
        st.info("Klik tombol di atas untuk menjalankan evaluasi.")

# ==========================================
# 7. MAIN APP
# ==========================================

def main():
    df = load_dataset()
    
    with st.sidebar:
        st.markdown("<div class='sidebar-header'><h2>MENU SKRIPSI</h2></div>", unsafe_allow_html=True)
        selected_page = st.radio("", ["🏠 Market Overview", "🔮 Forecast Simulator", "📉 Model Evaluation"], index=0)
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #666;'><p>Thesis Project 2025</p></div>", unsafe_allow_html=True)

    if selected_page == "🏠 Market Overview":
        show_market_overview(df)
    elif selected_page == "🔮 Forecast Simulator":
        show_forecast_simulator(df)
    elif selected_page == "📉 Model Evaluation":
        show_model_evaluation(df)

if __name__ == "__main__":
    main()