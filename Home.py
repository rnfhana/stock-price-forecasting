import streamlit as st
import pandas as pd
import os

# Import Modules dari utils
try:
    from utils.data_loader import load_dataset, load_metrics_data, get_ticker_data
    from utils.plots import plot_interactive_chart, plot_horizon_comparison
    from utils.model_handler import load_resources, run_multi_step_forecast
except ImportError:
    st.error("Gagal mengimport modul 'utils'. Pastikan folder 'utils' sudah di-push ke GitHub!")
    st.stop()

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS MEWAH
# ==========================================
st.set_page_config(
    page_title="Thesis Dashboard - GRU Fusion", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_css():
    """Load custom CSS styling (Versi Full Profesional)"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main { font-family: 'Inter', sans-serif; background-color: #f8f9fa; }
    
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
        color: white;
    }
    .main-header h1 { color: white; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
    .main-header p { color: rgba(255, 255, 255, 0.9); }
    
    /* Sidebar Styling */
    .sidebar-header {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sidebar-header h2 { color: white; margin: 0; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
    
    /* Page Headers */
    .page-header {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .page-header h2 { color: #2c3e50; font-weight: 600; margin-bottom: 0.5rem; }
    .page-header p { color: #5a6c7d; margin: 0; }
    
    /* Stats Card (Kotak Angka) */
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
        transition: transform 0.3s ease;
    }
    .stats-card:hover { transform: translateY(-5px); }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2c3e50; }
    .metric-label { font-size: 0.9rem; color: #666; font-weight: 500; }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Insight/Info Box */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-left: 4px solid #ff9f43;
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
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ==========================================
# 2. HALAMAN MARKET OVERVIEW
# ==========================================
def show_market_overview(df):
    st.sidebar.markdown("### ⚙️ Konfigurasi Market")
    selected_ticker = st.sidebar.selectbox("Pilih Emiten:", ["AKRA", "BBRI", "BMRI", "PGAS", "UNVR"])
    
    df_ticker = get_ticker_data(df, selected_ticker)
    if df_ticker.empty:
        st.warning("Data tidak ditemukan.")
        return

    st.markdown(f"""
    <div class='page-header'>
        <h2>Market Overview: {selected_ticker}</h2>
        <p>Analisis Teknikal & Indikator ATR</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Cards
    last = df_ticker.iloc[-1]
    prev = df_ticker.iloc[-2]
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    atr_val = last['ATR'] if 'ATR' in last else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='stats-card'><div class='metric-value'>Rp {last['Close']:,.0f}</div><div class='metric-label'>Last Close</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stats-card'><div class='metric-value' style='color: {'#4cd137' if change>=0 else '#e84118'}'>{change:,.0f} ({pct:.2f}%)</div><div class='metric-label'>Daily Change</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stats-card'><div class='metric-value'>{last['Volume']/1e6:.1f} M</div><div class='metric-label'>Volume</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='stats-card'><div class='metric-value'>{atr_val:.2f}</div><div class='metric-label'>ATR (Volatility)</div></div>", unsafe_allow_html=True)

    # Chart
    st.markdown("### 📊 Interactive Chart")
    st.plotly_chart(plot_interactive_chart(df_ticker, selected_ticker), use_container_width=True)

# ==========================================
# 3. HALAMAN FORECAST SIMULATOR
# ==========================================
def show_forecast_simulator(df):
    st.sidebar.markdown("### ⚙️ Konfigurasi Prediksi")
    ticker = st.sidebar.selectbox("Pilih Emiten:", ["AKRA", "BBRI", "BMRI", "PGAS", "UNVR"], key="sim")
    
    st.markdown(f"""
    <div class='page-header'>
        <h2>Forecast Simulator: {ticker}</h2>
        <p>Simulasi Prediksi 3 Hari ke Depan (Live Inference)</p>
    </div>
    """, unsafe_allow_html=True)

    df_ticker = get_ticker_data(df, ticker)
    
    # Load Model (Hanya saat tombol ditekan nanti, atau cache)
    model, scaler = load_resources(ticker)
    
    if not model:
        st.error(f"Model/Scaler untuk {ticker} tidak ditemukan. Pastikan file .h5 dan .pkl ada di folder models/ dan sudah di-push.")
        return

    st.markdown("""
    <div class="info-box">
        <h4>ℹ️ Skenario Prediksi</h4>
        <p>Model menggunakan <strong>GRU Fusion</strong>. Prediksi dilakukan secara recursive untuk mendapatkan Horizon H+1, H+2, dan H+3.</p>
    </div>
    <br>
    """, unsafe_allow_html=True)

    if st.button("🚀 Jalankan Prediksi (3 Days Horizon)"):
        with st.spinner("Sedang memproses GRU Fusion Model..."):
            try:
                preds = run_multi_step_forecast(df_ticker, model, scaler)
                
                # Tampilkan Hasil
                st.markdown("### 📅 Hasil Prediksi")
                c1, c2, c3 = st.columns(3)
                labels = ["Besok (H+1)", "Lusa (H+2)", "H+3"]
                
                last_price = df_ticker['Close'].iloc[-1]
                
                for i, (col, price) in enumerate(zip([c1,c2,c3], preds)):
                    prev = last_price if i==0 else preds[i-1]
                    chg = price - prev
                    pct = (chg/prev)*100
                    emoji = "📈" if chg >=0 else "📉"
                    bg_color = "rgba(76, 209, 55, 0.1)" if chg >= 0 else "rgba(232, 65, 24, 0.1)"
                    
                    col.markdown(f"""
                    <div style="background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); text-align: center; border-top: 5px solid #667eea;">
                        <h4 style="margin:0; color:#666; font-size: 1rem;">{labels[i]}</h4>
                        <h2 style="margin: 10px 0; font-size: 1.8rem; color: #2c3e50;">Rp {price:,.0f}</h2>
                        <div style="background: {bg_color}; padding: 5px 15px; border-radius: 20px; display: inline-block;">
                            <span style="font-weight: bold; font-size: 0.9rem; color: #333;">{emoji} {chg:+.0f} ({pct:+.2f}%)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")

# ==========================================
# 4. HALAMAN MODEL EVALUATION (VERSI RINGAN)
# ==========================================
def show_model_evaluation():
    st.markdown("""
    <div class='page-header'>
        <h2>Model Evaluation</h2>
        <p>Laporan Performa Model (Pre-calculated)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 1. Load Data dari Excel
    df_metrics = load_metrics_data()
    
    if df_metrics.empty:
        st.error("File metric_evaluate.xlsx belum ditemukan di folder data/. Pastikan sudah dibuat dan di-push.")
        return
        
    # 2. Tampilkan Tabel
    st.markdown("### 🏆 Scorecard (MAPE & RMSE)")
    
    st.dataframe(
        df_metrics.style.format("{:.3f}", subset=df_metrics.columns[1:])
                  .highlight_min(subset=['MAPE (%) H+1', 'RMSE H+1'], color='#d1e7dd', axis=0),
        use_container_width=True
    )
    
    # 3. Insight Visual
    st.markdown("### 📉 Analisis Degradasi Horizon")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.plotly_chart(plot_horizon_comparison(df_metrics), use_container_width=True)
        
    with c2:
        st.markdown("""
        <div class='info-box'>
            <h4>💡 Key Insights</h4>
            <p><strong>Stability:</strong> Perhatikan kemiringan garis di grafik. Garis yang landai menunjukkan model stabil terhadap waktu.</p>
            <p><strong>Performance:</strong> Emiten dengan MAPE terendah di H+3 adalah kandidat terbaik untuk investasi jangka panjang.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 5. MAIN CONTROLLER
# ==========================================
def main():
    df = load_dataset()
    
    with st.sidebar:
        st.markdown("<div style='text-align:center; padding:1rem;'><h2>MENU SKRIPSI</h2></div>", unsafe_allow_html=True)
        menu = st.radio("", ["🏠 Market Overview", "🔮 Forecast Simulator", "📊 Model Evaluation"])
        
        st.markdown("---")
        st.markdown("<div class='sidebar-header'><h5>Status: Online ✅</h5><p style='font-size:0.8em; color:white;'>GRU Fusion Model</p></div>", unsafe_allow_html=True)

    if menu == "🏠 Market Overview":
        show_market_overview(df)
    elif menu == "🔮 Forecast Simulator":
        show_forecast_simulator(df)
    elif menu == "📊 Model Evaluation":
        show_model_evaluation()

if __name__ == "__main__":
    main()