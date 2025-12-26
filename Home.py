import streamlit as st
import pandas as pd
import os

# Import Modules dari utils
from utils.data_loader import load_dataset, load_metrics_data, get_ticker_data
from utils.plots import plot_interactive_chart, plot_horizon_comparison
from utils.model_handler import load_resources, run_multi_step_forecast

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
st.set_page_config(page_title="Thesis Dashboard - GRU Fusion", page_icon="📈", layout="wide")

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main { font-family: 'Inter', sans-serif; }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 2rem; }
    .page-header { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; text-align: center; }
    .page-header h2 { color: #2c3e50; font-weight: 600; margin-bottom: 0.5rem; }
    .stats-card { background: white; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 1rem; border-left: 5px solid #667eea; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #2c3e50; }
    .metric-label { font-size: 0.9rem; color: #666; }
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

    st.markdown(f"<div class='page-header'><h2>Market Overview: {selected_ticker}</h2><p>Analisis Teknikal & Indikator ATR</p></div>", unsafe_allow_html=True)
    
    # KPI Cards
    last = df_ticker.iloc[-1]
    prev = df_ticker.iloc[-2]
    change = last['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='stats-card'><div class='metric-value'>Rp {last['Close']:,.0f}</div><div class='metric-label'>Last Close</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='stats-card'><div class='metric-value' style='color: {'green' if change>=0 else 'red'}'>{change:,.0f} ({pct:.2f}%)</div><div class='metric-label'>Daily Change</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='stats-card'><div class='metric-value'>{last['Volume']/1e6:.1f} M</div><div class='metric-label'>Volume</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='stats-card'><div class='metric-value'>{last['ATR']:.2f}</div><div class='metric-label'>ATR (Volatility)</div></div>", unsafe_allow_html=True)

    # Chart
    st.plotly_chart(plot_interactive_chart(df_ticker, selected_ticker), use_container_width=True)

# ==========================================
# 3. HALAMAN FORECAST SIMULATOR
# ==========================================
def show_forecast_simulator(df):
    st.sidebar.markdown("### ⚙️ Konfigurasi Prediksi")
    ticker = st.sidebar.selectbox("Pilih Emiten:", ["AKRA", "BBRI", "BMRI", "PGAS", "UNVR"], key="sim")
    
    st.markdown(f"<div class='page-header'><h2>Forecast Simulator: {ticker}</h2><p>Simulasi Prediksi 3 Hari ke Depan (Live Inference)</p></div>", unsafe_allow_html=True)

    df_ticker = get_ticker_data(df, ticker)
    
    # Load Model (Hanya saat tombol ditekan nanti, atau cache)
    model, scaler = load_resources(ticker)
    
    if not model:
        st.error("Model/Scaler tidak ditemukan. Pastikan file .h5 dan .pkl ada di folder models/")
        return

    if st.button("🚀 Jalankan Prediksi (3 Days Horizon)", use_container_width=True):
        with st.spinner("Menjalankan GRU Fusion Model..."):
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
                    color = "green" if chg >=0 else "red"
                    
                    col.markdown(f"""
                    <div class='stats-card' style='text-align:center;'>
                        <h4>{labels[i]}</h4>
                        <h2 style='color:#2c3e50;'>Rp {price:,.0f}</h2>
                        <div style='color:{color}; font-weight:bold;'>{chg:+.0f} ({pct:+.2f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error saat prediksi: {e}")

# ==========================================
# 4. HALAMAN MODEL EVALUATION (VERSI RINGAN)
# ==========================================
def show_model_evaluation():
    st.markdown("<div class='page-header'><h2>Model Evaluation</h2><p>Laporan Performa Model (Pre-calculated)</p></div>", unsafe_allow_html=True)
    
    # 1. Load Data dari Excel (Sangat Cepat & Stabil)
    df_metrics = load_metrics_data()
    
    if df_metrics.empty:
        st.error("File metrics_evaluate.xlsx tidak ditemukan di folder data/")
        return
        
    # 2. Tampilkan Tabel
    st.markdown("### 🏆 Scorecard (MAPE & RMSE)")
    
    # Highlight logic
    def highlight_best(s):
        is_min = s == s.min()
        return ['background-color: #d4edda' if v else '' for v in is_min]

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
        <div class='stats-card'>
            <h4>💡 Key Insights</h4>
            <p>1. <strong>Stability:</strong> PGAS memiliki MAPE terendah di semua horizon, menunjukkan model paling stabil.</p>
            <p>2. <strong>Degradation:</strong> UNVR mengalami penurunan akurasi paling tajam dari H+1 ke H+3.</p>
            <p>3. <strong>Best Model:</strong> Secara rata-rata, model PGAS dan BBRI memberikan performa terbaik.</p>
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
        st.info("System Status: Online ✅\nMode: GRU Fusion")

    if menu == "🏠 Market Overview":
        show_market_overview(df)
    elif menu == "🔮 Forecast Simulator":
        show_forecast_simulator(df)
    elif menu == "📊 Model Evaluation":
        show_model_evaluation() # Tidak perlu parameter df, dia baca excel sendiri

if __name__ == "__main__":
    main()