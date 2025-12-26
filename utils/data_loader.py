import pandas as pd
import os
import streamlit as st

def get_file_paths():
    """Mendapatkan path absolut ke folder data"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    return {
        "main": os.path.join(data_dir, "df_fusi_multimodal_final_hana.csv"),
        "metrics": os.path.join(data_dir, "metric_evaluate.xlsx"),
        "shap": os.path.join(data_dir, "shap_values_summary.csv")
    }

@st.cache_data
def load_dataset():
    """Load dataset utama (Harga saham)"""
    paths = get_file_paths()
    try:
        df = pd.read_csv(paths["main"])
        
        # Rename sesuai standar
        rename_map = {
            'date': 'Date', 'relevant_issuer': 'Stock',
            'Yt': 'Close', 'X1': 'Open', 'X2': 'High', 
            'X3': 'Low', 'X4': 'Volume', 'X7': 'ATR'
        }
        df.rename(columns=rename_map, inplace=True)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
        return df
    except Exception as e:
        st.error(f"Error loading main data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_metrics_data():
    """Load data evaluasi (Metric Excel)"""
    paths = get_file_paths()
    try:
        df = pd.read_excel(paths["metrics"])
        return df
    except Exception as e:
        st.error(f"Error loading metrics excel: {e}")
        return pd.DataFrame()

def get_ticker_data(df, ticker):
    """Filter data per emiten"""
    if df.empty: return df
    
    col_name = 'Stock' if 'Stock' in df.columns else 'relevant_issuer'
    df_ticker = df[df[col_name] == ticker].copy()
    
    if 'Date' in df_ticker.columns:
        df_ticker = df_ticker.sort_values('Date')
    
    return df_ticker