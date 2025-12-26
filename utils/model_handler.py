import os
import numpy as np
import tensorflow as tf
import joblib
import streamlit as st

def get_model_paths(ticker):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    return {
        "model": os.path.join(models_dir, f"model_fusion_{ticker}.h5"),
        "scaler": os.path.join(models_dir, f"scaler_{ticker}.pkl")
    }

@st.cache_resource
def load_resources(ticker):
    paths = get_model_paths(ticker)
    try:
        model = tf.keras.models.load_model(paths["model"], compile=False)
        scaler = joblib.load(paths["scaler"])
        return model, scaler
    except Exception as e:
        return None, None

def prepare_inputs(df_ticker, window_size=60):
    drop_cols = ['Date', 'Stock', 'relevant_issuer', 'Yt+1', 'Yt-1']
    features_df = df_ticker.drop(columns=drop_cols, errors='ignore')
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    last_window = features_df[numeric_cols].tail(window_size).values
    return last_window, numeric_cols

def run_multi_step_forecast(df_ticker, model, scaler, window_size=60, steps=3):
    """Logika prediksi 3 hari ke depan"""
    # 1. Prepare Data
    current_window, numeric_cols = prepare_inputs(df_ticker, window_size)
    if len(current_window) < window_size:
        raise ValueError("Data historis tidak cukup.")

    try:
        close_col_idx = numeric_cols.index('Close')
    except:
        close_col_idx = 0

    predictions = []
    
    # 2. Loop Prediction
    for _ in range(steps):
        # Scale
        input_scaled = scaler.transform(current_window)
        X_input = np.array([input_scaled]) # Shape (1, 60, 12)
        
        # Split Quant/Qual
        X_quant = X_input[:, :, :8]
        X_qual  = X_input[:, :, 8:]
        
        # Predict
        try:
            pred_scaled = model.predict([X_quant, X_qual], verbose=0)
        except:
            pred_scaled = model.predict(X_input, verbose=0)
            
        # Inverse
        dummy = np.zeros((1, 12))
        dummy[0, close_col_idx] = pred_scaled[0][0]
        pred_rupiah = scaler.inverse_transform(dummy)[0, close_col_idx]
        predictions.append(pred_rupiah)
        
        # Update Window (Recursive)
        new_row = current_window[-1].copy()
        new_row[close_col_idx] = pred_rupiah
        
        try:
            new_row[numeric_cols.index('Open')] = pred_rupiah
        except: pass
        
        current_window = np.vstack([current_window[1:], new_row])
        
    return predictions