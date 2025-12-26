import pandas as pd
import os

# Setup Path (Otomatis deteksi lokasi file ini)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "df_fusi_multimodal_final_hana.csv")

print("--- DIAGNOSA DATA FRAME ---")
try:
    # 1. Baca CSV Murni
    df = pd.read_csv(csv_path)
    print("\n1. NAMA KOLOM ASLI DI CSV:")
    print(df.columns.tolist())
    
    # 2. Simulasi Rename (Sesuai kode Home.py kita)
    rename_map = {
        'date': 'Date',
        'relevant_issuer': 'Stock',
        'Yt': 'Close',     
        'X1': 'Open',       
        'X2': 'High',       
        'X3': 'Low',        
        'X4': 'Volume',
        'X7': 'ATR'         
    }
    df.rename(columns=rename_map, inplace=True)
    
    print("\n2. NAMA KOLOM SETELAH RENAME (DI PYTHON):")
    cols = df.columns.tolist()
    print(cols)
    
    print("\n3. TIPE DATA SETIAP KOLOM:")
    print(df.dtypes)

    # 3. Cek Kandidat yang harus dibuang
    print("\n4. PENGECEKAN TARGET DROP:")
    for target in ['Yt+1', 'Yt-1', 'Yt+1 ', ' Yt+1']: # Cek variasi spasi
        if target in df.columns:
            print(f"   [DITEMUKAN] Kolom '{target}' ada di DataFrame.")
        else:
            print(f"   [MISSING] Kolom '{target}' TIDAK ditemukan.")

except Exception as e:
    print(f"Error membaca file: {e}")