from flask import Flask, render_template, request
import glob
import numpy as np
import pandas as pd
import joblib
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model dan scaler
model = load_model('model/lstm_gold.h5')
scaler = joblib.load('model/scaler_gold.pkl')

# Cari file .csv otomatis di folder 'data/' atau folder utama
csv_files = glob.glob('data/*.csv') + glob.glob('*.csv')
if not csv_files:
    raise FileNotFoundError("File CSV tidak ditemukan di folder 'data/' maupun folder utama.")

df = pd.read_csv(csv_files[0])
date_col = next((c for c in df.columns if c.lower() in ['date', 'tanggal', 'datetime', 'time', 'timestamp']), df.columns[0])
target_col = next((c for c in df.columns if c.lower() in ['close', 'price', 'adj close', 'last', 'harga', 'close/last']), df.columns[1])

df[date_col] = pd.to_datetime(df[date_col], format='mixed')
df.sort_values(date_col, inplace=True)
df.set_index(date_col, inplace=True)

if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].astype(str).str.replace(',', '').str.replace('$', '').str.strip().astype(float)

df = df[[target_col]].dropna()
df.rename(columns={target_col: 'Close'}, inplace=True)

TIMESTEPS = 60

def forecast_gold(days_ahead=7):
    scaled_data = scaler.transform(df.values)
    current_seq = scaled_data[-TIMESTEPS:].reshape(1, TIMESTEPS, 1)
    
    future_preds_scaled = []
    for _ in range(days_ahead):
        pred = model.predict(current_seq, verbose=0)[0, 0]
        future_preds_scaled.append(pred)
        current_seq = np.append(current_seq[:, 1:, :], [[[pred]]], axis=1)
        
    future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1)).flatten()
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
    
    return future_dates, future_preds

def create_gold_chart(future_dates, future_preds):
    plt.figure(figsize=(10, 4.5))
    recent_df = df.tail(90)
    plt.plot(recent_df.index, recent_df['Close'], label='Harga Penutupan Aktual (90 Hari Terakhir)', color='#ca8a04', linewidth=2)
    
    plot_dates = [recent_df.index[-1]] + list(future_dates)
    plot_values = [recent_df['Close'].iloc[-1]] + list(future_preds)
    plt.plot(plot_dates, plot_values, label='Prediksi Harga Emas (LSTM)', color='#0f172a', linestyle='--', marker='o', linewidth=2)
    
    plt.title('Peramalan Harga Penutupan Emas Menggunakan LSTM (Dataset Kaggle)')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga Penutupan (USD)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    days = int(request.form.get('days', 7))
    future_dates, future_preds = forecast_gold(days_ahead=days)
    plot_url = create_gold_chart(future_dates, future_preds)
    
    last_close = f"${df['Close'].iloc[-1]:,.2f}"
    last_date_str = df.index[-1].strftime('%Y-%m-%d')
    
    forecast_table = [
        {'date': d.strftime('%Y-%m-%d'), 'value': f"${v:,.2f}"}
        for d, v in zip(future_dates, future_preds)
    ]
    
    return render_template(
        'index.html',
        plot_url=plot_url,
        forecast_table=forecast_table,
        days=days,
        last_close=last_close,
        last_date=last_date_str
    )

if __name__ == '__main__':
    app.run(debug=True)