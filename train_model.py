import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

os.makedirs('data', exist_ok=True)
os.makedirs('model', exist_ok=True)

# 1. Cari file .csv secara otomatis di folder 'data/' atau folder utama
csv_files = glob.glob('data/*.csv') + glob.glob('*.csv')
if not csv_files:
    raise FileNotFoundError("File CSV tidak ditemukan! Silakan pindahkan file .csv dari Kaggle ke folder 'data/' atau folder utama proyek.")

CSV_PATH = csv_files[0]
print(f"[INFO] Membaca dataset: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

# Deteksi otomatis nama kolom waktu dan target umum di Kaggle
date_col = next((c for c in df.columns if c.lower() in ['date', 'tanggal', 'datetime', 'time', 'timestamp']), df.columns[0])
target_col = next((c for c in df.columns if c.lower() in ['close', 'price', 'adj close', 'last', 'harga', 'close/last']), df.columns[1])

print(f"[INFO] Menggunakan kolom waktu: '{date_col}' dan kolom target: '{target_col}'")

df[date_col] = pd.to_datetime(df[date_col], format='mixed')
df.sort_values(date_col, inplace=True)
df.set_index(date_col, inplace=True)

# Bersihkan simbol mata uang ($) dan koma ribuan jika tipe data masih string
if df[target_col].dtype == 'object':
    df[target_col] = df[target_col].astype(str).str.replace(',', '').str.replace('$', '').str.strip().astype(float)

df = df[[target_col]].dropna()
df.rename(columns={target_col: 'Close'}, inplace=True)

# Simpan plot garis awal (untuk bukti validasi time series di laporan)
plt.figure(figsize=(12, 5))
plt.plot(df.index, df['Close'], color='#ca8a04', linewidth=1.5)
plt.title('Grafik Historis Harga Penutupan Emas (Dataset Kaggle)')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (USD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/plot_validasi_awal.png')
plt.close()

# 2. Normalisasi Data ke Rentang [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df.values)

# 3. Membuat Sekuens (Windowing) dengan timesteps = 60
TIMESTEPS = 60
X, y = [], []
for i in range(TIMESTEPS, len(scaled_data)):
    X.append(scaled_data[i - TIMESTEPS:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split Data Latih (80%) dan Data Uji (20%) secara kronologis
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Membangun Arsitektur Model LSTM
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(TIMESTEPS, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=25, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 5. Melatih Model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 6. Evaluasi Model pada Data Uji
predictions = model.predict(X_test)
predictions_actual = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
mae = mean_absolute_error(y_test_actual, predictions_actual)
print(f"\n[HASIL EVALUASI] RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# Simpan grafik evaluasi aktual vs prediksi untuk laporan
plt.figure(figsize=(12, 5))
plt.plot(df.index[-len(y_test_actual):], y_test_actual, label='Harga Aktual', color='#1d4ed8')
plt.plot(df.index[-len(predictions_actual):], predictions_actual, label='Prediksi LSTM', color='#ea580c')
plt.title(f'Perbandingan Aktual vs Prediksi Harga Penutupan Emas (RMSE: {rmse:.2f})')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/plot_evaluasi_model.png')
plt.close()

# 7. Simpan Model (.h5) dan Scaler (.pkl)
model.save('model/lstm_gold.h5')
joblib.dump(scaler, 'model/scaler_gold.pkl')
print("[SUKSES] Model tersimpan di 'model/lstm_gold.h5' dan 'model/scaler_gold.pkl'")