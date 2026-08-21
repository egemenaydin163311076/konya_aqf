"""Merkezi yapılandırma. VS Code'da çalışırken sadece DATA_PATH'i kendi yolunla değiştir."""
from pathlib import Path

# --- Veri yolu (kendi makinende ./data/ altına koyup burayı güncelle) ---
DATA_PATH = Path(__file__).parent / "data" / "Konya_İstasyonları_Saatlik_Veriler__2017-2025_.xlsx"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# --- Ham veri -> sade ad eşlemesi (yoğun iki çapa istasyon) ---
SERIES_MAP = {
    'karkent_pm10': ('Konya - Karkent_PM10', None),
    'karkent_pm25': ('Konya - Karkent_PM 2.5', None),
    'karkent_so2':  ('Konya - Karkent_SO2', None),
    'karkent_co':   ('Konya - Karkent_CO', None),
    'karkent_no2':  ('Konya - Karkent_NO2', None),
    'karkent_nox':  ('Konya - Karkent_NOX', None),
    'karkent_no':   ('Konya - Karkent_NO', None),
    'karkent_o3':   ('Konya - Karkent_O3', None),
    'meram_pm10':   ('Konya - Meram_PM10', None),
    'meram_so2':    ('Konya - Meram_SO2_ppb', 'SO2'),   # ppb -> µg/m3
    'meram_co':     ('Konya - Meram_CO_ppb', 'CO'),     # ppb -> µg/m3
    'meram_no2':    ('Konya - Meram_NO2', None),
    'meram_nox':    ('Konya - Meram_NOX', None),
    'meram_no':     ('Konya - Meram_NO', None),
}

# ppb -> µg/m3 (25 °C, 1 atm; µg/m3 = ppb * MW / 24.45). AB için 24.04 kullan.
MOLAR_VOLUME = 24.45
MOLAR_WEIGHT = {'SO2': 64.066, 'CO': 28.010}

# Yüksek doluluklu kovaryatlar (her modele girdi olarak verilir)
COVARIATES = ['karkent_pm10', 'karkent_so2', 'karkent_pm25', 'karkent_o3',
              'karkent_co', 'meram_pm10', 'meram_so2', 'meram_co']

# Tahmin hedefleri (yoğun seriler)
TARGETS = ['karkent_pm10', 'karkent_so2', 'meram_pm10', 'meram_so2', 'meram_co']

# Ufuklar: ('ad', tip, parametre). tip: 'point'=t+h değeri, 'mean'=sonraki k saatin ort.
HORIZONS = [
    ('t+1',     'point', 1),
    ('t+6',     'point', 6),
    ('t+24',    'point', 24),
    ('gunluk',  'mean',  24),
    ('haftalik','mean',  168),
    ('aylik',   'mean',  720),
]

# Zaman bölmesi
TRAIN_END = 2023   # <= bu yıl: eğitim
VAL_YEAR  = 2024   # şampiyon seçimi
TEST_FROM = 2025   # >= bu yıl: test (raporlanan skor)

WINSOR_Q = 0.995   # uç değer kırpma üst kuantili (eğitimden)
SHORT_GAP = 6      # <= : lineer interpolasyon
MED_GAP = 24       # <= : eğitim saat-medyanı; > : eğitim genel medyanı
RANDOM_STATE = 42
