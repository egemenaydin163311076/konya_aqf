"""Meteoroloji entegrasyonu — akademik (ERA5) + operasyonel (gerçek gün-ileri tahmin).

İki senaryo, AYNI değişkenler:
  • ERA5 reanalizi (archive-api)               -> ÜST-SINIR / akademik
        "hava mükemmel bilinseydi" tavanı.
  • Open-Meteo PREVIOUS RUNS API (_previous_dayN) -> OPERASYONEL
        N gün önceden YAPILMIŞ tahmin (gerçek tahmin hatası içerir). Ocak 2024'ten itibaren.
        Bizim test dönemimiz (2025+) tam kapsanır.

NOT (önceki sorun): "historical-forecast-api" her koşunun ilk saatlerini birleştirip
neredeyse-analiz (≈0 lead-time) verir -> ERA5 ile özdeş çıkar. Doğru uç PREVIOUS RUNS'tır.

OPERASYONEL DENEY (sızıntısız, gerçekçi):
  - Eğitim (2017-2023) ve doğrulama: ERA5 (previous-runs 2024 öncesini kapsamaz, zaten doğru).
  - TEST (2025+): gün-ileri operasyonel meteo. Aynı şampiyon model iki kez puanlanır;
    R²(ERA5) - R²(operasyonel) = "hava tahmini hatasının maliyeti".
  Bu, sahadaki dağıtımın (reanalizde eğit, tahminde çalıştır) birebir karşılığıdır.

Ağ bu ortamda kapalı; fetch SENİN makinende çalışır (ücretsiz, anahtarsız). Cache outputs/ altına.
"""
import json, urllib.request, urllib.parse, urllib.error
from pathlib import Path
import numpy as np, pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as C

STATION_COORDS = {'karkent': (37.8810, 32.5210), 'meram': (37.8724, 32.4930)}

HOURLY_VARS = [
    'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
    'surface_pressure', 'pressure_msl',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'precipitation', 'cloud_cover', 'shortwave_radiation',
    'boundary_layer_height',
]
ERA5_ENDPOINT = 'https://archive-api.open-meteo.com/v1/archive'
PREVRUNS_ENDPOINT = 'https://previous-runs-api.open-meteo.com/v1/forecast'
OPERATIONAL_START = '2024-01-01'   # previous-runs API başlangıcı


def _cache_path(source, station):
    return C.OUTPUT_DIR / f'meteo_{source}_{station}.csv'


def _request(url):
    with urllib.request.urlopen(url, timeout=180) as r:
        return json.loads(r.read().decode())


def fetch(source, station, start='2017-01-01', end='2026-04-29', use_cache=True, lead_day=1):
    """source: 'era5' (üst-sınır) | 'operational' (gün-ileri previous-runs, lead_day gün)."""
    assert source in ('era5', 'operational'), source
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    cp = _cache_path(source, station)
    if use_cache and cp.exists():
        return pd.read_csv(cp, parse_dates=['ts']).set_index('ts')
    lat, lon = STATION_COORDS[station]

    if source == 'era5':
        params = {'latitude': lat, 'longitude': lon, 'start_date': start, 'end_date': end,
                  'hourly': ','.join(HOURLY_VARS), 'timezone': 'Europe/Istanbul', 'windspeed_unit': 'ms'}
        data = _request(ERA5_ENDPOINT + '?' + urllib.parse.urlencode(params))
        h = data['hourly']; df = pd.DataFrame(h); df['ts'] = pd.to_datetime(df.pop('time'))
    else:  # operational: previous-runs, _previous_dayN değişkenleri
        ostart = max(start, OPERATIONAL_START)
        varset = list(HOURLY_VARS)
        df = None
        for attempt in range(2):
            pv = [f'{v}_previous_day{lead_day}' for v in varset]
            params = {'latitude': lat, 'longitude': lon, 'start_date': ostart, 'end_date': end,
                      'hourly': ','.join(pv), 'timezone': 'Europe/Istanbul', 'windspeed_unit': 'ms'}
            try:
                data = _request(PREVRUNS_ENDPOINT + '?' + urllib.parse.urlencode(params))
                h = data['hourly']; df = pd.DataFrame(h); df['ts'] = pd.to_datetime(df.pop('time'))
                # _previous_dayN ekini sütun adlarından temizle -> ERA5 ile aynı şema
                df.columns = [c.replace(f'_previous_day{lead_day}', '') for c in df.columns]
                break
            except urllib.error.HTTPError as e:
                # bazı değişkenler previous-runs'ta olmayabilir (örn. boundary_layer_height) -> düşür, tekrar dene
                if attempt == 0 and 'boundary_layer_height' in varset:
                    varset.remove('boundary_layer_height')
                    print(f"  [operational] boundary_layer_height previous-runs'ta yok, düşürülüp tekrar deneniyor")
                else:
                    raise
    df = df.set_index('ts').sort_index()
    df.to_csv(cp)
    return df


def fetch_all_stations(source, **kw):
    frames = {s: fetch(source, s, **kw) for s in STATION_COORDS}
    common = None
    for s, f in frames.items():
        common = f if common is None else common.add(f, fill_value=np.nan)
    return common / len(frames)


def meteo_features(meteo):
    X = pd.DataFrame(index=meteo.index)
    for c in ['temperature_2m', 'relative_humidity_2m', 'surface_pressure',
              'wind_speed_10m', 'precipitation', 'cloud_cover',
              'shortwave_radiation', 'boundary_layer_height']:
        if c in meteo:
            X[c] = meteo[c]
    if 'wind_speed_10m' in meteo and 'wind_direction_10m' in meteo:
        rad = np.deg2rad(meteo['wind_direction_10m'])
        X['wind_u'] = -meteo['wind_speed_10m'] * np.sin(rad)
        X['wind_v'] = -meteo['wind_speed_10m'] * np.cos(rad)
    if 'wind_speed_10m' in meteo:
        X['calm'] = (meteo['wind_speed_10m'] < 1.5).astype(int)
    if 'boundary_layer_height' in meteo:
        X['low_pbl'] = (meteo['boundary_layer_height'] < 300).astype(int)
    if 'precipitation' in meteo:
        X['log_precip'] = np.log1p(meteo['precipitation'].clip(lower=0))
    if 'wind_speed_10m' in X.columns:
        X['wind_rm24'] = X['wind_speed_10m'].rolling(24, min_periods=6).mean()
    empty = [c for c in X.columns if X[c].isna().all()]
    if empty:
        print(f"[meteo] uyarı: tamamen boş sütunlar düşürüldü: {empty}")
        X = X.drop(columns=empty)
    return X


def _avg_raw(source):
    """İstasyon cache'lerini okuyup ortalama ham meteo döndürür (yoksa None)."""
    frames = []
    for s in STATION_COORDS:
        cp = _cache_path(source, s)
        if not cp.exists():
            return None
        frames.append(pd.read_csv(cp, parse_dates=['ts']).set_index('ts'))
    common = frames[0]
    for f in frames[1:]:
        common = common.add(f, fill_value=np.nan)
    return common / len(frames)


def load_cached_features(source):
    """era5: doğrudan. operasyonel: ERA5(eğitim/doğrulama) + operasyonel(test) EKLENMİŞ seri
    -> 'reanalizde eğit, gün-ileri tahminde çalıştır' deneyini sızıntısız kurar."""
    if source == 'era5':
        raw = _avg_raw('era5')
        return meteo_features(raw) if raw is not None else None
    if source == 'operational':
        era5, oper = _avg_raw('era5'), _avg_raw('operational')
        if era5 is None or oper is None:
            return None
        cols = [c for c in era5.columns if c in oper.columns]   # ortak şema
        test_from = pd.Timestamp(f'{C.TEST_FROM}-01-01')
        # operasyonel test döneminde TAMAMEN boş sütunları düş (örn. boundary_layer_height
        # previous-runs'ta gelmiyor) -> sahada gerçekten mevcut olmayan değişkenler dışlanır;
        # aksi halde router'ın notna().all() maskesi tüm test satırlarını eler.
        otest = oper[oper.index >= test_from]
        cols = [c for c in cols if not otest[c].isna().all()]
        era5, oper = era5[cols].copy(), oper[cols]
        spliced = era5.copy()
        m = spliced.index >= test_from
        spliced.loc[m] = oper.reindex(spliced.index).loc[m]      # test döneminde operasyonel
        return meteo_features(spliced)
    return None


def target_time_features(mfeat, kind, param):
    out = pd.DataFrame(index=mfeat.index)
    if kind == 'point':
        for c in mfeat.columns:
            out[f'met_{c}_h{param}'] = mfeat[c].shift(-param)
    else:
        a = mfeat.shift(-1)
        for c in mfeat.columns:
            out[f'met_{c}_fw{param}'] = (a[c][::-1].rolling(param, min_periods=max(3, param // 4))
                                         .mean()[::-1])
    return out


if __name__ == '__main__':
    print("Meteoroloji çekiliyor (ağ gerekli)...")
    for src in ['era5', 'operational']:
        try:
            m = fetch_all_stations(src)
            print(f"  {src}: {m.shape} | {m.index.min()} -> {m.index.max()}")
        except Exception as e:
            print(f"  {src}: çekilemedi ({e}). Ağ erişimi olan ortamda çalıştır.")
