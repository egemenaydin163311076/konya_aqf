"""WHO günlük hava kalitesi + gün-ileri tahmin (arkadaşın 7-node dosyasıyla, adil kıyas).
- Günlük değer: takvim-günü ortalaması, WHO/EEA %75 tamlık kuralı (>=18/24 GÖZLENMİŞ saat).
- Girdi öznitelikleri: doldurulmuş günlük seri (her zaman mevcut). Hedef: gözlenmiş günlük.
- Bölme: arkadaşın 'split' kolonu (train 2021-23 / val 2024 / test 2025-26).
- WHO 2021 AQG 24s eşikleri: PM10 45, SO2 40, NO2 25 µg/m3 -> aşım sınıflandırması.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

CSV = "data/05_air_quality_7node_long.csv"   # <-- arkadaşın 7-node dosyası   # <-- kendi yolunla değiştir
STATIONS = {1: 'Karkent', 3: 'Meram'}
POLLUTANTS = ['PM10', 'SO2', 'NO2', 'NOX', 'NO']
ALL_POLL = ['PM10', 'PM25', 'SO2', 'CO', 'NO2', 'NOX', 'NO', 'O3']
WHO_AQG = {'PM10': 45, 'SO2': 40, 'NO2': 25}    # 2021, 24-saat ortalama (µg/m3)
MIN_HOURS = 18                                   # %75 tamlık
RANDOM_STATE = 42


def _gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
        max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=RANDOM_STATE)


def build_daily(df):
    """Her (node, kirletici) için günlük doldurulmuş (imp) ve gözlenmiş (obs) seriler."""
    daily = {}
    for nid in STATIONS:
        sub = df[df.node_id == nid].set_index('timestamp').sort_index()
        d = pd.DataFrame()
        for p in ALL_POLL:
            imp = sub[p].resample('D').mean()                       # tüm saatler (doldurulmuş)
            obs_vals = sub[p].where(sub[p + '_observed'] == 1)
            obs_mean = obs_vals.resample('D').mean()
            obs_cnt = obs_vals.resample('D').count()
            obs = obs_mean.where(obs_cnt >= MIN_HOURS)               # WHO %75 kuralı
            d[p + '_imp'] = imp
            d[p + '_obs'] = obs
        # split (günlük): o güne düşen ilk split etiketi
        d['split'] = sub['split'].resample('D').first()
        daily[nid] = d
    return daily


def features(daily, nid, p):
    d = daily[nid]
    other = 3 if nid == 1 else 1
    X = pd.DataFrame(index=d.index)
    s = d[p + '_imp']
    for lag in [1, 2, 3, 7, 14]:
        X[f'{p}_lag{lag}'] = s.shift(lag)
    X[f'{p}_rm7'] = s.shift(1).rolling(7).mean()
    X[f'{p}_rm30'] = s.shift(1).rolling(30).mean()
    for q in ALL_POLL:                                # aynı istasyon diğer kirleticiler
        X[f'{q}_lag1'] = d[q + '_imp'].shift(1)
    X[f'other_{p}_lag1'] = daily[other][p + '_imp'].shift(1)   # diğer istasyon
    doy = d.index.dayofyear
    X['doy_sin'] = np.sin(2*np.pi*doy/365); X['doy_cos'] = np.cos(2*np.pi*doy/365)
    X['dow'] = d.index.dayofweek; X['month'] = d.index.month
    return X


CV_FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]   # (train_end, val) genişleyen pencere


def forecast_series(daily, nid, p):
    d = daily[nid]
    X = features(daily, nid, p)
    anchor = d[p + '_imp']                       # son günün değeri
    y = d[p + '_obs'].shift(-1)                   # ertesi gün GÖZLENMİŞ günlük
    yr = pd.Series(d.index.year, index=d.index)
    fok = X.notna().all(axis=1)
    teh = (yr >= 2025) & fok & y.notna()

    # aday ÜRETİCİLERİ (CV'de fold başına yeniden eğitilir)
    def b_persist(m): return anchor
    def b_seasonal(m): return d[p + '_imp'].shift(6)
    def b_clim(m):
        clim = y[m].groupby(d.index[m].month).mean()
        return pd.Series([clim.get(mo, np.nan) for mo in d.index.month], index=d.index)
    def b_ridge(m):
        rm = make_pipeline(StandardScaler(), Ridge(alpha=10.0)); rm.fit(X[m], y[m])
        return pd.Series(rm.predict(X.fillna(0)), index=d.index)
    def b_gbmd(m):
        g = _gbm(); g.fit(X[m], (y - anchor)[m])
        return anchor + pd.Series(g.predict(X.fillna(0)), index=d.index)
    builders = {'persistence': b_persist, 'seasonal_naive': b_seasonal,
                'climatology': b_clim, 'ridge': b_ridge, 'gbm_delta': b_gbmd}

    def r2(pred, m):
        mm = m & pred.notna()
        return r2_score(y[mm], pred[mm]) if mm.sum() > 20 else np.nan

    # CV: genişleyen pencere foldlarında ortalama val R2
    scores = {}
    for name, build in builders.items():
        fr = []
        for te, va in CV_FOLDS:
            trm = (yr <= te) & fok & y.notna()
            if trm.sum() < 200:
                continue
            r = r2(build(trm), (yr == va) & fok & y.notna())
            if r == r:
                fr.append(r)
        if fr:
            scores[name] = float(np.mean(fr))
    champ = max(scores, key=scores.get)

    # final: train+val (<=2024) üzerinde yeniden eğit, test'te raporla
    final_m = (yr <= 2024) & fok & y.notna()
    pred = builders[champ](final_m)
    mm = teh & pred.notna()
    r2t = r2_score(y[mm], pred[mm]); mae = mean_absolute_error(y[mm], pred[mm])
    r2_persist = r2_score(y[mm], anchor[mm])      # kıyas için persistence test R2
    # WHO aşım sınıflandırma — tam metrikler (halk sağlığı kararı)
    who = {}
    if p in WHO_AQG:
        thr = WHO_AQG[p]
        yp = (pred[mm] > thr).astype(int).values
        yt = (y[mm] > thr).astype(int).values
        tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
        n = tp + fp + tn + fn
        acc = (tp + tn) / n if n else np.nan
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else np.nan
        base = (tp + fn) / n if n else np.nan          # gerçek aşım taban oranı
        who = dict(WHO_esik=thr, WHO_acc=round(acc, 3),
                   WHO_precision=round(prec, 3) if prec == prec else None,
                   WHO_recall=round(rec, 3) if rec == rec else None,
                   WHO_F1=round(f1, 3) if f1 == f1 else None,
                   asim_taban_orani=round(base, 3), TP=tp, FP=fp, TN=tn, FN=fn)
    out = dict(sampiyon=champ, R2_test=round(r2t, 3), R2_persistence=round(r2_persist, 3),
               MAE=round(mae, 2), n_test=int(mm.sum()))
    out.update(who)
    return out


def main():
    cols = ['timestamp', 'node_id', 'node_name', 'split'] + ALL_POLL + [p+'_observed' for p in ALL_POLL]
    df = pd.read_csv(CSV, usecols=cols)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    daily = build_daily(df)
    rows = []
    for nid, nm in STATIONS.items():
        for p in POLLUTANTS:
            r = forecast_series(daily, nid, p)
            r.update(istasyon=nm, kirletici=p)
            rows.append(r)
            print(f"  {nm:8s} {p:4s} -> {r['sampiyon']:12s} R2={r['R2_test']:.3f} "
                  f"(persist {r['R2_persistence']:.3f})")
    res = pd.DataFrame(rows)
    res.to_csv("outputs/who_daily_results.csv", index=False)

    # WHO sınıflandırma tablosu (yalnız PM10/SO2/NO2 — kılavuz eşiği olanlar)
    who = res[res['WHO_acc'].notna()][['istasyon', 'kirletici', 'WHO_esik', 'WHO_acc',
            'WHO_precision', 'WHO_recall', 'WHO_F1', 'asim_taban_orani', 'TP', 'FP', 'TN', 'FN']]
    who.to_csv("outputs/who_classification.csv", index=False)
    print("\n================ WHO EŞİK-AŞIMI SINIFLANDIRMA (test 2025+) ================")
    print(who.to_string(index=False))
    print(f"\nOrtalama WHO sınıflandırma doğruluğu: {who['WHO_acc'].mean():.3f}")
    print("Kaydedildi: who_daily_results.csv, who_classification.csv")
    return res


if __name__ == '__main__':
    main()
