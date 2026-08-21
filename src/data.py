"""Ham xlsx -> temiz saatlik ızgara: birim çevrimi, winsor, boşluk bayrakları,
ve EĞİTİM istatistiklerini bozan anomali yıl-dilimlerinin otomatik tespiti."""
import numpy as np
import pandas as pd
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as C


def load_raw():
    df = pd.read_excel(C.DATA_PATH, header=[0, 1])
    df.columns = ['_'.join([str(a), str(b)])
                  .replace(' ( µg/m3 )', '').replace(' ( ppb )', '_ppb')
                  .replace('Unnamed', '').strip('_') for a, b in df.columns]
    df = df.rename(columns={df.columns[0]: 'ts'})
    df['ts'] = pd.to_datetime(df['ts']).dt.floor('h')
    df = df.drop_duplicates('ts').set_index('ts').sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq='h')
    return df.reindex(full).rename_axis('ts')


def _gap_hours(series):
    """Her eksik satır için ait olduğu kesintisiz boşluğun uzunluğu (gözlemde 0)."""
    isna = series.isna().values
    out = np.zeros(len(series), dtype=int)
    i, n = 0, len(series)
    while i < n:
        if isna[i]:
            j = i
            while j < n and isna[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def detect_anomalous_years(series, train_mask):
    """Bir serinin EĞİTİM yıllarından, ya çok eksik ya da dağılımı kayık olanları
    döndürür. Bunlar winsor/impute istatistiklerinden ve hedef eğitiminden dışlanır."""
    s = series[train_mask]
    yrs = s.index.year
    bad = set()
    stats = []
    for y in np.unique(yrs):
        yv = s[yrs == y]
        frac = yv.notna().mean()
        med = yv.median()
        stats.append((y, frac, med))
        if frac < 0.30:          # yılın >%70'i eksik -> güvenilmez
            bad.add(int(y))
    # yıllık medyanlarda robust aykırılık (MAD z-skoru)
    meds = np.array([m for _, _, m in stats if not np.isnan(m)])
    yrs_ok = [y for (y, f, m) in stats if not np.isnan(m)]
    if len(meds) >= 4:
        med0 = np.median(meds)
        mad = np.median(np.abs(meds - med0)) + 1e-9
        for y, m in zip(yrs_ok, meds):
            if abs(m - med0) / (1.4826 * mad) > 3.5:
                bad.add(int(y))
    return bad


def clean():
    raw = load_raw()
    idx = raw.index
    train_mask = idx.year <= C.TRAIN_END

    clean = pd.DataFrame(index=idx)
    meta = {}
    for name, (col, conv) in C.SERIES_MAP.items():
        v = pd.to_numeric(raw[col], errors='coerce')
        if conv:  # ppb -> µg/m3
            v = v * C.MOLAR_WEIGHT[conv] / C.MOLAR_VOLUME

        bad_years = detect_anomalous_years(v, train_mask)
        # eğitim-istatistik uygunluğu: anomali yıllar hariç eğitim satırları
        stat_mask = train_mask & ~idx.year.isin(bad_years)

        cap = v[stat_mask].quantile(C.WINSOR_Q)
        n_clip = int((v > cap).sum())
        v = v.clip(upper=cap)

        clean[name] = v
        clean[name + '_observed'] = v.notna().astype(int)
        clean[name + '_gaphours'] = _gap_hours(v)
        # uzun-vadeli lineer trend (durağanlık bayrağı): trend kayması > 0.5 std mı?
        tf = idx.year + (idx.month - 1) / 12.0
        ok = stat_mask & v.notna()
        slope = np.polyfit(tf[ok], v[ok], 1)[0] if ok.sum() > 100 else 0.0
        span = tf[ok].max() - tf[ok].min() if ok.sum() > 100 else 1.0
        sd = v[ok].std() + 1e-9
        is_trending = abs(slope * span) / sd > 0.5
        meta[name] = dict(cap=round(float(cap), 2), n_clipped=n_clip,
                          bad_years=sorted(bad_years),
                          observed_pct=round(100 * v.notna().mean(), 1),
                          trend_slope=round(float(slope), 2), is_trending=bool(is_trending))

    # takvim
    t = idx
    clean['hour_sin'] = np.sin(2*np.pi*t.hour/24); clean['hour_cos'] = np.cos(2*np.pi*t.hour/24)
    clean['dow_sin']  = np.sin(2*np.pi*t.dayofweek/7); clean['dow_cos'] = np.cos(2*np.pi*t.dayofweek/7)
    clean['month_sin'] = np.sin(2*np.pi*(t.month-1)/12); clean['month_cos'] = np.cos(2*np.pi*(t.month-1)/12)
    clean['year'] = t.year
    return clean, meta


if __name__ == '__main__':
    df, meta = clean()
    print(f"Temiz: {df.shape} | {df.index.min()} -> {df.index.max()}")
    for k, v in meta.items():
        flag = f"  ⚠ anomali yıl: {v['bad_years']}" if v['bad_years'] else ""
        print(f"  {k:14s} %{v['observed_pct']:<5} cap={v['cap']:<9} kırpıldı={v['n_clipped']}{flag}")
