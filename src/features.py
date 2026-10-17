"""Kademeli doldurma (sadece GİRDİ için) + lag/rolling/takvim öznitelikleri."""
import os
import numpy as np
import pandas as pd
import config as C

# Sensitivity arm only: AQF_TWO_SIDED_FILL=1 restores the earlier two-sided (non-causal) short-gap interpolation.
TWO_SIDED = os.environ.get("AQF_TWO_SIDED_FILL", "0") == "1"


def elapsed_gap(values):
    """Number of consecutive missing hours up to and including each hour (past information only)."""
    miss = values.isna().to_numpy()
    out = np.zeros(len(miss), dtype=int)
    run = 0
    for k, m in enumerate(miss):
        run = run + 1 if m else 0
        out[k] = run
    return pd.Series(out, index=values.index)


def tiered_impute(values, gaphours, stat_mask):
    """Causal tiered filling for the INPUT matrix only. At each missing hour the fill depends on the
    number of consecutive missing hours elapsed so far (never on the length of the whole gap):
    <= SHORT_GAP: last observed value carried forward; <= MED_GAP: training hour-of-day median;
    otherwise: training overall median. The argument `gaphours` is kept for compatibility and unused."""
    if TWO_SIDED:
        o = values.copy()
        interp = values.interpolate('linear', limit_direction='both')
        short = (gaphours > 0) & (gaphours <= C.SHORT_GAP)
        o[short] = interp[short]
        hod = pd.Series(values.index.hour, index=values.index)
        hod_med = values[stat_mask].groupby(values.index[stat_mask].hour).median()
        medium = (gaphours > C.SHORT_GAP) & (gaphours <= C.MED_GAP)
        o[medium] = hod.map(hod_med)[medium]
        gmed = values[stat_mask].median()
        o[gaphours > C.MED_GAP] = gmed
        return o.fillna(gmed)
    o = values.copy()
    el = elapsed_gap(values)
    ffill = values.ffill()
    short = (el > 0) & (el <= C.SHORT_GAP)
    o[short] = ffill[short]

    hod = pd.Series(values.index.hour, index=values.index)
    hod_med = values[stat_mask].groupby(values.index[stat_mask].hour).median()
    medium = (el > C.SHORT_GAP) & (el <= C.MED_GAP)
    o[medium] = hod.map(hod_med)[medium]

    gmed = values[stat_mask].median()
    o[el > C.MED_GAP] = gmed
    return o.fillna(gmed)


def impute_all(clean, stat_masks):
    imp = pd.DataFrame(index=clean.index)
    for name in C.SERIES_MAP:
        imp[name] = tiered_impute(clean[name], clean[name + '_gaphours'], stat_masks[name])
    return imp


def build_features(imp, clean, target, meteo_feats=None, kind=None, param=None):
    """target için zaman-nedenselliğe (<=t) uyan öznitelik matrisi.
    meteo_feats verilirse HEDEF-ZAMAN havası (bilinen-gelecek kovaryat) eklenir."""
    X = pd.DataFrame(index=imp.index)
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        X[f'{target}_lag{lag}'] = imp[target].shift(lag)
    for w in [24, 168]:
        X[f'{target}_rmean{w}'] = imp[target].shift(1).rolling(w).mean()
        X[f'{target}_rstd{w}']  = imp[target].shift(1).rolling(w).std()
    for p in C.COVARIATES:
        X[f'{p}_lag1']  = imp[p].shift(1)
        X[f'{p}_lag24'] = imp[p].shift(24)
    for c in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'year']:
        X[c] = clean[c]
    if meteo_feats is not None and kind is not None:
        # Meteoroloji yalnızca <= günlük ufuklarda: uzun ufukta (mean, k>=168) gelecek hava
        # bilinemez; ileri-ortalanmış meteo CV seçimini küçük örneklemde yanıltıp test'i çökertir.
        if not (kind == 'mean' and param >= 168):
            from src.meteo import target_time_features
            X = X.join(target_time_features(meteo_feats, kind, param))
    return X
