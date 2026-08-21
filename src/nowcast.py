"""Nowcasting (eşzamanlı spasyo-temporal yeniden-üretim).
Bir hedef sensörün t anındaki değerini, DİĞER sensörlerin t anındaki değerleri
+ hedefin kendi yakın geçmişi + takvim'den tahmin eder (hedefin kendi t değeri GİRDİDE YOK).
Metrik: saatlik R² + günlük/haftalık/aylık zaman-kovalarına göre R² (arkadaşın kurgusu)."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as C
from src.data import clean as load_clean
from src.features import impute_all
from src.models import _gbm, GBM_NAME


def build_nowcast_X(imp, clean, target):
    """Eşzamanlı (t) diğer seriler + hedefin kendi gecikmeleri + takvim.
    HEDEFİN kendi t değeri DAHİL DEĞİL (aksi halde trivial)."""
    X = pd.DataFrame(index=imp.index)
    for s in C.SERIES_MAP:
        if s == target:
            continue
        X[f'{s}_t'] = imp[s]                      # eşzamanlı çapraz-kesit
    for lag in [1, 24, 168]:                      # hedefin kendi geçmişi (spasyo-TEMPORAL)
        X[f'{target}_lag{lag}'] = imp[target].shift(lag)
    for c in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'year']:
        X[c] = clean[c]
    return X


def bucketed_r2(pred, ytrue, mask, freq):
    """test satırlarını periyoda göre grupla, periyot-ortalamaları üzerinde R²."""
    m = mask & ytrue.notna() & pred.notna()
    d = pd.DataFrame({'p': pred[m], 'y': ytrue[m]}, index=ytrue.index[m])
    if freq == 'H':
        if len(d) < 30:
            return np.nan, len(d)
        return r2_score(d['y'], d['p']), len(d)
    g = d.groupby(d.index.to_period(freq)).mean()
    if len(g) < 4:
        return np.nan, len(g)
    return r2_score(g['y'], g['p']), len(g)


def run_nowcast(targets=None):
    clean, meta = load_clean()
    idx = clean.index; yr = idx.year
    train_base = yr <= C.TRAIN_END; val_mask = yr == C.VAL_YEAR; test_mask = yr >= C.TEST_FROM
    stat_masks = {n: (train_base & ~np.isin(yr, meta[n]['bad_years'])) for n in C.SERIES_MAP}
    imp = impute_all(clean, stat_masks)

    rows = []
    for T in (targets or C.TARGETS):
        X = build_nowcast_X(imp, clean, T)
        feat_ok = X.notna().all(axis=1)
        y = clean[T]
        target_train = train_base & ~np.isin(yr, meta[T]['bad_years'])
        fit = target_train & feat_ok & y.notna()

        cands = {
            'nowcast_gbm': _gbm(),
            'nowcast_ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        }
        for cname, model in cands.items():
            model.fit(X[fit], y[fit])
            pred = pd.Series(model.predict(X.fillna(0)), index=idx)
            # doğrulama saatlik R² (şampiyon seçimi)
            r2v, _ = bucketed_r2(pred, y, val_mask & feat_ok, 'H')
            tm = test_mask & feat_ok
            r2h, nh = bucketed_r2(pred, y, tm, 'H')
            r2d, nd = bucketed_r2(pred, y, tm, 'D')
            r2w, nw = bucketed_r2(pred, y, tm, 'W')
            r2mn, nmn = bucketed_r2(pred, y, tm, 'M')
            mae = mean_absolute_error(y[tm & y.notna()], pred[tm & y.notna()])
            rows.append(dict(hedef=T, model=cname, R2_val_saatlik=round(r2v, 3),
                             R2_saatlik=round(r2h, 3), R2_gunluk=round(r2d, 3),
                             R2_haftalik=round(r2w, 3), R2_aylik=round(r2mn, 3),
                             MAE=round(mae, 2), n_ay=nmn))
    res = pd.DataFrame(rows)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    res.to_csv(C.OUTPUT_DIR / 'nowcast_results.csv', index=False)
    return res


if __name__ == '__main__':
    print(f"GBM: {GBM_NAME}\n")
    r = run_nowcast()
    print(r.to_string(index=False))
    # her hedef için en iyi (gbm vs ridge) saatlik
    best = r.loc[r.groupby('hedef')['R2_saatlik'].idxmax()]
    print("\n=== Hedef başına en iyi nowcast (saatlik / aylık) ===")
    print(best[['hedef','model','R2_saatlik','R2_gunluk','R2_haftalik','R2_aylik']].to_string(index=False))
