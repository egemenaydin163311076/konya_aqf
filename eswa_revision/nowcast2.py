"""Concurrent spatial reconstruction under the final protocol (fold-wise preprocessing, raw evaluation targets)."""
import numpy as np, pandas as pd, os
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import config as C, proto2
from src.nowcast import build_nowcast_X, bucketed_r2
from src.models import _gbm

def fit_eval(T, end_year, eval_mask_fn):
    P = proto2.prepare_period(end_year)
    imp, clean, yr, idx, raw = P['imp'], P['clean'], P['yr'], P['idx'], P['raw']
    X = build_nowcast_X(imp, clean, T); ok = X.notna().all(axis=1)
    y = clean[T]; y_ev = raw[T]
    fit = (yr <= end_year) & ~np.isin(yr, P['bad'][T]) & ok & y.notna()
    out = {}
    for cname, mdl in {'nowcast_gbm': _gbm(), 'nowcast_ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0))}.items():
        mdl.fit(X[fit], y[fit]); pred = pd.Series(mdl.predict(X.fillna(0)), index=idx)
        out[cname] = (pred, ok & eval_mask_fn(yr), y_ev)
    return out

rows = []
for T in C.TARGETS:
    val = fit_eval(T, C.VAL_YEAR - 1, lambda yr: yr == C.VAL_YEAR)
    test = fit_eval(T, C.VAL_YEAR, lambda yr: yr >= C.TEST_FROM)
    for c in val:
        r2v, _ = bucketed_r2(val[c][0], val[c][2], val[c][1], 'H')
        pred, m, y = test[c]
        d = dict(hedef=T, model=c, R2_val_saatlik=round(r2v, 3))
        for k, f in (('R2_saatlik', 'H'), ('R2_gunluk', 'D'), ('R2_haftalik', 'W'), ('R2_aylik', 'M')):
            d[k] = round(bucketed_r2(pred, y, m, f)[0], 4)
        rows.append(d); print(d, flush=True)
pd.DataFrame(rows).to_csv('outputs/nowcast_proto2.csv', index=False)
