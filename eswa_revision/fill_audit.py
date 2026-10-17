"""Causal (backward-only) versus two-sided short-gap filling: delta-parameterized ridge regression, next-hour horizon,
final model fitted on the years up to 2024 under the protocol of proto2.py. Writes outputs/fill_audit.csv."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import config as C, proto2 as P2
import src.features as F
from src.targets import make_target
from src.models import candidates
from sklearn.metrics import r2_score

SERIES = ['karkent_pm10', 'karkent_so2', 'meram_pm10', 'meram_so2', 'meram_co', 'karkent_co']

def run(T, two_sided):
    F.TWO_SIDED = two_sided; P2._PREP.clear()
    P = P2.prepare_period(C.VAL_YEAR)
    X = F.build_features(P['imp'], P['clean'], T)
    y = make_target(P['clean'][T], 'point', 1); y_ev = make_target(P['raw'][T], 'point', 1)
    anchor = P['clean'][T].fillna(P['imp'][T]); yr = P['yr']
    ok = X.notna().all(axis=1); pos = np.arange(len(yr)); end_pos = pos[yr <= C.VAL_YEAR].max()
    fm = (yr <= C.VAL_YEAR) & ~np.isin(yr, P['bad'][T]) & ok & y.notna() & (pos + 1 <= end_pos)
    ctx = dict(X=X, y=y, anchor=anchor, anchor_raw=anchor, fit_mask=fm, index=P['idx'], clean=P['clean'], imp=P['imp'],
               target=T, kind='point', param=1)
    mdl = [m for m in candidates('point', 1) if m.name == 'ridge_lags'][0]
    pred = mdl.fit_predict(ctx)
    tm = (yr >= C.TEST_FROM) & ok & y_ev.notna() & pred.notna()
    lag1 = P['imp'][T].shift(1)
    return r2_score(y_ev[tm], pred[tm]), int(tm.sum()), lag1, tm, P['clean'][T].notna().mean()

if __name__ == '__main__':
    rows = []
    for T in SERIES:
        r_two, n, lag_two, tm, comp = run(T, True)
        r_cau, _, lag_cau, _, _ = run(T, False)
        aff = float(((lag_two - lag_cau).abs() > 1e-9)[tm].mean() * 100)
        rows.append(dict(hedef=T, completeness=round(comp * 100, 1), n_test=n, affected_lag1_pct=round(aff, 1),
                         R2_two_sided=round(r_two, 4), R2_causal=round(r_cau, 4), delta=round(r_cau - r_two, 4)))
        print(rows[-1], flush=True)
    F.TWO_SIDED = False
    pd.DataFrame(rows).to_csv(C.OUTPUT_DIR / 'fill_audit.csv', index=False)
