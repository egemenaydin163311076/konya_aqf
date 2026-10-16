import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import config as C, proto2 as P2
from src.features import build_features
from src.targets import make_target
from src.models import candidates
from sklearn.metrics import r2_score

def final_r2(T, hname, kind, param, model_name, excl=True):
    P = P2.prepare_period(C.VAL_YEAR, excl)
    X = build_features(P['imp'], P['clean'], T)
    y = make_target(P['clean'][T], kind, param); anchor = P['clean'][T].fillna(P['imp'][T])
    yr = P['yr']; pos = np.arange(len(yr)); end_pos = pos[yr <= C.VAL_YEAR].max()
    ok = X.notna().all(axis=1)
    fm = (yr <= C.VAL_YEAR) & ~np.isin(yr, P['bad'][T]) & ok & y.notna() & (pos + param <= end_pos)
    ctx = dict(X=X, y=y, anchor=anchor, fit_mask=fm, index=P['idx'], clean=P['clean'], imp=P['imp'],
               target=T, kind=kind, param=param)
    mdl = [m for m in candidates(kind, param) if m.name == model_name][0]
    pred = mdl.fit_predict(ctx)
    tm = (yr >= C.TEST_FROM) & ok & y.notna() & pred.notna()
    return r2_score(y[tm], pred[tm]), int(tm.sum())

if __name__ == '__main__':
    base = pd.read_csv(C.OUTPUT_DIR / 'proto2_baseline.csv')
    ms = base[(base.hedef == 'meram_so2') & (base.ufuk == 't+1')].iloc[0]
    out = []
    w, n = final_r2('meram_so2', 't+1', 'point', 1, ms.sampiyon, True)
    wo, _ = final_r2('meram_so2', 't+1', 'point', 1, ms.sampiyon, False)
    out.append(dict(ablation='anomalous_year_exclusion', cell='meram_so2 t+1', model=ms.sampiyon, without=round(wo, 4), with_=round(w, 4), n=n))
    d, n = final_r2('karkent_pm10', 't+1', 'point', 1, 'gbm_delta')
    l, _ = final_r2('karkent_pm10', 't+1', 'point', 1, 'gbm_level')
    out.append(dict(ablation='difference_parameterization', cell='karkent_pm10 t+1', model='gbm_level vs gbm_delta', without=round(l, 4), with_=round(d, 4), n=n))
    pd.DataFrame(out).to_csv(C.OUTPUT_DIR / 'ablate2.csv', index=False); print(pd.DataFrame(out))
