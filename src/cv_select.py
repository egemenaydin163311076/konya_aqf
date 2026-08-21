"""Zaman-serisi CV ile ufuk-başına şampiyon seçimi.
select_one(): TEK bir (hedef, ufuk) pipeline'ı -> CV ile şampiyon + final fit + test.
run_cv(): tüm kombinasyonları sırayla koşar. (Paralel sürüm: src/router.py)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as C
from src.data import clean as load_clean
from src.features import impute_all, build_features
from src.targets import make_target
from src.models import candidates, GBM_NAME

CV_FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]   # (train_end, val) genişleyen pencere


def _r2(pred, ytrue, mask):
    m = mask & ytrue.notna() & pred.notna()
    return r2_score(ytrue[m], pred[m]) if m.sum() >= 30 else np.nan


def prepare(meteo_feats=None):
    """Veriyi bir kez hazırla (router ve run_cv paylaşır).
    meteo_feats: meteo.meteo_features(...) çıktısı (opsiyonel; verilmezse meteo'suz)."""
    clean, meta = load_clean()
    idx = clean.index; yr = idx.year
    stat = {n: (yr <= C.TRAIN_END) & ~np.isin(yr, meta[n]['bad_years']) for n in C.SERIES_MAP}
    imp = impute_all(clean, stat)
    return dict(clean=clean, meta=meta, imp=imp, idx=idx, yr=yr, meteo=meteo_feats)


def select_one(T, hname, kind, param, S, folds=CV_FOLDS):
    """Tek (hedef, ufuk): CV ile şampiyon seç, train+val'de yeniden eğit, test'te raporla."""
    clean, meta, imp, idx, yr = S['clean'], S['meta'], S['imp'], S['idx'], S['yr']
    X = build_features(imp, clean, T, meteo_feats=S.get('meteo'), kind=kind, param=param)
    feat_ok = X.notna().all(axis=1)
    anchor = clean[T].fillna(imp[T])
    bad = meta[T]['bad_years']
    ytrue = make_target(clean[T], kind, param)

    scored = []
    for model in candidates(kind, param):
        fr = []
        for te_year, val_year in folds:
            if val_year in bad:
                continue
            fit_mask = (yr <= te_year) & ~np.isin(yr, bad) & feat_ok & ytrue.notna()
            if fit_mask.sum() < 500:
                continue
            ctx = dict(X=X, y=ytrue, anchor=anchor, fit_mask=fit_mask, index=idx,
                       clean=clean, imp=imp, target=T, kind=kind, param=param)
            try:
                r = _r2(model.fit_predict(ctx), ytrue, (yr == val_year) & feat_ok)
            except Exception:
                r = np.nan
            if r == r:
                fr.append(r)
        if fr:
            scored.append((model, float(np.mean(fr)), len(fr)))
    if not scored:
        return None
    champ, cv_r2, nf = max(scored, key=lambda s: s[1])

    final_fit = (yr <= C.VAL_YEAR) & ~np.isin(yr, bad) & feat_ok & ytrue.notna()
    ctx = dict(X=X, y=ytrue, anchor=anchor, fit_mask=final_fit, index=idx,
               clean=clean, imp=imp, target=T, kind=kind, param=param)
    pred = champ.fit_predict(ctx)
    tm = (yr >= C.TEST_FROM) & feat_ok & ytrue.notna()
    r2t = _r2(pred, ytrue, (yr >= C.TEST_FROM) & feat_ok)
    mae = mean_absolute_error(ytrue[tm], pred[tm]) if tm.sum() else np.nan
    return dict(hedef=T, ufuk=hname, sampiyon=champ.name, CV_R2=round(cv_r2, 4),
                R2_test=round(r2t, 4), MAE_test=round(float(mae), 3), n_fold=nf, pred=pred)


def run_cv(targets=None, horizons=None, folds=CV_FOLDS, verbose=True):
    S = prepare()
    rows = []
    for T in (targets or C.TARGETS):
        for hname, kind, param in (horizons or C.HORIZONS):
            r = select_one(T, hname, kind, param, S, folds)
            if r is None:
                continue
            r.pop('pred')
            rows.append(r)
            if verbose:
                print(f"  {T:13s} {hname:9s} -> {r['sampiyon']:15s} "
                      f"CV={r['CV_R2']:.3f}  test={r['R2_test']:.3f}")
    champ_df = pd.DataFrame(rows)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    champ_df.to_csv(C.OUTPUT_DIR / 'champions_cv.csv', index=False)
    return champ_df, S['meta']


if __name__ == '__main__':
    print(f"GBM: {GBM_NAME} | CV folds: {CV_FOLDS}\n")
    champ, _ = run_cv()
    print("\n=== CV ŞAMPİYONLARI ===")
    print(champ.to_string(index=False))
