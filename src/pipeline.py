"""Her (hedef × ufuk) için model zoo'yu yarıştırır, 2024 doğrulamasıyla şampiyon seçer,
2025+ test üzerinde raporlar. Çıktı: lider tablosu."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as C
from src.data import clean as load_clean
from src.features import impute_all, build_features
from src.targets import make_target
from src.models import ZOO, candidates, GBM_NAME


def _metrics(yhat, ytrue, mask):
    m = mask & ytrue.notna() & yhat.notna()
    yt, yp = ytrue[m], yhat[m]
    if m.sum() < 30:
        return dict(r2=np.nan, mae=np.nan, rmse=np.nan, n=int(m.sum()))
    return dict(r2=r2_score(yt, yp), mae=mean_absolute_error(yt, yp),
                rmse=np.sqrt(mean_squared_error(yt, yp)), n=int(m.sum()))


def run(targets=None, horizons=None, verbose=True):
    clean, meta = load_clean()
    idx = clean.index
    yr = idx.year
    train_base = yr <= C.TRAIN_END
    val_mask = yr == C.VAL_YEAR
    test_mask = yr >= C.TEST_FROM

    # impute istatistikleri: her seri için anomali yıllar hariç eğitim
    stat_masks = {n: (train_base & ~np.isin(yr, meta[n]['bad_years'])) for n in C.SERIES_MAP}
    imp = impute_all(clean, stat_masks)

    targets = targets or C.TARGETS
    horizons = horizons or C.HORIZONS
    rows, champions = [], []

    for tname in targets:
        X = build_features(imp, clean, tname)
        feat_ok = X.notna().all(axis=1)
        anchor = clean[tname].fillna(imp[tname])
        bad_years = meta[tname]['bad_years']
        target_train = train_base & ~np.isin(yr, bad_years)   # bozuk yıllar eğitimden dışla

        for hname, kind, param in horizons:
            ytrue = make_target(clean[tname], kind, param)
            fit_mask = target_train & feat_ok & ytrue.notna()
            ctx = dict(X=X, y=ytrue, anchor=anchor, fit_mask=fit_mask, index=idx,
                       clean=clean, imp=imp, target=tname, kind=kind, param=param)

            scored = []
            for model in candidates(kind, param):
                try:
                    pred = model.fit_predict(ctx)
                except Exception as e:
                    if verbose: print(f"    [{model.name}] hata: {e}")
                    continue
                val = _metrics(pred, ytrue, val_mask & feat_ok)
                test = _metrics(pred, ytrue, test_mask & feat_ok)
                scored.append((model.name, val['r2'], test))
                rows.append(dict(hedef=tname, ufuk=hname, model=model.name,
                                 R2_val=round(val['r2'], 4) if val['r2']==val['r2'] else None,
                                 R2_test=round(test['r2'], 4) if test['r2']==test['r2'] else None,
                                 MAE_test=round(test['mae'], 3) if test['mae']==test['mae'] else None,
                                 RMSE_test=round(test['rmse'], 3) if test['rmse']==test['rmse'] else None,
                                 n_test=test['n']))
            # şampiyon: en yüksek doğrulama R2'si
            scored = [s for s in scored if s[1] == s[1]]
            if scored:
                champ = max(scored, key=lambda s: s[1])
                champions.append(dict(hedef=tname, ufuk=hname, sampiyon=champ[0],
                                      R2_val=round(champ[1], 4),
                                      R2_test=round(champ[2]['r2'], 4),
                                      MAE_test=round(champ[2]['mae'], 3),
                                      n_test=champ[2]['n']))
                if verbose:
                    print(f"  {tname:13s} {hname:9s} -> {champ[0]:15s} "
                          f"val={champ[1]:.3f}  test={champ[2]['r2']:.3f}")

    full = pd.DataFrame(rows)
    champ_df = pd.DataFrame(champions)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    full.to_csv(C.OUTPUT_DIR / 'leaderboard_full.csv', index=False)
    champ_df.to_csv(C.OUTPUT_DIR / 'champions.csv', index=False)
    return full, champ_df, meta
