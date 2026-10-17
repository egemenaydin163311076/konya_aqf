"""Pencere tabanli sinir agi taban modelleri — arindirilmis protokol (proto2) ile.
Sampiyon ve taban modeller ayni on isleme (<=2024) ve ayni egitim satirlariyla egitilir."""
import sys, os, time, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import proto2 as P2
import baselines_dl as B      # dm_test, metrics, sk_models, write_latex
from src.models import candidates

CELLS = [('karkent_pm10', 't+1'), ('karkent_pm10', 'gunluk'), ('meram_pm10', 't+1'), ('meram_pm10', 'gunluk')]
CAL6 = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
L = 24

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=3); a = ap.parse_args()
    hz = {h[0]: h for h in C.HORIZONS}; rows = []
    for T, hn in CELLS:
        _, kind, param = hz[hn]; t0 = time.time()
        r = P2.run_cell(T, hn, kind, param, return_pred=True)
        Pp = P2.prepare_period(C.VAL_YEAR); idx = Pp['idx']; yr = Pp['yr']
        y, anchor, tm, fm = r['y'], r['anchor'], r['test_mask'], r['fit_mask']      # evaluation (raw) target and last observation
        y_fit, anchor_fit = r['y_fit'], r['anchor_fit']                              # series used for fitting
        preds = {'champion': [r['pred']], 'persistence': [anchor.copy()]}
        X = P2.build_features(Pp['imp'], Pp['clean'], T)
        ctx = dict(X=X, y=y_fit, anchor=anchor_fit, anchor_raw=anchor, fit_mask=fm, index=idx, clean=Pp['clean'], imp=Pp['imp'],
                   target=T, kind=kind, param=param)
        for m in candidates(kind, param):
            if m.name == 'ridge_lags': preds['ridge_lags'] = [m.fit_predict(ctx)]
        cols = [T] + [c for c in C.COVARIATES if c != T]
        stat = (yr <= C.VAL_YEAR) & ~np.isin(yr, Pp['bad'][T])
        Zdf = Pp['imp'][cols].copy(); mu = Zdf[stat].mean(); sd = Zdf[stat].std() + 1e-9
        Z = np.hstack([((Zdf - mu) / sd).values, Pp['clean'][CAL6].values]).astype(np.float32)
        delta = (y_fit - anchor_fit); dsd = float(delta[fm].std() + 1e-9); yd = (delta / dsd).fillna(0.0).values
        pos = np.arange(len(idx)); okp = pos >= L - 1
        fit_pos = pos[fm.values & okp]; all_pos = pos[okp]
        for s_ in range(a.seeds):
            res = B.sk_models(Z, yd, fit_pos, all_pos, 1000 + s_, L=L)
            for name, out in res.items():
                full = pd.Series(np.nan, index=idx); full.iloc[all_pos] = out
                preds.setdefault(name, []).append(anchor_fit + dsd * full)
        ref_idx = y.index[tm.values]
        err_c = (y[ref_idx] - r['pred'][ref_idx]).values
        h_dm = 24 if kind == 'mean' else param
        for name, pl in preds.items():
            ms = [B.metrics(p, y, tm) for p in pl]
            ens = sum(pl) / len(pl); me = B.metrics(ens, y, tm)
            z = p = np.nan
            if name != 'champion':
                z, p = B.dm_test((y[ref_idx] - ens[ref_idx]).values, err_c, h_dm)
            rows.append(dict(hedef=T, ufuk=hn, model=name, sampiyon=r['sampiyon'],
                             R2_mean=np.mean([m['r2'] for m in ms]),
                             R2_sd=np.std([m['r2'] for m in ms], ddof=1) if len(ms) > 1 else np.nan,
                             MAE_mean=np.mean([m['mae'] for m in ms]), R2_ens=me['r2'], n_test=me['n'],
                             DM_z_vs_champ=z, DM_p_vs_champ=p))
        print(T, hn, r['sampiyon'], round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows); C.OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(C.OUTPUT_DIR / 'dl_baselines2.csv', index=False)
    B.write_latex(df); print(df.round(4).to_string(index=False))

if __name__ == '__main__':
    main()
