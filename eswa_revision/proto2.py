"""Arindirilmis protokol: (1) her CV katinin TUM on isleme adimlari (anomali-yil tespiti, kirpma esigi,
doldurma istatistikleri) yalnizca o katin egitim doneminden ogrenilir; nihai model icin ayni adimlar
<=2024 egitim doneminden ogrenilir. (2) Egitim satirlarinin ileriye donuk hedef penceresi egitim
sinirini asmaz (purge). Ham xlsx varsa onu, yoksa saklanan temiz CSV'yi girdi olarak kullanir."""
import sys, os, pickle, argparse, time
import numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from src.data import detect_anomalous_years, _gap_hours
from src.features import impute_all, build_features
from src.targets import make_target
from src.models import candidates

CSV = C.DATA_PATH.parent / 'konya_clean_2stations.csv'
PURGE = True   # False only for the purge sensitivity comparison (--no-purge)
CV_FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]
CAL = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'year']

_RAW = None
def raw_series():
    """Kirpilmamis ham seri varsa (xlsx) onu, yoksa CSV'deki saklanan seriyi dondurur."""
    global _RAW
    if _RAW is not None:
        return _RAW
    if C.DATA_PATH.exists():
        from src.data import load_raw
        raw = load_raw(); df = pd.DataFrame(index=raw.index)
        for name, (col, conv) in C.SERIES_MAP.items():
            v = pd.to_numeric(raw[col], errors='coerce')
            if conv: v = v * C.MOLAR_WEIGHT[conv] / C.MOLAR_VOLUME
            df[name] = v
        t = df.index
        for k, f in (('hour', 24), ('dayofweek', 7)):
            pass
        cal = pd.DataFrame(index=t)
        cal['hour_sin'] = np.sin(2*np.pi*t.hour/24); cal['hour_cos'] = np.cos(2*np.pi*t.hour/24)
        cal['dow_sin'] = np.sin(2*np.pi*t.dayofweek/7); cal['dow_cos'] = np.cos(2*np.pi*t.dayofweek/7)
        cal['month_sin'] = np.sin(2*np.pi*(t.month-1)/12); cal['month_cos'] = np.cos(2*np.pi*(t.month-1)/12)
        cal['year'] = t.year
        _RAW = (df, cal, 'xlsx')
    else:
        d = pd.read_csv(CSV, parse_dates=['ts']).drop_duplicates('ts').set_index('ts').sort_index()
        d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='h')).rename_axis('ts')
        _RAW = (d[list(C.SERIES_MAP)].copy(), d[CAL].copy(), 'csv')   # stored series: already clipped, see README
    return _RAW

_PREP = {}
def prepare_period(end_year, excl=True):
    """end_year'a kadar olan donemden ogrenilen on isleme (aykiri yil, kirpma, doldurma)."""
    key = (end_year, excl)
    if key in _PREP:
        return _PREP[key]
    raw, cal, _ = raw_series()
    idx = raw.index; yr = idx.year
    train = yr <= end_year
    clean = pd.DataFrame(index=idx); stat = {}; bad_all = {}
    for n in C.SERIES_MAP:
        v = raw[n]
        bad = detect_anomalous_years(v, train) if excl else set()
        sm = train & ~np.isin(yr, bad)
        cap = v[sm].quantile(C.WINSOR_Q)
        v = v.clip(upper=cap)
        clean[n] = v; clean[n + '_gaphours'] = _gap_hours(v)
        stat[n] = sm; bad_all[n] = sorted(bad)
    for c in cal.columns:
        clean[c] = cal[c]
    imp = impute_all(clean, stat)
    _PREP[key] = dict(clean=clean, imp=imp, bad=bad_all, idx=idx, yr=yr, raw=raw, src=_)
    return _PREP[key]

def _r2(pred, y, m):
    m = m & y.notna() & pred.notna()
    return r2_score(y[m], pred[m]) if m.sum() >= 30 else np.nan

def run_cell(T, hname, kind, param, meteo=None, folds=CV_FOLDS, return_pred=False):
    def fitmask(P, X, y, end_year):
        yr = P['yr']; pos = np.arange(len(yr))
        end_pos = pos[yr <= end_year].max()
        ok = X.notna().all(axis=1)
        m = (yr <= end_year) & ~np.isin(yr, P['bad'][T]) & ok & y.notna() & ((pos + param <= end_pos) | (not PURGE))
        return m, ok
    scores = {}; models = {m.name: m for m in candidates(kind, param)}
    for te, va in folds:
        P = prepare_period(te)
        X = build_features(P['imp'], P['clean'], T, meteo_feats=meteo, kind=kind, param=param)
        y = make_target(P['clean'][T], kind, param); anchor = P['clean'][T].fillna(P['imp'][T])
        y_ev = make_target(P['raw'][T], kind, param); anchor_ev = P['raw'][T].fillna(P['imp'][T])
        fm, ok = fitmask(P, X, y, te)
        if fm.sum() < 500: continue
        for name, mdl in models.items():
            ctx = dict(X=X, y=y, anchor=anchor, anchor_raw=anchor_ev, fit_mask=fm, index=P['idx'], clean=P['clean'],
                       imp=P['imp'], target=T, kind=kind, param=param)
            try: r = _r2(mdl.fit_predict(ctx), y_ev, (P['yr'] == va) & ok)
            except Exception: r = np.nan
            if r == r: scores.setdefault(name, []).append(r)
    cv = {k: float(np.mean(v)) for k, v in scores.items()}
    champ = max(cv, key=cv.get)
    P = prepare_period(C.VAL_YEAR)
    X = build_features(P['imp'], P['clean'], T, meteo_feats=meteo, kind=kind, param=param)
    y = make_target(P['clean'][T], kind, param); anchor = P['clean'][T].fillna(P['imp'][T])
    y_ev = make_target(P['raw'][T], kind, param); anchor_ev = P['raw'][T].fillna(P['imp'][T])
    fm, ok = fitmask(P, X, y, C.VAL_YEAR)
    ctx = dict(X=X, y=y, anchor=anchor, anchor_raw=anchor_ev, fit_mask=fm, index=P['idx'], clean=P['clean'],
               imp=P['imp'], target=T, kind=kind, param=param)
    pred = models[champ].fit_predict(ctx)
    tm = (P['yr'] >= C.TEST_FROM) & ok & y_ev.notna() & pred.notna()
    out = dict(hedef=T, ufuk=hname, sampiyon=champ, CV_R2=round(cv[champ], 4),
               R2_test=round(r2_score(y_ev[tm], pred[tm]), 4), MAE_test=round(mean_absolute_error(y_ev[tm], pred[tm]), 3),
               R2_persist=round(r2_score(y_ev[tm], anchor_ev[tm]), 4), n_test=int(tm.sum()), target_src=P['src'])
    if return_pred:
        # 'y' and 'anchor' are the evaluation (raw) target and last observation; 'y_fit' is the series used for fitting
        out['pred'] = pred; out['fit_mask'] = fm; out['y'] = y_ev; out['y_fit'] = y; out['test_mask'] = tm
        out['anchor'] = anchor_ev; out['anchor_fit'] = anchor; out['ctx_ok'] = ok
    return out

def _work(args):
    scen, T, hn, kind, param = args
    os.environ['OMP_NUM_THREADS'] = '1'
    meteo = None
    if scen != 'baseline':
        from src.meteo import load_cached_features
        meteo = load_cached_features(scen)
    t0 = time.time()
    r = run_cell(T, hn, kind, param, meteo=meteo)
    r['scenario'] = scen; r['sec'] = round(time.time() - t0)
    print(r, flush=True)
    return r

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--scen', default='baseline'); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--no-purge', action='store_true', help='keep training rows whose target window crosses the training boundary')
    a = ap.parse_args()
    PURGE = not a.no_purge
    if raw_series()[2] == 'csv':
        print('WARNING: the stored series is already clipped; evaluation targets are NOT raw. Provide the raw xlsx (config.DATA_PATH) for raw-target evaluation.', flush=True)
    tag = 'nopurge_' if a.no_purge else 'proto2_'
    from multiprocessing import Pool
    H = {h[0]: h for h in C.HORIZONS}
    if a.scen == 'baseline':
        tasks = [(a.scen, T, hn, k, p) for T in C.TARGETS for hn, k, p in C.HORIZONS]
    else:   # meteoroloji karsilastirmasinda raporlanan hucreler (Tablo 3 ve Sekil 5)
        cells = [(T, h) for T in C.TARGETS for h in ('t+24', 'gunluk')] + [('meram_co', 't+6')]
        tasks = [(a.scen, T, hn, H[hn][1], H[hn][2]) for T, hn in cells]
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    with Pool(a.workers) as pool:
        res = pool.map(_work, tasks, chunksize=1)
    pd.DataFrame(res).to_csv(C.OUTPUT_DIR / f'{tag}{a.scen}.csv', index=False)
