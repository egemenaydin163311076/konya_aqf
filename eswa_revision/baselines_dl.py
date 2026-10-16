"""Derin ogrenme taban modelleri ve istatistiksel karsilastirma (ESWA revizyonu icin).

Makaledeki sizintisiz protokolun AYNISI kullanilir: src.data.clean (egitim-only kirpma ve
anomali yil dislama), src.features.impute_all, ayni hedefler/ufuklar, ayni egitim/dogrulama/test
bolmesi (<=2023 / 2024 / >=2025), delta parametrizasyonu, yalnizca gozlenmis hedefler.

Calistirma (proje kok klasorunden):
    pip install torch            # yalnizca bir kez
    python baselines_dl.py               # varsayilan: 3 tohum, 4 hucre
    python baselines_dl.py --quick       # hizli deneme (1 tohum, az epoch)
Cikti: outputs/dl_baselines.csv, outputs/dl_baselines_table.tex, outputs/dl_baselines_summary.txt
"""
import argparse, time, math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

import config as C
from src.data import detect_anomalous_years


def load_clean():
    path = C.DATA_PATH.parent / 'konya_clean_2stations.csv'
    df = pd.read_csv(path, parse_dates=['ts']).drop_duplicates('ts').set_index('ts').sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='h')).rename_axis('ts')
    train_mask = df.index.year <= C.TRAIN_END
    meta = {}
    for n in C.SERIES_MAP:
        meta[n] = dict(bad_years=sorted(detect_anomalous_years(df[n], train_mask)))
    return df, meta
from src.features import impute_all, build_features
from src.targets import make_target
from src.models import candidates
import src.cv_select as _cv
from src.cv_select import prepare, select_one
_cv.load_clean = load_clean

CELLS = [('karkent_pm10', 't+1'), ('karkent_pm10', 'gunluk'),
         ('meram_pm10', 't+1'), ('meram_pm10', 'gunluk')]
CAL = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
LOOKBACK = 48


# ------------------------------------------------------------------ istatistik
def dm_test(err_a, err_b, h):
    """Diebold-Mariano (kare hata); Bartlett cekirdekli HAC varyans, bant genisligi h-1.
    Negatif istatistik: a modeli b'den daha iyi. Iki yonlu p degeri."""
    d = np.asarray(err_a) ** 2 - np.asarray(err_b) ** 2
    n = len(d)
    dm = d.mean()
    lag = max(int(h) - 1, 0)
    g0 = np.mean((d - dm) ** 2)
    var = g0
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        gk = np.mean((d[k:] - dm) * (d[:-k] - dm))
        var += 2 * w * gk
    se = math.sqrt(max(var, 1e-12) / n)
    z = dm / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def metrics(yhat, ytrue, mask):
    m = mask & ytrue.notna() & yhat.notna()
    return dict(r2=r2_score(ytrue[m], yhat[m]), mae=mean_absolute_error(ytrue[m], yhat[m]),
                n=int(m.sum()), idx=ytrue.index[m])


# ------------------------------------------------------------------ torch modeller
def build_nets(F, L):
    import torch.nn as nn
    import torch

    class RNN(nn.Module):
        def __init__(self, kind):
            super().__init__()
            cell = nn.LSTM if kind == 'lstm' else nn.GRU
            self.rnn = cell(F, 64, batch_first=True)
            self.out = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        def forward(self, x):
            h, _ = self.rnn(x)
            return self.out(h[:, -1]).squeeze(-1)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            ch, layers, cin = 32, [], F
            for dil in (1, 2, 4, 8):
                layers += [nn.ConstantPad1d(((3 - 1) * dil, 0), 0.0),
                           nn.Conv1d(cin, ch, 3, dilation=dil), nn.ReLU()]
                cin = ch
            self.net = nn.Sequential(*layers)
            self.out = nn.Linear(ch, 1)
        def forward(self, x):
            h = self.net(x.transpose(1, 2))
            return self.out(h[:, :, -1]).squeeze(-1)

    class TF(nn.Module):
        def __init__(self):
            super().__init__()
            d = 32
            self.inp = nn.Linear(F, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            enc = nn.TransformerEncoderLayer(d, 4, 64, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, 2)
            self.out = nn.Linear(d, 1)
        def forward(self, x):
            h = self.enc(self.inp(x) + self.pos)
            return self.out(h[:, -1]).squeeze(-1)

    class DLinear(nn.Module):
        """Zeng ve ark. (2023) DLinear: hareketli ortalama ayristirmasi + dogrusal katmanlar."""
        def __init__(self, k=25):
            super().__init__()
            self.k = k
            self.lt = nn.Linear(F * L, 1)
            self.ls = nn.Linear(F * L, 1)
        def forward(self, x):
            pad = (self.k - 1) // 2
            xt = x.transpose(1, 2)
            xp = torch.cat([xt[:, :, :1].repeat(1, 1, pad), xt, xt[:, :, -1:].repeat(1, 1, pad)], 2)
            trend = nn.functional.avg_pool1d(xp, self.k, stride=1)
            seas = xt - trend
            return (self.lt(trend.flatten(1)) + self.ls(seas.flatten(1))).squeeze(-1)

    return {'lstm': lambda: RNN('lstm'), 'gru': lambda: RNN('gru'), 'tcn': TCN,
            'transformer': TF, 'dlinear': DLinear}


def train_predict(name, make_net, Z, y_delta, fit_pos, val_pos, all_pos, seed, max_epochs, patience):
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    offs = torch.arange(-LOOKBACK + 1, 1)
    Zt = torch.from_numpy(Z)
    yt = torch.from_numpy(y_delta.astype(np.float32))

    def batch(pos):
        p = torch.as_tensor(pos)
        return Zt[p[:, None] + offs[None, :]]

    def predict(pos, bs=2048):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(pos), bs):
                out.append(net(batch(pos[i:i + bs])).numpy())
        return np.concatenate(out)

    best, best_state, bad = 1e18, None, 0
    for ep in range(max_epochs):
        net.train()
        perm = np.random.permutation(len(fit_pos))
        for i in range(0, len(perm), 256):
            b = fit_pos[perm[i:i + 256]]
            opt.zero_grad()
            loss = torch.mean((net(batch(b)) - yt[torch.as_tensor(b)]) ** 2)
            loss.backward(); opt.step()
        v = float(np.mean((predict(val_pos) - y_delta[val_pos]) ** 2))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return predict(all_pos)



# ------------------------------------------------------------------ torch yoksa (Termux vb.)
def sk_models(Z, yd, fit_pos, all_pos, seed, L=24):
    """PyTorch kurulamayan ortamlar icin scikit-learn tabanli sinir agi ve DLinear benzeri
    dogrusal taban modeller. Girdi: son L saatlik pencere (duzlestirilmis)."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge
    offs = np.arange(-L + 1, 1)
    def win(pos):
        return Z[pos[:, None] + offs[None, :]]            # (B, L, F)
    def flat(pos):
        return win(pos).reshape(len(pos), -1)
    def decomp(pos, k=13):
        w = win(pos)
        ker = np.ones(k) / k
        pad = (k - 1) // 2
        wp = np.concatenate([np.repeat(w[:, :1], pad, 1), w, np.repeat(w[:, -1:], pad, 1)], 1)
        tr = np.stack([np.apply_along_axis(lambda v: np.convolve(v, ker, 'valid'), 1, wp[:, :, j])
                       for j in range(w.shape[2])], 2)
        return np.concatenate([tr.reshape(len(pos), -1), (w - tr).reshape(len(pos), -1)], 1)
    out = {}
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, validation_fraction=0.15,
                       n_iter_no_change=5, max_iter=60, random_state=seed)
    mlp.fit(flat(fit_pos), yd[fit_pos])
    out['mlp'] = mlp.predict(flat(all_pos))
    rd = Ridge(alpha=10.0).fit(decomp(fit_pos), yd[fit_pos])
    out['dlinear_sk'] = rd.predict(decomp(all_pos))
    return out

# ------------------------------------------------------------------ ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--no-torch', action='store_true', help='yalnizca klasik modeller')
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.patience = 1, 6, 2

    use_torch = not a.no_torch
    if use_torch:
        try:
            import torch  # noqa
        except Exception:
            print('PyTorch bulunamadi: scikit-learn tabanli sinir agi ve DLinear benzeri modellerle devam ediliyor.')
            use_torch = False
    clean, meta = load_clean()
    idx = clean.index; yr = idx.year
    train_base = yr <= C.TRAIN_END
    val_mask = yr == C.VAL_YEAR
    test_mask = yr >= C.TEST_FROM
    stat_masks = {n: (train_base & ~np.isin(yr, meta[n]['bad_years'])) for n in C.SERIES_MAP}
    imp = impute_all(clean, stat_masks)
    hz = {h[0]: h for h in C.HORIZONS}

    rows, dm_rows = [], []
    for tname, hname in CELLS:
        _, kind, param = hz[hname]
        t0 = time.time()
        X = build_features(imp, clean, tname)
        feat_ok = X.notna().all(axis=1)
        anchor = clean[tname].fillna(imp[tname])
        ytrue = make_target(clean[tname], kind, param)
        target_train = train_base & ~np.isin(yr, meta[tname]['bad_years'])
        fit_mask = target_train & feat_ok & ytrue.notna()
        ctx = dict(X=X, y=ytrue, anchor=anchor, fit_mask=fit_mask, index=idx, clean=clean,
                   imp=imp, target=tname, kind=kind, param=param)
        eval_mask = test_mask & feat_ok
        preds = {}
        S = prepare()
        sel = select_one(tname, hname, kind, param, S)
        champ_name = sel['sampiyon']
        print(f"  {tname} {hname}: CV sampiyon {champ_name} (CV R2 {sel['CV_R2']:.3f}, test {sel['R2_test']:.3f})", flush=True)
        preds['champion'] = [sel['pred']]
        final_fit = (yr <= C.VAL_YEAR) & ~np.isin(yr, meta[tname]['bad_years']) & feat_ok & ytrue.notna()
        ctx2 = dict(ctx, fit_mask=final_fit)
        for m in candidates(kind, param):
            if m.name in ('persistence', 'ridge_lags'):
                preds[m.name] = [m.fit_predict(ctx2)]
        # --- derin modeller
        if True:
            cols = [tname] + [c for c in C.COVARIATES if c != tname]
            F = len(cols) + len(CAL)
            Zdf = imp[cols].copy()
            mu = Zdf[stat_masks[tname]].mean(); sd = Zdf[stat_masks[tname]].std() + 1e-9
            Z = np.hstack([((Zdf - mu) / sd).values, clean[CAL].values]).astype(np.float32)
            delta = (ytrue - anchor)
            dsd = float(delta[fit_mask].std() + 1e-9)
            yd = (delta / dsd).fillna(0.0).values
            pos_all = np.arange(len(idx))
            ok_pos = pos_all >= LOOKBACK - 1
            fit_pos = pos_all[fit_mask.values & ok_pos]
            val_pos = pos_all[(val_mask & feat_ok & ytrue.notna()).values & ok_pos]
            all_pos = pos_all[ok_pos]
            if use_torch:
                nn_defs = build_nets(F, LOOKBACK)
                for name, mk in nn_defs.items():
                    preds[name] = []
                    for s_ in range(a.seeds):
                        out = train_predict(name, mk, Z, yd, fit_pos, val_pos, all_pos, 1000 + s_, a.epochs, a.patience)
                        full = pd.Series(np.nan, index=idx)
                        full.iloc[all_pos] = out
                        preds[name].append(anchor + dsd * full)
                    print(f"  {tname} {hname} {name} tamam ({time.time()-t0:.0f}s)", flush=True)
            elif not a.no_torch:
                for s_ in range(a.seeds):
                    res = sk_models(Z, yd, fit_pos, all_pos, 1000 + s_)
                    for name, out in res.items():
                        full = pd.Series(np.nan, index=idx)
                        full.iloc[all_pos] = out
                        preds.setdefault(name, []).append(anchor + dsd * full)
                print(f"  {tname} {hname} sinir agi tabanlari tamam ({time.time()-t0:.0f}s)", flush=True)
        # --- metrikler
        champ_ens = preds['champion'][0]
        ref = metrics(champ_ens, ytrue, eval_mask)
        err_champ = (ytrue[ref['idx']] - champ_ens[ref['idx']]).values
        h_dm = 24 if kind == 'mean' else param
        for name, plist in preds.items():
            ms = [metrics(p, ytrue, eval_mask) for p in plist]
            ens = sum(p for p in plist) / len(plist)
            m_ens = metrics(ens, ytrue, eval_mask)
            assert m_ens['n'] == ref['n']
            err = (ytrue[ref['idx']] - ens[ref['idx']]).values
            if name != 'champion':
                z, p = dm_test(err, err_champ, h_dm)
            else:
                z, p = np.nan, np.nan
            rows.append(dict(hedef=tname, ufuk=hname, model=name, sampiyon=champ_name,
                             R2_mean=np.mean([m['r2'] for m in ms]),
                             R2_sd=np.std([m['r2'] for m in ms], ddof=1) if len(ms) > 1 else np.nan,
                             MAE_mean=np.mean([m['mae'] for m in ms]),
                             R2_ens=m_ens['r2'], n_test=ref['n'], DM_z_vs_champ=z, DM_p_vs_champ=p))
        print(f"{tname} {hname}: tamam ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(C.OUTPUT_DIR / 'dl_baselines.csv', index=False)
    write_latex(df)
    print(df.round(4).to_string(index=False))


NAMES = {'persistence': 'Persistence', 'ridge_lags': 'Ridge regression', 'champion': 'Selected champion (framework)',
         'lstm': 'LSTM', 'gru': 'GRU', 'tcn': 'TCN', 'transformer': 'Transformer encoder', 'dlinear': 'DLinear', 'mlp': 'MLP (scikit-learn)', 'dlinear_sk': 'DLinear-style ridge'}
TLAB = {'karkent_pm10': 'Karkent PM10', 'meram_pm10': 'Meram PM10'}
HLAB = {'t+1': 'Next hour', 'gunluk': 'Daily'}


def write_latex(df):
    cells = list(dict.fromkeys(zip(df.hedef, df.ufuk)))
    lines = ["\\begin{tabular}{l" + "r" * len(cells) + "}", "\\toprule",
             "\\textbf{Model} & " + " & ".join(f"\\textbf{{{TLAB[t]}, {HLAB[h].lower()}}}" for t, h in cells) + " \\\\",
             "\\midrule"]
    for m in NAMES:
        if m not in set(df.model):
            continue
        parts = []
        for t, h in cells:
            r = df[(df.hedef == t) & (df.ufuk == h) & (df.model == m)].iloc[0]
            sd = '' if np.isnan(r.R2_sd) else f"$\\pm${r.R2_sd:.3f}"
            star = ''
            if not np.isnan(r.DM_p_vs_champ) and r.DM_p_vs_champ < 0.05:
                star = '$^{\\mathrm{a}}$' if r.DM_z_vs_champ > 0 else '$^{\\mathrm{b}}$'
            parts.append(f"{r.R2_mean:.3f}{sd}{star}")
        lines.append(f"{NAMES[m]} & " + " & ".join(parts) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (C.OUTPUT_DIR / 'dl_baselines_table.tex').write_text("\n".join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
