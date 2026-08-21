"""Yayın-kalitesinde figürler — outputs/*.csv dosyalarından üretir.
Tek başına: python src/plots.py   (önce diğer adımlar koşmuş olmalı)
Hangi CSV varsa o figürü üretir; eksikse atlar (uyarı verir)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as C
FIG = C.OUTPUT_DIR / 'figures'
HORIZON_ORDER = ['t+1', 't+6', 't+24', 'gunluk', 'haftalik', 'aylik']
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})


def _csv(name):
    p = C.OUTPUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def fig_availability():
    """1) Veri doluluk ısı haritası (istasyon×kirletici×yıl) — ham veriden."""
    from src.data import clean
    df, meta = clean()
    yr = df.index.year
    series = [c for c in C.SERIES_MAP]
    years = sorted(set(yr))
    M = np.array([[df.loc[yr == y, s].notna().mean() * 100 for y in years] for s in series])
    fig, ax = plt.subplots(figsize=(1.1 * len(years) + 2, 0.42 * len(series) + 1.5))
    im = ax.imshow(M, aspect='auto', cmap='YlGnBu', vmin=0, vmax=100)
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years, rotation=0)
    ax.set_yticks(range(len(series))); ax.set_yticklabels(series, fontsize=8)
    for i in range(len(series)):
        for j in range(len(years)):
            ax.text(j, i, f'{M[i,j]:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if M[i, j] > 55 else 'black')
    ax.set_title('Veri doluluk oranı (%) — istasyon×kirletici×yıl')
    fig.colorbar(im, ax=ax, label='% gözlem'); fig.tight_layout()
    fig.savefig(FIG / '1_doluluk_isi.png'); plt.close(fig)
    return '1_doluluk_isi.png'


def fig_horizon_curve():
    """2) Ufuk-bozulma eğrisi: R² vs ufuk, forecasting vs nowcasting (ANA figür)."""
    fc = _csv('routing_table.csv')
    if fc is None:
        fc = _csv('champions_cv.csv')
    if fc is None:
        return None
    rcol = 'R2_test'; mcol = 'secilen_model' if 'secilen_model' in fc else 'sampiyon'
    nc = _csv('nowcast_results.csv')
    targets = sorted(fc['hedef'].unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(2.7 * len(targets) + 1, 3.4), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for ax, t in zip(axes, targets):
        sub = fc[fc['hedef'] == t].set_index('ufuk').reindex(HORIZON_ORDER)
        ax.plot(range(len(HORIZON_ORDER)), sub[rcol].values, 'o-', label='Forecasting', color='#1f4e79')
        if nc is not None and t in set(nc['hedef']):
            ncs = nc[nc['hedef'] == t].iloc[0]
            ncmap = {'t+1': ncs.get('R2_saatlik'), 'gunluk': ncs.get('R2_gunluk'),
                     'haftalik': ncs.get('R2_haftalik'), 'aylik': ncs.get('R2_aylik')}
            xs = [i for i, h in enumerate(HORIZON_ORDER) if ncmap.get(h) is not None]
            ys = [ncmap[HORIZON_ORDER[i]] for i in xs]
            ax.plot(xs, ys, 's--', label='Nowcasting', color='#c00000')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_xticks(range(len(HORIZON_ORDER))); ax.set_xticklabels(HORIZON_ORDER, rotation=45, fontsize=7)
        ax.set_title(t, fontsize=9); ax.set_ylim(-0.3, 1.0)
    axes[0].set_ylabel('R² (test)'); axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Ufuk-bozulma: Forecasting vs Nowcasting', y=1.02)
    fig.tight_layout(); fig.savefig(FIG / '2_ufuk_egrisi.png', bbox_inches='tight'); plt.close(fig)
    return '2_ufuk_egrisi.png'


def fig_meteo():
    """3) Meteo katkısı: günlük R² temel vs era5 vs operational."""
    m = _csv('meteo_karsilastirma.csv')
    if m is None:
        return None
    sub = m[m['ufuk'].isin(['t+24', 'gunluk'])]
    if sub.empty:
        return None
    labels = sub['hedef'] + '\n' + sub['ufuk']
    x = np.arange(len(sub)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.4 * len(sub) + 2, 3.6))
    ax.bar(x - w, sub['R2_temel'], w, label='Temel (meteosuz)', color='#9dc3e6')
    if 'R2_era5' in sub:
        ax.bar(x, sub['R2_era5'], w, label='ERA5 (üst-sınır)', color='#1f4e79')
    if 'R2_oper' in sub and sub['R2_oper'].notna().any():
        ax.bar(x + w, sub['R2_oper'], w, label='Operasyonel', color='#c55a11')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('R² (test)'); ax.set_title('Meteorolojinin katkısı (≤günlük ufuk)')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / '3_meteo_katki.png'); plt.close(fig)
    return '3_meteo_katki.png'


def fig_champion_heat():
    """4) Şampiyon ısı haritası: hedef×ufuk -> model ailesi."""
    fc = _csv('routing_table.csv')
    if fc is None:
        fc = _csv('champions_cv.csv')
    if fc is None:
        return None
    mcol = 'secilen_model' if 'secilen_model' in fc else 'sampiyon'
    fams = ['persistence', 'seasonal_naive', 'climatology', 'ridge_lags',
            'detrend_seasonal', 'gbm_delta', 'gbm_level', 'xgb_delta', 'xgb_level']
    fid = {f: i for i, f in enumerate(fams)}
    targets = sorted(fc['hedef'].unique())
    M = np.full((len(targets), len(HORIZON_ORDER)), np.nan)
    lab = {}
    for i, t in enumerate(targets):
        for j, h in enumerate(HORIZON_ORDER):
            r = fc[(fc['hedef'] == t) & (fc['ufuk'] == h)]
            if not r.empty:
                mdl = r.iloc[0][mcol]; M[i, j] = fid.get(mdl, -1); lab[(i, j)] = mdl
    fig, ax = plt.subplots(figsize=(1.0 * len(HORIZON_ORDER) + 3, 0.5 * len(targets) + 2))
    im = ax.imshow(M, aspect='auto', cmap='tab10', vmin=0, vmax=9)
    ax.set_xticks(range(len(HORIZON_ORDER))); ax.set_xticklabels(HORIZON_ORDER, rotation=45, fontsize=8)
    ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets, fontsize=8)
    for (i, j), mdl in lab.items():
        ax.text(j, i, mdl.replace('_', '\n'), ha='center', va='center', fontsize=5.5)
    ax.set_title('Ufuk-başına seçilen model (şampiyon)'); fig.tight_layout()
    fig.savefig(FIG / '4_sampiyon_isi.png'); plt.close(fig)
    return '4_sampiyon_isi.png'


def fig_who():
    """5) WHO karışıklık matrisleri (PM10/NO2)."""
    w = _csv('who_classification.csv')
    if w is None:
        return None
    sel = w[w['kirletici'].isin(['PM10', 'NO2'])].reset_index(drop=True)
    if sel.empty:
        return None
    n = len(sel); fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.6))
    if n == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, sel.iterrows()):
        cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
        im = ax.imshow(cm, cmap='Blues')
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha='center', va='center',
                    color='white' if v > cm.max() / 2 else 'black', fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Aşım yok', 'Aşım'], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['Aşım yok', 'Aşım'], fontsize=7)
        ax.set_xlabel('Tahmin'); ax.set_ylabel('Gerçek')
        ax.set_title(f"{r['istasyon']} {r['kirletici']}\nacc={r['WHO_acc']:.2f} F1={r.get('WHO_F1','-')}", fontsize=8)
    fig.suptitle('WHO eşik-aşımı sınıflandırma (test)', y=1.04)
    fig.tight_layout(); fig.savefig(FIG / '5_who_confusion.png', bbox_inches='tight'); plt.close(fig)
    return '5_who_confusion.png'


def fig_agent():
    """7) Agent PoC: artık zaman serisi + alarm eşiği."""
    import glob
    files = glob.glob(str(C.OUTPUT_DIR / 'agent_faultlog_*.csv'))
    if not files:
        return None
    d = pd.read_csv(files[0], parse_dates=['ts']).set_index('ts').iloc[:800]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(d.index, d['gercek'], lw=0.7, label='Gerçek ölçüm', color='#333')
    ax.plot(d.index, d['nowcast'], lw=0.9, label='Mekânsal nowcast', color='#1f4e79')
    if 'alarm_surekli' in d:
        al = d[d['alarm_surekli'] == True]
        ax.scatter(al.index, al['gercek'], s=14, color='#c00000', zorder=5, label='Alarm')
    ax.set_title('Agent: sensör sağlığı (gerçek vs bağımsız nowcast)')
    ax.set_ylabel('µg/m³'); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG / '7_agent_residual.png'); plt.close(fig)
    return '7_agent_residual.png'


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in [fig_availability, fig_horizon_curve, fig_meteo,
               fig_champion_heat, fig_who, fig_agent]:
        try:
            r = fn()
            if r:
                made.append(r); print(f"  ✓ {r}")
            else:
                print(f"  – {fn.__name__}: gerekli CSV yok, atlandı")
        except Exception as e:
            print(f"  ! {fn.__name__}: hata ({e})")
    print(f"\n{len(made)} figür üretildi -> {FIG}")


if __name__ == '__main__':
    main()
