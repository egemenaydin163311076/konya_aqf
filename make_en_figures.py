"""İngilizce etiketli yayın figürleri — yetkili nihai verilerden.
Kaynaklar: /tmp/meteo_recon.csv (kullanıcının nihai forecasting sonucu),
outputs/nowcast_results.csv, outputs/who_classification.csv, ana xlsx (doluluk)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config as C

OUT = '/tmp/figs_en'; os.makedirs(OUT, exist_ok=True)
HO = ['t+1', 't+6', 't+24', 'gunluk', 'haftalik', 'aylik']
HO_EN = ['t+1h', 't+6h', 't+24h', 'daily', 'weekly', 'monthly']
plt.rcParams.update({'figure.dpi': 150, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False})
meteo = pd.read_csv('/tmp/meteo_recon.csv')
TARGETS = ['karkent_pm10', 'karkent_so2', 'meram_co', 'meram_pm10', 'meram_so2']
TLAB = {'karkent_pm10': 'Karkent PM$_{10}$', 'karkent_so2': 'Karkent SO$_2$',
        'meram_co': 'Meram CO', 'meram_pm10': 'Meram PM$_{10}$', 'meram_so2': 'Meram SO$_2$'}


def fig1_availability():
    from src.data import clean
    df, _ = clean(); yr = df.index.year
    series = list(C.SERIES_MAP); years = sorted(set(yr))
    M = np.array([[df.loc[yr == y, s].notna().mean()*100 for y in years] for s in series])
    lbl = [s.replace('karkent_', 'Karkent ').replace('meram_', 'Meram ').upper()
           .replace('KARKENT', 'Karkent').replace('MERAM', 'Meram') for s in series]
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(M, aspect='auto', cmap='YlGnBu', vmin=0, vmax=100)
    ax.set_xticks(range(len(years))); ax.set_xticklabels(years)
    ax.set_yticks(range(len(series))); ax.set_yticklabels(lbl, fontsize=8)
    for i in range(len(series)):
        for j in range(len(years)):
            ax.text(j, i, f'{M[i,j]:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if M[i, j] > 55 else 'black')
    ax.set_title('Data availability (%) by station-pollutant series and year')
    fig.colorbar(im, ax=ax, label='% observed'); fig.tight_layout()
    fig.savefig(f'{OUT}/fig1_availability.png'); plt.close(fig)


def fig2_horizon():
    nc = pd.read_csv('outputs/nowcast_results.csv')
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(15, 3.4), sharey=True)
    for ax, t in zip(axes, TARGETS):
        sub = meteo[meteo['hedef'] == t].set_index('ufuk').reindex(HO)
        ax.plot(range(len(HO)), sub['R2_temel'].values, 'o-', color='#1f4e79', label='Forecasting')
        if t in set(nc['hedef']):
            r = nc[nc['hedef'] == t].iloc[0]
            nmap = {'t+1': r.get('R2_saatlik'), 'gunluk': r.get('R2_gunluk'),
                    'haftalik': r.get('R2_haftalik'), 'aylik': r.get('R2_aylik')}
            xs = [i for i, h in enumerate(HO) if pd.notna(nmap.get(h))]
            ax.plot(xs, [nmap[HO[i]] for i in xs], 's--', color='#c00000', label='Nowcasting')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_xticks(range(len(HO))); ax.set_xticklabels(HO_EN, rotation=45, fontsize=7)
        ax.set_title(TLAB[t], fontsize=9); ax.set_ylim(-0.3, 1.0)
    axes[0].set_ylabel('R² (test)'); axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Skill decay across horizons: forecasting vs nowcasting', y=1.02)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig2_horizon.png', bbox_inches='tight'); plt.close(fig)


def fig3_meteo():
    sub = meteo[meteo['ufuk'].isin(['t+24', 'gunluk'])].copy()
    sub['lab'] = sub['hedef'].map(lambda x: TLAB[x].replace('$_{10}$', '10').replace('$_2$', '2')) \
        + '\n' + sub['ufuk'].map({'t+24': 't+24h', 'gunluk': 'daily'})
    x = np.arange(len(sub)); w = 0.38
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.bar(x - w/2, sub['R2_temel'], w, label='Baseline (no meteorology)', color='#9dc3e6')
    ax.bar(x + w/2, sub['R2_era5'], w, label='With ERA5 meteorology', color='#1f4e79')
    ax.set_xticks(x); ax.set_xticklabels(sub['lab'], fontsize=7)
    ax.set_ylabel('R² (test)'); ax.set_title('Contribution of meteorology (≤ daily horizons)')
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f'{OUT}/fig3_meteo.png'); plt.close(fig)


def fig4_champion():
    fams = ['persistence', 'seasonal_naive', 'climatology', 'ridge_lags',
            'detrend_seasonal', 'gbm_delta', 'gbm_level', 'xgb_delta', 'xgb_level']
    fid = {f: i for i, f in enumerate(fams)}
    M = np.full((len(TARGETS), len(HO)), np.nan); lab = {}
    for i, t in enumerate(TARGETS):
        for j, h in enumerate(HO):
            r = meteo[(meteo['hedef'] == t) & (meteo['ufuk'] == h)]
            if not r.empty:
                m = r.iloc[0]['model_era5']; M[i, j] = fid.get(m, -1); lab[(i, j)] = m
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(M, aspect='auto', cmap='tab10', vmin=0, vmax=9)
    ax.set_xticks(range(len(HO))); ax.set_xticklabels(HO_EN, rotation=45, fontsize=9)
    ax.set_yticks(range(len(TARGETS))); ax.set_yticklabels([TLAB[t] for t in TARGETS], fontsize=9)
    for (i, j), m in lab.items():
        ax.text(j, i, m.replace('_', '\n'), ha='center', va='center', fontsize=6)
    ax.set_title('Selected champion model per horizon')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig4_champion.png'); plt.close(fig)


def fig5_who():
    w = pd.read_csv('outputs/who_classification.csv')
    sel = w[w['kirletici'].isin(['PM10', 'NO2'])].reset_index(drop=True)
    n = len(sel); fig, axes = plt.subplots(1, n, figsize=(2.5*n, 2.7))
    if n == 1: axes = [axes]
    for ax, (_, r) in zip(axes, sel.iterrows()):
        cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
        ax.imshow(cm, cmap='Blues')
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha='center', va='center',
                    color='white' if v > cm.max()/2 else 'black', fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['No exc.', 'Exceed'], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(['No exc.', 'Exceed'], fontsize=7)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
        poll = 'PM$_{10}$' if r['kirletici'] == 'PM10' else 'NO$_2$'
        st = r['istasyon']
        ax.set_title(f"{st} {poll}\nacc={r['WHO_acc']:.2f}, F1={r['WHO_F1']:.2f}", fontsize=8)
    fig.suptitle('WHO 24-h exceedance classification (test period)', y=1.05)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig5_who.png', bbox_inches='tight'); plt.close(fig)


for fn in [fig1_availability, fig2_horizon, fig3_meteo, fig4_champion, fig5_who]:
    try:
        fn(); print(f'  ✓ {fn.__name__}')
    except Exception as e:
        print(f'  ! {fn.__name__}: {e}')
print('İngilizce figürler ->', OUT)
