"""AI-Agent Karar Katmanı — Proof of Concept.

Üç kural-tabanlı yetenek, nowcast + forecast çıktıları üzerine kurulu:
  1) SENSÖR ARIZA/SÜRÜKLENME TESPİTİ: bağımsız MEKÂNSAL nowcast (yalnız DİĞER istasyonlar
     + takvim; hedef sensörün kendi geçmişi KULLANILMAZ) ile gerçek ölçüm arasındaki artık
     robust eşiği aşarsa alarm. Hedef sensör bozulsa bile referans bağımsız kalır.
  2) SANAL SENSÖR: istasyon düştüğünde nowcast ile değeri doldur (kapsama kazancı + hata).
  3) WHO UYARISI: günlük forecast'tan ertesi gün eşik-aşımı alarmı (precision/recall).

PoC kanıtı: test dönemine yapay bir arıza (sabit ofset = sürüklenme) enjekte edip
tespit sisteminin onu yakaladığını gösteriyoruz.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import config as C
from src.cv_select import prepare
from src.models import _gbm


def spatial_nowcast(S, target):
    """Hedef sensörün KENDİ geçmişini kullanmayan bağımsız mekânsal kestirim.
    (Yalnız diğer istasyonların eşzamanlı değerleri + takvim.) Sensör sağlığı referansı."""
    clean, imp, idx, yr = S['clean'], S['imp'], S['idx'], S['yr']
    X = pd.DataFrame(index=idx)
    for s in C.SERIES_MAP:
        if s == target:
            continue
        X[f'{s}_t'] = imp[s]
    for c in ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'year']:
        X[c] = clean[c]
    y = clean[target]
    fok = X.notna().all(axis=1)
    tr = (yr <= C.TRAIN_END) & fok & y.notna()
    m = _gbm(); m.fit(X[tr], y[tr])
    pred = pd.Series(m.predict(X.fillna(0)), index=idx)
    return pred, y, fok, tr


def fault_detector(S, target, k=4.0):
    """Robust artık eşiğiyle sensör arıza/sürüklenme bayrağı.
    eşik = medyan(|artık_train|) + k·1.4826·MAD(artık_train)."""
    pred, y, fok, tr = spatial_nowcast(S, target)
    resid = (y - pred)
    rt = resid[tr & resid.notna()]
    med = rt.median(); mad = (rt - med).abs().median()
    thr = abs(med) + k * 1.4826 * mad
    return pred, y, resid, thr, fok


def run_poc(target='karkent_pm10'):
    S = prepare()
    yr = S['yr']
    test = np.asarray(yr >= C.TEST_FROM)
    pred, y, resid, thr, fok = fault_detector(S, target)
    fok = np.asarray(fok)
    trv = np.asarray(yr <= C.TRAIN_END)

    # sürekli arıza için yuvarlanan-ortalama artık eşiği (CUSUM benzeri, gürültüyü siler)
    roll = resid.rolling(24, min_periods=12).mean()
    rr = roll[trv & roll.notna().values]
    thr_roll = abs(rr.median()) + 4.0 * 1.4826 * (rr - rr.median()).abs().median()

    print(f"=== AI-AGENT PoC | hedef: {target} ===\n")
    print(f"[1] SENSÖR ARIZA TESPİTİ  (bağımsız mekânsal referans)")
    print(f"    anlık eşik |artık|>{thr:.1f} | sürekli eşik |24s-ort artık|>{thr_roll:.1f} µg/m³")
    tm = test & fok & y.notna().values
    flags_i = (resid.abs().values > thr) & tm
    flags_r = (roll.abs().values > thr_roll) & tm
    print(f"    test saati {int(tm.sum())} | doğal alarm: anlık %{100*flags_i.sum()/tm.sum():.2f}, "
          f"sürekli %{100*flags_r.sum()/tm.sum():.2f}")

    print(f"\n[1b] KANIT — yapay sürüklenme enjeksiyonu (yakalama / yanlış-alarm)")
    test_idx = y.index[tm]
    win = test_idx[500:500 + 168]
    win_set = y.index.isin(win)
    for drift in [20, 40, 80]:
        yf = y.copy(); yf.loc[win] += drift
        rf = (yf - pred)
        rollf = rf.rolling(24, min_periods=12).mean()
        rec_i = (rf.loc[win].abs() > thr).mean()
        rec_r = (rollf.loc[win].abs() > thr_roll).mean()
        outside = tm & ~win_set
        fp_i = ((rf.abs().values > thr) & outside).sum() / outside.sum()
        fp_r = ((rollf.abs().values > thr_roll) & outside).sum() / outside.sum()
        print(f"    +{drift:>2}: ANLIK recall %{100*rec_i:.0f}/FP %{100*fp_i:.1f}  |  "
              f"SÜREKLI recall %{100*rec_r:.0f}/FP %{100*fp_r:.2f}")

    print(f"\n[2] SANAL SENSÖR (istasyon düştüğünde doldurma)")
    obs_rate = y.values[test].__ne__(y.values[test]).mean()  # placeholder
    mm = tm
    mae = mean_absolute_error(y.values[mm], pred.values[mm])
    cov = np.isfinite(y.values[test]).mean()
    print(f"    gerçek kapsama %{100*cov:.1f} -> nowcast ile %100 | "
          f"yeniden-üretim MAE {mae:.2f} (ort {y.values[mm].mean():.1f})")

    print(f"\n[3] WHO ERTESİ-GÜN UYARISI")
    print(f"    who_classification.csv: ort. doğruluk %86.6 (PM10 F1≈0.66-0.75, NO2 F1≈0.90)")

    out = pd.DataFrame({'ts': y.index, 'gercek': y.values, 'nowcast': pred.values,
                        'artik': resid.values, 'artik_24s': roll.values})
    out['alarm_surekli'] = (roll.abs().values > thr_roll) & test & fok & y.notna().values
    out[tm].to_csv(C.OUTPUT_DIR / f'agent_faultlog_{target}.csv', index=False)
    print(f"\nKaydedildi: agent_faultlog_{target}.csv")
    return dict(thr=float(thr), thr_roll=float(thr_roll), mae=float(mae))


if __name__ == '__main__':
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    run_poc('karkent_pm10')
