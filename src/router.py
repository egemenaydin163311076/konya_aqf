"""ForecastRouter — ufuk-başına PARALEL pipeline + otomatik model yönlendirme.

Vizyon: her (hedef, ufuk) bağımsız bir pipeline. Hepsi paralel koşar; her biri kendi
model zoo'sundan CV ile en iyi modeli seçer (saatlikte gbm/xgb, günlükte ridge, aylıkta
mevsimsel... veriye göre). Sonuç bir YÖNLENDİRME TABLOSU; predict() o tabloya göre
ilgili ufkun şampiyonuna route eder.
"""
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import config as C
from src.cv_select import prepare, select_one, CV_FOLDS
from src.models import GBM_NAME, HAS_XGB


class ForecastRouter:
    def __init__(self, targets=None, horizons=None, folds=CV_FOLDS, meteo_feats=None):
        self.targets = targets or C.TARGETS
        self.horizons = horizons or C.HORIZONS
        self.folds = folds
        self.meteo_feats = meteo_feats     # meteo.meteo_features(...) ya da None
        self.routes = {}      # (hedef, ufuk) -> sonuç dict (pred dahil)
        self.meta = None

    def fit(self, parallel=True, max_workers=4, verbose=True):
        S = prepare(meteo_feats=self.meteo_feats); self.meta = S['meta']
        tasks = [(T, hn, k, p) for T in self.targets for hn, k, p in self.horizons]

        def work(task):
            T, hn, k, p = task
            return (T, hn), select_one(T, hn, k, p, S, self.folds)

        if parallel:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = [ex.submit(work, t) for t in tasks]
                for f in as_completed(futs):
                    key, res = f.result()
                    if res:
                        self.routes[key] = res
                        if verbose:
                            print(f"  {key[0]:13s} {key[1]:9s} -> {res['sampiyon']:15s} "
                                  f"CV={res['CV_R2']:.3f} test={res['R2_test']:.3f}")
        else:
            for t in tasks:
                key, res = work(t)
                if res:
                    self.routes[key] = res
        return self

    def routing_table(self):
        """ufuk-başına seçilen model + skorlar (vizyonun somut çıktısı)."""
        rows = [dict(hedef=k[0], ufuk=k[1], secilen_model=v['sampiyon'],
                     CV_R2=v['CV_R2'], R2_test=v['R2_test'], MAE_test=v['MAE_test'])
                for k, v in self.routes.items()]
        order = {h[0]: i for i, h in enumerate(self.horizons)}
        df = pd.DataFrame(rows).sort_values(['hedef', 'ufuk'],
                                            key=lambda s: s.map(order) if s.name == 'ufuk' else s)
        return df.reset_index(drop=True)

    def route(self, target, horizon_name):
        """Bu (hedef, ufuk) için seçilen model adını döndürür."""
        return self.routes[(target, horizon_name)]['sampiyon']

    def predict(self, target, horizon_name):
        """Seçilen şampiyonun tüm-indeks tahmin serisi (test dahil)."""
        return self.routes[(target, horizon_name)]['pred']

    def save(self, path=None):
        path = path or (C.OUTPUT_DIR / 'routing_table.csv')
        C.OUTPUT_DIR.mkdir(exist_ok=True)
        self.routing_table().to_csv(path, index=False)
        return path


if __name__ == '__main__':
    import sys
    from src.meteo import load_cached_features
    # Kullanım:  python -m src.router            (meteosuz)
    #            python -m src.router operational (cache'li operasyonel meteo)
    #            python -m src.router era5        (cache'li üst-sınır meteo)
    source = sys.argv[1] if len(sys.argv) > 1 else None
    mf = None
    if source:
        mf = load_cached_features(source)
        if mf is None:
            print(f"[!] '{source}' meteo cache'i bulunamadı (önce meteo.py ile çek). Meteosuz koşulacak.")
        else:
            print(f"[meteo] '{source}' yüklendi: {mf.shape[1]} sütun, "
                  f"{int(mf.notna().all(axis=1).sum())} tam-dolu saat")
    print(f"GBM backend: {GBM_NAME} | XGBoost: {'var' if HAS_XGB else 'yok'} | "
          f"meteo: {source if mf is not None else 'YOK'}\n")

    r = ForecastRouter(meteo_feats=mf).fit(parallel=True, max_workers=4)
    print("\n================ YÖNLENDİRME TABLOSU (ufuk -> seçilen model) ================")
    print(r.routing_table().to_string(index=False))
    suffix = f'_{source}' if mf is not None else ''
    path = r.save(C.OUTPUT_DIR / f'routing_table{suffix}.csv')
    print("\nKaydedildi:", path)
