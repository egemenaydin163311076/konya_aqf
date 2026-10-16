import os
os.environ['OMP_NUM_THREADS'] = '1'; os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import config as C, proto2 as P2
from multiprocessing import Pool
H = {h[0]: h for h in C.HORIZONS}
TASKS = [('era5','meram_so2','gunluk'),('operational','meram_so2','gunluk')]
def work(t):
    scen, T, hn = t
    from src.meteo import load_cached_features
    m = load_cached_features(scen); t0 = time.time()
    r = P2.run_cell(T, hn, H[hn][1], H[hn][2], meteo=m); r['scenario'] = scen; r['sec'] = round(time.time() - t0)
    return r
if __name__ == '__main__':
    out = C.OUTPUT_DIR / 'proto2_meteo_extra.csv'; rows = []
    with Pool(2) as p:
        for r in p.imap_unordered(work, TASKS):
            print(r, flush=True); rows.append(r); pd.DataFrame(rows).to_csv(out, index=False)
