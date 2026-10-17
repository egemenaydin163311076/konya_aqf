"""Tum hucrelerde taban modellerinin (persistence, seasonal_naive, climatology) test R2 degerleri, proto2 ile ayni on isleme ve test maskesi."""
import os, sys
os.environ['OMP_NUM_THREADS']='1'
import numpy as np, pandas as pd
sys.path.insert(0,'.')
import config as C, proto2 as P2
from src.features import build_features
from src.targets import make_target
from src.models import candidates
from sklearn.metrics import r2_score
def cell(a):
    T,hn,kind,param=a
    P=P2.prepare_period(C.VAL_YEAR)
    X=build_features(P['imp'],P['clean'],T,meteo_feats=None,kind=kind,param=param)
    y=make_target(P['clean'][T],kind,param); anchor=P['clean'][T].fillna(P['imp'][T])
    y_ev=make_target(P['raw'][T],kind,param); anchor_ev=P['raw'][T].fillna(P['imp'][T])
    yr=P['yr']; pos=np.arange(len(yr)); end_pos=pos[yr<=C.VAL_YEAR].max()
    ok=X.notna().all(axis=1)
    fm=(yr<=C.VAL_YEAR)&~np.isin(yr,P['bad'][T])&ok&y.notna()&(pos+param<=end_pos)
    ctx=dict(X=X,y=y,anchor=anchor,anchor_raw=anchor_ev,fit_mask=fm,index=P['idx'],clean=P['clean'],imp=P['imp'],target=T,kind=kind,param=param)
    out=dict(hedef=T,ufuk=hn)
    ms={m.name:m for m in candidates(kind,param)}
    tm=(yr>=C.TEST_FROM)&ok&y_ev.notna()
    for n in ('persistence','seasonal_naive','climatology'):
        if n in ms:
            p=ms[n].fit_predict(ctx); m=tm&p.notna(); out[n]=round(r2_score(y_ev[m],p[m]),4)
        else: out[n]=np.nan
    # anchor tabanli persistence (son gozlem), havuzda olmayan ufuklar icin referans
    m=tm&anchor_ev.notna(); out['persistence_anchor']=round(r2_score(y_ev[m],anchor_ev[m]),4)
    out['n_test']=int(tm.sum()); print(out,flush=True); return out
if __name__=='__main__':
    from multiprocessing import Pool
    tasks=[(T,hn,k,p) for T in C.TARGETS for hn,k,p in C.HORIZONS]
    with Pool(2) as pool: res=pool.map(cell,tasks,chunksize=1)
    pd.DataFrame(res).to_csv(C.OUTPUT_DIR/'proto2_refs.csv',index=False)
