"""Hedef pencerelerinin CV katlari ve nihai egitim sinirini asip asmadigini olcer:
orijinal (purge yok) ile purge'lu (egitim satirlarinin hedef penceresi sinirin icinde) karsilastirir."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,'.')
import config as C
import baselines_dl as B                      # load_clean yamasi
import src.cv_select as cv
from src.features import build_features
from src.targets import make_target
from src.models import candidates

def select(T, hname, kind, param, S, purge, folds=cv.CV_FOLDS):
    clean, meta, imp, idx, yr = S['clean'], S['meta'], S['imp'], S['idx'], S['yr']
    X = build_features(imp, clean, T); feat_ok = X.notna().all(axis=1)
    anchor = clean[T].fillna(imp[T]); bad = meta[T]['bad_years']
    y = make_target(clean[T], kind, param)
    span = param  # saat: nokta icin h, ortalama icin k
    pos = pd.Series(np.arange(len(idx)), index=idx)
    def fitm(end_year):
        m = (yr <= end_year) & ~np.isin(yr, bad) & feat_ok & y.notna()
        if purge:
            end_pos = pos[yr <= end_year].max()
            m = m & (pos.values + span <= end_pos)
        return m
    scored=[]
    for model in candidates(kind, param):
        fr=[]
        for te, va in folds:
            if va in bad: continue
            fm = fitm(te)
            if fm.sum()<500: continue
            ctx=dict(X=X,y=y,anchor=anchor,fit_mask=fm,index=idx,clean=clean,imp=imp,target=T,kind=kind,param=param)
            try: r=cv._r2(model.fit_predict(ctx),y,(yr==va)&feat_ok)
            except Exception: r=np.nan
            if r==r: fr.append(r)
        if fr: scored.append((model,float(np.mean(fr))))
    champ,cvr=max(scored,key=lambda s:s[1])
    ctx=dict(X=X,y=y,anchor=anchor,fit_mask=fitm(C.VAL_YEAR),index=idx,clean=clean,imp=imp,target=T,kind=kind,param=param)
    r2=cv._r2(champ.fit_predict(ctx),y,(yr>=C.TEST_FROM)&feat_ok)
    return champ.name, cvr, r2

if __name__=='__main__':
    T=sys.argv[1]
    cv.load_clean=B.load_clean
    S=cv.prepare()
    out=[]
    for hname,kind,param in sorted(C.HORIZONS,key=lambda h:-h[2] if h[1]=='mean' else 0):
        a=select(T,hname,kind,param,S,False); b=select(T,hname,kind,param,S,True)
        out.append(dict(hedef=T,ufuk=hname,champ_orig=a[0],cv_orig=round(a[1],4),test_orig=round(a[2],4),
                        champ_purged=b[0],cv_purged=round(b[1],4),test_purged=round(b[2],4)))
        print(out[-1],flush=True); pd.DataFrame(out).to_csv(f'outputs/audit_purge_{T}.csv',index=False)
