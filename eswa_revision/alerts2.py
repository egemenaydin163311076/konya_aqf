"""Gun-oncesi esik asimi uyarilari: ana protokolun (proto2) 'gunluk' ufku sampiyonundan, her gunun 23:00 kaynak saatinde
verilen ertesi-gun ortalama tahmini WHO 24 saatlik sinirla karsilastirilir. Gerceklesen gun yalnizca ertesi takvim gununde >=18 gozlenmis saat varsa degerlendirilir."""
import os, sys
os.environ['OMP_NUM_THREADS']='1'
import numpy as np, pandas as pd
sys.path.insert(0,'.')
import config as C, proto2 as P2
LIM={'PM10':45,'SO2':40,'NO2':25}
CELLS=[('karkent_pm10','Karkent','PM10'),('karkent_so2','Karkent','SO2'),('karkent_no2','Karkent','NO2'),
       ('meram_pm10','Meram','PM10'),('meram_so2','Meram','SO2'),('meram_no2','Meram','NO2')]
def one(a):
    T,st,p=a
    r=P2.run_cell(T,'gunluk','mean',24,return_pred=True)
    P=P2.prepare_period(C.VAL_YEAR)
    obs=P['clean'][T].notna().astype(float)
    cnt=obs.shift(-1)[::-1].rolling(24,min_periods=1).sum()[::-1]   # (t,t+24] icindeki gozlenmis saat sayisi
    idx=P['idx']; m=r['test_mask']&(idx.hour==23)&(cnt>=18)&r['pred'].notna()
    y=r['y'][m]; pr=r['pred'][m]; thr=LIM[p]
    yt=(y>thr).astype(int); yp=(pr>thr).astype(int)
    tp=int(((yp==1)&(yt==1)).sum()); fp=int(((yp==1)&(yt==0)).sum()); tn=int(((yp==0)&(yt==0)).sum()); fn=int(((yp==0)&(yt==1)).sum())
    n=tp+fp+tn+fn; acc=(tp+tn)/n; base=(tp+fn)/n
    prec=tp/(tp+fp) if tp+fp else np.nan; rec=tp/(tp+fn) if tp+fn else np.nan
    f1=2*prec*rec/(prec+rec) if prec==prec and rec==rec and prec+rec>0 else np.nan
    out=dict(istasyon=st,kirletici=p,sampiyon=r['sampiyon'],limit=thr,n_gun=n,acc=round(acc,3),prec=round(prec,3) if prec==prec else np.nan,
             rec=round(rec,3) if rec==rec else np.nan,f1=round(f1,3) if f1==f1 else np.nan,base=round(base,3),always=round(base,3),never=round(1-base,3),TP=tp,FP=fp,TN=tn,FN=fn,
             R2_gunluk=r['R2_test'])
    print(out,flush=True); return out
if __name__=='__main__':
    from multiprocessing import Pool
    with Pool(1) as pool: res=pool.map(one,CELLS,chunksize=1)
    pd.DataFrame(res).to_csv(C.OUTPUT_DIR/'alerts_proto2.csv',index=False)
