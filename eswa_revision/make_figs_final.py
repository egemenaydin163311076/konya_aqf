"""Figures 4-7 of the manuscript, generated only from outputs/*.csv of the final protocol."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
O='outputs/'; F='figs_final/'
plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.grid':True,'grid.alpha':.3,'axes.spines.top':False,'axes.spines.right':False})
b=pd.read_csv(O+'proto2_baseline.csv'); era=pd.read_csv(O+'proto2_era5.csv'); op=pd.read_csv(O+'proto2_operational.csv')
nw=pd.read_csv(O+'nowcast_proto2.csv'); al=pd.read_csv(O+'alerts_proto2.csv')
HO=['t+1','t+6','t+24','gunluk','haftalik','aylik']; HOL=['t+1','t+6','t+24','Daily','Weekly','Monthly']
TL={'karkent_pm10':'Karkent PM10','karkent_so2':'Karkent SO$_2$','meram_co':'Meram CO','meram_pm10':'Meram PM10','meram_so2':'Meram SO$_2$'}
# Figure 4: forecasting (hourly=t+1, daily, weekly, monthly) vs nowcasting (champion by validation)
fig,axes=plt.subplots(1,5,figsize=(16,3.6),sharey=True)
sc=['t+1','gunluk','haftalik','aylik']; sl=['Hourly','Daily','Weekly','Monthly']
for ax,t in zip(axes,TL):
    f=[b[(b.hedef==t)&(b.ufuk==h)].R2_test.iloc[0] for h in sc]
    s=nw[nw.hedef==t]; c=s.loc[s.R2_val_saatlik.idxmax()]
    n=[c.R2_saatlik,c.R2_gunluk,c.R2_haftalik,c.R2_aylik]
    ax.plot(range(4),f,'o-',color='#1f4e79',lw=2,label='Forecasting'); ax.plot(range(4),n,'s--',color='#c0392b',lw=2,label='Nowcasting')
    ax.axhline(0,color='gray',lw=.6); ax.set_xticks(range(4)); ax.set_xticklabels(sl,rotation=30); ax.set_title(TL[t],fontsize=10)
axes[0].set_ylabel('R² (test)'); axes[0].legend(fontsize=8,loc='lower left')
fig.suptitle('Forecasting and concurrent spatial reconstruction skill by aggregation scale'); fig.tight_layout(); fig.savefig(F+'Figure_4.png',bbox_inches='tight'); plt.close()
# Figure 5
cells=[('karkent_pm10','t+24'),('karkent_pm10','gunluk'),('karkent_so2','t+24'),('meram_pm10','t+24'),('meram_pm10','gunluk'),('meram_co','t+6'),('meram_so2','t+24'),('meram_so2','gunluk')]
g=lambda d,t,h: d[(d.hedef==t)&(d.ufuk==h)].R2_test.iloc[0]
lab=[f"{TL[t]}\n{dict(zip(HO,HOL))[h]}" for t,h in cells]; x=np.arange(len(cells)); w=.27
fig,ax=plt.subplots(figsize=(13,4.4))
ax.bar(x-w,[g(b,*c) for c in cells],w,color='#aac4e0',label='Baseline (no meteorology)')
ax.bar(x,[g(era,*c) for c in cells],w,color='#1f4e79',label='ERA5 reanalysis')
ax.bar(x+w,[g(op,*c) for c in cells],w,color='#c55a11',label='Operational (day-ahead forecasts)')
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=8.5); ax.set_ylabel('R² (test)'); ax.axhline(0,color='k',lw=.6)
ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.14),frameon=False); fig.tight_layout(); fig.savefig(F+'Figure_5.png',bbox_inches='tight'); plt.close()
# Figure 6
nm={'gbm_delta':'GBM-Δ','gbm_level':'GBM-level','ridge_lags':'Ridge','seasonal_naive':'Seas.-naive','detrend_seasonal':'Detrend-seas.','persistence':'Persistence','climatology':'Climatology'}
col={'gbm_delta':'#8c564b','gbm_level':'#e377c2','ridge_lags':'#d62728','seasonal_naive':'#ff7f0e','detrend_seasonal':'#2ca02c','persistence':'#7f7f7f','climatology':'#17becf'}
T6=['karkent_pm10','karkent_so2','meram_co','meram_pm10','meram_so2']
fig,ax=plt.subplots(figsize=(10,4.4))
for i,t in enumerate(T6):
    for j,h in enumerate(HO):
        m=b[(b.hedef==t)&(b.ufuk==h)].sampiyon.iloc[0]
        ax.add_patch(plt.Rectangle((j-.5,i-.5),.97,.97,color=col.get(m,'#999')))
        ax.text(j,i,nm.get(m,m),ha='center',va='center',color='white',fontsize=9)
ax.set_xlim(-.5,5.5); ax.set_ylim(4.5,-.5); ax.grid(False)
ax.set_xticks(range(6)); ax.set_xticklabels(HOL); ax.set_yticks(range(5)); ax.set_yticklabels([TL[t] for t in T6])
for s in ax.spines.values(): s.set_visible(False)
ax.set_title('Champion model selected by horizon (no-meteorology scenario)'); fig.tight_layout(); fig.savefig(F+'Figure_6.png',bbox_inches='tight'); plt.close()
# Figure 7: all six panels from alerts_proto2.csv (same file as Table 4)
fig,axes=plt.subplots(2,3,figsize=(13,7.6))
order=[('Karkent','PM10'),('Meram','PM10'),('Karkent','SO2'),('Meram','SO2'),('Karkent','NO2'),('Meram','NO2')]
for ax,(s,p) in zip(axes.ravel(),order):
    r=al[(al.istasyon==s)&(al.kirletici==p)].iloc[0]
    cm=np.array([[r.TN,r.FP],[r.FN,r.TP]]); ax.grid(False)
    ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(cm.max(),1))
    for i in range(2):
        for j in range(2): ax.text(j,i,int(cm[i,j]),ha='center',va='center',fontsize=14,color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set_xticks([0,1]); ax.set_xticklabels(['No exceed.','Exceed.']); ax.set_yticks([0,1]); ax.set_yticklabels(['No exceed.','Exceed.'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
    f1='undefined' if pd.isna(r.f1) else f'{r.f1:.2f}'
    ax.set_title(f"{s} {p.replace('SO2','SO$_2$').replace('NO2','NO$_2$')}  (n={int(r.n_gun)} days)\nAcc={r.acc:.2f}  F1={f1}",fontsize=10)
fig.tight_layout(); fig.savefig(F+'Figure_7.png',bbox_inches='tight'); plt.close()
print('ok')
