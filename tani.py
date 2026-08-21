# Operasyonel meteo tanı script'i — proje kökünde çalıştır: python tani.py
import os, pandas as pd, numpy as np
print("=== 1) Operasyonel meteo cache var mı? ===")
for s in ['karkent','meram']:
    p=f'outputs/meteo_operational_{s}.csv'
    if os.path.exists(p):
        d=pd.read_csv(p, parse_dates=['ts']).set_index('ts')
        print(f"  {p}: VAR | {d.shape[0]} satır | {d.index.min()} -> {d.index.max()}")
        print(f"     sütunlar: {list(d.columns)}")
        print(f"     test dönemi (2025+) satır: {(d.index>='2025-01-01').sum()} | NaN oranı: {d[d.index>='2025-01-01'].isna().mean().mean():.2%}")
    else:
        print(f"  {p}: YOK")
print("\n=== 2) ERA5 cache (kıyas için) ===")
for s in ['karkent','meram']:
    p=f'outputs/meteo_era5_{s}.csv'
    print(f"  {p}: {'VAR' if os.path.exists(p) else 'YOK'}")
print("\n=== 3) routing_table_operational.csv içeriği ===")
p='outputs/routing_table_operational.csv'
if os.path.exists(p):
    d=pd.read_csv(p)
    print(f"  satır sayısı: {len(d)}")
    print(f"  ufuklar: {sorted(d['ufuk'].unique())}")
    sub=d[d['ufuk'].isin(['t+1','t+6','t+24','gunluk'])]
    print(f"  ≤günlük satır sayısı: {len(sub)} (0 ise operasyonel ≤günlük koşmamış)")
else:
    print("  YOK — operasyonel routing hiç üretilmemiş")
