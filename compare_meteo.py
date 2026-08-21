"""Akademik (ERA5 üst-sınır) vs Operasyonel (arşiv-tahmin) meteo senaryolarını kıyaslar.

Önce şunları üret:
    python src/router.py                -> outputs/routing_table.csv            (meteosuz temel)
    python src/router.py era5           -> outputs/routing_table_era5.csv       (üst-sınır)
    python src/router.py operational    -> outputs/routing_table_operational.csv(operasyonel)
Sonra:
    python compare_meteo.py

Üretir: meteo_karsilastirma.csv + ekrana özet.
  meteo_kazanc_era5  = R2(era5) - R2(temel)           (havanın teorik katkısı)
  meteo_kazanc_oper  = R2(operational) - R2(temel)    (sahadaki gerçek katkı)
  tahmin_maliyeti    = R2(era5) - R2(operational)     (hava TAHMİN hatasının maliyeti)
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'outputs')


def _load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)[['hedef', 'ufuk', 'secilen_model', 'R2_test']]
    return df.rename(columns={'secilen_model': f'model', 'R2_test': 'R2'})


def main():
    base = _load('routing_table.csv')
    era5 = _load('routing_table_era5.csv')
    oper = _load('routing_table_operational.csv')

    if base is None:
        print("[!] routing_table.csv yok. Önce: python src/router.py")
        return
    m = base.rename(columns={'R2': 'R2_temel', 'model': 'model_temel'})
    if era5 is not None:
        m = m.merge(era5.rename(columns={'R2': 'R2_era5', 'model': 'model_era5'}),
                    on=['hedef', 'ufuk'], how='left')
        m['meteo_kazanc_era5'] = (m['R2_era5'] - m['R2_temel']).round(3)
    if oper is not None:
        m = m.merge(oper.rename(columns={'R2': 'R2_oper', 'model': 'model_oper'}),
                    on=['hedef', 'ufuk'], how='left')
        m['meteo_kazanc_oper'] = (m['R2_oper'] - m['R2_temel']).round(3)
    if era5 is not None and oper is not None:
        m['tahmin_maliyeti'] = (m['R2_era5'] - m['R2_oper']).round(3)

    m.to_csv(os.path.join(OUT, 'meteo_karsilastirma.csv'), index=False)
    pd.set_option('display.width', 200)
    print("======== METEO SENARYO KIYASLAMASI (test R²) ========")
    print(m.to_string(index=False))

    # özet: yalnız meteo'nun anlamlı olduğu <= günlük ufuklar
    short = m[m['ufuk'].isin(['t+1', 't+6', 't+24', 'gunluk'])]
    if 'meteo_kazanc_era5' in m:
        print(f"\n≤günlük ufuklarda ortalama meteo kazancı (ERA5/üst-sınır): "
              f"+{short['meteo_kazanc_era5'].mean():.3f} R²")
    if 'meteo_kazanc_oper' in m:
        print(f"≤günlük ufuklarda ortalama meteo kazancı (operasyonel):    "
              f"+{short['meteo_kazanc_oper'].mean():.3f} R²")
    if 'tahmin_maliyeti' in m:
        print(f"≤günlük ufuklarda ort. hava-tahmini hatasının maliyeti:    "
              f"-{short['tahmin_maliyeti'].mean():.3f} R²")
    print("\nKaydedildi: outputs/meteo_karsilastirma.csv")


if __name__ == '__main__':
    main()
