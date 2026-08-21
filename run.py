"""ANA BORU HATTI — tüm adımları sırayla çalıştırır.

  python run.py            # tam boru hattı (meteo dahil, ağ varsa)
  python run.py --no-meteo # meteo çekmeyi atla (sadece temel senaryo)

Her aşama bağımsız try/except içinde; biri başarısız olursa diğerleri devam eder.
Tüm çıktılar outputs/ (figürler outputs/figures/) altına düşer.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import time
import config as C

NO_METEO = '--no-meteo' in _sys.argv


def stage(name, fn):
    print(f"\n{'='*64}\n▶ {name}\n{'='*64}")
    t0 = time.time()
    try:
        fn()
        print(f"  ✓ tamam ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  ! atlandı/hata: {e}")


def s1_data():
    from src.data import clean
    _, meta = clean()
    flagged = {k: v for k, v in meta.items() if v['bad_years'] or v.get('is_trending')}
    print(f"  {len(meta)} seri temizlendi | bayraklı: {len(flagged)}")


def s2_meteo():
    if NO_METEO:
        print("  --no-meteo: atlandı"); return
    from src.meteo import fetch_all_stations
    for src in ['era5', 'operational']:   # operasyonel = gerçek gün-ileri (previous-runs)
        try:
            m = fetch_all_stations(src)
            print(f"  {src}: {m.shape}")
        except Exception as e:
            print(f"  {src}: çekilemedi ({e})")


def s3_forecast():
    from src.router import ForecastRouter
    from src.meteo import load_cached_features
    # temel (meteosuz)
    ForecastRouter().fit(parallel=True, verbose=False).save(C.OUTPUT_DIR / 'routing_table.csv')
    print("  routing_table.csv")
    if not NO_METEO:
        for src in ['era5', 'operational']:
            mf = load_cached_features(src)
            if mf is None:
                print(f"  {src}: meteo cache yok, atlandı"); continue
            ForecastRouter(meteo_feats=mf).fit(parallel=True, verbose=False)\
                .save(C.OUTPUT_DIR / f'routing_table_{src}.csv')
            print(f"  routing_table_{src}.csv")


def s4_compare():
    import compare_meteo
    compare_meteo.main()


def s5_nowcast():
    from src.nowcast import run_nowcast
    run_nowcast()
    print("  nowcast_results.csv")


def s6_who():
    import who_daily_forecast
    who_daily_forecast.main()


def s7_agent():
    from src.agent import run_poc
    run_poc('karkent_pm10')


def s8_plots():
    from src.plots import main as plots_main
    plots_main()


def main():
    C.OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"KONYA HAVA KALİTESİ — TAM BORU HATTI  (meteo: {'KAPALI' if NO_METEO else 'açık'})")
    stage("1/8  Temizleme + veri kalite", s1_data)
    stage("2/8  Meteoroloji çekme", s2_meteo)
    stage("3/8  Forecasting (router, 1-3 senaryo)", s3_forecast)
    stage("4/8  Meteo senaryo kıyaslaması", s4_compare)
    stage("5/8  Nowcasting", s5_nowcast)
    stage("6/8  WHO günlük + aşım sınıflandırma", s6_who)
    stage("7/8  AI-agent karar katmanı PoC", s7_agent)
    stage("8/8  Figürler", s8_plots)
    print(f"\n{'='*64}\nBORU HATTI TAMAM. Çıktılar: {C.OUTPUT_DIR}\n{'='*64}")


if __name__ == '__main__':
    main()
