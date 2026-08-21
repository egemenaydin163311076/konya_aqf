# Konya Hava Kalitesi Tahmin Sistemi (AQF)

Çok-istasyonlu, çok-kirleticili, **çok-ufuklu** hava kalitesi tahmin çatısı.
Her `(kirletici × istasyon × ufuk)` için bir **model zoo** yarışır, doğrulama yılı (2024)
R²'sine göre **şampiyon** seçilir, test (2025+) üzerinde raporlanır.

## Hızlı başlangıç (VS Code)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# config.py içindeki DATA_PATH'i kendi xlsx yoluna ayarla
python run.py
```
Çıktılar `outputs/` altına yazılır: `champions.csv`, `champion_matrix.csv`, `leaderboard_full.csv`.

## Proje yapısı
```
konya_aqf/
├── config.py          # tüm sabitler: yollar, seriler, ufuklar, bölme, eşikler
├── run.py             # giriş noktası
├── src/
│   ├── data.py        # ham xlsx -> temiz ızgara; birim çevrimi; winsor;
│   │                  #   boşluk bayrakları; ANOMALİ YIL otomatik tespiti
│   ├── features.py    # kademeli doldurma (sadece girdi) + lag/rolling/takvim
│   ├── targets.py     # noktasal + ileri-ortalama hedefler (gözlem-maskeli)
│   ├── models.py      # model zoo: persistence, seasonal_naive, climatology,
│   │                  #   ridge_lags, gbm_delta, gbm_level
│   └── pipeline.py    # yarış + şampiyon seçimi + metrikler
└── outputs/
```

## Tasarım ilkeleri (makale için kritik)
- **Sızıntısızlık:** doldurma yalnızca *girdi* matrisini besler; metrik yalnızca
  *gerçekten gözlenmiş* hedef satırlarında ölçülür. Winsor eşikleri ve impute
  istatistikleri yalnızca eğitim penceresinden (≤2023), anomali yıllar hariç.
- **Anomali yıl tespiti:** bir seri-yılı çok eksikse (>%70) veya yıllık medyanı
  robust-z > 3.5 ile aykırıysa eğitim istatistiklerinden ve hedef eğitiminden
  otomatik dışlanır (örn. Meram SO2 2023, Meram CO 2017-2019).
- **Δ-modelleme:** kısa ufukta GBM, seviye yerine değişimi (y₍t+h₎ − yₜ) tahmin
  eder → persistence'ı geçer.

## Doğrulanmış sonuç (özet)
| Ufuk | En iyi | Test R² |
|------|--------|---------|
| t+1 | karkent_pm10 / gbm_delta | **0.92** |
| t+6 | karkent_pm10 / gbm_delta | 0.55 |
| günlük | meram_so2 / ridge_lags | 0.62 |
| aylık | karkent_pm10 / seasonal_naive | 0.37 |

PM kısa ufukta çok iyi; SO2/CO birkaç saatten sonra persistence-baskın; uzun ufuk
forecasting'in tavanı düşük.

## İki hat
**1. Forecasting** (`run.py`) — geçmişten geleceği tahmin. Kısa ufuk güçlü
(Karkent PM10 t+1 = 0.92), uzun ufuk tavanı düşük.

**2. Nowcasting** (`python -m src.nowcast`) — bir sensörün t anındaki değerini
DİĞER sensörlerin t anındaki değerlerinden + kendi yakın geçmişinden yeniden-üretir.
Çıktı `outputs/nowcast_results.csv`. Saatlik R² + günlük/haftalık/aylık kovalara göre R².
Sanal sensör / boşluk-doldurma / QA-QC için; aynı zamanda AI-agent katmanının temeli.

Doğrulanmış nowcast (test 2025+):
| Hedef | Saatlik | Günlük | Aylık |
|-------|---------|--------|-------|
| karkent_pm10 | 0.93 | 0.98 | 0.97 |
| meram_so2 | 0.83 | 0.97 | 0.98 |
| meram_co | 0.86 | 0.97 | 0.98 |

> Not: aylık R² yalnızca ~16 test ayı üzerinden (gösterge). Asıl sağlam zemin günlük.
> Nowcasting "geleceği tahmin" değildir — mekânsal yeniden-üretimdir; ayrı çerçevelenmeli.

## Sağlam seçim: zaman-serisi CV
`python -m src.cv_select` — şampiyonu tek yıl yerine genişleyen-pencere foldları
(val 2022/2023/2024) ortalamasıyla seçer; durağan olmayan serilerde test'e daha iyi
genelleşir. Uzun ufukta (mean, k≥168) aday kümesi otomatik olarak **ekstrapole edebilen
+ mevsimsel** modellere daralır (`ridge_lags`, `detrend_seasonal`, `climatology`,
`seasonal_naive`), çünkü ağaç-seviye modelleri gelecekteki sürüklenmeyi ekstrapole edemez.
Etki: Meram CO aylık −0.41 → **+0.52**, Karkent SO2 aylık −0.85 → **+0.28**.
Ayrıca `data.py` her seri için `is_trending` bayrağı üretir.

## İki hat — nihai özet (test 2025+, R²)
| Hedef | FC saatlik | FC günlük | FC aylık | NC saatlik | NC günlük | NC aylık |
|-------|-----------|-----------|----------|-----------|-----------|----------|
| karkent_pm10 | 0.92 | 0.61 | 0.38 | 0.93 | 0.98 | 0.97 |
| meram_so2 | 0.82 | 0.62 | – | 0.83 | 0.97 | 0.98 |
| meram_co | 0.83 | 0.65 | 0.52 | 0.86 | 0.97 | 0.98 |
| meram_pm10 | 0.60 | 0.42 | – | 0.67 | 0.93 | 0.87 |
| karkent_so2 | 0.43 | 0.46 | 0.28 | 0.50 | 0.85 | 0.77 |

FC = forecasting (geleceği tahmin), NC = nowcasting (eşzamanlı yeniden-üretim).
Kısa ufuk → FC güçlü; uzun ufuk → NC hâkim. İki hat birbirini tamamlıyor.

## Paralel yönlendirme: ForecastRouter (`src/router.py`)
Senin vizyonun: her (hedef × ufuk) bağımsız bir pipeline, hepsi **paralel** koşar, her
biri model zoo'sundan CV ile en iyi modeli seçer; sonuç bir **yönlendirme tablosu**,
`predict()` ilgili ufkun şampiyonuna route eder.
```python
from src.router import ForecastRouter
r = ForecastRouter().fit(parallel=True, max_workers=4)
print(r.routing_table())          # ufuk -> seçilen model + skor
r.predict('karkent_pm10', 't+1')  # şampiyonun tahmin serisi
```
Zoo: persistence, seasonal_naive, climatology, ridge_lags, gbm_delta/level,
detrend_seasonal (+ kuruluysa xgb_delta/level).

## Meteoroloji: akademik + operasyonel denge (`src/meteo.py`)
İki kaynak, aynı değişkenler, iki senaryo — ikisi de raporlanır:
- **ERA5 reanalizi** (`fetch('era5', ...)`) → **üst-sınır** ("hava bilinseydi" tavanı).
- **Open-Meteo arşiv-tahmini** (`fetch('operational', ...)`) → **operasyonel**, sızıntısız,
  sahadaki gerçek performans.
- İkisinin **farkı = hava tahmini hatasının maliyeti** (yayınlanabilir bulgu).

Sızıntı kuralı: hedef y₍t+h₎ için meteo, HEDEF zamandaki (t+h) havadır — operasyonelde
"o an yapılmış tahmin", üst-sınırda reanaliz. `target_time_features()` bunu otomatik kurar.
İlçe merkezleri: Karkent→Karatay (37.881, 32.521), Meram (37.872, 32.493) — `STATION_COORDS`.

Kullanım (kendi makinende, ağ gerekli):
```python
from src.meteo import fetch_all_stations, meteo_features
from src.router import ForecastRouter
for src in ['era5', 'operational']:
    raw = fetch_all_stations(src)            # cache -> outputs/meteo_*.csv
    mf  = meteo_features(raw)
    rt  = ForecastRouter(meteo_feats=mf).fit().routing_table()
    rt.to_csv(f'outputs/routing_{src}.csv')  # iki senaryoyu kıyasla
```

## Karar katmanı ve değerlendirme (makale-öncesi modüller)

**WHO eşik-aşımı sınıflandırması** (`who_daily_forecast.py` → `outputs/who_classification.csv`):
gün-ileri günlük tahminden WHO 2021 24-saat eşiklerine (PM10 45, SO2 40, NO2 25 µg/m³) göre
aşım sınıflandırması; doğruluk + precision/recall/F1 + karışıklık matrisi + taban oran.
Ortalama doğruluk **%86.6** (PM10 F1≈0.66-0.75, NO2 F1≈0.90). Not: Karkent SO2'de test
döneminde hiç aşım yok (taban oran 0) → doğruluk 1.0 trivialdir, F1 anlamlıdır.

**AI-agent karar katmanı PoC** (`src/agent.py` → `outputs/agent_faultlog_*.csv`):
- *Sensör arıza/sürüklenme tespiti:* hedefin kendi geçmişini KULLANMAYAN bağımsız mekânsal
  nowcast ile gerçek ölçüm arasındaki artık; anlık ve sürekli (24s-ort, CUSUM benzeri) eşik.
  Enjekte-arıza testi: +80 µg/m³ sürüklenme %89-93 yakalanır (büyük arızalar güvenilir;
  ince sürüklenme hassasiyeti mekânsal yeniden-üretim hatasıyla sınırlı — namuslu PoC).
- *Sanal sensör:* istasyon düştüğünde nowcast ile %100 kapsama (MAE ~19 µg/m³).
- *WHO uyarısı:* ertesi-gün aşım alarmı (yukarıdaki sınıflandırma metrikleri).

**Meteo senaryo kıyaslaması** (`compare_meteo.py` → `outputs/meteo_karsilastirma.csv`):
üç routing tablosunu (temel/era5/operational) birleştirir; `meteo_kazanc_era5` (teorik katkı),
`meteo_kazanc_oper` (saha katkısı) ve `tahmin_maliyeti = R2(era5) − R2(operational)`
("hava tahmini hatasının maliyeti") üretir.

## Sıradaki adımlar
- Meteoroloji çekip iki senaryoyu (üst-sınır vs operasyonel) raporlamak → günlük forecast'ı
  persistence üstüne gerçekten taşımak.
- Log dönüşümü + WHO aşım sınıflandırma kafası.
- Bir ileri model (ST-GNN / TFT) — bayrak gemisi makale.
- Nowcast + forecast'ı birleştiren AI-agent karar katmanı.
