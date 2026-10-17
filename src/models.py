"""Model zoo: her aday ortak arayüzle çalışır -> tüm indeks için tahmin Serisi.
GBM varsayılan sklearn HistGradientBoosting; LightGBM kuruluysa otomatik kullanılır."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import config as C

try:
    from lightgbm import LGBMRegressor
    def _gbm():
        return LGBMRegressor(n_estimators=400, learning_rate=0.06, num_leaves=63,
                             reg_lambda=1.0, random_state=C.RANDOM_STATE, verbose=-1)
    GBM_NAME = 'lightgbm'
except Exception:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def _gbm():
        return HistGradientBoostingRegressor(max_iter=350, learning_rate=0.06,
                    max_leaf_nodes=63, l2_regularization=1.0, early_stopping=True,
                    validation_fraction=0.15, random_state=C.RANDOM_STATE)
    GBM_NAME = 'hist_gbm'


class Persistence:
    name = 'persistence'
    def fit_predict(self, ctx):
        return ctx.get('anchor_raw', ctx['anchor']).copy()   # son gözlenen (ham) değeri taşı


class SeasonalNaive:
    name = 'seasonal_naive'
    def fit_predict(self, ctx):
        if ctx['kind'] == 'point' and ctx['param'] < 168:
            # bir hafta önce, hedef saatindeki gerçekleşmiş (doldurulmuş) değer
            return ctx['imp'][ctx['target']].shift(168 - ctx['param'])
        # uzun ufuklarda klimatolojiye bırak
        return Climatology().fit_predict(ctx)


class Climatology:
    name = 'climatology'
    def fit_predict(self, ctx):
        clean, idx, fm = ctx['clean'], ctx['index'], ctx['fit_mask']
        tcol = ctx['target']
        if ctx['kind'] == 'point':
            key = list(zip(clean['month'] if 'month' in clean else idx.month, idx.hour))
            tab = pd.Series(ctx['y'][fm].values,
                            index=pd.MultiIndex.from_arrays([idx[fm].month, idx[fm].hour])
                            ).groupby(level=[0, 1]).mean()
            return pd.Series([tab.get((m, h), np.nan) for m, h in zip(idx.month, idx.hour)], index=idx)
        else:  # mean: ay bazlı klimatoloji
            tab = pd.Series(ctx['y'][fm].values, index=idx[fm].month).groupby(level=0).mean()
            return pd.Series([tab.get(m, np.nan) for m in idx.month], index=idx)


class RidgeLags:
    name = 'ridge_lags'
    def fit_predict(self, ctx):
        X, y, fm = ctx['X'], ctx['y'], ctx['fit_mask']
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(X[fm], y[fm])
        return pd.Series(model.predict(X.fillna(0)), index=ctx['index'])


class GBMDelta:
    name = f'gbm_delta'
    def fit_predict(self, ctx):
        X, anchor, fm = ctx['X'], ctx['anchor'], ctx['fit_mask']
        delta = ctx['y'] - anchor
        m = _gbm(); m.fit(X[fm], delta[fm])
        return anchor + pd.Series(m.predict(X.fillna(0)), index=ctx['index'])


class GBMLevel:
    name = f'gbm_level'
    def fit_predict(self, ctx):
        X, fm = ctx['X'], ctx['fit_mask']
        m = _gbm(); m.fit(X[fm], ctx['y'][fm])
        return pd.Series(m.predict(X.fillna(0)), index=ctx['index'])


class DetrendSeasonal:
    """Lineer trend (year_frac) + artığın mevsimsel klimatolojisi -> trendi ekstrapole eder.
    Durağan olmayan (trendli) seriler için uzun ufukta kritik."""
    name = 'detrend_seasonal'
    def fit_predict(self, ctx):
        idx, y, fm = ctx['index'], ctx['y'], ctx['fit_mask']
        tf = pd.Series(idx.year + (idx.month - 1) / 12.0, index=idx)
        b = np.polyfit(tf[fm], y[fm], 1)
        trend = pd.Series(np.polyval(b, tf), index=idx)
        resid = y - trend
        if ctx['kind'] == 'point':
            tab = pd.Series(resid[fm].values,
                            index=pd.MultiIndex.from_arrays([idx[fm].month, idx[fm].hour])
                            ).groupby(level=[0, 1]).mean()
            clim = pd.Series([tab.get((m, h), 0.0) for m, h in zip(idx.month, idx.hour)], index=idx)
        else:
            tab = pd.Series(resid[fm].values, index=idx[fm].month).groupby(level=0).mean()
            clim = pd.Series([tab.get(m, 0.0) for m in idx.month], index=idx)
        return trend + clim


try:
    from xgboost import XGBRegressor
    def _xgb():
        return XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                            random_state=C.RANDOM_STATE, n_jobs=2, verbosity=0)
    HAS_XGB = True
except Exception:
    HAS_XGB = False


class XGBDelta:
    name = 'xgb_delta'
    def fit_predict(self, ctx):
        X, anchor, fm = ctx['X'], ctx['anchor'], ctx['fit_mask']
        m = _xgb(); m.fit(X[fm], (ctx['y'] - anchor)[fm])
        return anchor + pd.Series(m.predict(X.fillna(0)), index=ctx['index'])


class XGBLevel:
    name = 'xgb_level'
    def fit_predict(self, ctx):
        X, fm = ctx['X'], ctx['fit_mask']
        m = _xgb(); m.fit(X[fm], ctx['y'][fm])
        return pd.Series(m.predict(X.fillna(0)), index=ctx['index'])


ZOO = [Persistence(), SeasonalNaive(), Climatology(), RidgeLags(),
       GBMDelta(), GBMLevel(), DetrendSeasonal()]
if HAS_XGB:
    ZOO += [XGBDelta(), XGBLevel()]      # kuruluysa otomatik yarışa katılır
# trendli seri + uzun ufukta yalnızca ekstrapole edebilen modeller
EXTRAPOLATING = {'ridge_lags', 'detrend_seasonal'}


def candidates(kind, param):
    """Uzun ufuk (mean, k>=168): ağaç-seviye modelleri gelecekteki sürüklenmeyi
    ekstrapole edemez -> aday kümesini ekstrapole eden + mevsimsel modellere daralt.
    Hem hızlı (pipeline) hem CV (cv_select) modları bunu kullanır."""
    if kind == 'mean' and param >= 168:
        allowed = EXTRAPOLATING | {'climatology', 'seasonal_naive'}
        return [m for m in ZOO if m.name in allowed]
    return ZOO
