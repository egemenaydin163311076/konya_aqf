"""Hedef üretimi — yalnızca GERÇEK gözlenmiş değerlerle (sızıntısız metrik için)."""
import numpy as np
import pandas as pd


def point_target(observed_series, h):
    """t+h anındaki gerçek (gözlenmiş) değer."""
    return observed_series.shift(-h)


def forward_mean_target(observed_series, k, min_frac=0.5):
    """(t, t+k] aralığındaki gözlenmiş değerlerin ortalaması; <%50 gözlemde NaN."""
    a = observed_series.shift(-1)               # a[t] = v[t+1]
    val = a.fillna(0.0)[::-1].rolling(k, min_periods=1).sum()[::-1]
    cnt = a.notna().astype(float)[::-1].rolling(k, min_periods=1).sum()[::-1]
    out = val / cnt
    out[cnt < min_frac * k] = np.nan
    return out


def make_target(observed_series, kind, param):
    return point_target(observed_series, param) if kind == 'point' \
        else forward_mean_target(observed_series, param)
