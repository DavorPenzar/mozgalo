# -*- coding: utf-8 -*-

import numpy as _np
import pandas as _pd

def divida (df, p = 2.0 / 3.0, target = None, epsilon = 1.0e-5):
    df_0 = df
    df_1 = df

    N = int(_np.round(p * df.shape[0]))

    while True:
        I = _np.sort(_np.random.choice(df.shape[0], size = N, replace = False))

        df_0 = df.iloc[I]
        df_1 = df.iloc[_np.setdiff1d(_np.arange(df.shape[0]), I)]

        if target is None:
            break
        elif _np.abs(
            float(df_0[target].sum()) / df_0.shape[0] -
            float(df_1[target].sum()) / df_1.shpe[0]
        ) < epsilon:
            break

    return (df_0.copy(), df_1.copy())
