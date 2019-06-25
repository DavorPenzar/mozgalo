# -*- coding: utf-8 -*-

import copy as _copy
import functools as _functools
import importlib as _importlib
import multiprocessing as _multiprocessing

import numpy as _np
import pandas as _pd
from pandas.core.arrays.categorical import Categorical as _Categorical
from pandas.core.frame import DataFrame as _DataFrame
from pandas.core.indexes.base import Index as _Index
from pandas.core.series import Series as _Series

_feature_engineering = None
try:
    _feature_engineering = _importlib.import_module('feature_engineering')
except ModuleNotFoundError:
    _feature_engineering = _importlib.import_module(
        'script.feature_engineering'
    )

def pojednostavi_indeks (df):
    df.reset_index(drop = True, inplace = True)

    return df

def izbaci_nepotrebne_stupce (df):
    stupci = ['Unnamed', 'Unnamed: 0', 'Unnamed: 0.1']

    df.drop(columns = stupci, errors = 'ignore', inplace = True)

    return df

def kategoriziraj (df):
    stupci = [
        'KLIJENT_ID',
        'OZNAKA_PARTIJE',
        'PRVA_VALUTA',
        'ZADNJA_VALUTA',
        'VRSTA_KLIJENTA',
        'PROIZVOD',
        'VRSTA_PROIZVODA',
        'PRVI_TIP_KAMATE',
        'ZADNJI_TIP_KAMATE'
    ]

    for stupac in stupci:
        vrijednosti = df[stupac].dropna().unique()

        if isinstance(vrijednosti, _Categorical):
            vrijednosti = _np.asarray(vrijednosti)

        try:
            if isinstance(vrijednosti, _np.ndarray):
                vrijednosti.sort()
            else:
                vrijednosti = vrijednosti.sort_values()
        except (TypeError, ValueError, AttributeError):
            pass

        df[stupac] = df[stupac].astype(
            _pd.api.types.CategoricalDtype(
                categories = vrijednosti,
                ordered = False
            )
        )

    return df

def sredi_prijevremeni_raskid (df):
    interval = _pd.Timedelta(10, 'D')

    df.PRIJEVREMENI_RASKID = 0
    df.PRIJEVREMENI_RASKID = df.PRIJEVREMENI_RASKID.astype(int)

    df.loc[df.UGOVORENI_IZNOS <= 0, 'PRIJEVREMENI_RASKID'] = 1
    df.loc[df.DATUM_OTVARANJA + interval >= df.PLANIRANI_DATUM_ZATVARANJA] = 0

def spljosti (df, n_proc = 4):
    interval = _pd.Timedelta(10, 'D')

    bas_prvi = lambda df, stupac : df[stupac].iloc[0]
    bas_zadnji = lambda df, stupac : df[stupac].iloc[-1]

    znacajke = [
        (
            'instance_id',
            _functools.partial(bas_zadnji, stupac = 'instance_id')
        ),
        ('KLIJENT_ID', _functools.partial(bas_prvi, stupac = 'KLIJENT_ID')),
        (
            'OZNAKA_PARTIJE',
            _functools.partial(bas_prvi, stupac = 'OZNAKA_PARTIJE')
        ),
        (
            'PRVI_DATUM_OTVARANJA',
            _functools.partial(bas_prvi, stupac = 'DATUM_OTVARANJA')
        ),
        (
            'ZADNJI_DATUM_OTVARANJA',
            _functools.partial(bas_zadnji, stupac = 'DATUM_OTVARANJA')
        ),
        (
            'PRVI_PLANIRANI_DATUM_ZATVARANJA',
            _functools.partial(
                _feature_engineering.firstie,
                column = 'PLANIRANI_DATUM_ZATVARANJA'
            )
        ),
        (
            'ZADNJI_PLANIRANI_DATUM_ZATVARANJA',
            _functools.partial(
                _feature_engineering.lastie,
                column = 'PLANIRANI_DATUM_ZATVARANJA'
            )
        ),
        (
            'VRSTA_KLIJENTA',
            _functools.partial(bas_prvi, stupac = 'VRSTA_KLIJENTA')
        ),
        ('PRVA_STAROST', _functools.partial(bas_prvi, stupac = 'STAROST')),
        ('STAROST', _functools.partial(bas_prvi, stupac = 'STAROST')),
        ('ZADNJA_STAROST', _functools.partial(bas_zadnji, stupac = 'STAROST')),
        (
            'VRSTA_PROIZVODA',
            _functools.partial(bas_prvi, stupac = 'VRSTA_PROIZVODA')
        ),
        ('PROIZVOD', _functools.partial(bas_prvi, stupac = 'PROIZVOD')),
        (
            'PRVI_UGOVORENI_IZNOS',
            _functools.partial(bas_prvi, stupac = 'UGOVORENI_IZNOS')
        ),
        (
            'ZADNJI_UGOVORENI_IZNOS',
            _functools.partial(bas_zadnji, stupac = 'UGOVORENI_IZNOS')
        ),
        ('PRVA_VALUTA', _functools.partial(bas_prvi, stupac = 'VALUTA')),
        ('ZADNJA_VALUTA', _functools.partial(bas_zadnji, stupac = 'VALUTA')),
        (
            'PRVI_TIP_KAMATE',
            _functools.partial(bas_prvi, stupac = 'TIP_KAMATE')
        ),
        (
            'ZADNJI_TIP_KAMATE',
            _functools.partial(bas_zadnji, stupac = 'TIP_KAMATE')
        ),
        (
            'PRVA_VISINA_KAMATE',
            _functools.partial(
                _feature_engineering.firstie,
                column = 'VISINA_KAMATE'
            )
        ),
        (
            'ZADNJA_VISINA_KAMATE',
            _functools.partial(
                _feature_engineering.lastie,
                column = 'VISINA_KAMATE'
            )
        ),
        ('PRIJEVREMENI_RASKID', lambda df : 0)
    ]

    po_partijama = df.groupby('OZNAKA_PARTIJE')

    partije = _np.random.permutation(list(po_partijama.groups.keys()))

    granice = _np.round(
        _np.linspace(0, partije.shape[0], num = n_proc + 1)
    ).astype(int)
    granice[0] = 0
    granice[-1] = partije.shape[0]

    def proces (id, df, grupe, kljucevi, spljostenja):
        spljostenja[id] = _feature_engineering.feat(
            tuple((kljuc, grupe.get_group(kljuc)) for kljuc in iter(kljucevi)),
            znacajke
        )

    glavni = _multiprocessing.Manager()
    spljostenja = glavni.dict()
    procesi = list()

    for i in iter(range(n_proc)):
        procesi.append(
            _multiprocessing.Process(
                target = proces,
                args = (
                    i,
                    df,
                    po_partijama,
                    partije[int(granice[i]):int(granice[i + 1])],
                    spljostenja
                )
            )
        )

        procesi[i].start()

    for i in iter(range(n_proc)):
        procesi[i].join()

    znacajcki_df = _pd.concat(tuple(spljostenja.values()), axis = 0)
    znacajcki_df.drop(columns = 'ID', inplace = True)

    znacajcki_df = znacajcki_df.loc[
        (znacajcki_df.ZADNJI_UGOVORENI_IZNOS > 0) &
        (
            znacajcki_df.ZADNJI_DATUM_OTVARANJA + interval <
            znacajcki_df.ZADNJI_PLANIRANI_DATUM_ZATVARANJA
        )
    ].copy()
    pojednostavi_indeks(znacajcki_df)

    return znacajcki_df
