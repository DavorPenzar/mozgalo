# -*- coding: utf-8 -*-

import copy as _copy
import functools as _functools
import multiprocessing as _multiprocessing

import numpy as _np
import pandas as _pd
from pandas.core.arrays.categorical import Categorical as _Categorical
from pandas.core.frame import DataFrame as _DataFrame
from pandas.core.indexes.base import Index as _Index
from pandas.core.series import Series as _Series

from feature_engineering import *

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

    for stupac in iter(stupci):
        vrijednosti = df[stupac].dropna().unique()

        if isinstance(vrijednosti, _Categorical):
            vrijednosti = _np.asarray(vrijednosti)

        try:
            if isinstance(vrijednosti, _np.ndarray):
                vrijednosti.sort()
            else:
                vrijednosti.sort_values(inplace = True)
        except (TypeError, ValueError, AttributeError):
            pass

        df[stupac] = df[stupac].astype(_pd.api.types.CategoricalDtype(categories = vrijednosti, ordered = False))

    return df

def poredaj (df):
    stupci = [
        'DATUM_IZVJESTAVANJA',
        'STAROST',
        'DATUM_OTVARANJA',
        'PLANIRANI_DATUM_ZATVARANJA',
        'DATUM_ZATVARANJA',
        'UGOVORENI_IZNOS',
        'VISINA_KAMATE'
    ]
    uzlazno = [True, True, True, True, True, False, False]

    df.sort_values(by = stupci, ascending = uzlazno, inplace = True)

    return df

def sredi_prijevremeni_raskid (df):
    interval = _pd.Timedelta(10, 'D')

    df.PRIJEVREMENI_RASKID = (df.DATUM_ZATVARANJA + interval < df[['PRVI_PLANIRANI_DATUM_ZATVARANJA', 'ZADNJI_PLANIRANI_DATUM_ZATVARANJA']].min(axis = 1))

    return df

def spljosti (df, n_proc = 4):
    bas_prvi = lambda df, stupac : df[stupac].iloc[0]
    bas_zadnji = lambda df, stupac : df[stupac].iloc[-1]

    znacajke = [
        ('DATUM_IZVJESTAVANJA', _functools.partial(bas_zadnji, stupac = 'DATUM_IZVJESTAVANJA')),
        ('KLIJENT_ID', _functools.partial(bas_prvi, stupac = 'KLIJENT_ID')),
        ('OZNAKA_PARTIJE', _functools.partial(bas_prvi, stupac = 'OZNAKA_PARTIJE')),
        ('PRVI_DATUM_OTVARANJA', _functools.partial(bas_prvi, stupac = 'DATUM_OTVARANJA')),
        ('ZADNJI_DATUM_OTVARANJA', _functools.partial(bas_zadnji, stupac = 'DATUM_OTVARANJA')),
        ('PRVI_PLANIRANI_DATUM_ZATVARANJA', _functools.partial(firstie, column = 'PLANIRANI_DATUM_ZATVARANJA')),
        ('ZADNJI_PLANIRANI_DATUM_ZATVARANJA', _functools.partial(lastie, column = 'PLANIRANI_DATUM_ZATVARANJA')),
        ('VRSTA_KLIJENTA', _functools.partial(bas_prvi, stupac = 'VRSTA_KLIJENTA')),
        ('STAROST', _functools.partial(bas_zadnji, stupac = 'STAROST')),
        ('VRSTA_PROIZVODA', _functools.partial(bas_prvi, stupac = 'VRSTA_PROIZVODA')),
        ('PROIZVOD', _functools.partial(bas_prvi, stupac = 'PROIZVOD')),
        ('PRVI_UGOVORENI_IZNOS', _functools.partial(bas_prvi, stupac = 'UGOVORENI_IZNOS')),
        ('ZADNJI_UGOVORENI_IZNOS', _functools.partial(bas_zadnji, stupac = 'UGOVORENI_IZNOS')),
        ('PRVA_VALUTA', _functools.partial(bas_prvi, stupac = 'VALUTA')),
        ('ZADNJA_VALUTA', _functools.partial(bas_zadnji, stupac = 'VALUTA')),
        ('PRVI_TIP_KAMATE', _functools.partial(bas_prvi, stupac = 'TIP_KAMATE')),
        ('ZADNJI_TIP_KAMATE', _functools.partial(bas_zadnji, stupac = 'TIP_KAMATE')),
        ('PRVA_VISINA_KAMATE', _functools.partial(firstie, column = 'VISINA_KAMATE')),
        ('ZADNJA_VISINA_KAMATE', _functools.partial(lastie, column = 'VISINA_KAMATE')),
        ('DATUM_ZATVARANJA', _functools.partial(lastie, column = 'DATUM_ZATVARANJA')),
        ('PRIJEVREMENI_RASKID', lambda df : False)
    ]

    po_partijama = df.groupby('OZNAKA_PARTIJE')

    partije = _np.random.permutation(list(po_partijama.groups.keys()))

    granice = _np.round(_np.linspace(0, partije.shape[0], num = n_proc + 1)).astype(int)
    granice[0] = 0
    granice[-1] = partije.shape[0]

    def proces (id, df, grupe, kljucevi, spljostenja):
        spljostenja[id] = feat(tuple((kljuc, grupe.get_group(kljuc)) for kljuc in iter(kljucevi)), znacajke)

    glavni = _multiprocessing.Manager()
    spljostenja = glavni.dict()
    procesi = list()

    for i in iter(range(n_proc)):
        procesi.append(_multiprocessing.Process(target = proces, args = (i, df, po_partijama, partije[int(granice[i]):int(granice[i + 1])], spljostenja)))

        procesi[i].start()

    for i in iter(range(n_proc)):
        procesi[i].join()

    znacajcki_df = _pd.concat(tuple(spljostenja.values()), axis = 0)
    znacajcki_df.drop(columns = 'ID', inplace = True)

    return znacajcki_df

def lose_oznake_partija (df, n_proc = 4):
    def kriterij (df):
        return (
            df.KLIJENT_ID.unique().size > 1 or
            (df.STAROST < 0).any() or
            (df.STAROST >= 500).any() or
            (df.UGOVORENI_IZNOS <= 0).any() or
#           _pd.isnull(df.PLANIRANI_DATUM_ZATVARANJA).any() or
            (df.PLANIRANI_DATUM_ZATVARANJA < df.DATUM_OTVARANJA).any() or
            (df.DATUM_ZATVARANJA < df.DATUM_OTVARANJA).any()
        )

    df = df.copy()

    po_partijama = df.groupby('OZNAKA_PARTIJE')

    partije = _np.random.permutation(list(po_partijama.groups.keys()))

    granice = _np.round(_np.linspace(0, partije.shape[0], num = n_proc + 1)).astype(int)
    granice[0] = 0
    granice[-1] = partije.shape[0]

    def proces (id, df, grupe, kljucevi, losi, kriterij):
        losi[id] = _np.asarray([partija for partija in iter(kljucevi) if kriterij(grupe.get_group(partija))])

    glavni = _multiprocessing.Manager()
    losi = glavni.dict()
    procesi = list()

    for i in iter(range(n_proc)):
        procesi.append(_multiprocessing.Process(target = proces, args = (i, df, po_partijama, partije[int(granice[i]):int(granice[i + 1])], losi, kriterij)))

        procesi[i].start()

    for i in iter(range(n_proc)):
        procesi[i].join()

    return _np.concatenate(tuple(losi.values()))

def pocisti_drugacije (df, n_proc = 4):
    interval = _pd.Timedelta(10, 'D')

    df = df.loc[~df.OZNAKA_PARTIJE.isin(lose_oznake_partija(df, n_proc))].copy()
    pojednostavi_indeks(poredaj(izbaci_nepotrebne_stupce(df)))

    spljosteni = spljosti(df, n_proc)
    spljosteni = spljosteni.loc[(spljosteni.PRVI_PLANIRANI_DATUM_ZATVARANJA - spljosteni.PRVI_DATUM_OTVARANJA > interval) & (spljosteni.ZADNJI_PLANIRANI_DATUM_ZATVARANJA - spljosteni.ZADNJI_DATUM_OTVARANJA > interval)].copy()
    sredi_prijevremeni_raskid(spljosteni)
    spljosteni = spljosteni.loc[spljosteni.PRIJEVREMENI_RASKID | (spljosteni.ZADNJI_PLANIRANI_DATUM_ZATVARANJA - spljosteni.DATUM_IZVJESTAVANJA <= interval)].copy()
    pojednostavi_indeks(spljosteni)

    spljosteni.drop(columns = ['DATUM_IZVJESTAVANJA'], inplace = True)

    kategoriziraj(spljosteni)

    return spljosteni
