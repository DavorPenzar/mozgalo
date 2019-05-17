#!/usr/bin/env python

# -*- coding: utf-8 -*-

"""
Skripta za automatizirano ucitavanje tablica.

"""

##  PRIPREMA OKRUZENJA

# Standardna Python biblioteka.
import sys

# SciPy paketi.
import numpy as np
import pandas as pd
from pandas.core.arrays.categorical import Categorical as Categorical
from pandas.core.frame import DataFrame as DataFrame
from pandas.core.indexes.base import Index as Index
from pandas.core.series import Series as Series

##  PRIPREMA OBLIKOVANJA
##      *   Sljedece varijable namjesti po izboru.

# Znak za razdvajanje direktorija u lokacijama datoteka.
DIR_DELIMITER = '/'
#DIR_DELIMITER = "\\"

# Pretpotavljena ekstenzija ulazne datoteke.
DEFAULT_IN_EXTENSION = 'csv'

# Ekstenzija izlazne datoteke.
OUT_EXTENSION = 'pkl'

# Funkcije za ucitavanje i zapisivanje podataka.
READER = pd.read_csv
    # poziv:
    #     >>> READER(
    #     ...     in_name,
    #     ...     header = HEADER,
    #     ...     index_col = INDEX_COL,
    #     ...     parse_dates = DATE_COLUMNS,
    #     ...     infer_datetime_format = True,
    #     ...     false_values = BOOLEAN_VALUES[False],
    #     ...     true_values = BOOLEAN_VALUES[True]
    #     ... )
    # gdje je in_name "string" (objekt klase `str') s vrijednosti lokacije
    # datoteke iz koje se cita tablica, a ostale vrijednosti zadane su
    # varijablama nize.
WRITER = DataFrame.to_pickle
    # poziv:
    #     >>> WRITER(df, out_name)
    # gdje je df tablica (objekt klase `pandas.DataFrame'), a out_name "string"
    # (objekt klase `str') s vrijednosti lokacije datoteke u koju se tablica
    # sprema.

# Redak s nazivima stupaca.
HEADER = 'infer'
#HEADER = 0

# Stupac s indeksima redaka.
INDEX_COL = None
#INDEX_COL = 0

# Oznake istina i lazi.
BOOLEAN_VALUES = {
    False : ['N'],
    True : ['Y']
}

# Stupci s datumima.
DATE_COLUMNS = [
#   'DATUM_IZVJESTAVANJA',
#   'DATUM_ZATVARANJA',
    'DATUM_OTVARANJA',
    'PLANIRANI_DATUM_ZATVARANJA'
]

# Stupci s kategorickim vrijednstima.
CATEGORICAL_COLUMNS = [
    'KLIJENT_ID',
    'OZNAKA_PARTIJE',
    'VALUTA',
    'VRSTA_KLIJENTA',
    'PROIZVOD',
    'VRSTA_PROIZVODA',
    'TIP_KAMATE'
]

# Funkcija za indeksiranje tablice.
INDEXER = lambda df : df
#INDEXER = lambda df : df.reset_index(inplace = False)
#INDEXER = lambda df : df.reset_index(drop = True, inplace = False)

# Stupci za obrisati.
DROP_COLUMNS = [
    'Unnamed',
    'Unnamed: 0',
    'Unnamed: 0.1'
]

##  CITANJE ARGUMENATA S KOMANDNE LINIJE

# Provjeri je li program glavni.
assert isinstance(__name__, str)
assert __name__ == '__main__', 'Pokreni me kao glavni program!'

# Provjeri argumente dane preko komandne linije.
assert isinstance(sys.argv, list)
assert len(sys.argv) == 2, 'Zadaj mi tocno 1 argument preko komandne linije!'
assert isinstance(sys.argv[0], str) and isinstance(sys.argv[1], str)

# Deduciraj ime ulazne datoteke i ekstenziju.
in_name = sys.argv[1]
last_dir = in_name.rfind(DIR_DELIMITER)
partial = in_name[last_dir + 1:].split('.')
if last_dir != -1:
    partial[0] = in_name[:last_dir + 1] + partial[0]
extension = DEFAULT_IN_EXTENSION
if len(partial) > 1:
    extension = partial[-1]
    partial = partial[:-1]

# Oslobodi memoriju.
del in_name
del last_dir

# Spremi ime tablice.
name = '.'.join(partial)

# Oslobodi memoriju.
del partial

# Spremi ime ulazne i izlazne datoteke.
in_name = '{name:s}.{extension:s}'.format(name = name, extension = extension)
out_name = '{name:s}.{extension:s}'.format(
    name = name,
    extension = OUT_EXTENSION
)

# Oslobodi memoriju.
del extension

##  UCITAVANJE I SPREMANJE TABLICE

# Ucitaj tablicu.
df = READER(
    in_name,
    header = HEADER,
    index_col = INDEX_COL,
    parse_dates = DATE_COLUMNS,
    infer_datetime_format = True,
    false_values = BOOLEAN_VALUES[False],
    true_values = BOOLEAN_VALUES[True]
)

WRITER(df, out_name)

# Provjeri je li tablica objekt klase pandas.DataFrame.
assert isinstance(df, DataFrame)

# Indeksiraj tablicu.
df = INDEXER(df)

# Provjeri je li tablica objekt klase pandas.DataFrame.
assert isinstance(df, DataFrame)

# Izbaci stupce za izbaciti.
df.drop(columns = DROP_COLUMNS, inplace = True, errors = 'ignore')

# Postavi kategoricke stupce na kategoricki tip vrijednosti.
for col in iter(CATEGORICAL_COLUMNS):
    # Dohvati jedinstvene vrijednosti u stupcu.
    valids = df[col].dropna().unique()

    # Po potrebi pretvori niz valids u numpy.ndarray.
    if isinstance(valids, Categorical):
        valids = np.asarray(valids)

    # Po mogucnosti sortiraj jedinstvene vrijednosti u stupcu.
    try:
        if isinstance(valids, np.ndarray):
            valids.sort()
        else:
            valids.sort_values(inplace = True)
    except (TypeError, ValueError, AttributeError):
        pass

    # Postavi tip vrijednosti stupca na kategoricki tip vrijednosti.
    df[col] = df[col].astype(
        pd.api.types.CategoricalDtype(categories = valids, ordered = False)
    )

# Spremi tablicu.
WRITER(df, out_name)
