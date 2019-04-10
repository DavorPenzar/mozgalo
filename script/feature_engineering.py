# -*- coding: utf-8 -*-

"""
Skripta s funkcijama korisnima za "feature-engineering".

"""

# Standardna Python biblioteka.
import math as _math
import six as _six

# SciPy paketi.
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import scipy as _sp

# Seaborn.
import seaborn as _sns

# Definicija funkcije grouppie.
def grouppie (df, columns, values, final_groups):
    """
    Dohvati slozene grupe iz tablice.

    Pretpostavimo da je values jedinsvena vrijednost takva da je poziv
        >>> df.groupby(columns).get_group(values)
    legalan.  Tada je poziv funkcije ekvivalentan izrazu
        >>> df.groupby(columns).get_group(values).groupby(final_groups)
    Inace, ako values nije jedinsvena vrijednost, poziv funkcije ekvivalentan je
    izrazu
        >>> tuple(df.groupby(columns).get_group(value).groupby(final_groups) for value in values)

    Povratna vrijednost je objekt klase pandas.GroupBy ili tuple takvih
    objekata.

    """

    # Dohvati grupe tablice po stupcu/stupcima columns.
    columns_df = df.groupby(columns)

    # Vrati jedinsvenu grupiranu tablicu ako je moguce.
    try:
        return columns_df.get_group(values).groupby(final_groups)
    except KeyError:
        pass

    # Vrati tuple grupiranih tablica.
    return tuple(
        columns_df.get_group(value).groupby(final_groups)
            for value in iter(values)
    )

# Definicija funkcije plots.
def plots (df, plotters, axes = None, axes_kwargs = None):
    """
    Konstruiraj grafove za zadanu tablicu.

    Za svaki uredeni par (title, plotter) u objektu plotters (dict ili lista
    uredenih parova --- ne smije biti obicni iterator jer je za ispravan rad
    funkcije potrebno svojstvo len(plotters)) na odgovarajuce osi konstruira se
    graf dobiven pozivom
        >>> ax.plot(*plotter(df))
    tako da te osi imaju naslov title.  Objekt axes (opcionalan; zadana
    vrijednost je None) moze biti:
        --  None    --  deducira se optimalni raspored osi tako da svaki graf
                        ima odgovarajuce osi i tada je povratna vrijednost
                        tuple figure i konstruiranih osi,
        --  matplotlib.axes.Axes    --  svi se grafovi prikazuju na istim osima,
        --  numpy.ndarray za dtype = matplotlib.axes.Axes
            --  svaki graf ima svoje odgovarajuce osi (osi moze biti i viska ---
                uzima se prvih len(plotters), ali ne manjka).

    Dodatno, svaki se graf moze urediti argumentom axes_kwargs (opcionalan;
    zadana vrijednost je None) pri cemu on moze biti:
        --  None    --  grafovi se ne ureduju dodatnim imenovanim argumentima,
        --  dict    --  svaki se graf ureduje istim dodatnim imenovanim
                        argumentima,
        --  lista rjecnika (dict)   --  svaki graf ima vlastite dodatne
                                        imenovane argumente za uredivanje.

    Ako je axes objekt klase numpy.ndarray, liste (nuzno jednodimenzionalne)
    plotters i axes_kwargs moraju pratiti raspored osi u axes.ravel().

    Povratna vrijednost je None osim ako argument axes nije None (vidi gore).

    """

    # Incijaliziraj objekt figure na None i original_axes na argument axes.
    figure = None
    original_axes = axes

    # Ako je argument axes None, konstruiraj nove osi.
    if axes is None:

        # Izracunaj optimalan raspored osi.
        nrows = int(_math.floor(_math.sqrt(len(plotters))))
        if nrows:
            ncols = int(_math.ceil(float(len(plotters)) / nrows))
        else:
            ncols = 0

        # Konstruiraj novu figuru i osi.  Kosnstruirane osi spremi u objekt
        # original_axes.
        figure, axes = (
            _plt.subplots(nrows = nrows, ncols = ncols) if nrows
                else _plt.subplots(nrows = 0, ncols = 1)
        )
        original_axes = axes

    # Deduciraj funkciju za iteriranje po objektu plotters ovisno o njegovoj
    # klasi.
    plotters_iter = iter
    if isinstance(plotters, dict):
        plotters_iter = _six.iteritems

    # Ako je axes jedinsvena vrijednost (ne lista vrijednosti), prosiri taj
    # objekt na listu istih vrijednosti duljine len(plotters).
    if not isinstance(axes, _np.ndarray):
        axes = _np.array(len(plotters) * [axes])

    # "Izravnaj" listu axes i uzmi samo prvih len(plotters) osi u toj listi.
    axes = axes.ravel()
    axes = axes[:len(plotters)]

    # Ako je axes_kwargs None, pretvori ga u listu praznih rjecnika duljine
    # axes.size.  Ako je axes_kwargs rjecnik, pretvori ga u listu od axes.size
    # kopija tog rjecnika.
    if axes_kwargs is None:
        axes_kwargs = axes.size * [dict()]
    if isinstance(axes_kwargs, dict):
        axes_kwargs = axes.size * [axes_kwargs]

    # Konstruiraj trazene grafove.
    for axis, plotter, kwargs in zip(
        iter(axes),
        plotters_iter(plotters),
        iter(axes_kwargs)
    ):
        axis.set_title(plotter[0])
        axis.plot(*plotter[1](df), **kwargs)

    # Ako je konstruirana neka figura (ako figure nije None), vrati uredeni
    # par (tuple) objekata figure, original_axes.
    if figure is not None:
        return (figure, original_axes)

# Definicija funkcije split.
def split (df, names = None):
    """
    Dohvati tablicu dobivenu sjecenjem originalne tablice/stupca.

    Funkcija je zapravo omotac poziva
        >>> pandas.DataFrame(df.tolist(), index = df.index, columns = names)

    Povratna vrijednost je objekt klase pandas.DataFrame.

    """

    # Vrati trazenu tablicu.
    return _pd.DataFrame(df.tolist(), index = df.index, columns = names)

# Definicija funkcije transform.
def transform (df, transformers):
    """
    Konstruiraj tablicu dobivenu transformacijom originalne tablice.

    Povratna tablica dobivena je konkatenacijom po stupcima povratnih
    vrijednosti poziva
        >>> transformer(df[column])
    za svaki uredeni par (column, transformer) u objektu transformers (dict ili
    lista uredenih parova).  Specijalno, transformer moze biti i None cime se
    stupac samo "prepisuje" u povratnu tablicu bez transformacije, to jest,
    to je ekvivalentno sa slucajem da je transformer "identiteta".

    Ako transformers nije dict, od istog se stupca originalne tablice moze
    u povratnoj tablici moze konstruirati vise transformacija (u objektu
    transformers moze se pojaviti vise uredenih parova cija je prva komponenta
    jednaka).

    Povratna vrijednost je objekt klase pandas.DataFrame.

    """

    # Deduciraj funkciju za iteriranje po objektu transformers ovisno o njegovoj
    # klasi.
    transformers_iter = iter
    if isinstance(transformers, dict):
        transformers_iter = _six.iteritems

    # Vrati tablicu trazenih transformacija.
    return _pd.concat(
        tuple(
            df[column] if transformer is None else transformer(df[column])
                for column, transformer in transformers_iter(transformers)
        ),
        axis = 1
    )

# Definicija funkcije dummify.
def dummify (df, columns):
    """
    Dohvati pojavnost kategorija u tablici.

    Funkcija je zapravo omotac poziva
        >>> tuple(pandas.get_dummies(df[columns])).any(axis = 0).astype(numpy.uint8))

    Povratna vrijednost je tuple ciji su objekti 0 ili 1 klase numpy.uint8.

    """

    # Vrati trazeni tuple.
    return tuple(_pd.get_dummies(df[columns])).any(axis = 0).astype(_np.uint8))

# Definicija funkcije feat.
def feat (groups, features, row_id = 'ID'):
    """
    Konstruiraj tablicu svojstava grupa.

    Svakom uredenom paru (name, group) u objektu groups (pandas.GroupBy, dict
    ili lista uredenih parova) pridruzen je jedinstveni redak u povratnoj
    tablici.  Retci se konstruiraju pomocu objekta features (dict ili lista
    uredenih parova --- ne smije biti obicni iterator jer se za svaku grupu
    iterira po cijelom objektu features) tako da se za svaki uredeni par
    (column, feature) u objektu features u stupac column u redak pridruzen
    promatranoj grupi sprema vrijednost poziva
        >>> feature(group)

    Argument row_id (opcionalan; zadana vrijednost je 'ID') zadaje naziv prvog
    stupca u kojemu su spremljena imena grupa, to jest, vrijednosti name za
    uredene parove (name, group) u objektu groups.

    Povratna vrijednost je objekt klase pandas.DataFrame.

    """

    # Deduciraj funkciju za iteriranje po objektu groups ovisno o njegovoj
    # klasi.
    groups_iter = iter
    if isinstance(groups, dict):
        groups_iter = _six.iteritems

    # Deduciraj funkcieu za iteriranje po objektu features ovisno o njegovoj
    # klasi.
    features_iterkeys = lambda features : iter(
        feature[0] for feature in iter(features)
    )
    features_itervalues = lambda features : iter(
        feature[1] for feature in iter(features)
    )
    if isinstance(features, dict):
        features_iterkeys = _six.iterkeys
        features_itervalues = _six.itervalues

    # Vrati tablicu trazenih svojstava.
    return _pd.DataFrame(
        [
            [name] + [
                feature(group)
                    for feature in features_itervalues(features)
            ] for name, group in groups_iter(groups)
        ],
        columns = [row_id] + list(features_iterkeys(features))
    )
