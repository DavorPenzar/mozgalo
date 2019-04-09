# -*- coding: utf-8 -*-

"""
Skripta s funkcijama korisnima za "feature-engineering".

"""

# Standardna Python biblioteka
import math as _math
import six as _six

# SciPy paketi
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import scipy as _sp

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

    # Vrati jedinsvenu grupiranu tablicu ili tuple grupiranih tablica.
    try:
        return columns_df.get_group(values).groupby(final_groups)
    except KeyError:
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
        --  numpy.ndarray za dtype = matplotlib.axes.Axes   --  svaki graf ima
                                                                svoje
                                                                odgovarajuce osi
                                                                (osi moze biti i
                                                                viska ---
                                                                uzima se prvih
                                                                len(plotters),
                                                                potrebno) ali ne
                                                                manjka).

    Dodatno, svaki se graf moze urediti argumentom axes_kwargs (opcionalan;
    zadana vrijednost je None) pri cemu on moze biti:
        --  None    --  grafovi se ne ureduju dodatnim imenovanim argumentima,
        --  dict    --  svaki se graf ureduje istim dodatnim imenovanim
                        argumentima,
        --  lista rjecnika (dict)   --  svaki graf ima vlastite dodatne
                                        imenovane argumente za uredivanje.

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
    if isinstance(plotter, dict):
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

# Definicija funkcije feat.
def feat (groups, features, row_id = 'ID'):
    """
    Konstruiraj tablicu svojstava grupa.

    Svakom uredenom paru (name, group) u objektu groups (pandas.GroupBy, dict
    ili lista uredenih parova) pridruzen je jedinstveni redak u povratnoj
    tablici.  Retci se konstruiraju pomocu objekta features (dict ili lista
    uredenih parova) tako da se za svaki uredeni par (column, feature) u objektu
    features u stupac column u redak pridruzen promatranoj grupi sprema
    vrijednost poziva
        >>> feature(group)

    Argument row_id (opcionalan; zadana vrijednost je 'ID') zadaje naziv prvog
    stupca u kojemu su spremljena imena grupa, to jest, vrijednosti name za
    uredene parove (name, group) u objektu groups.

    Povratna vrijednost je objekt klase pandas.DataFrame.

    """

    # Definicija pomocne funkcije features_iterkeys za iteriranje po
    # "kljucevima" (prvi element) u listi uredenih parova (key, value).
    def features_iterkeys (features):
        """
        Dohvacaj sve druge elemente tuple-ova u iterabilnom argumentu features.

        """

        for feature in iter(features):
            yield feature[0]

    # Definicija pomocne funkcije features_itervalues za iteriranje po
    # "vrijednostima" (drugi element) u listi uredenih parova (key, value).
    def features_itervalues (features):
        """
        Dohvacaj sve druge elemente tuple-ova u iterabilnom argumentu features.

        """

        for feature in iter(features):
            yield feature[1]

    # Deduciraj funkciju za iteriranje po objektu groups ovisno o njegovoj
    # klasi.
    groups_iter = iter
    if isinstance(groups, dict):
        groups_iter = _six.iteritems

    # Ako je objekt features rjecnik, azuriraj funkcije features_iterkeys i
    # features_itervalues adekvatno.
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
