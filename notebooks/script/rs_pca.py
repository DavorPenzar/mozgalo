# -*- coding: utf-8 -*-

"""
Skripta s klasom za RS-PCA.

"""

# Standardna Python biblioteka.
import copy as _copy
import datetime as _datetime
import functools as _functools
import math as _math
import os as _os
import random as _random
import six as _six
import string as _string
import sys as _sys
import time as _time
import warnings as _warnings

# SciPy paketi.
import matplotlib as _mpl
import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd
import scipy as _sp
import sympy as _sym
from pandas.core.dtypes.dtypes import CategoricalDtype as _CategoricalDtype
from sympy.logic.boolalg import Boolean as _Boolean
from sympy.core.numbers import Integer as _Integer

# Seaborn.
import seaborn as _sns

# Definicija klase RS_PCA.
class RS_PCA (object):
    """
    Analiza glavnih komponenti koristeci regularne simplekse za kategoricke podatke.

    Acronyms
    --------
    RS
        Regular simplex; hrv. regularni simpleks.  Konveksna ljuska konacnog
        skupa tocaka u euklidskom prostoru tako da su radijvektori svake od tih
        tocaka norme 1 i da je kut izmedu svaka dva njihova radijvektora isti.

    LRSV
        List of regular simplex vertices; hrv. lista vrhova regularnih
        simpleksa. Za vektore koordinata vrhova regularnih simpleksa
            (v_{0, 0}, v_{0, 1}, ..., v_{0, n_0}),
            (v_{1, 0}, v_{1, 1}, ..., v_{1, n_1}),
            ...,
            (v_{m - 1, 0}, v_{m - 1, 1}, ..., v_{m - 1, n_{m - 1}})
        LRSV je njihova konkatenacija
            (v_{0, 0}, v_{0, 1}, ..., v_{0, n_0}), v_{1, 0}, v_{1, 1}, ...,
             v_{1, n_1}, ..., v_{m - 1, 0}, v_{m - 1, 1}, ...,
             v_{m - 1, n_{m - 1}}).

    RS-PCA
        Regular simplex -- principal components analysis (principal components
        analysis using the regular simplex for categorical data); hrv. regularni
        simpleks -- analiza glavnih komponenti (analiza glavnih komponenti
        koristeci reglurne simplekse za kategoricke podatke).  Vidi [1].

    Remarks
    -------
    (1) U primjerima koda pretpostavlja se da je su ranije bile izvrsene
        komande
            >>> import numpy as np
            >>> import pandas as pd
        i da je df neki objekt klase pandas.DataFrame ili pandas.Series (kao
        stupac neke tablice klase pandas.DataFrame).

    (2) Ako se neki atribut objekta klase RS_PCA koristi vise puta, efikasnije
        ga je spremiti u vanjsku varijablu nego ga svaki put dohvacati jer se
        pri svakom pozivu dohvacanja atributa generira i vraca njegova duboka
        kopija.  Ukratko, postupak
            >>> pca = RS_PCA()
            >>> pca.fit(df)
            <RS_PCA: (...)>
            >>> shape = pca.shape
            >>> components = pca.components
            >>> for i in range(shape[2][0]):
            ...     for j in range(i, shape[2][1]):
            ...         print((i, j, np.dot(components[:, i], components[:, j])))
        je mnogo efikasniji nego
            >>> pca = RS_PCA()
            >>> pca.fit(df)
            <RS_PCA: (...)>
            >>> for i in range(pca.shape[2][0]):
            ...     for j in range(i, pca.shape[2][1]):
            ...         print((i, j, np.dot(pca.components[:, i], pca.components[:, j])))
        jer se u drugom postupku, na primjer, u svakoj iteraciji unutarnje
        petlje generira duboka kopija atributa pca.shape, a pri svakom pozivu
        funkcije print generiraju se dvije duboke kopije atributa
        pca.components.

    Bibliography
    ------------
    [1] Hirotaka Niitsuma, Takashi Okada. "Covariance and PCA for Categorical
        Variables". 2007. arXiv:0711.4452 [cs.LG]. URL:
        http://arxiv.org/abs/0711.4452

    """

    # Bazna bijekcija za prijevod kategorickih vrijednosti u indekse.
    @classmethod
    def translate (cls, cat, x):
        """
        Vrati indeks elementa x u konacnom nizu cat.

        Returns
        -------
        i : int
            Indeks i takav da vrijedi cat[i] == x.

        """

        assert issubclass(cls, RS_PCA)

        # Provjeri i saniraj sve argumenta.

        assert isinstance(cat, _CategoricalDtype)

        I = _np.where(cat == x)

        i = None
        try:
            i = _copy.deepcopy(int(I[0][0]))
        except (IndexError, KeyError, ValueError):
            pass

        del I

        return i

    @classmethod
    def simplex (cls, n):
        """
        Dohvati koordinate vrhova rehularnog n-simpleksa.

        Arguments
        ---------
        n : int
            Broj vrhova simpleksa.  Uzima se
                >>> n = max(n, 0)

        Returns
        -------
        S : (n, n + 1) array
            Stupci povratne matrice predstavljaju vrhove regularnog n-simpleksa
            u prostoru R^n za n >= 0.

        """

        assert issubclass(cls, RS_PCA)

        # Provjeri i saniraj argument.

        assert (
            isinstance(n, _six.integer_types) or
            isinstance(n, (bool, _np.bool, _Boolean, _np.integer, _Integer))
        )

        # Ako je n <= 0, vrati "prazni" simpleks (matrica dimenzija 0 x 1).
        if n <= 0:
            return _np.zeros((0, 1), dtype = float, order = 'F')

        # Izracunaj 1 / n.
        k = -(float(n) ** -1)

        # Izracunaj koordinate vrhova simpleksa.
        S = _np.zeros((n, n + 1), dtype = float, order = 'F')
        for i in iter(range(int(n))):
            # Izracunaj p = || S[:i, i] ||^2.

            x = S[:i, i]

            p = _np.dot(x, x)

            del x

            # Izracunaj S[i, i] i S[i, i + 1:].

            S[i, i] = _np.sqrt(1.0 - p)
            S[i, i + 1:] = (k - p) / S[i, i]

            del p

        try:
            del i
        except (NameError, UnboundLocalError):
            pass

        del k

        del n

        # Vrati izracunatu matricu koordinata vrhova.
        return S

    @classmethod
    def _lrsv (cls, df, simplices, translators):
        """
        Prevedi kategoricku tablicu u realnu matricu LRSV.

        Arguments
        ---------
        df : DataFrame
            Tablica kategorickih stupaca.  U stupcima se ne bi smjela pojaviti
            nedefinirana vrijednost (NaN).

        simplices : sequence of returns of simplex
            Niz duljine df.shape[1] tako da na i-tom mjestu stoji matrica
            koordinata regularnog (n - 1)-simpleksa gdje je n broj jedinstvenih
            vrijednosti u i-tom stupcu tablice df.

        translators : sequence of partials (argument x missing) of translate
            Niz duljine df.shape[1] tako da na i-tom mjestu stoji objekt ciji
            pozivi cine bijekciju izmedu jedinstvenih vrijednosti u i-tom stupcu
            tablice df i skupa {0, 1, ..., n - 1} (tipa int) gdje je n broj
            jedinstvenih vrijednosti u i-tom stupcu tablice df.

        Returns
        -------
        X : (K, N) array
            Povratna vrijednost je realna matrica dimenzija K x N gdje je K
            zbroj sim.shape[0] po svim sim u simplices, a N jednak df.shape[0].
            Stupci matrice X predstavljaju konkatenaciju pridruzenih vrhova
            (po objektima iz translators) simpleksa kategorickim vrijednostima
            u retcima tablice df istim redom.

        """

        assert issubclass(cls, RS_PCA)

        # Izracunaj dimenzije LRSV matrice.
        K = int(_np.sum([sim.shape[0] for sim in iter(simplices)]))
        N = int(df.shape[0])

        try:
            del sim
        except (NameError, UnboundLocalError):
            pass

        # Konkatenacijom izracunaj trazenu LRSV matricu.
        X = _np.concatenate(
            tuple(
                _np.concatenate(
                    tuple(
                        simplices[i][
                            :,
                            translators[i](df.loc[a, df.columns[i]])
                        ]
                            for i in iter(range(int(df.shape[1])))
                    )
                ).reshape((K, 1)) for a in iter(df.index)
            ),
            axis = 1
        ).copy(order = 'F')

        try:
            del i
        except (NameError, UnboundLocalError):
            pass
        try:
            del a
        except (NameError, UnboundLocalError):
            pass

        del N
#       del df

        # Vrati izracunatu LRSV matricu.
        return X

    @classmethod
    def _lrsv_cov (
        cls,
        df,
        simplices,
        translators,
        save_temp = False,
        temp_file_name_base = str(),
        n_simultaneous_instances = 1
    ):
        """
        Dohvati kovarijacijsku matricu LRSV matrice tablice df.

        Arguments
        ---------
        df : DataFrame
            Tablica kategorickih stupaca.  U stupcima se ne bi smjela pojaviti
            nedefinirana vrijednost (NaN).

        simplices : sequence of returns of simplex
            Niz duljine df.shape[1] tako da na i-tom mjestu stoji matrica
            koordinata regularnog (n - 1)-simpleksa gdje je n broj jedinstvenih
            vrijednosti u i-tom stupcu tablice df.

        translators : sequence of partials (argument x missing) of translate
            Niz duljine df.shape[1] tako da na i-tom mjestu stoji objekt ciji
            pozivi cine bijekciju izmedu jedinstvenih vrijednosti u i-tom stupcu
            tablice df i skupa {0, 1, ..., n - 1} (tipa int) gdje je n broj
            jedinstvenih vrijednosti u i-tom stupcu tablice df.

        save_temp : boolean, optional
            Ako True, dijelovi LRSV matrice tablice df spremaju se u datoteke
            tako da se u izbjegne "pretrpavanje" radne memorije prevelikom
            matricom odjednom.  Ako False, argumenti temp_file_name_base i
            n_simultaneous_instances se ignoriraju.  Zadana vrijednost je False.

        temp_file_name_base : str, optional
            Bazno ime datoteka za spremanje dijelova LRSV matrice tablice df.
            Na bazno ime nadodaje se '_{}.npy', gdje {} predstavlja indeks (od 0
            do df.shape[0]) prvog retka kojem pripada prvi stupac fragmenta LRSV
            matrice.  Zadana vrijednost je ''.

        n_simultaneous_instances : int, optional
            Najveci broj stupaca LRSV matrice tablice df kojim se u danom
            trenutku racuna.  Zadana vrijednost je 1.

        Returns
        -------
        A : (K, K) array
            Kovarijacijska matrica LRSV stupaca tablice df gdje je K zbroj
            sim.shape[0] po svim sim u simplices.

        """

        assert issubclass(cls, RS_PCA)

        # Izracunaj dimenzije LRSV matrice.
        K = int(_np.sum([sim.shape[0] for sim in iter(simplices)]))
        N = int(df.shape[0])

        try:
            del sim
        except (NameError, UnboundLocalError):
            pass

        # Ovisno o vrijednosti same_temp, izracunaj LRSV kovarijacijsku matricu.
        A = None
        if save_temp:
            # U varijabli X_mean prvo ce se racnati suma svih stupaca LRSV
            # matrice paralelno s racunanjem fragmenata LRSV matrice i njihovog
            # spremanja u vanjsku memoriju, a zatim ce se X_mean podijeliti s N.
            # Konacno ce se LRSV Kovarijacijska matrica A (pomnozena s N)
            # izracunati ucitavanjem fragmenata LRSV matrice i sumiranjem
            # odgovarajucih produkata.

            del save_temp

            X_mean = _np.zeros((K, 1), dtype = float, order = 'F')

            for n in iter(range(0, N, n_simultaneous_instances)):
                X = RS_PCA._lrsv(
                    df.iloc[n:n + n_simultaneous_instances],
                    simplices,
                    translators
                )

                X_mean += X.sum(axis = 1).reshape((K, 1))

                _np.save('{0:s}_{1:d}.npy'.format(temp_file_name_base, n), X)

                del X

            X_mean /= N

            try:
                del n
            except (NameError, UnboundLocalError):
                pass

            del df
            del simplices
            del translators

            A = _np.zeros((K, K), dtype = float, order = 'F')
            for n in iter(range(0, N, n_simultaneous_instances)):
                X = _np.load('{0:s}_{1:d}.npy'.format(temp_file_name_base, n))
                X -= X_mean

                A += _np.sum(
                    [
                        _np.matmul(
                            X[:, i].reshape((K, 1)),
                            X[:, i].reshape((K, 1)).T
                        ) for i in iter(range(int(X.shape[1])))
                    ],
                    axis = 0
                )

                try:
                    del i
                except (NameError, UnboundLocalError):
                    pass

                del X

            try:
                del n
            except (NameError, UnboundLocalError):
                pass

            del X_mean

            del temp_file_name_base
            del n_simultaneous_instances
        else:
            # U varijabli X bit ce zapisana LRSV matrica, a zatim ce se
            # odgovarajucom sumom produkata izracunati LRSV kovarijacijska
            # matrica A (pomnozena s N).

            del save_temp
            del n_simultaneous_instances
            del temp_file_name_base

            X = RS_PCA._lrsv(df, simplices, translators)

            del df
            del simplices
            del translators

            A = _np.sum(
                [
                    _np.matmul(
                        X[:, i].reshape((K, 1)),
                        X[:, i].reshape((K, 1)).T
                    ) for i in iter(range(int(X.shape[1])))
                ],
                axis = 0
            ).copy(order = 'F')

            try:
                del i
            except (NameError, UnboundLocalError):
                pass

            del X

        del K

        # Podijeli A s N.
        A /= N

        del N

        return A

    def __new__ (cls):
        assert issubclass(cls, RS_PCA)

        instance = super(RS_PCA, cls).__new__(cls)

        instance._shape = (0, 0, (0, 0), tuple())
        instance._columns = tuple()
        instance._categories = tuple()
        instance._indices = tuple()
        instance._simplices = tuple()
        instance._translators = tuple()

        instance._A = _np.zeros((0, 0), dtype = float, order = 'F')
        instance._L = _np.zeros((0, 0), dtype = float, order = 'F')
        instance._cov = _np.zeros((0, 0), dtype = float, order = 'F')

        instance._explained_variance = _np.zeros(
            0,
            dtype = complex,
            order = 'F'
        )
        instance._explained_variance_ratios = _np.zeros(
            0,
            dtype = float,
            order = 'F'
        )
        instance._components = _np.zeros((0, 0), dtype = complex, order = 'F')

        return instance

    def __init__ (self):
        assert isinstance(self, RS_PCA)

        self.reset()

    def reset (self):
        """
        Ponisti sve spremljene rezultate eventualne prethodne RS-PCA.

        Returns
        -------
        self : RS_PCA
            Povratna vrijednost je self.

        """

        assert isinstance(self, RS_PCA)

        # Postavi sve atribute na prazne vrijednosti.

        self._shape = (0, 0, (0, 0), tuple())
        self._columns = tuple()
        self._categories = tuple()
        self._indices = tuple()
        self._simplices = tuple()
        self._translators = tuple()

        self._A = _np.zeros((0, 0), dtype = float, order = 'F')
        self._L = _np.zeros((0, 0), dtype = float, order = 'F')
        self._cov = _np.zeros((0, 0), dtype = float, order = 'F')

        self._explained_variance = _np.zeros(0, dtype = complex, order = 'F')
        self._explained_variance_ratios = _np.zeros(
            0,
            dtype = float,
            order = 'F'
        )
        self._components = _np.zeros((0, 0), dtype = complex, order = 'F')

        # Vrati self.
        return self

    def fit (
        self,
        df,
        save_temp = None,
        temp_file_name_base = None,
        n_simultaneous_instances = None
    ):
        """
        Provedi RS-PCA na tablici df i spremi rezultate u atribute objekta self.

        Prije RS-PCA poziva se metoda RS_PCA.reset stoga je nemoguce u istom
        objektu spremiti rezultate dvije ili vise RS-PCA (za svaku RS-PCA
        potrebno je kreirati zasebni objekt klase RS_PCA).

        Arguments
        ---------
        df : DataFrame
            Tablica kategorickih stupaca.  Iz tablice se prije analize brisu
            svi retci u kojima postoji barem jedna nedefinirana vrijednost
            (NaN) i takva "prociscena" tablica ne smije biti prazna.  Za
            tretiranje i nedefiniranih vrijednosti kao zasebnih kategorickih
            vrijednosti moguce je nedefinirane vrijednosti zamijeniti novim
            vrijednostima (na primjer, 'NaN').

        save_temp : None or boolean, optional
            Ako True, dijelovi LRSV matrice tablice df (ako je redaka tablice df
            strogo vise od n_simultaneous_instances) spremaju se u datoteke tako
            da se u izbjegne "pretrpavanje" radne memorije prevelikom matricom u
            istom trenutku.  Ako False, argumenti temp_file_name_base i
            n_simultaneous_instances se ignoriraju.  None automatski zakljucuje
            vrijednost tako da se uzima
                >>> save_temp = df.shape[0] > n_simultaneous_instances
            pri cemu su iz tablice df vec izbaceni retci u kojima se pojavljuje
            barem jedna nedefinirana vrijednost (NaN), a
            n_simultaneous_instances je sanirana u slucaju da je dana vrijednost
            None (v. opis argumenta n_simultaneous_instances).  Zadana
            vrijednost je None.

        temp_file_name_base : None or str, optional
            Bazno ime datoteka za spremanje dijelova LRSV matrice tablice df.
            Na bazno ime nadodaje se '_{}.npy', gdje {} predstavlja indeks (od 0
            do df.shape[0]) prvog retka kojem pripada prvi stupac fragmenta LRSV
            matrice.  Ako None, generira se slucajno odabrani naziv od 6
            znakova (slova engleske abecede, velika i mala, i znamenke od 0 do
            9) s vremenskim zigom poziva funkcije.  Zadana vrijednost je None.

        n_simultaneous_instances : None or int, optional
            Najveci broj stupaca LRSV matrice tablice df kojim se u danom
            trenutku racuna.  None automatski zakljucuje vrijednost da se cijela
            LRSV matrica racuna odjednom, to jest, u tom se slucaju uzima
                >>> n_simultaneous_instances = df.shape[0]
            pri cemu su iz df vec izbaceni retci u kojima se pojavljuje barem
            jedna nedefinirana vrijednost (NaN).  Zadana vrijednost je None.

        Returns
        -------
        self : RS_PCA
            Povratna vrijednost je self.

        """

        assert isinstance(self, RS_PCA)

        # Provjeri i saniraj sve argumente.

        assert isinstance(
            df,
            (_pd.core.frame.DataFrame, _pd.core.series.Series)
        )

        assert (
            save_temp is None or
            isinstance(save_temp, _six.integer_types) or
            isinstance(
                save_temp,
                (
                    bool,
                    _np.bool,
                    _Boolean,
                    _np.integer,
                    _Integer
                )
            )
        )
        assert (
            temp_file_name_base is None or
            isinstance(temp_file_name_base, _six.string_types) or
            isinstance(temp_file_name_base, _six.text_type) or
            isinstance(temp_file_name_base, (_np.str_, _np.unicode_))
        )
        assert (
            n_simultaneous_instances is None or
            isinstance(n_simultaneous_instances, _six.integer_types) or
            isinstance(
                n_simultaneous_instances,
                (
                    bool,
                    _np.bool,
                    _Boolean,
                    _np.integer,
                    _Integer
                )
            )
        )

        if isinstance(df, _pd.core.series.Series):
            df = df.to_frame().T

        df = df.loc[_pd.notnull(df).all(axis = 1)]

        assert bool(df.size)

        n_simultaneous_instances = _copy.deepcopy(
            int(
                df.shape[0] if n_simultaneous_instances is None
                    else n_simultaneous_instances
            )
        )

        temp_file_name_base = (
            '{0:s}_{1:s}'.format(
                str().join(
                    _random.choice(_string.ascii_letters + _string.digits)
                        for i in range(6)
                ),
                _datetime.datetime.now().strftime('%y%m%d%H%M%S')
            ) if temp_file_name_base is None
                else _copy.deepcopy(str(temp_file_name_base))
        )

        try:
            del i
        except (NameError, UnboundLocalError):
            pass

        save_temp = (
            (int(df.shape[0]) > n_simultaneous_instances) if (
                save_temp is None or
                save_temp
            ) else _copy.deepcopy(bool(save_temp))
        )

        assert n_simultaneous_instances > 0

        # Ponisti sve atribute.
        self.reset()

        # Deduciraj sve stupce.
        self._columns = tuple(_copy.deepcopy(col) for col in df.columns)

        try:
            del col
        except (NameError, UnboundLocalError):
            pass

        # Deduciraj sve kategorije odnosno njihove vrijednosti.
        self._categories = tuple(
            _pd.api.types.CategoricalDtype(
                categories = _copy.deepcopy(df[col].unique().sort_values()),
                ordered = False
            ) for col in iter(self._columns)
        )

        try:
            del col
        except (NameError, UnboundLocalError):
            pass

        # Izracunaj regularne simplekse za sve stupce.
        self._simplices = tuple(
            RS_PCA.simplex(int(cat.categories.size - 1))
                for cat in iter(self._categories)
        )

        try:
            del cat
        except (NameError, UnboundLocalError):
            pass

        # Izracunaj indekse stupaca tablice u rezultantnim matricama/vektorima.
        self._indices = tuple(
            _copy.deepcopy(int(i)) for i in iter(
                _np.cumsum(
                    [0] + [
                        sim.shape[0]
                            for sim in iter(self._simplices)
                    ]
                )
            )
        )

        try:
            del i
        except (NameError, UnboundLocalError):
            pass
        try:
            del sim
        except (NameError, UnboundLocalError):
            pass

        # Generiraj sve prevoditeljske objekte.
        self._translators = tuple(
            _copy.deepcopy(
                _functools.partial(
                    RS_PCA.simplex,
                    _copy.deepcopy(cat.categories)
                )
            ) for cat in iter(self._categories)
        )

        try:
            del cat
        except (NameError, UnboundLocalError):
            pass

        # Izracunaj, osim na indeksu 2, atribut self._shape.

        self._shape = (
            _copy.deepcopy(len(self._columns)),
            0,
            (0, 0),
            _copy.deepcopy(
                tuple(int(cat.categories.size) for cat in self._categories)
            )
        )

        try:
            del cat
        except (NameError, UnboundLocalError):
            pass

        self._shape = (
            self._shape[0],
            _copy.deepcopy(int(_np.sum(self._shape[3]))),
            (0, 0),
            self._shape[3]
        )

        # Izracunaj LRSV kovarijacijsku matricu.
        self._A = RS_PCA._lrsv_cov(
            df,
            self._simplices,
            self._translators,
            save_temp,
            temp_file_name_base,
            n_simultaneous_instances
        )

#       del df
        del save_temp
        del n_simultaneous_instances
        del temp_file_name_base

        # Izracunaj maksimizirajuce matrice L i kovarijance stupaca.

        self._L = _np.zeros(
            self._A.shape,
            dtype = float,
            order = 'F'
        )
        self._cov = _np.zeros(
            (self._shape[0], self._shape[0]),
            dtype = float,
            order = 'F'
        )

        for i in iter(range(self._shape[0])):
            # Prvo se racunaju L_{i, i} i sigma_{i, i}, a zatim se, za svaki
            # j > i racunaju L_{i, j} i sigma_{i, j} od kojih se onda,
            # transponiranjem L_{i, j} i doslovnim prijepisom sigma_{i, j},
            # racunaju L{j, i} i sigma_{j, i}.  L i sigma racunaju se SVD
            # odgovarajucih pod-blok-matrica matrice A.

            try:
                U, S, V = _np.linalg.svd(
                    self._A[
                        self._indices[i]:self._indices[i + 1],
                        self._indices[i]:self._indices[i + 1]
                    ],
                    full_matrices = False
                )

                self._L[
                    self._indices[i]:self._indices[i + 1],
                    self._indices[i]:self._indices[i + 1]
                ] = _np.matmul(U, V)
                self._cov[i, i] = S.sum()

                del U
                del S
                del V
            except (_np.linalg.linalg.LinAlgError):
                self._cov[i, i] = 0

            for j in iter(range(i + 1, self._shape[0])):
                try:
                    U, S, V = _np.linalg.svd(
                        self._A[
                            self._indices[i]:self._indices[i + 1],
                            self._indices[j]:self._indices[j + 1]
                        ],
                        full_matrices = False
                    )

                    self._L[
                        self._indices[i]:self._indices[i + 1],
                        self._indices[j]:self._indices[j + 1]
                    ] = _np.matmul(U, V)
                    self._cov[i, j] = S.sum()

                    del U
                    del S
                    del V

                    self._L[
                        self._indices[j]:self._indices[j + 1],
                        self._indices[i]:self._indices[i + 1]
                    ] = self._L[
                        self._indices[i]:self._indices[i + 1],
                        self._indices[j]:self._indices[j + 1]
                    ].T
                    self._cov[j, i] = self._cov[i, j]
                except (_np.linalg.linalg.LinAlgError):
                    self._cov[i, j] = 0
                    self._cov[j, i] = 0

            try:
                del j
            except (NameError, UnboundLocalError):
                pass

        try:
            del i
        except (NameError, UnboundLocalError):
            pass

        # Izracunaj objasnjene varijance i glavne komponente.  Atributi
        # self._explained_variance, self._explained_variance_ratios i
        # self._components poredani su tako da objasnjenost varijance opada
        # od prvog prema zadnjemu.  Dodatno je garantirana orijentiranost
        # glavnih komponenti tako da prva koordinata koja je "osjetno" razlicita
        # od 0, gledajuci od dijagonale prema dolje pa od vrha do dijagonale,
        # ima pozitivnu vrijednost (ako je kompleksna, da pozitivnu vrijednost
        # ima apsolutni maksimum od njezinih realnog i kompleksnog dijela ili,
        # ako su jednaki, realni dio.

        # Izracunaj svojstvene vrijednosti i svojstvene vektore LRSV
        # kovarijacijske matrice.

        self._explained_variance, self._components = _np.linalg.eig(self._A)

        self._explained_variance_ratios = _np.abs(self._explained_variance)

        # Poredaj nizove self._explained_variance,
        # self._explained_variance_ratios i self._components.

        I_sort = _np.flipud(_np.argsort(self._explained_variance_ratios))

        with _warnings.catch_warnings():
            _warnings.filterwarnings('error')
            with _np.errstate(invalid = 'raise'):
                try:
                    self._explained_variance = _np.array(
                        self._explained_variance[I_sort],
                        dtype = float,
                        copy = True,
                        order = 'F'
                    )
                except (
                    TypeError,
                    ValueError,
                    RuntimeWarning,
                    _np.ComplexWarning
                ):
                    self._explained_variance = _np.array(
                        self._explained_variance[I_sort],
                        dtype = complex,
                        copy = True,
                        order = 'F'
                    )

                self._explained_variance_ratios = _np.array(
                    self._explained_variance_ratios[I_sort],
                    dtype = float,
                    copy = True,
                    order = 'F'
                )
                try:
                    self._explained_variance_ratios /= (
                        self._explained_variance_ratios.sum()
                    )
                except (TypeError, ValueError, RuntimeWarning):
                    self._explained_variance_ratios[:] = 0

                try:
                    self._components = _np.array(
                        self._components[:, I_sort],
                        dtype = float,
                        copy = True,
                        order = 'F'
                    )
                except (
                    TypeError,
                    ValueError,
                    RuntimeWarning,
                    _np.ComplexWarning
                ):
                    self._components = _np.array(
                        self._components[:, I_sort],
                        dtype = complex,
                        copy = True,
                        order = 'F'
                    )

        del I_sort

        # Orijentiraj vektore u self._components.
        with _warnings.catch_warnings():
            _warnings.filterwarnings('error')
            with _np.errstate(invalid = 'raise'):
                for i in iter(range(self._shape[2][0])):
                    j = i

                    abs_comp = _np.roll(_np.abs(self._components[:, i]), -i)

                    J = None
                    try:
                        J = _np.where(
                            ~_np.isclose(1, 1 + abs_comp / abs_comp.max())
                        )

                        j = (int(J[0][0]) + i) % self._shape[2][1]
                    except (IndexError, KeyError, ValueError, RuntimeWarning):
                        pass

                    del J

                    del abs_comp

                    if (
                        self._components[j, i].real
                            if (
                                _np.abs(self._components[j, i].real) >=
                                _np.abs(self._components[j, i].imag)
                            ) else self._components[j, i].imag
                    ) < 0:
                        self._components[:, i] = -self._components[:, i]

                    del j

                try:
                    del i
                except (NameError, UnboundLocalError):
                    pass

        # Izracunaj do kraja atribut self._shape.
        self._shape = (
            self._shape[0],
            self._shape[1],
            _copy.deepcopy(
                (
                    int(self._components.shape[0]),
                    int(self._components.shape[1])
                )
            ),
            self._shape[3]
        )

        # Vrati self.
        return self

    def reinterpret (self, n_first = None):
        """
        Reinterpretiraj glavne komponente.

        Arguments
        ---------
        n_first : None or int, optional
            Broj prvih glavnih komponenti koje se reinterpretiraju.  Ako None,
            reinterpretiraju se sve glavne komponente.  Zadana vrijednost je
            None.

        Returns
        coeff : (K, n_first) array
            Matrica ciji stupci predstavljaju koeficijente pridruzene radij-
            vektorima vrhova simpleksa pridruzenih kategorickim vrijednostima
            iz tablice nad kojom je pozvana metoda RS_PCA.fit (K je zbroj
            sim.shape[0] po svim sim u self.simplices).  U tom smislu i-ti
            stupac matrice coeff zadaje koeficijente u linearnoj kombinaciji
            spomenutih radij-vektora koja rezultira i-tom glavnom komponentom.
            Ako je n_first None, povratna matrica je dimenzija K x K.

        values : (K, n_first) DataFrame
            Tablica koja tumaci kojim kategorickim vrijednostima odgovaraju
            koeficijenti u povratnoj matrici coeff (K je zbroj sim.shape[0]
            po svim sim u self.simplices).  Na mjestu (i, j) povratne tablice
            (values.iloc[i, j]) stoji kategoricka vrijednost tako da coeff[i, j]
            predstavlja koeficijent za radij-vektor vrha simpleksa koji odgovara
            toj kategorickoj vrijednosti. Ako je n_first None, povratna matrica
            je dimenzija K x K.

        """

        assert isinstance(self, RS_PCA)

        # Provjeri i saniraj argument.

        assert (
            n_first is None or
            isinstance(n_first, _six.integer_types) or
            isinstance(
                n_first,
                (
                    bool,
                    _np.bool,
                    _Boolean,
                    _np.integer,
                    _Integer
                )
            )
        )

        if n_first is None:
            n_first = self._shape[2][1]

        n_first = _copy.deepcopy(int(n_first))

        assert n_first > 0 and n_first <= self._shape[2][1]

        # Inicijaliziraj varijable coeff i values na prazne vrijednosti
        # odgovarajucih velicina.
        coeff = _np.zeros(
            (self._shape[2][0], n_first),
            dtype = self._components.dtype,
            order = 'F'
        )
        values = [
            [None for i in iter(range(int(self._shape[2][0])))]
                for j in iter(range(n_first))
        ]

        try:
            del i
        except (NameError, UnboundLocalError):
            pass
        try:
            del j
        except (NameError, UnboundLocalError):
            pass

        # Za svaku glavnu komponentu pronadi odabir vrijednosti iz svakog
        # kategorickog stupca tako da padajuce uredeni konacni niz
        # (|a_{i, 0}|, |a_{i, 1}|, ..., |a_{i, m_i - 2}|) koeficijenata za i-ti
        # kategoricki stupac bude najveci po leksikografskom uredaju.  Paralelno
        # spremaj odgovarajuce kateogircke vrijednosti kojima koeficijeni
        # odgovaraju.
        for i in iter(range(self._shape[0])):
            if not self._simplices[i].size:
                continue

            choice = _np.array(
                [False] + int(self._simplices[i].shape[1] - 1) * [True],
                dtype = bool,
                order = 'F'
            )
            choice = _np.array(
                [
                    _np.roll(choice, -r)
                        for r in iter(range(int(choice.size)))
                ],
                dtype = bool,
                order = 'C'
            )

            try:
                del r
            except (NameError, UnboundLocalError):
                pass

            orig_values = _copy.deepcopy(self._categories[i].categories)

            for j in iter(range(n_first)):
                best_values = _copy.deepcopy(orig_values)
                best_coeff = -_np.inf * _np.ones(
                    self._simplices[i].shape[0],
                    dtype = float,
                    order = 'F'
                )
                best_abs_coeff = best_coeff

                for k in iter(range(int(choice.shape[0]))):
                    try:
                        choice_coeff = _np.linalg.solve(
                            self._simplices[i][:, choice[k, :]],
                            self._components[
                                self._indices[i]:self._indices[i + 1],
                                j
                            ]
                        )

                        abs_choice_coeff = _np.abs(choice_coeff)

                        I_sort = _np.flipud(_np.argsort(abs_choice_coeff))

                        choice_values = orig_values[choice[k, :]][I_sort].copy()
                        choice_coeff = choice_coeff[I_sort].copy(order = 'F')
                        abs_choice_coeff = (
                            abs_choice_coeff[I_sort].copy(order = 'F')
                        )

                        del I_sort

                        if abs_choice_coeff.tolist() > best_abs_coeff.tolist():
                            best_values = choice_values
                            best_coeff = choice_coeff
                            best_abs_coeff = abs_choice_coeff

                        del choice_values
                        del choice_coeff
                        del abs_choice_coeff
                    except (_np.linalg.linalg.LinAlgError):
                        pass

                try:
                    del k
                except (NameError, UnboundLocalError):
                    pass

                del best_abs_coeff

                coeff[self._indices[i]:self._indices[i + 1], j] = best_coeff
                values[j][
                    self._indices[i]:self._indices[i + 1]
                ] = _copy.deepcopy(best_values.tolist())

                del best_values
                del best_coeff

            try:
                del j
            except (NameError, UnboundLocalError):
                pass

            del choice
            del orig_values

        try:
            del i
        except (NameError, UnboundLocalError):
            pass

        del n_first

        # Pretvori values u objekt klase DataFrame.
        values = _pd.DataFrame(
            data = values,
            columns = sum(
                iter(
                    [
                        (i, _copy.deepcopy(self._columns[i]), j)
                            for j in iter(range(self._shape[3][i] - 1))
                    ] for i in iter(range(self._shape[0]))
                ),
                []
            )
        ).T.copy(deep = True)

        try:
            del i
        except (NameError, UnboundLocalError):
            pass
        try:
            del j
        except (NameError, UnboundLocalError):
            pass

        # Vrati izracunate koeficijente i kateogircke vrijednosti
        # reinterpretacije.
        return (coeff, values)

    def transform (self, df, n_first = None):
        """
        Transformiraj tablicu kategorickih vrijednosti df u koordinate glavnih komponenti.

        Arguments
        ---------
        df : DataFrame
            Tablica kategorickih stupaca.  Stupci tablice df i vrijednosti
            koji se u njima pojavljuju trebali bi odgovarati formatu tablice
            nad kojom je bila vrsena RS-PCA pozivom metode RS_PCA.fit (ako se
            metoda RS_PCA.fit, na primjer, pozvala nad tablicom ciji su stupci
            'A', 'B' i u kojima se pojavljuju jedinstvene vrijednosti
            'A': ['a1', 'a2', 'a3'] i 'B': ['b1', 'b2', 'b3', 'b4'], onda bi i
            tablica df trebala imati samo dva stupca i to takva da se u prvom
            stupcu pojavljuju samo vrijednosti iz skupa {'a1', 'a2', 'a3'}, a u
            drugom iz skupa {'b1', 'b2', 'b3', 'b4'}).

        n_first : None or int, optional
            Broj prvih glavnih komponenti u cijim se koordinatama tablica df
            transformira.  Ako None, racunaju se koordinate po svim glavnim
            komponentama.  Zadana vrijednost je None.

        Returns
        -------
        Y : (df.shape[0], n_first) array
            Matrica ciji retci odgovaraju redom koordinatama kategorickih redaka
            tablice df u projekciji na prvih n_first glavnih kompone. Ako je
            n_first None, povratna matrica je dimenzija df_shape[0] x K gdje je
            K zbroj sim.shape[0] po svim sim u self.simplices.

        """

        assert isinstance(self, RS_PCA)

        # Provjeri i saniraj sve argumente.

        assert isinstance(
            df,
            (_pd.core.frame.DataFrame, _pd.core.series.Series)
        )

        assert (
            n_first is None or
            isinstance(n_first, _six.integer_types) or
            isinstance(
                n_first,
                (
                    bool,
                    _np.bool,
                    _Boolean,
                    _np.integer,
                    _Integer
                )
            )
        )

        if isinstance(df, _pd.core.series.Series):
            df = df.to_frame().T

        # Ako je n_first None, pomnozi LRSV matricu tablice df sa
        # self._components i vrati rezultat.
        if n_first is None:
            return _np.matmul(
                RS_PCA._lrsv(df, self._simplices, self._translators).T,
                self._components
            ).copy(order = 'F')

        n_first = _copy.deepcopy(int(n_first))

        assert n_first > 0 and n_first <= self._shape[2][1]

        # Pomnozi LRSV matricu tablice df sa self._components[:, :n_first] i
        # vrati rezultat.
        return _np.matmul(
            RS_PCA._lrsv(df, self._simplices, self._translators).T,
            self._components[:, :n_first]
        ).copy(order = 'F')

    def __repr__ (self):
        assert isinstance(self, RS_PCA)

        return '<{0:s}: {1:s}>'.format(
            self.__class__.__name__,
            repr(self._columns)
        )

    def __copy__ (self):
        assert isinstance(self, RS_PCA)

        instance = RS_PCA()

        instance._shape = self._shape
        instance._columns = self._columns
        instance._categories = self._categories
        instance._indices = self._indices
        instance._simplices = self._simplices
        instance._translators = self._translators

        instance._A = self._A
        instance._L = self._L
        instance._cov = self._cov

        instance._explained_variance = self._explained_variance
        instance._explained_variance_ratios = self._explained_variance_ratios
        instance._components = self._components

        return instance

    def __deepcopy__ (self, memo = dict()):
        assert isinstance(self, RS_PCA)

        instance = RS_PCA()

        instance._shape = _copy.deepcopy(self._shape, memo)
        instance._columns = _copy.deepcopy(self._columns, memo)
        instance._categories = _copy.deepcopy(self._categories, memo)
        instance._indices = _copy.deepcopy(self._indices, memo)
        instance._simplices = _copy.deepcopy(self._simplices, memo)
        instance._translators = _copy.deepcopy(self._translators, memo)

        instance._A = _copy.deepcopy(self._A, memo)
        instance._L = _copy.deepcopy(self._L, memo)
        instance._cov = _copy.deepcopy(self._cov, memo)

        instance._explained_variance = _copy.deepcopy(
            self._explained_variance,
            memo
        )
        instance._explained_variance_ratios = _copy.deepcopy(
            self._explained_variance_ratios,
            memo
        )
        instance._components = _copy.deepcopy(self._components, memo)

        return instance

    @property
    def shape (self):
        """
        Tuple (n, M, (K, K), (m_0, m_1, ..., m_{n - 1})).

        Vrijednosti:
            n   --  broj analiziranih stupaca,
            M   --  zbroj m_0 + m_1 + ... + m_{n - 1},
            K   --  zbroj sim.shape[0] po svim sim u self.simplices,
            m_i --  broj jedinstvenih vrijednosti u i-tom analiziranom stupcu.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._shape)

    @property
    def columns (self):
        """
        Tuple (col_0, col_1, ..., col_{n - 1}).

        Vrijednosti:
            n   --  broj analiziranih stupaca,
            col_i   --  naziv i-tog analiziranog stupca.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._columns)

    @property
    def categories (self):
        """
        Tuple (pandas.CategoricalDtype([cat_{0, 0}, cat_{0, 1}, ..., cat_{0, m_0 - 1}]), pandas.CategoricalDtype([cat_{1, 0}, cat_{1, 1}, ..., cat_{1, m_1 - 1}]), ..., pandas.CategoricalDtype([cat_{n - 1, 0}, cat_{n - 1, 1}, ..., cat_{0, m_{n - 1} - 1}])).

        Vrijednosti:
            n   --  broj analiziranih stupaca,
            m_i --  broj jedinstvenih vrijednosti u i-tom analiziranom stupcu,
            cat_{i, j}  --  j-ta jedinstvena vrijednost u i-tom analiziranom
                            stupcu.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._categories)

    @property
    def indices (self):
        """
        Tuple (0, k_0, k_0 + k_1, ..., k_o + k_1 + ... + k_{n - 1}).

        Vrijednosti:
            n   --  broj analiziranih stupaca,
            k_i --  self.simplices[i].shape[0], to jest, m_i - 1 gdje je m_i
                    broj jedinstvenih vrijednosti u i-tom analiziranom stupcu.

        Za ekstrakciju podataka o i-tom analiziranom stupcu u povratnim
        vrijednostima metode RS_PCA.reinterpret korisno je koristiti
        self.indices i to na nacin
            >>> pca = RS_PCA
            >>> pca.fit(df)
            <RS_PCA: (...)>
            >>> coeff, values = pca.reinterpret()
            >>> coeff[indices[i]:indices[i + 1], j]
            array(...)
            >>> values.iloc[indices[i]:indices[i + 1], j]
            Series(...)
        Posljednje dvije naredbe vratit ce redom koeficijente i kategoricke
        vrijednosti koji se odnose na reinterpretaciju j-te glavne komponente
        preko i-tog analiziranog stupca.  Slicno se self.indices moze koristiti
        za dohvacanje kovarijacijskih matrica iz self.A odnosno maksimizirajucih
        matrica iz self.L.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._indices)

    @property
    def simplices (self):
        """
        Tuple ((m_0 - 1, m_0) array, (m_1 - 1, m_1) array, ..., (m_{n - 1} - 1, m_{n - 1}) array) kojemu su na i-tom mjestu koordinate vrhova regularnog (m_i - 1)-simpleksa u R^{m_i - 1}.

        Vrijednosti:
            n   --  broj analiziranih stupaca,
            m_i --  broj jedinstvenih vrijednosti u i-tom analiziranom stupcu.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._simplices)

    @property
    def translators (self):
        """
        Tuple (t_0, t_1, ..., t_{n - 1}).

        Vrijednosti:
            n   --  broj analiziranih stupaca,
            m_i --  broj jedinstvenih vrijednosti u i-tom analiziranom stupcu,
            t_i --  funkcijski objekt koji kategoricke vrijednosti iz i-tog
                    analiziranog stupca bijektivno (domena je skup svih
                    jedinstvenih vrijednosti u i-tom analiziranom stupcu)
                    preslikava u skup {0, 1, ..., m_i - 1}.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._translators)

    @property
    def A (self):
        """
        (K, K) array koji je LRSV kovarijacijska matrica analizirane tablice.

        Vrijednosti:
            K   --  zbroj sim.shape[0] po svim sim u self.simplices.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._A)

    @property
    def L (self):
        """
        (K, K) array koji je konkatenacija maksimizirajucih matrica za analiziranu tablicu.

        Vrijednosti:
            K   --  zbroj sim.shape[0] po svim sim u self.simplices.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._L)

    @property
    def cov (self):
        """
        (n, n) array koji je kovarijacijska matrica analizirane tablice.

        Vrijednosti:
            n   --  broj analiziranih stupaca.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._cov)

    @property
    def explained_variance (self):
        """
        (K,) array [var_0, var_1, ..., var_{K - 1}] koji predstavlja objasnjene varijance glavnih komponenti.

        Vrijednosti:
            K   --  zbroj sim.shape[0] po svim sim u self.simplices,
            var_i   --  objasnjena varijanca i-te glavne komponente.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._explained_variance)

    @property
    def explained_variance_ratios (self):
        """
        (K,) array [relvar_0, relvar_1, ..., relvar_{K - 1}] koji predstavlja udjele objasnjenih varijanci glavnih komponenti.

        Ako postoji vrijednost razlicita od 0 u self.explained_variance, onda
        je
            self.explained_variance ==
                np.abs(self.explained_variance) /
                np.sum(np.abs(self.explained_variance)).

        Vrijednosti:
            K   --  zbroj sim.shape[0] po svim sim u self.simplices,
            relvar_i    --  udio objasnjene varijance i-te glavne komponente u
                            odnosu na kompletnu varijabilnost podataka.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._explained_variance_ratios)

    @property
    def components (self):
        """
        (K, K) array koji po stupcima predstavlja koordinate normiranih vektora smjerova glavnih komponenti u R^K (array oblika (K, K)).

        Vrijednosti:
            K   --  zbroj sim.shape[0] po svim sim u self.simplices.

        """

        assert isinstance(self, RS_PCA)

        return _copy.deepcopy(self._components)
