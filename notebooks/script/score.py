# -*- coding: utf-8 -*-

"""
Skripta s klasom za racunanje tocnosti prediktivnog modela.

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

# Seaborn.
import seaborn as _sns

# Definicija klase Score.
class Score (object):
    """
    Ocjenjivac tocnosti prediktivnog modela (binarnog klasifikatora).

    Arguments
    ---------
    n_positives : int
        Ukupni broj pozitivnih instanci u skupu svih instanci.

    n_negatives : int
        Ukupni broj negativnih instanci u skupu svih instanci.

    """

    def __new__ (cls, n_positives, n_negatives):
        instance = super(Score, cls).__new__(cls)

        instance._n_positives = _copy.deepcopy(int(n_positives))
        instance._n_negatives = _copy.deepcopy(int(n_negatives))

        instance._true_positives = 0
        instance._false_positives = instance._n_positives
        instance._true_negatives = 0
        instance._false_negatives = instance._n_negatives

        instance._accuracy = 0.0
        instance._precision = 0.0
        instance._recall = 0.0
        instance._specificity = 0.0

        return instance

    def __init__ (self, n_positives, n_negatives):
        pass

    def reset (self):
        """
        Ponisti spremljenu eventualnu prethodnu ocjenu.

        Arguments
        ---------
        true_positives : int
            Broj tocno predvidenih pozitivnih instanci.

        true_negatives : int
            Broj tocno predvidenih negativnih instanci.

        Returns
        -------
        self : Score
            Povratna vrijednost je self.

        """

        # Ponisti ocjenu.

        self._true_positives = 0
        self._false_positives = self._n_positives
        self._true_negatives = 0
        self._false_negatives = self._n_negatives

        self._accuracy = 0.0
        self._precision = 0.0
        self._recall = 0.0
        self._specificity = 0.0

        return self

    def score (
        self,
        true_positives,
        false_positives,
        true_negatives,
        false_negatives
    ):
        """
        Ocijeni prediktivni model.

        Prije ocjenjivanja poziva se metoda Score.reset stoga je nemoguce u
        istom objektu spremiti ocjene dva ili vise prediktivna modela (za svaki
        model potrebno je kreirati zasebni objekt klase Score).

        Ne provjerava se je li
            true_positives +
                false_positives +
                true_negatives +
                false_negatives ==
            == n_positives + n_negatives.

        Arguments
        ---------
        true_positives : int
            Broj tocno predvidenih pozitivnih instanci.

        false_positives : int
            Broj krivo predvidenih pozitivnih instanci.

        true_negatives : int
            Broj tocno predvidenih negativnih instanci.

        false_negatives : int
            Broj krivo predvidenih negativnih instanci.

        Returns
        -------
        self : Score
            Povratna vrijednost je self.

        """

        # Ponisti ocjenu.
        self.reset()

        # Izracunaj novu ocjenu.

        self._true_positives = _copy.deepcopy(int(true_positives))
        self._false_positives = _copy.deepcopy(int(false_positives))
        self._true_negatives = _copy.deepcopy(int(true_negatives))
        self._false_negatives = _copy.deepcopy(int(false_positives))

        self._accuracy = (
            float(self._true_positives + self._true_negatives) /
            (self._n_positives + self._n_negatives)
        )
        self._precision = float(self._true_positives) / self._n_positives
        self._recall = (
            float(self._true_positives) /
            (self._true_positives + self._false_negatives)
        )
        self._specificity = (
            float(self._true_negatives) /
            (self._false_positives + self._true_negatives)
        )

        return self

    def F (self, beta = 1.0):
        """
        Izracunaj F_beta vrijednost ocjene.

        Arguments
        ---------
        beta : float, optional
            Parametar beta za F_beta ocjenu.  Zadana vrijednost je 1.0.

        Returns
        -------
        f : float
            F_beta vrijednost spremljene ocjene prediktivnog modela.

        """

        # Izracunaj beta^2.
        beta2 = _copy.deepcopy(float(beta)) ** 2

        # Izracunaj i vrati F_beta vrijednost ocjene.
        return (
            (1 + beta2) *
            self._precision *
            self._recall /
            (beta2 ** self._precision + self._recall)
        )

    def __repr__ (self):
        return (
            '<'
                '{name:s}: ('
                    'P: {P:d}, N: {N:d}; '
                    'TP: {TP:d}, FP: {FP:d}, TN: {TN:d}, FN: {FN:d}'
                ')'
            '>'.format(
                name = self.__class__.__name__,
                P = self._n_positives,
                N = self._n_negatives,
                TP = self._true_positives,
                FP = self._false_positives,
                TN = self._true_negatives,
                FN = self._false_negatives
            )
        )

    def __copy__ (self):
        instance = Score(self._n_positives, self._n_negatives)

        instance._true_positives = self._true_positives
        instance._false_positives = self._false_positives
        instance._true_negatives = self._true_negatives
        instance._false_negatives = self._false_negatives

        instance._accuracy = self._accuracy
        instance._precision = self._precision
        instance._recall = self._recall
        instance._specificity = self._specificity

        return instance

    def __deepcopy__ (self, memo = dict()):
        instance = Score(
            _copy.deepcopy(self._n_positives, memo),
            _copy.deepcopy(self._n_negatives, memo)
        )

        instance._true_positives = _copy.deepcopy(self._true_positives, memo)
        instance._false_positives = _copy.deepcopy(self._false_positives, memo)
        instance._true_negatives = _copy.deepcopy(self._true_negatives, memo)
        instance._false_negatives = _copy.deepcopy(self._false_negatives, memo)

        instance._accuracy = _copy.deepcopy(self._accuracy, memo)
        instance._precision = _copy.deepcopy(self._precision, memo)
        instance._recall = _copy.deepcopy(self._recall, memo)
        instance._specificity = _copy.deepcopy(self._specificity, memo)

        return instance

    @property
    def n_positives (self):
        """
        Ukupni broj pozitivnih instanci.

        """

        return _copy.deepcopy(self._n_positives)

    @property
    def n_negatives (self):
        """
        Ukupni broj negativnih instanci.

        """

        return _copy.deepcopy(self._n_negatives)

    @property
    def true_positives (self):
        """
        Broj tocno predvidenih pozitivnih instanci.

        """

        return _copy.deepcopy(self._true_positives)

    @property
    def false_positives (self):
        """
        Broj krivo predvidenih pozitivnih instanci.

        """

        return _copy.deepcopy(self._false_positives)

    @property
    def true_negatives (self):
        """
        Broj tocno predvidenih negativnih instanci.

        """

        return _copy.deepcopy(self._true_negatives)

    @property
    def false_negatives (self):
        """
        Broj krivo predvidenih negativnih instanci.

        """

        return _copy.deepcopy(self._false_negatives)

    @property
    def accuracy (self):
        """
        Tocnost prediktivnog modela.

        """

        return _copy.deepcopy(self._accuracy)

    @property
    def precision (self):
        """
        Preciznost prediktivnog modela.

        """

        return _copy.deepcopy(self._precision)

    @property
    def recall (self):
        """
        Odziv prediktivnog modela.

        """

        return _copy.deepcopy(self._recall)

    @property
    def specificity (self):
        """
        Specificnost prediktivnog modela.

        """

        return _copy.deepcopy(self._specificity)
